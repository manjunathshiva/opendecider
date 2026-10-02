// A fake `opendecider serve` and a fake OpenAI-compatible server, as fetch functions: no network, deterministic.

export interface Call {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
  redirect?: string;
}

/** A fake model: an option scores high when the state mentions its name or
 * description; a yes/no question is "yes" for "ignore" or
 * "disregard" in the state, else "no". */
export function fakeProbabilities(state: unknown, options: Record<string, unknown>): Record<string, number> {
  const text = (typeof state === "string" ? state : JSON.stringify(state)).toLowerCase();
  const attack = /ignore|disregard/.test(text);
  const yesNo = Object.keys(options).join() === "yes,no";
  const w = Object.entries(options).map(
    ([name, desc]) =>
      1 +
      9 *
        Number(
          yesNo
            ? (name === "yes") === attack
            : text.includes(name.toLowerCase()) || (typeof desc === "string" && text.includes(desc.toLowerCase())),
        ),
  );
  const z = w.reduce((a, b) => a + b, 0);
  return Object.fromEntries(Object.keys(options).map((n, i) => [n, w[i]! / z]));
}

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

function answerAll(state: unknown, questions: Record<string, { criteria: Record<string, unknown> }>) {
  return {
    model: "manjunathshiva/opendecider-small-td",
    answers: Object.fromEntries(
      Object.entries(questions).map(([k, q]) => {
        const p = fakeProbabilities(state, q.criteria);
        const top = Object.keys(p).reduce((a, b) => (p[b]! > p[a]! ? b : a));
        return [k, { type: "choice", choice: top, probabilities: p, confidence: p[top] }];
      }),
    ),
    usage: { input_tokens: 10, output_tokens: 0 },
  };
}

export interface FakeServe {
  fetch: typeof fetch;
  calls: Call[];
  /** Answer the next requests with these responses first (status codes or functions), then normally. */
  queue: (Response | (() => Response | Promise<Response>))[];
  name: string;
}

export function fakeServe(name = "manjunathshiva/opendecider-small-td"): FakeServe {
  const fake: FakeServe = { calls: [], queue: [], name, fetch: undefined as unknown as typeof fetch };
  fake.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    fake.calls.push({
      url,
      method: init?.method ?? "GET",
      headers: { ...(init?.headers as Record<string, string>) },
      body,
      redirect: init?.redirect,
    });
    if (init?.signal?.aborted) throw init.signal.reason;
    const next = fake.queue.shift();
    if (next) return typeof next === "function" ? next() : next;
    const path = new URL(url).pathname;
    if (path.endsWith("/v1/models")) return json({ models: [{ name: fake.name, kind: "small" }] });
    if (path.endsWith("/ready")) return json({ ready: true });
    if (path.endsWith("/v1/systemone")) return json(answerAll(body.state, body.questions));
    if (path.endsWith("/v1/systemone/batch")) {
      return json({ model: fake.name, results: body.states.map((s: unknown) => answerAll(s, body.questions)) });
    }
    return json({ detail: "Not Found" }, 404);
  }) as typeof fetch;
  return fake;
}

/** A fake OpenAI-compatible server (Ollama, LM Studio, vLLM) for the log-probability backend. */
export function fakeOpenAI() {
  const calls: Call[] = [];
  const queue: (Response | (() => Response))[] = [];
  const fetchFn = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, method: init?.method ?? "GET", headers: { ...(init?.headers as Record<string, string>) }, body });
    const next = queue.shift();
    if (next) return typeof next === "function" ? next() : next;
    if (url.endsWith("/models")) return json({ data: [] });
    const prompt: string = body.messages[1].content;
    const state = prompt.split("\n\nQuestion:")[0]!.replace("Input:\n", "");
    const letters = [...prompt.matchAll(/^([A-Z])\) ([^:\n]+)/gm)].map((m) => [m[1]!, m[2]!] as const);
    const p = fakeProbabilities(state, Object.fromEntries(letters.map(([, n]) => [n, null])));
    const top = letters.map(([l, n]) => ({ token: l, logprob: Math.log(p[n]!) }));
    return json({ choices: [{ logprobs: { content: [{ top_logprobs: top }] } }], usage: { prompt_tokens: 7 } });
  }) as typeof fetch;
  return { fetch: fetchFn, calls, queue };
}

export { json };
