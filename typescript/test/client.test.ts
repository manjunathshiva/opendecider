// The client against a fake opendecider serve and a fake OpenAI-compatible server: answers, batching, the HTTP rules
// (retries, timeouts, the fail-fast window, cancellation, credentials) and loading.
import { afterEach, describe, expect, it, vi } from "vitest";
import { Decider, InputError, ServerError, choice, load, noul, score } from "../src/index.js";
import { ServedBackend } from "../src/backends.js";
import { fakeOpenAI, fakeServe, json } from "./helpers.js";

const URL_ = "http://127.0.0.1:8000";
const TICKET = "Hi, we were billed twice for March. Please refund the duplicate today.";
const QS = {
  team: choice("Which team?", { billing: "invoices, refunds", tech: "bugs" }),
  urgent: noul("Does this need a reply today?"),
  mood: score("How upset?", ["calm", "annoyed", "angry"]),
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("opendecider serve by URL", () => {
  it("answers typed questions in one request, as named options", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    expect(model.name).toBe("manjunathshiva/opendecider-small-td");
    expect(model.meta).toMatchObject({ kind: "served", served_kind: "small", base_url: URL_ });
    const r = await model.systemOne(TICKET, QS);
    expect(r.model).toBe(fake.name);
    expect(r.answers.team).toMatchObject({ type: "choice", choice: "billing" });
    expect(r.answers.urgent).toMatchObject({ type: "noul" });
    expect(r.answers.mood).toMatchObject({ type: "score", legend: { 0: "calm", 1: "annoyed", 2: "angry" } });
    expect(r.usage).toEqual({ input_tokens: 10, output_tokens: 0 });
    expect(r.latency_ms).toBeGreaterThanOrEqual(0);
    const posts = fake.calls.filter((c) => c.method === "POST");
    expect(posts).toHaveLength(1);
    expect(posts[0]!.body).toEqual({
      state: TICKET,
      questions: {
        q0: { type: "choice", instructions: "Which team?", criteria: { billing: "invoices, refunds", tech: "bugs" } },
        q1: { type: "choice", instructions: "Does this need a reply today?", criteria: { yes: "Yes", no: "No" } },
        q2: { type: "choice", instructions: "How upset?", criteria: { 0: "calm", 1: "annoyed", 2: "angry" } },
      },
    });
  });

  it("sends JSON states as JSON and reports truncation", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    fake.queue.push(() =>
      json({
        answers: { q0: { probabilities: { billing: 0.9, tech: 0.1 }, truncated: true } },
        usage: { input_tokens: 5 },
      }),
    );
    const state = { subject: "Refund?", when: new Date(0) };
    const r = await model.systemOne(state, { team: QS.team });
    expect(fake.calls.at(-1)!.body).toMatchObject({ state: { subject: "Refund?", when: "1970-01-01T00:00:00.000Z" } });
    expect(r.answers.team!.truncated).toBe(true);
    expect(r.warnings).toEqual(["question 'team': the state was truncated to fit the model's input length"]);
  });

  it("batches states through the batch endpoint, split by count and size", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    const b = model.backend as ServedBackend;
    b.maxBatchStates = 2;
    const rs = await model.systemOneBatch(["refund please", "the app crashes (tech)", "billing", "tech"], {
      team: QS.team,
    });
    expect(rs.map((r) => (r.answers.team as { choice: string }).choice)).toEqual([
      "billing",
      "tech",
      "billing",
      "tech",
    ]);
    expect(
      fake.calls.filter((c) => c.url.endsWith("/batch")).map((c) => (c.body as { states: unknown[] }).states.length),
    ).toEqual([2, 2]);
    b.maxBatchStates = 256;
    b.maxBodyBytes = 400;
    await model.systemOneBatch(["x".repeat(150), "y".repeat(150), "z".repeat(150)], { team: QS.team });
    expect(
      fake.calls
        .filter((c) => c.url.endsWith("/batch"))
        .slice(2)
        .map((c) => (c.body as { states: unknown[] }).states.length),
    ).toEqual([1, 1, 1]);
    expect(await model.systemOneBatch([], { team: QS.team })).toEqual([]);
  });

  it("splits more questions than the server takes per request (64) into several requests", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    const many = Object.fromEntries(Array.from({ length: 65 }, (_, i) => [`q${i}`, noul(`ignore ${i}?`)]));
    const one = await model.systemOne("ignore me", many);
    expect(Object.keys(one.answers)).toHaveLength(65);
    const batch = await model.systemOneBatch(["ignore me", "hello"], many);
    expect(batch.map((r) => Object.keys(r.answers).length)).toEqual([65, 65]);
    expect(batch.map((r) => r.answers.q64!.noul > 0.5)).toEqual([true, false]);
    const sizes = fake.calls
      .filter((c) => c.method === "POST")
      .map((c) => Object.keys((c.body as { questions: object }).questions).length);
    expect(sizes).toEqual([64, 1, 64, 1, 64, 1]);
  });

  it("validates questions before sending anything", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    await expect(model.systemOne("x", { q: choice("Which?", ["only"]) })).rejects.toThrow(InputError);
    expect(fake.calls.filter((c) => c.method === "POST")).toHaveLength(0);
  });

  it("maps the server's errors: 422 to InputError with its detail, 401 to a credentials message", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    fake.queue.push(json({ detail: "at most 64 questions per state (got 65)" }, 422));
    await expect(model.systemOne("x", { team: QS.team })).rejects.toThrow(
      new InputError(`${URL_} rejected the request: at most 64 questions per state (got 65)`),
    );
    fake.queue.push(json({ detail: "invalid or missing bearer token" }, 401));
    await expect(model.systemOne("x", { team: QS.team })).rejects.toThrow(/rejected the credentials \(HTTP 401\)/);
    fake.queue.push(new Response("<html>bad gateway</html>", { status: 500 }));
    await expect(model.systemOne("x", { team: QS.team })).rejects.toThrow(
      `${URL_} answered HTTP 500: <html>bad gateway</html>`,
    );
  });

  it("retries a busy server, honouring Retry-After", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    fake.queue.push(
      json({ detail: "busy" }, 503, { "retry-after": "0" }),
      json({ detail: "busy" }, 429, { "retry-after": "0.01" }),
    );
    const r = await model.systemOne(TICKET, { team: QS.team });
    expect(r.answers.team).toMatchObject({ choice: "billing" });
    expect(fake.calls.filter((c) => c.method === "POST")).toHaveLength(3);
    fake.queue.push(...[1, 2, 3].map(() => json({ detail: "busy" }, 503, { "retry-after": "0" })));
    await expect(model.systemOne(TICKET, { team: QS.team })).rejects.toThrow(`${URL_} answered HTTP 503: busy`);
  });

  it("starts no new request once one of a call's requests has failed", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch, workers: 1 });
    (model.backend as ServedBackend).maxBatchStates = 1;
    fake.queue.push(json({ detail: "boom" }, 500));
    await expect(model.systemOneBatch(["a", "b", "c", "d"], { team: QS.team })).rejects.toThrow(/HTTP 500/);
    expect(fake.calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });

  it("does not retry a 504 from opendecider serve (it already waited)", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    fake.queue.push(json({ detail: "no answer within 30 s" }, 504));
    await expect(model.systemOne(TICKET, { team: QS.team })).rejects.toThrow(ServerError);
    expect(fake.calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });

  it("fails fast for 5 s after the server cannot be reached", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    fake.queue.push(() => {
      throw new TypeError("fetch failed", { cause: { code: "ECONNREFUSED" } });
    });
    const err = await model.systemOne(TICKET, { team: QS.team }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ServerError);
    expect((err as ServerError).unreachable).toBe(true);
    expect((err as Error).message).toBe(
      `cannot reach ${URL_} (ECONNREFUSED); is the server running and the model loaded?`,
    );
    const posts = () => fake.calls.filter((c) => c.method === "POST").length;
    await expect(model.systemOne(TICKET, { team: QS.team })).rejects.toThrow("(calls fail at once for 5 s)");
    expect(posts()).toBe(1);
    now += 5001;
    await expect(model.systemOne(TICKET, { team: QS.team })).resolves.toBeTruthy();
    expect(posts()).toBe(2);
  });

  it("retries a connection reset", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    fake.queue.push(() => {
      throw new TypeError("fetch failed", { cause: { code: "ECONNRESET" } });
    });
    await expect(model.systemOne(TICKET, { team: QS.team })).resolves.toBeTruthy();
  });

  it("leaves no timer or listener behind once a request is done (nothing outlives the request)", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch, timeoutMs: 120_000 });
    const ac = new AbortController();
    const add = vi.spyOn(ac.signal, "addEventListener");
    const remove = vi.spyOn(ac.signal, "removeEventListener");
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      await model.systemOne(TICKET, { team: QS.team }, { signal: ac.signal });
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
    expect(add.mock.calls.length).toBeGreaterThan(0);
    expect(remove.mock.calls.length).toBe(add.mock.calls.length);
  });

  it("times out a server that does not answer", async () => {
    const hang = fakeServe().fetch;
    const slow = (async (input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === "POST") {
        return new Promise<Response>((_, reject) =>
          init.signal!.addEventListener("abort", () => reject(init.signal!.reason)),
        );
      }
      return hang(input, init);
    }) as typeof fetch;
    const m2 = await load(URL_, { fetch: slow, timeoutMs: 50 });
    await expect(m2.systemOne(TICKET, { team: QS.team })).rejects.toThrow(`${URL_} gave no answer within 0.05 s`);
  });

  it("times out a server that sends its headers and then stalls", async () => {
    const hang = fakeServe().fetch;
    const stall = (async (input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method !== "POST") return hang(input, init);
      const body = new ReadableStream({
        start(c) {
          c.enqueue(new TextEncoder().encode('{"answers":'));
          init.signal!.addEventListener("abort", () => c.error(init.signal!.reason));
        },
      });
      return new Response(body, { status: 200 });
    }) as typeof fetch;
    const m = await load(URL_, { fetch: stall, timeoutMs: 50 });
    const e = await m.systemOne(TICKET, { team: QS.team }).catch((x: unknown) => x);
    expect(e).toBeInstanceOf(ServerError);
    expect((e as ServerError).message).toBe(`${URL_} gave no answer within 0.05 s`);
    expect((e as ServerError).unreachable).toBe(true);
  });

  it("rejects with the caller's abort reason when cancelled, without the fail-fast window", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    const ac = new AbortController();
    ac.abort();
    await expect(model.systemOne(TICKET, { team: QS.team }, { signal: ac.signal })).rejects.toHaveProperty(
      "name",
      "AbortError",
    );
    await expect(model.systemOne(TICKET, { team: QS.team })).resolves.toBeTruthy();
  });

  it("reports an answer without probabilities (a Jev or Laya server) and a body that is not JSON", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    fake.queue.push(json({ answers: { q0: { type: "choice", choice: "billing" } } }));
    await expect(model.systemOne(TICKET, { team: QS.team })).rejects.toThrow(
      /a Jev or Laya server returns only the top answer/,
    );
    fake.queue.push(new Response("not json", { status: 200 }));
    await expect(model.systemOne(TICKET, { team: QS.team })).rejects.toThrow(
      /answered with something that is not JSON: "not json"/,
    );
  });

  it("reports a server that cannot be reached at load, and its credentials", async () => {
    const down = (async () => {
      throw new TypeError("fetch failed", { cause: { code: "ECONNREFUSED" } });
    }) as typeof fetch;
    await expect(load(URL_, { fetch: down })).rejects.toThrow(
      `cannot reach opendecider serve at ${URL_} (ECONNREFUSED); is it running?`,
    );
    const fake = fakeServe();
    fake.queue.push(json({ detail: "invalid or missing bearer token" }, 401));
    await expect(load(URL_, { fetch: fake.fetch })).rejects.toThrow(/rejected the credentials/);
    fake.queue.push(new Response("<html>", { status: 200 }));
    await expect(load(URL_, { fetch: fake.fetch })).rejects.toThrow(/is not opendecider serve/);
  });

  it("says how to reach Ollama, LM Studio or vLLM when given their URL", async () => {
    const fake = fakeServe();
    const id = "hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0";
    fake.queue.push(json({ object: "list", data: [{ id }] }));
    const e = (await load("http://127.0.0.1:11434", { fetch: fake.fetch }).catch((x: Error) => x)) as Error;
    expect(e).toBeInstanceOf(InputError);
    expect(e.message).toBe(
      "http://127.0.0.1:11434 is an OpenAI-compatible server (Ollama, LM Studio or vLLM), not opendecider serve: " +
        `load "openai:${id}" with baseUrl "http://127.0.0.1:11434/v1" ` +
        `(or "ollama:${id}" / "lmstudio:${id}" at their default local addresses)`,
    );
  });

  it("pings /ready", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    expect(await model.ping()).toBe(true);
    fake.queue.push(json({ ready: false }, 503));
    expect(await model.ping()).toBe(false);
  });
});

describe("credentials", () => {
  it("sends the API key as a bearer token, and refuses redirects with it", async () => {
    const fake = fakeServe();
    const model = await load("https://decider.example", { fetch: fake.fetch, apiKey: "k1" });
    await model.systemOne(TICKET, { team: QS.team });
    expect(fake.calls.every((c) => c.headers["authorization"] === "Bearer k1" && c.redirect === "manual")).toBe(true);
    fake.queue.push(
      new Response(null, { status: 307, headers: { location: "https://elsewhere.example/v1/systemone" } }),
    );
    await expect(model.systemOne(TICKET, { team: QS.team })).rejects.toThrow(
      "https://decider.example redirected the request to https://elsewhere.example/v1/systemone; " +
        "credentials are sent only to the URL given, so use the final URL",
    );
  });

  it("never follows a redirect with a credential given as a header either", async () => {
    const fake = fakeServe();
    await (
      await load(URL_, { fetch: fake.fetch, headers: { Authorization: "Basic abc" } })
    ).systemOne(TICKET, { team: QS.team });
    expect(fake.calls.every((c) => c.redirect === "manual" && c.headers["Authorization"] === "Basic abc")).toBe(true);
    const plain = fakeServe();
    await (await load(URL_, { fetch: plain.fetch, headers: { "x-trace": "1" } })).systemOne(TICKET, { team: QS.team });
    expect(plain.calls.every((c) => c.redirect === "follow")).toBe(true);
  });

  it("follows redirects without a key", async () => {
    const fake = fakeServe();
    await (await load(URL_, { fetch: fake.fetch })).systemOne(TICKET, { team: QS.team });
    expect(fake.calls.every((c) => c.redirect === "follow" && !("authorization" in c.headers))).toBe(true);
  });

  it("never sends a key over plain HTTP to another host, unless allowed", async () => {
    const fake = fakeServe();
    await expect(load("http://gpu-box:8000", { fetch: fake.fetch, apiKey: "k" })).rejects.toThrow(
      "refusing to send an API key over plain HTTP to gpu-box; use https, or set allowInsecureHttp " +
        "(OPENDECIDER_REMOTE_ALLOW_HTTP=1) if the network path is trusted",
    );
    expect(fake.calls).toHaveLength(0);
    await expect(
      load("http://gpu-box:8000", { fetch: fake.fetch, apiKey: "k", allowInsecureHttp: true }),
    ).resolves.toBeTruthy();
    await expect(load("http://localhost:8000", { fetch: fake.fetch, apiKey: "k" })).resolves.toBeTruthy();
    await expect(load("http://[::1]:8000", { fetch: fake.fetch, apiKey: "k" })).resolves.toBeTruthy();
  });

  it("reads the key and the HTTP opt-in from the environment", async () => {
    vi.stubEnv("OPENDECIDER_REMOTE_API_KEY", "from-env");
    const fake = fakeServe();
    await expect(load("http://gpu-box:8000", { fetch: fake.fetch })).rejects.toThrow(/refusing/);
    vi.stubEnv("OPENDECIDER_REMOTE_ALLOW_HTTP", "1");
    await load("http://gpu-box:8000", { fetch: fake.fetch });
    expect(fake.calls[0]!.headers["authorization"]).toBe("Bearer from-env");
  });

  it("rejects a key or a header that fetch cannot send, without quoting it", async () => {
    const f = fakeServe().fetch;
    const e = (await load(URL_, { fetch: f, apiKey: "sk-SECRET\nX" }).catch((x: Error) => x)) as Error;
    expect(e).toBeInstanceOf(InputError);
    expect(e.message).toBe("the API key is not a valid header value (a line break or a control character in it?)");
    const h = (await load(URL_, { fetch: f, headers: { "x-token": "tok-SECRET\r\nX" } }).catch(
      (x: Error) => x,
    )) as Error;
    expect(h.message).toBe('the "x-token" header is not a valid header (a line break or a control character in it?)');
    expect(`${e.message} ${h.message}`).not.toContain("SECRET");
  });

  it("never puts the key in an error message", async () => {
    const fake = fakeServe();
    const model = await load("https://decider.example", { fetch: fake.fetch, apiKey: "secret-key-123" });
    fake.queue.push(json({ detail: "nope" }, 401));
    const e = (await model.systemOne(TICKET, { team: QS.team }).catch((x: Error) => x)) as Error;
    expect(String(e.message)).not.toContain("secret-key-123");
  });
});

describe("Ollama, LM Studio and OpenAI-compatible servers", () => {
  it("reads option letters from the next-token log-probabilities", async () => {
    const fake = fakeOpenAI();
    const model = await load("ollama:opendecider-small", { fetch: fake.fetch });
    expect(model.meta).toEqual({ name: "opendecider-small", kind: "remote", base_url: "http://127.0.0.1:11434/v1" });
    const r = await model.systemOne(TICKET, QS);
    expect(r.answers.team).toMatchObject({ choice: "billing" });
    expect(r.usage.input_tokens).toBe(21);
    const sent = fake.calls[0]!.body as Record<string, unknown>;
    expect(fake.calls[0]!.url).toBe("http://127.0.0.1:11434/v1/chat/completions");
    expect(sent).toMatchObject({
      model: "opendecider-small",
      max_tokens: 1,
      temperature: 0,
      logprobs: true,
      top_logprobs: 20,
    });
    expect((sent["messages"] as { content: string }[])[0]!.content).toBe(
      "You make one decision for a software system.",
    );
  });

  it("uses LM Studio's address, OPENDECIDER_REMOTE_URL, and baseUrl for openai:", async () => {
    const fake = fakeOpenAI();
    expect((await load("lmstudio:m", { fetch: fake.fetch })).meta["base_url"]).toBe("http://127.0.0.1:1234/v1");
    await expect(load("openai:m", { fetch: fake.fetch })).rejects.toThrow(
      "openai:<model> needs baseUrl (or OPENDECIDER_REMOTE_URL)",
    );
    expect((await load("openai:m", { fetch: fake.fetch, baseUrl: "https://vllm.example/v1/" })).meta["base_url"]).toBe(
      "https://vllm.example/v1",
    );
    vi.stubEnv("OPENDECIDER_REMOTE_URL", "http://gpu:9000/v1");
    expect((await load("ollama:m", { fetch: fake.fetch })).meta["base_url"]).toBe("http://gpu:9000/v1");
  });

  it("explains a server without log-probabilities, and more than 26 options", async () => {
    const fake = fakeOpenAI();
    const model = await load("lmstudio:m", { fetch: fake.fetch });
    fake.queue.push(json({ choices: [{ message: { content: "A" } }] }));
    await expect(model.systemOne(TICKET, { team: QS.team })).rejects.toThrow(
      /returned no token log-probabilities for 'm'/,
    );
    const many = Array.from({ length: 27 }, (_, i) => `o${i}`);
    await expect(model.systemOne(TICKET, { q: choice("Which?", many) })).rejects.toThrow(
      "at most 26 options per question through a model server (got 27); use the PyTorch model for more",
    );
  });

  it("retries 500 and 504 from a model server", async () => {
    const fake = fakeOpenAI();
    const model = await load("ollama:m", { fetch: fake.fetch });
    fake.queue.push(json({}, 500, { "retry-after": "0" }), json({}, 504, { "retry-after": "0" }));
    await expect(model.systemOne(TICKET, { team: QS.team })).resolves.toBeTruthy();
  });
});

describe("load and Decider", () => {
  it("never takes credentials inside the URL, and never echoes them", async () => {
    const e = (await load("https://user:s3cret@decider.example", { fetch: fakeServe().fetch }).catch(
      (x: Error) => x,
    )) as Error;
    expect(e).toBeInstanceOf(InputError);
    expect(e.message).toBe("the server URL must not contain a user name or password; pass the key as apiKey");
    expect(() => new Decider("constructor:x")).toThrow(/calls served models only/);
    await expect(load("toString:x")).rejects.toThrow(/calls served models only/);
  });

  it("takes a path prefix (a reverse proxy), and refuses a query string or fragment without echoing it", async () => {
    const f = fakeServe();
    await (await load("HTTPS://Decider.Example/opendecider/", { fetch: f.fetch })).systemOne("x", { q: noul("y?") });
    expect(f.calls.map((c) => c.url)).toEqual([
      "https://decider.example/opendecider/v1/models",
      "https://decider.example/opendecider/v1/systemone",
    ]);
    for (const u of ["http://h:8000?token=SECRET", "http://h:8000/#SECRET"]) {
      const e = (await load(u, { fetch: f.fetch }).catch((x: Error) => x)) as Error;
      expect(e.message).toBe("the server URL must not have a query string or a fragment; pass a key as apiKey");
    }
  });

  it("validates the connection options", async () => {
    const f = fakeServe().fetch;
    await expect(load(URL_, { fetch: f, timeoutMs: 0 })).rejects.toThrow(
      "timeoutMs must be above 0 and at most 2147483647 (got 0)",
    );
    await expect(load(URL_, { fetch: f, timeoutMs: Infinity })).rejects.toThrow(InputError);
    await expect(load(URL_, { fetch: f, workers: NaN })).rejects.toThrow(
      "workers must be a whole number of at least 1 (got NaN)",
    );
    await expect(load(URL_, { fetch: f, workers: 0 })).rejects.toThrow(InputError);
    await expect(load(URL_, { fetch: f, workers: 2 })).resolves.toBeTruthy();
  });

  it("keeps labels and question names such as __proto__ as plain keys", async () => {
    const fake = fakeServe();
    const model = await load(URL_, { fetch: fake.fetch });
    const r = await model.systemOne("__proto__ please", {
      ["__proto__"]: choice("Which?", ["__proto__", "constructor"]),
    });
    const a = r.answers["__proto__"] as { choice: string; probabilities: Record<string, number> };
    expect(Object.hasOwn(r.answers, "__proto__")).toBe(true);
    expect(a.choice).toBe("__proto__");
    expect(Object.hasOwn(a.probabilities, "__proto__")).toBe(true);
    expect(a.probabilities["__proto__"]).toBeCloseTo(10 / 11, 6);
    expect(Object.getPrototypeOf(r.answers)).toBe(Object.prototype);
  });

  it("refuses a model that would run on this machine, with what to use instead", async () => {
    await expect(load("manjunathshiva/opendecider-nano")).rejects.toThrow(/calls served models only/);
    expect(() => new Decider("manjunathshiva/opendecider-nano")).toThrow(/pip install opendecider/);
    await expect(load("ftp://x")).rejects.toThrow(InputError);
    await expect(load("http://")).rejects.toThrow("baseUrl must be an http or https URL, got 'http://'");
  });

  it("connects once for concurrent first calls, then fails fast for 5 s after a failed connection", async () => {
    const fake = fakeServe();
    const d = new Decider(URL_, { fetch: fake.fetch });
    expect(d.status()).toEqual({ model: URL_, loaded: false });
    const [a, b] = await Promise.all([d.model(), d.model()]);
    expect(a).toBe(b);
    expect(fake.calls.filter((c) => c.url.endsWith("/v1/models"))).toHaveLength(1);
    expect(d.label).toBe(fake.name);
    expect(d.status()).toEqual({ model: URL_, loaded: true, kind: "served", device: `served (${URL_})` });

    let now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    let up = false;
    const flaky = (async (i: RequestInfo | URL, init?: RequestInit) => {
      if (!up) throw new TypeError("fetch failed", { cause: { code: "ECONNREFUSED" } });
      return fake.fetch(i, init);
    }) as typeof fetch;
    const d2 = new Decider(URL_, { fetch: flaky });
    await expect(d2.model()).rejects.toThrow(/cannot reach opendecider serve/);
    up = true;
    await expect(d2.model()).rejects.toThrow(/cannot reach opendecider serve/); // within 5 s: no new attempt
    now += 5001;
    await expect(d2.model()).resolves.toBeTruthy();
  });
});
