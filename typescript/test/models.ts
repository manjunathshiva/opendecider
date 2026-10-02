// A scripted language model for the AI SDK and Mastra tests, in the model specification the installed `ai` speaks
// (v2 for ai 5, v3 for ai 6 and 7). Our own, rather than `ai/test`'s: those import packages that some ai releases
// leave out (ai 5.0.207's need msw).
import { createRequire } from "node:module";

const aiMajor = Number(
  (createRequire(import.meta.url)("ai/package.json") as { version: string }).version.split(".")[0],
);
/** The specification version the installed `ai` uses for its models. */
export const SPEC: "v2" | "v3" = aiMajor >= 6 ? "v3" : "v2";

/** One step of the script: a tool call, or text. */
export type Step = { toolName: string; input: unknown } | { text: string };

const usage =
  SPEC === "v3"
    ? {
        inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 1, text: 1, reasoning: 0 },
      }
    : { inputTokens: 1, outputTokens: 1, totalTokens: 2 };
const finish = (reason: string) => (SPEC === "v3" ? { unified: reason, raw: reason } : reason);
const content = (s: Step) =>
  "text" in s
    ? [{ type: "text", text: s.text }]
    : [{ type: "tool-call", toolCallId: "call-1", toolName: s.toolName, input: JSON.stringify(s.input) }];

/** A model that answers with `steps` in turn (the last one repeats), through generate or stream. */
export function scriptedModel(steps: Step[]) {
  let i = 0;
  const next = () => steps[Math.min(i++, steps.length - 1)]!;
  const doGenerateCalls: unknown[] = [];
  const doStreamCalls: unknown[] = [];
  return {
    specificationVersion: SPEC,
    provider: "scripted",
    modelId: "scripted",
    supportedUrls: {},
    doGenerateCalls,
    doStreamCalls,
    async doGenerate(options: unknown) {
      doGenerateCalls.push(options);
      const s = next();
      return { content: content(s), finishReason: finish("text" in s ? "stop" : "tool-calls"), usage, warnings: [] };
    },
    async doStream(options: unknown) {
      doStreamCalls.push(options);
      const s = next();
      const parts: unknown[] = [{ type: "stream-start", warnings: [] }];
      for (const p of content(s) as { type: string; text?: string }[]) {
        if (p.type === "text") {
          parts.push(
            { type: "text-start", id: "t" },
            { type: "text-delta", id: "t", delta: p.text },
            { type: "text-end", id: "t" },
          );
        } else parts.push(p);
      }
      parts.push({ type: "finish", finishReason: finish("text" in s ? "stop" : "tool-calls"), usage });
      return { stream: new ReadableStream({ start: (c) => (parts.forEach((p) => c.enqueue(p)), c.close()) }) };
    },
  };
}
