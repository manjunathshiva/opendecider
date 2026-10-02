/**
 * OpenDecider in the Vercel AI SDK (`ai` 5 or later): the seven tools of `opendecider mcp`, and a guard for any model.
 *
 *     import { generateText, stepCountIs, wrapLanguageModel } from "ai";
 *     import { opendeciderTools, guardMiddleware } from "@opendecider/client/ai-sdk";
 *
 *     const tools = opendeciderTools({ model: "http://localhost:8000" });         // decide, choose, yes_no, ...
 *     const guarded = wrapLanguageModel({ model, middleware: guardMiddleware({ model: "http://localhost:8000" }) });
 *     await generateText({ model: guarded, tools, stopWhen: stepCountIs(5), prompt });
 *
 * A tool call with invalid input, or one the server cannot answer, returns {error: "<what to fix>"} to the model. The
 * guard middleware screens each new user message before the model sees it and throws GuardrailError when it is
 * blocked; steps that carry tool results pass unchecked.
 */
import { jsonSchema, tool, type Tool } from "ai";
import { asGuard, type Guard, type GuardOptions } from "./guard.js";
import { lastUserText, toolSpecs, type ToolsOptions } from "./toolkit.js";
import type { ToolName } from "./tools.js";

export { GuardrailError } from "./guard.js";

/** The tools as an AI SDK tool set: `generateText({ tools: opendeciderTools({ model }) })`. */
export function opendeciderTools(options: ToolsOptions): Record<ToolName, Tool> {
  return Object.fromEntries(
    toolSpecs(options).map((spec) => [
      spec.name,
      tool({
        description: spec.description,
        inputSchema: jsonSchema<Record<string, unknown>>(spec.inputSchema as Parameters<typeof jsonSchema>[0]),
        execute: (input: Record<string, unknown>, opts: { abortSignal?: AbortSignal }) =>
          spec.run(input, opts?.abortSignal),
      } as Tool),
    ]),
  ) as Record<ToolName, Tool>;
}

/** The shape of a language-model middleware this guard needs (the AI SDK's own types differ between major versions). */
export interface GuardMiddleware {
  /** The middleware specification it follows ("v3": its params shape is the same in ai 5, 6 and 7). */
  readonly specificationVersion: "v3";
  transformParams<P extends { prompt: readonly unknown[]; abortSignal?: AbortSignal | undefined }>(options: {
    params: P;
  }): Promise<P>;
}

/** A middleware for `wrapLanguageModel`: screens each new user message for jailbreaks and prompt injection before
 * the model sees it, and throws GuardrailError (from `generateText` / `streamText`) when it is blocked. The
 * GuardResult goes to the guard's `onDecision` hooks.
 *
 * guard: a Guard, or its options ({ model, threshold, onError, onDecision, ... }). */
export function guardMiddleware(guard: Guard | GuardOptions): GuardMiddleware {
  const screen = asGuard(guard);
  return {
    specificationVersion: "v3",
    async transformParams({ params }) {
      const text = lastUserText(params.prompt);
      if (text !== null) await screen.enforce(text, { signal: params.abortSignal });
      return params;
    },
  };
}
