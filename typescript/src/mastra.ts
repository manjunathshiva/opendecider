/**
 * OpenDecider in Mastra (`@mastra/core` 1.x): the seven tools of `opendecider mcp`, and a guard as an input processor.
 *
 *     import { Agent } from "@mastra/core/agent";
 *     import { opendeciderTools, GuardProcessor } from "@opendecider/client/mastra";
 *
 *     const agent = new Agent({
 *       name: "support", instructions: "...", model,
 *       tools: opendeciderTools({ model: "http://localhost:8000" }),               // decide, choose, yes_no, ...
 *       inputProcessors: [new GuardProcessor({ model: "http://localhost:8000" })],        // block jailbreaks
 *     });
 *
 * A tool call with invalid input, or one the server cannot answer, returns {error: "<what to fix>"} to the model.
 * GuardProcessor screens the newest user message before the agent runs and, when it is blocked, stops the run through
 * Mastra's tripwire (the result's `tripwire` carries the reason and the GuardResult as metadata). It replaces Mastra's
 * LLM-based PromptInjectionDetector with one small model.
 */
import type { MastraDBMessage } from "@mastra/core/agent/message-list";
import type { ProcessInputArgs, Processor } from "@mastra/core/processors";
import { createTool } from "@mastra/core/tools";
import { Guard, GuardrailError, type GuardOptions, type GuardResult } from "./guard.js";
import { log } from "./log.js";
import { textOf, toolSpecs, type ToolsOptions } from "./toolkit.js";
import type { ToolName } from "./tools.js";

/** The tools for a Mastra agent's `tools`, keyed by the names the model sees (ids `opendecider_<name>`). */
export function opendeciderTools(options: ToolsOptions): Record<ToolName, ReturnType<typeof createTool>> {
  return Object.fromEntries(
    toolSpecs(options).map((spec) => [
      spec.name,
      createTool({
        id: `opendecider_${spec.name}`,
        description: spec.description,
        inputSchema: spec.inputSchema as Parameters<typeof createTool>[0]["inputSchema"],
        execute: async (input: unknown, context?: { abortSignal?: AbortSignal }) =>
          spec.run((input ?? {}) as Record<string, unknown>, context?.abortSignal),
      } as Parameters<typeof createTool>[0]),
    ]),
  ) as Record<ToolName, ReturnType<typeof createTool>>;
}

/** The text of a Mastra message: its text parts, else its plain content. */
function messageText(m: MastraDBMessage): string | null {
  const content = m.content as unknown as { parts?: unknown; content?: unknown } | string | undefined;
  if (typeof content === "string") return content;
  return textOf(content?.parts) ?? (typeof content?.content === "string" ? content.content : null);
}

/** The settings of a GuardProcessor, beside the Guard's own. */
export interface GuardProcessorOptions {
  /** "block" (the default) stops the run through the tripwire; "warn" logs and lets it run; "filter" drops the
   * flagged messages. */
  strategy?: "block" | "warn" | "filter";
  /** Screen only the newest message, when it is the user's (the default); false screens every user message. */
  lastMessageOnly?: boolean;
}

/** A Mastra input processor that screens user messages for jailbreaks and prompt injection. */
export class GuardProcessor implements Processor<"opendecider-guard", GuardResult> {
  readonly id = "opendecider-guard";
  readonly name = "OpenDecider guard";
  readonly guard: Guard;
  readonly strategy: "block" | "warn" | "filter";
  readonly lastMessageOnly: boolean;

  /** guard: a Guard, or its options ({ model, threshold, onError, onDecision, ... }), plus the processor's own. */
  constructor(guard: Guard | (GuardOptions & GuardProcessorOptions), options: GuardProcessorOptions = {}) {
    let own: GuardProcessorOptions = {};
    if (guard instanceof Guard) {
      this.guard = guard;
    } else {
      const { strategy, lastMessageOnly, ...settings } = guard;
      this.guard = new Guard(settings);
      own = { strategy, lastMessageOnly };
    }
    this.strategy = options.strategy ?? own.strategy ?? "block";
    this.lastMessageOnly = options.lastMessageOnly ?? own.lastMessageOnly ?? true;
    if (!["block", "warn", "filter"].includes(this.strategy)) {
      throw new TypeError(`strategy must be "block", "warn" or "filter" (got '${String(this.strategy)}')`);
    }
  }

  async processInput({ messages, abort, abortSignal }: ProcessInputArgs<GuardResult>): Promise<MastraDBMessage[]> {
    const picked = (this.lastMessageOnly ? messages.slice(-1) : messages)
      .filter((m) => m.role === "user")
      .flatMap((m) => {
        const t = messageText(m);
        return t !== null && t.trim() ? [{ m, t }] : [];
      });
    if (!picked.length) return messages;
    const results = await this.guard.checkMany(
      picked.map((p) => p.t),
      { signal: abortSignal },
    );
    const blocked = new Set<MastraDBMessage>();
    picked.forEach(({ m }, i) => {
      if (!results[i]!.passed) blocked.add(m);
    });
    if (!blocked.size) return messages;
    const first = results.find((r) => !r.passed) as GuardResult;
    const reason = new GuardrailError(first).message;
    if (this.strategy === "block") abort(reason, { metadata: first });
    log("warn", `${reason}${this.strategy === "filter" ? " (message removed)" : ""}`);
    return this.strategy === "filter" ? messages.filter((m) => !blocked.has(m)) : messages;
  }
}
