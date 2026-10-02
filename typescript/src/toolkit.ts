/** The seven agent tools, framework-neutral: name, description, JSON Schema and how to run. The AI SDK and Mastra
 * adapters wrap these, so every framework gets the same tools as `opendecider mcp`. */
import type { ConnectionOptions } from "./http.js";
import { Guard, guardToolResult, type GuardOptions } from "./guard.js";
import { asDecider, type Decider, type OpenDecider } from "./model.js";
import type { Criteria, Levels, Questions } from "./questions.js";
import { DESCRIPTIONS, asResult, choose, decide, decideBatch, score, status, yesNo, type ToolName } from "./tools.js";

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** The seven tools, in order. */
export const TOOL_NAMES: readonly ToolName[] = [
  "decide",
  "choose",
  "yes_no",
  "score",
  "decide_batch",
  "guard",
  "status",
];

const STATE = {
  anyOf: [{ type: "string" }, { type: "object", additionalProperties: true }, { type: "array", items: {} }],
};
const QUESTIONS = { type: "object", additionalProperties: { type: "object", additionalProperties: true } };

/** The tools' input schemas: the ones `opendecider mcp` publishes. */
export const SCHEMAS: Readonly<Record<ToolName, Record<string, unknown>>> = {
  decide: { type: "object", properties: { state: STATE, questions: QUESTIONS }, required: ["state", "questions"] },
  choose: {
    type: "object",
    properties: {
      state: STATE,
      question: { type: "string" },
      options: {
        anyOf: [
          { type: "array", items: { type: "string" } },
          { type: "object", additionalProperties: { type: "string" } },
        ],
      },
    },
    required: ["state", "question", "options"],
  },
  yes_no: {
    type: "object",
    properties: { state: STATE, question: { type: "string" } },
    required: ["state", "question"],
  },
  score: {
    type: "object",
    properties: { state: STATE, question: { type: "string" }, levels: { type: "array", items: { type: "string" } } },
    required: ["state", "question", "levels"],
  },
  decide_batch: {
    type: "object",
    properties: { states: { type: "array", items: STATE }, questions: QUESTIONS },
    required: ["states", "questions"],
  },
  guard: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
  status: { type: "object", properties: {} },
};

/** One tool, framework-neutral: its name, description, input schema and how to run it. */
export interface ToolSpec {
  name: ToolName;
  description: string;
  inputSchema: Record<string, unknown>;
  /** Runs the tool; input errors and an unavailable server come back as {error}, so the agent can fix its call. */
  run(input: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>;
}

/** The settings of a tool set. */
export interface ToolsOptions {
  /** An `opendecider serve` URL, `ollama:...`, `lmstudio:...`, `openai:...`, or a loaded OpenDecider / Decider. */
  model: string | OpenDecider | Decider;
  /** How to reach the model, when `model` is a name. */
  connection?: ConnectionOptions;
  /** The `guard` tool's Guard or its options (default: the default checks on the same model); false leaves it out. */
  guard?: Guard | Omit<GuardOptions, "model" | "connection"> | false;
  /** Which tools to make (default: all seven). */
  include?: readonly ToolName[];
}

/** The tools as framework-neutral specs, for a framework without an adapter here (wrap `run` in its tool type). */
export function toolSpecs(options: ToolsOptions): ToolSpec[] {
  const decider = asDecider(options.model, options.connection);
  const include = options.include ?? TOOL_NAMES;
  for (const name of include) {
    if (!TOOL_NAMES.includes(name))
      throw new TypeError(`unknown tool '${name}'; the tools are ${TOOL_NAMES.join(", ")}`);
  }
  const screen =
    options.guard === false || !include.includes("guard")
      ? null
      : options.guard instanceof Guard
        ? options.guard
        : new Guard({ ...options.guard, model: decider });
  const run: Record<ToolName, ToolSpec["run"]> = {
    decide: (i, signal) => asResult(() => decide(decider, i["state"], i["questions"] as Questions, { signal })),
    choose: (i, signal) =>
      asResult(() => choose(decider, i["state"], i["question"] as string, i["options"] as Criteria, { signal })),
    yes_no: (i, signal) => asResult(() => yesNo(decider, i["state"], i["question"] as string, { signal })),
    score: (i, signal) =>
      asResult(() => score(decider, i["state"], i["question"] as string, i["levels"] as Levels, { signal })),
    decide_batch: (i, signal) =>
      asResult(() => decideBatch(decider, i["states"] as unknown[], i["questions"] as Questions, { signal })),
    guard: async (i, signal) => guardToolResult(await screen!.check(i["text"] as string, { signal })),
    status: async () => status(decider),
  };
  return include
    .filter((n) => n !== "guard" || screen)
    .map((name) => ({
      name,
      description: DESCRIPTIONS[name],
      inputSchema: SCHEMAS[name],
      // the frameworks may pass what the model sent unvalidated: anything but an object reads as no arguments
      run: (input, signal) => run[name](isRecord(input) ? input : {}, signal),
    }));
}

/** The text of the newest user message, joined parts; null when the last message is not the user's (a tool result or
 * the model's own turn), so only new user input is screened. */
export function lastUserText(messages: readonly unknown[]): string | null {
  const last = messages.at(-1) as { role?: unknown; content?: unknown } | undefined;
  if (!last || last.role !== "user") return null;
  return textOf(last.content);
}

/** The text in a message's content: a string, or the text parts of a list. */
export function textOf(content: unknown): string | null {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;
  const texts = content.flatMap((p: { type?: unknown; text?: unknown }) =>
    p?.type === "text" && typeof p.text === "string" ? [p.text] : [],
  );
  return texts.length ? texts.join("\n") : null;
}
