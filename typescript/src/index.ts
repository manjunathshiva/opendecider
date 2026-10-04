/**
 * OpenDecider for TypeScript: typed questions about a state, answered with a calibrated probability for every option,
 * by small open decision models served by `opendecider serve`, Ollama, LM Studio or vLLM.
 *
 *     import { load, choice, noul, score } from "@opendecider/client";
 *
 *     const model = await load("http://localhost:8000");          // or "ollama:opendecider-small"
 *     const r = await model.systemOne(
 *       { subject: "Refund?", body: "I was charged twice for order 1182." },
 *       {
 *         team: choice("Which team should handle this?", { billing: "charges, refunds", tech: "bugs" }),
 *         urgent: noul("Does this need a reply today?"),
 *         anger: score("How upset is the customer?", ["calm", "annoyed", "angry"]),
 *       });
 *     r.answers.team;   // { type: "choice", choice: "billing", probabilities: {...}, confidence: 0.97 }
 *
 * Agent tools: `@opendecider/client/ai-sdk` (Vercel AI SDK) and `@opendecider/client/mastra` (Mastra).
 */
export { type Backend, type Decided, type Info, type Item } from "./backends.js";
export { type JsonValue } from "./json.js";
export { type Option } from "./prompt.js";
export { InputError, ServerError } from "./errors.js";
export { type ConnectionOptions } from "./http.js";
export { setLogger, type Logger } from "./log.js";
export { Decider, OpenDecider, load, type CallOptions, type SystemOneResult } from "./model.js";
export {
  choice,
  noul,
  score,
  type AnswerFor,
  type AnswersFor,
  type AnyAnswer,
  type ChoiceAnswer,
  type ChoiceQuestion,
  type Criteria,
  type Levels,
  type NoulAnswer,
  type NoulQuestion,
  type Question,
  type Questions,
  type ScoreAnswer,
  type ScoreQuestion,
} from "./questions.js";
export { Router, type Decision, type RouterOptions } from "./router.js";
export {
  ATTACK_CHECKS,
  BLOCKED_MESSAGE,
  DEFAULT_THRESHOLD,
  Guard,
  GuardrailError,
  THRESHOLDS,
  WINDOW_CHARS,
  WINDOW_OVERLAP,
  type GuardOptions,
  type GuardResult,
} from "./guard.js";
export {
  DESCRIPTIONS,
  INSTRUCTIONS,
  MAX_BATCH_ITEMS,
  MAX_BATCH_STATES,
  MAX_OPTIONS,
  MAX_QUESTIONS,
  MAX_STATE_CHARS,
  type ToolName,
} from "./tools.js";
export { SCHEMAS, TOOL_NAMES, toolSpecs, type ToolSpec, type ToolsOptions } from "./toolkit.js";
export { VERSION } from "./version.js";
