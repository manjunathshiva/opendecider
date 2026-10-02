/**
 * Typed questions (choice / score / noul) and typed answers: a port of opendecider/questions.py.
 *
 *     { type: "choice", instructions: "Which team?", criteria: { billing: "Charges", tech: "Bugs" } }
 *     { type: "score",  instructions: "How urgent?", criteria: ["not urgent", "soon", "now"] }
 *     { type: "noul",   instructions: "Is this spam?" }                                       // yes/no
 *
 * Every model sees a question as named options; this module turns the typed form into those options exactly as the
 * Python package does, and turns option probabilities back into a typed answer.
 */
import { InputError } from "./errors.js";
import { pyRepr } from "./json.js";
import type { Option } from "./prompt.js";

/** The question types. */
export const TYPES = ["choice", "score", "noul"] as const;
/** choice, score or noul (yes/no). */
export type QuestionType = (typeof TYPES)[number];

/** Option labels, or {label: "short description"} (descriptions help with terse labels). */
export type Criteria = readonly string[] | Readonly<Record<string, string | number | null | undefined>>;
/** Ordered levels, lowest first. */
export type Levels = readonly (string | number)[];

/** Pick one of the options. */
export interface ChoiceQuestion {
  type: "choice";
  instructions: string;
  criteria: Criteria;
}
/** Rate on ordered levels, lowest first. */
export interface ScoreQuestion {
  type: "score";
  instructions: string;
  criteria: Levels;
}
/** A yes/no question, optionally with what yes and no mean. */
export interface NoulQuestion {
  type: "noul";
  instructions: string;
  criteria?: { true?: string | null; false?: string | null };
}
/** A typed question. */
export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;
/** Named questions about one state. */
export type Questions = Readonly<Record<string, Question>>;

/** A choice question: pick one of the options. */
export function choice(instructions: string, criteria: Criteria): ChoiceQuestion {
  return { type: "choice", instructions, criteria };
}

/** A score question: rate on ordered levels, lowest first. */
export function score(instructions: string, levels: Levels): ScoreQuestion {
  return { type: "score", instructions, criteria: levels };
}

/** A yes/no question, optionally with descriptions of what yes and no mean. */
export function noul(instructions: string, criteria?: { true?: string; false?: string }): NoulQuestion {
  return criteria ? { type: "noul", instructions, criteria } : { type: "noul", instructions };
}

/** What every typed answer has. */
export interface Answer {
  type: QuestionType;
  probabilities: Record<string, number>;
  /** The probability of the top answer. */
  confidence: number;
  /** Present (true) when the state was shortened to fit the model's input. */
  truncated?: true;
}
/** The answer to a choice question: the most likely option. */
export interface ChoiceAnswer extends Answer {
  type: "choice";
  choice: string;
}
/** The answer to a score question. */
export interface ScoreAnswer extends Answer {
  type: "score";
  /** The most likely level (0 = the first). */
  score: number;
  /** The expected level, a probability-weighted mean that can fall between levels. */
  expected: number;
  legend: Record<string, string | number>;
}
/** The answer to a yes/no question: `noul` is the probability that the answer is yes. */
export interface NoulAnswer extends Answer {
  type: "noul";
  noul: number;
  probabilities: { true: number; false: number };
}
/** Any typed answer. */
export type AnyAnswer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

/** The answer type for a question type: `choice(...)` gets a ChoiceAnswer, and so on. */
export type AnswerFor<Q> = Q extends { type: "choice" }
  ? ChoiceAnswer
  : Q extends { type: "score" }
    ? ScoreAnswer
    : Q extends { type: "noul" }
      ? NoulAnswer
      : AnyAnswer;
/** The answers for named questions, each typed by its question. */
export type AnswersFor<Q extends Questions> = { [K in keyof Q]: AnswerFor<Q[K]> };

/** A validated question: its type, instructions and options in order (and the levels of a score question). */
export interface Prepared {
  type: QuestionType;
  instructions: string;
  options: Option[];
  levels?: Levels;
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** Validate one question and turn it into named options (raises InputError with a readable message). */
export function prepareQuestion(q: unknown): Prepared {
  const raw = isRecord(q) ? q : {};
  const type = raw["type"];
  if (!(TYPES as readonly unknown[]).includes(type)) {
    throw new InputError(`question type must be one of ('choice', 'score', 'noul'), got ${pyRepr(type)}`);
  }
  const instructions = raw["instructions"];
  if (typeof instructions !== "string" || !instructions.trim()) {
    throw new InputError("question needs non-empty 'instructions'");
  }
  const crit = raw["criteria"];
  if (type === "choice") {
    let options: Option[] | null = null;
    if (Array.isArray(crit)) {
      if (!crit.every((c) => typeof c === "string")) throw new InputError("choice options must be text");
      options = [...new Set(crit as string[])].map((c) => [c, null] as const);
    } else if (isRecord(crit)) {
      // (numbers print the same in both languages; true/false would not, so they are refused)
      if (!Object.values(crit).every((d) => d == null || typeof d === "string" || Number.isFinite(d))) {
        throw new InputError("choice option descriptions must be text or numbers");
      }
      options = Object.entries(crit).map(([k, d]) => [k, d ?? null] as const);
    }
    if (!options || options.length < 2)
      throw new InputError("choice question needs 'criteria' with at least 2 options");
    return { type, instructions, options };
  }
  if (type === "score") {
    if (!Array.isArray(crit) || crit.length < 2) {
      throw new InputError("score question needs 'criteria' as an ordered list of at least 2 levels");
    }
    if (!crit.every((c) => typeof c === "string" || typeof c === "number")) {
      throw new InputError("score levels must be text or numbers");
    }
    return { type, instructions, options: crit.map((lvl, i) => [String(i), lvl] as const), levels: crit as Levels };
  }
  const c = isRecord(crit) ? crit : {};
  return {
    type: "noul",
    instructions,
    options: [
      ["yes", "true" in c ? c["true"] : "Yes"],
      ["no", "false" in c ? c["false"] : "No"],
    ],
  };
}

/** Validate named questions (OpenDecider.prepare in Python). */
export function prepare(questions: unknown): Map<string, Prepared> {
  if (!isRecord(questions) || Object.keys(questions).length === 0) {
    throw new InputError("questions must be a non-empty object of named questions");
  }
  const out = new Map<string, Prepared>();
  for (const [k, q] of Object.entries(questions)) {
    try {
      out.set(k, prepareQuestion(q));
    } catch (e) {
      if (e instanceof InputError) throw new InputError(`question ${pyRepr(k)}: ${e.message}`);
      throw e;
    }
  }
  return out;
}

/** Option probabilities -> typed answer. */
export function answer(q: Prepared, probs: Record<string, number>): AnyAnswer {
  if (q.type === "noul") {
    const p = Number(probs["yes"]);
    return { type: "noul", noul: p, probabilities: { true: p, false: 1 - p }, confidence: Math.max(p, 1 - p) };
  }
  // entries, then fromEntries: a label such as "__proto__" stays a plain key
  const entries = q.options.map(
    ([name]) => [name, Number(Object.hasOwn(probs, name) ? probs[name] : NaN)] as [string, number],
  );
  let [top, best] = entries[0]!;
  for (const [name, p] of entries) {
    // the first of equally likely options wins, as Python's max() picks it
    if (p > best) [top, best] = [name, p];
  }
  const ordered = Object.fromEntries(entries);
  if (q.type === "score") {
    const legend = Object.fromEntries(q.levels!.map((lvl, i) => [String(i), lvl]));
    const expected = entries.reduce((s, [k, v]) => s + Number(k) * v, 0);
    return { type: "score", score: Number(top), expected, legend, probabilities: ordered, confidence: best };
  }
  return { type: "choice", choice: top, probabilities: ordered, confidence: best };
}
