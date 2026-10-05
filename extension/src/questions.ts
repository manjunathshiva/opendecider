/** The questions opendecider-nano is asked about each video, and how its answers become "show" or "hide". */
import { KIND_NAMES, KIND_QUESTION_TEXT, KINDS, oneLine, type Kind, type Settings } from "./settings.js";

export interface Video {
  id: string;
  title: string;
  channel: string;
}

/** The kind of video: one choice question; the answer is a probability for every kind. */
export const KIND_QUESTION = {
  type: "choice",
  instructions: KIND_QUESTION_TEXT,
  criteria: KINDS,
} as const;

/** The viewer's rule as a yes/no question about the video: topics become "Is this video about ...?", a question
 * (ending in "?") is asked as written. */
export function ruleQuestion(rule: string) {
  const r = oneLine(rule);
  return { type: "noul", instructions: r.endsWith("?") ? r : `Is this video about ${r}?` } as const;
}

/** What the model reads: the video is the state (opendecider-nano judges the state; a title written into the question
 * gets the same answer for every video). */
export const videoState = (v: Video) => `YouTube video. Title: ${v.title}. Channel: ${v.channel || "unknown"}.`;

/** A video's answers: the probability of each kind, and the rule's yes probability. */
export interface Answers {
  kinds?: Record<Kind, number>;
  rule?: number;
}

/** Hide the video? Each filter that is on can hide it; one without an answer (not asked) never does. */
export function hides(s: Settings, a: Answers): boolean {
  if (s.hide.length && a.kinds) {
    const wanted = KIND_NAMES.filter((k) => !s.hide.includes(k)).reduce((sum, k) => sum + (a.kinds![k] ?? 0), 0);
    if (wanted < s.strictness) return true; // the kinds kept hold too little of the probability
  }
  if (s.rule && a.rule !== undefined) {
    if (s.ruleMode === "hide" ? a.rule > 1 - s.strictness : a.rule < s.strictness) return true;
  }
  return false;
}

/** Short, stable id of a text (FNV-1a, 32 bits): cache keys for a rule, never a security boundary. */
export function hash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return (h >>> 0).toString(16).padStart(8, "0");
}
