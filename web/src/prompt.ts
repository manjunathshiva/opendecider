/** opendecider-nano's input as token ids: a port of opendecider/prompt.py's `nano_ids` (parity-tested). */
import type { JsonValue, Option } from "@opendecider/client";

/** The token ids nano's prompt is built from (opendecider-nano's tokenizer; the tokenizer never produces them). */
export const SPECIAL = { cls: 50281, sep: 50282, pad: 50283, mask: 50284 } as const;

/** Text to token ids, with no special tokens: "[MASK]" in the text stays text. */
export type Encode = (text: string) => readonly number[];

/** Python's `json.dumps(value, ensure_ascii=False)`: ", " and ": " separators, and numbers as Python writes them. */
export function pyJson(value: JsonValue): string {
  if (typeof value === "number") return pyNumber(value);
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(pyJson).join(", ")}]`;
  return `{${Object.entries(value)
    .map(([k, v]) => `${JSON.stringify(k)}: ${pyJson(v)}`)
    .join(", ")}}`;
}

/** A number as Python's json.dumps writes it. They differ only below 1e-4 (Python: 1e-05, JavaScript: 0.00001); a whole
 * number is written as an integer, as JavaScript cannot tell 3 from 3.0. */
function pyNumber(v: number): string {
  if (!Number.isFinite(v) || Number.isInteger(v) || Math.abs(v) >= 1e-4) return JSON.stringify(v);
  const [m, e] = v.toExponential().split("e") as [string, string];
  const exp = Number(e);
  return `${m}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
}

/**
 * [CLS] question: ... [SEP] [MASK] opt1 [MASK] opt2 ... [SEP] input: <state> [SEP] as token ids, and whether the state
 * was shortened to fit `maxLen` (it goes first, so every option survives).
 */
export function nanoIds(
  encode: Encode,
  state: JsonValue,
  instructions: string,
  options: readonly Option[],
  maxLen: number,
): { ids: number[]; truncated: boolean } {
  const st = typeof state === "string" ? state : pyJson(state);
  const q = encode(`question: ${instructions}`);
  const opts: number[] = [];
  for (const [k, v] of options) opts.push(SPECIAL.mask, ...encode(v && v !== k ? ` ${k}: ${String(v)}` : ` ${k}`));
  const s = encode(`input: ${st}`);
  const room = Math.max(maxLen - q.length - opts.length - 4, 0);
  const ids = [SPECIAL.cls, ...q, SPECIAL.sep, ...opts, SPECIAL.sep, ...s.slice(0, room), SPECIAL.sep];
  return { ids, truncated: s.length > room };
}
