/** JSON and text helpers that reproduce what the Python package measures and prints, so limits and messages match. */
import { InputError } from "./errors.js";

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

/** `value` as plain JSON data (what the server receives): toJSON applied, undefined and functions dropped. */
export function jsonValue(value: unknown, what = "the state"): JsonValue {
  if (value === undefined) return null;
  let text: string | undefined;
  try {
    text = JSON.stringify(value);
  } catch (e) {
    // a BigInt or a cycle
    throw new InputError(`${what} must be text or JSON-serialisable (${(e as Error).message})`);
  }
  return text === undefined ? null : (JSON.parse(text) as JsonValue);
}

/** Python's `json.dumps(value, ensure_ascii=False)`: ", " and ": " separators (the size the limits count). */
export function pyJson(value: JsonValue): string {
  if (typeof value === "number") return pyNumber(value);
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(pyJson).join(", ")}]`;
  return `{${Object.entries(value)
    .map(([k, v]) => `${JSON.stringify(k)}: ${pyJson(v)}`)
    .join(", ")}}`;
}

/** Python's `json.dumps(value, indent=1, ensure_ascii=False)`: what the Qwen-based models' prompt shows. */
export function pyJsonIndented(value: JsonValue, indent = ""): string {
  if (typeof value === "number") return pyNumber(value);
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  const inner = `${indent} `;
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    return `[\n${value.map((v) => inner + pyJsonIndented(v, inner)).join(",\n")}\n${indent}]`;
  }
  const entries = Object.entries(value);
  if (!entries.length) return "{}";
  return `{\n${entries.map(([k, v]) => `${inner}${JSON.stringify(k)}: ${pyJsonIndented(v, inner)}`).join(",\n")}\n${indent}}`;
}

/** A number as Python's json.dumps writes it. They differ only below 1e-4 (Python: 1e-05, JavaScript: 0.00001); a whole
 * number is written as an integer, as JavaScript cannot tell 3 from 3.0. */
export function pyNumber(v: number): string {
  if (!Number.isFinite(v) || Number.isInteger(v) || Math.abs(v) >= 1e-4) return JSON.stringify(v);
  const [m, e] = v.toExponential().split("e") as [string, string];
  const exp = Number(e);
  return `${m}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
}

/** The length of `text` in Unicode code points, as Python's `len` counts it. */
export function codePoints(text: string): number {
  let n = 0;
  for (const _ of text) n++;
  return n;
}

/** Python's `repr` of a value in an error message: 'text', None, True, 3. */
export function pyRepr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return String(value);
  if (typeof value !== "string") return JSON.stringify(value) ?? String(value);
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  let out = "";
  for (const ch of value) {
    const c = ch.codePointAt(0)!;
    if (ch === "\\") out += "\\\\";
    else if (ch === quote) out += `\\${quote}`;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (c < 0x20 || c === 0x7f) out += `\\x${c.toString(16).padStart(2, "0")}`;
    else out += ch;
  }
  return quote + out + quote;
}

/** `x` rounded to 4 decimals, as Python's `round(x, 4)` gives it. */
export function round4(x: number): number {
  return Number(x.toFixed(4));
}
