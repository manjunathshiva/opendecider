/** The prompt the Qwen-based OpenDecider models were trained on (a port of opendecider/prompt.py; parity-tested). */
import { pyJsonIndented, type JsonValue } from "./json.js";

export const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
export const SYSTEM = "You make one decision for a software system.";

/** One option of a question: its name and its description (or null). */
export type Option = readonly [name: string, description: unknown];

export function render(state: unknown, instructions: string, options: readonly Option[]): string {
  const st = typeof state === "string" ? state : pyJsonIndented((state ?? null) as JsonValue); // json.dumps(..., indent=1)
  const lines = options.map(([k, v], i) => `${LETTERS[i]}) ${v && v !== k ? `${k}: ${String(v)}` : k}`);
  return (
    `Input:\n${st}\n\nQuestion: ${instructions}\n\nOptions:\n${lines.join("\n")}\n\n` +
    "Answer with the letter of the correct option only."
  );
}

/** The prompt as sent to a server that applies the chat template and tokenises it itself (Ollama, LM Studio, vLLM),
 * which reads "<|im_end|>" written in a message as a control token: a zero-width space after the "<" of anything
 * shaped like a Qwen special token (<|name|>) keeps it plain text there. Any other text is sent unchanged. */
export function forServer(prompt: string): string {
  return prompt.replace(/<\|(?=[A-Za-z0-9_]+\|>)/g, "<\u200b|");
}
