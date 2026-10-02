/** Where the package writes warnings and errors: `console` by default; `setLogger` sends them elsewhere (or
 * nowhere). */

export interface Logger {
  warn(message: string): void;
  error(message: string): void;
}

let logger: Logger | null = console;
const seen = new Map<string, number>();
const REPEAT_MS = 5000; // the same warning or error is written once per this window

/** Send the package's warnings and errors to `target` (e.g. a pino or winston logger), or nowhere with null. */
export function setLogger(target: Logger | null): void {
  logger = target;
}

export function log(level: "warn" | "error", message: string): void {
  try {
    logger?.[level](`[opendecider] ${message}`);
  } catch {
    /* a logger must never break a decision */
  }
}

/** `log`, but a message repeated within 5 s is dropped: an outage that fails every request at once must not write one
 * line per request. */
export function logOnce(level: "warn" | "error", message: string): void {
  const now = performance.now();
  const key = `${level}:${message}`;
  if (now - (seen.get(key) ?? -Infinity) < REPEAT_MS) return;
  seen.set(key, now);
  if (seen.size > 1000) {
    // bounded: drop the oldest entries
    for (const k of [...seen.keys()].slice(0, 500)) seen.delete(k);
  }
  log(level, message);
}
