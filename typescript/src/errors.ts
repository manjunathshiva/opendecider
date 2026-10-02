/** A caller's input that cannot be answered (Python's ValueError): the message says what to fix. */
export class InputError extends Error {
  override name = "InputError";
}

/** The server failed to answer: unreachable, timed out, rejected the request, or answered something unusable. */
export class ServerError extends Error {
  override name = "ServerError";
  /** True when the server timed out or could not be reached (calls then fail at once for a few seconds). */
  readonly unreachable: boolean;
  /** The low-level reason, when there is one (e.g. ECONNREFUSED, or "gave no answer within 30 s"). */
  readonly reason: string | undefined;

  constructor(message: string, options: { unreachable?: boolean; reason?: string; cause?: unknown } = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.unreachable = options.unreachable ?? false;
    this.reason = options.reason;
  }
}
