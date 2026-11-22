/** Shared runtime type guards and abort-error helpers across ACP and provider modules. */

export function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

export function makeAbortError(message = "Operation aborted."): Error {
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}
