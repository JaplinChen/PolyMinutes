// localStorage throws when site data is blocked (SecurityError) or full (QuotaExceededError).
export function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStorage(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Persisting is best-effort.
  }
}
