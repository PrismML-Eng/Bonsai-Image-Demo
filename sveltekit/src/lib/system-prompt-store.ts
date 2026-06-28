const STORAGE_KEY = 'bonsai-speech2speech-system-prompt';

export function readSystemPrompt(): string | null {
  if (typeof localStorage === 'undefined') return null;
  return localStorage.getItem(STORAGE_KEY);
}

export function writeSystemPrompt(prompt: string) {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, prompt);
}

export function clearSystemPrompt() {
  if (typeof localStorage === 'undefined') return;
  localStorage.removeItem(STORAGE_KEY);
}
