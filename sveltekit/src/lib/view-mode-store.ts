export type ViewMode = 'simple' | 'advanced';

const STORAGE_KEY = 'bonsai-speech2speech-view-mode';

export function readViewMode(): ViewMode | null {
  if (typeof localStorage === 'undefined') return null;
  const value = localStorage.getItem(STORAGE_KEY);
  return value === 'simple' || value === 'advanced' ? value : null;
}

export function writeViewMode(mode: ViewMode) {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, mode);
}
