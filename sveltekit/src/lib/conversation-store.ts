import {
  DEFAULT_SYSTEM_PROMPT,
  type ConversationTurn,
  type ModelId,
  type SpeechBreakMode,
  type TtsLanguage,
  TTS_LANGUAGE_CODES,
  type VoiceId,
} from '$lib/speech2speech';

const DB_NAME = 'bonsai-speech2speech';
const DB_VERSION = 1;
const STORE = 'session';
const SESSION_KEY = 'current';

export type ConversationSession = {
  turns: ConversationTurn[];
  systemPrompt: string;
  speechBreak: SpeechBreakMode;
  modelId: ModelId;
  voice: VoiceId;
  ttsLanguage: TtsLanguage;
  updatedAt: number;
};

function normalizeSession(raw: unknown): ConversationSession | null {
  if (!raw || typeof raw !== 'object') return null;
  const session = raw as Partial<ConversationSession>;
  return {
    turns: Array.isArray(session.turns) ? session.turns : [],
    systemPrompt:
      typeof session.systemPrompt === 'string' ? session.systemPrompt : DEFAULT_SYSTEM_PROMPT,
    speechBreak:
      session.speechBreak === 'sentence' ||
      session.speechBreak === 'paragraph' ||
      session.speechBreak === 'none'
        ? session.speechBreak
        : 'paragraph',
    modelId: session.modelId ?? '4b',
    voice: session.voice ?? 'F1',
    ttsLanguage:
      typeof session.ttsLanguage === 'string' &&
      (TTS_LANGUAGE_CODES as readonly string[]).includes(session.ttsLanguage)
        ? (session.ttsLanguage as TtsLanguage)
        : 'na',
    updatedAt: typeof session.updatedAt === 'number' ? session.updatedAt : Date.now(),
  };
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error ?? new Error('Failed to open IndexedDB'));
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
  });
}

export async function loadSession(): Promise<ConversationSession | null> {
  if (typeof indexedDB === 'undefined') return null;
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const request = tx.objectStore(STORE).get(SESSION_KEY);
    request.onerror = () => reject(request.error ?? new Error('Failed to read session'));
    request.onsuccess = () => {
      resolve(normalizeSession(request.result));
    };
    tx.oncomplete = () => db.close();
  });
}

function cloneForStorage(session: ConversationSession): ConversationSession {
  return {
    turns: session.turns.map((turn) => ({
      id: String(turn.id),
      userText: String(turn.userText),
      assistantText: String(turn.assistantText),
      ...(typeof turn.llmPrompt === 'string' ? { llmPrompt: turn.llmPrompt } : {}),
    })),
    systemPrompt: String(session.systemPrompt),
    speechBreak: session.speechBreak,
    modelId: session.modelId,
    voice: session.voice,
    ttsLanguage: session.ttsLanguage,
    updatedAt: Date.now(),
  };
}

export async function saveSession(session: ConversationSession): Promise<void> {
  if (typeof indexedDB === 'undefined') return;
  const db = await openDb();
  const storable = cloneForStorage(session);
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(storable, SESSION_KEY);
    tx.onerror = () => reject(tx.error ?? new Error('Failed to save session'));
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
  });
}

export async function clearSession(): Promise<void> {
  if (typeof indexedDB === 'undefined') return;
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(SESSION_KEY);
    tx.onerror = () => reject(tx.error ?? new Error('Failed to clear session'));
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
  });
}
