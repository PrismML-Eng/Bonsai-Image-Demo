export const BONSAI_WORKER_URL = '/bonsai-llm-webgpu/assets/worker-CM5jAItV.js';
export const SUPERTONIC_WORKER_URL = '/speech2speech/supertonic-worker.js';
export const PAUSE_MS = 500;
export const POST_TTS_COOLDOWN_MS = 800;

export const MODELS = [
  { id: '1.7b', name: '1.7B', size: '470 MB' },
  { id: '4b', name: '4B', size: '1.1 GB' },
  { id: '8b', name: '8B', size: '2.2 GB' },
] as const;

export const VOICES = ['F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', 'M4', 'M5'] as const;

/** Supertonic 3 ISO codes plus language-agnostic `na`. */
export const TTS_LANGUAGE_CODES = [
  'na',
  'en',
  'es',
  'fr',
  'de',
  'it',
  'nl',
  'pt',
  'sv',
  'fi',
  'da',
  'el',
  'pl',
  'cs',
  'hu',
  'ro',
  'sk',
  'sl',
  'ru',
  'uk',
  'bg',
  'et',
  'lv',
  'lt',
  'hr',
  'ar',
  'tr',
  'hi',
  'vi',
  'ja',
  'ko',
  'id',
] as const;

export const TTS_LANGUAGE_GROUPS = [
  {
    label: 'Default',
    languages: [{ code: 'na', label: 'Language-agnostic (na)' }],
  },
  {
    label: 'Western / Central Europe',
    languages: [
      { code: 'en', label: 'English' },
      { code: 'es', label: 'Spanish' },
      { code: 'fr', label: 'French' },
      { code: 'de', label: 'German' },
      { code: 'it', label: 'Italian' },
      { code: 'nl', label: 'Dutch' },
      { code: 'pt', label: 'Portuguese' },
      { code: 'sv', label: 'Swedish' },
      { code: 'fi', label: 'Finnish' },
      { code: 'da', label: 'Danish' },
      { code: 'el', label: 'Greek' },
      { code: 'pl', label: 'Polish' },
      { code: 'cs', label: 'Czech' },
      { code: 'hu', label: 'Hungarian' },
      { code: 'ro', label: 'Romanian' },
      { code: 'sk', label: 'Slovak' },
      { code: 'sl', label: 'Slovenian' },
    ],
  },
  {
    label: 'Eastern Europe / Asia',
    languages: [
      { code: 'ru', label: 'Russian' },
      { code: 'uk', label: 'Ukrainian' },
      { code: 'bg', label: 'Bulgarian' },
      { code: 'et', label: 'Estonian' },
      { code: 'lv', label: 'Latvian' },
      { code: 'lt', label: 'Lithuanian' },
      { code: 'hr', label: 'Croatian' },
    ],
  },
  {
    label: 'Global',
    languages: [
      { code: 'ar', label: 'Arabic' },
      { code: 'tr', label: 'Turkish' },
      { code: 'hi', label: 'Hindi' },
      { code: 'vi', label: 'Vietnamese' },
      { code: 'ja', label: 'Japanese' },
      { code: 'ko', label: 'Korean' },
      { code: 'id', label: 'Indonesian' },
    ],
  },
] as const;

export type ModelId = (typeof MODELS)[number]['id'];
export type VoiceId = (typeof VOICES)[number];
export type TtsLanguage = (typeof TTS_LANGUAGE_CODES)[number];
export type Role = 'user' | 'assistant';

export type WorkerRole = 'system' | Role;

export type WorkerMessage = {
  role: WorkerRole;
  content: string;
};

export const OPENING_USER_MESSAGE =
  'Begin the conversation. Speak first with your opening line.';

export const DEFAULT_SYSTEM_PROMPT = `You are a friendly and helpful AI assistant who enjoys conversations.

Continue the conversation with a response of up to three sentences.

Output text rules:
- Plain text only.
- No Markdown.
- No bullet points.
- Use normal ASCII apostrophes and quotes.
- Use punctuation to control pacing.
- Do not use SSML.`;

export const SPEECH_BREAK_OPTIONS = [
  { value: 'sentence', label: 'Sentence' },
  { value: 'paragraph', label: 'Paragraph' },
  { value: 'none', label: 'None' },
] as const;

export type SpeechBreakMode = (typeof SPEECH_BREAK_OPTIONS)[number]['value'];

export type TtsSegment = {
  text: string;
  pauseAfterMs: number;
};

const ALLOWED_EXPRESSION_TAGS = new Set([
  'laugh',
  'breath',
  'sigh',
  'surprise',
  'scream',
  'throatclear',
  'sad',
  'angry',
  'cough',
  'yawn',
]);

const EXPRESSION_TAG_PATTERN = [...ALLOWED_EXPRESSION_TAGS].join('|');
const EXPRESSION_TAG_RE = new RegExp(`<(?:${EXPRESSION_TAG_PATTERN})>`, 'i');
const LAUGH_TAG_RE = /(?:<laugh>\s*)+/gi;

/** Supertonic 3 inline expression tags (experimental — work best in en/ko/ja). */
export const SUPERTONIC_EXPRESSION_TAGS = [...ALLOWED_EXPRESSION_TAGS] as const;

export type ConversationTurn = {
  id: string;
  userText: string;
  assistantText: string;
  /** Full prompt sent to Bonsai when this turn was generated. */
  llmPrompt?: string;
};

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
};

export type BonsaiWorkerMessage =
  | { status: 'progress_total'; progress?: number; loaded?: number; total?: number }
  | { status: 'loading'; data?: string }
  | { status: 'ready' }
  | { status: 'start' }
  | { status: 'update'; output?: string; tps?: number }
  | { status: 'complete'; output?: string }
  | { status: 'error'; data?: string };

export type SupertonicWorkerMessage =
  | { status: 'loading'; data?: string }
  | {
      status: 'progress';
      progress?: number;
      loaded?: number;
      total?: number;
      file?: string;
      phase?: 'start' | 'done';
    }
  | { status: 'ready'; backend?: string }
  | { status: 'speaking' }
  | { status: 'audio'; audio: ArrayBuffer; mimeType?: string }
  | { status: 'error'; data?: string };

export function messageId(prefix: string) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function workerMessages(messages: ChatMessage[]): WorkerMessage[] {
  return messages.map(({ role, content }) => ({ role, content }));
}

const NO_PRIOR_CONVERSATION = '(No prior conversation.)';

function buildBonsaiSystemPrompt(systemPrompt: string): string {
  const trimmed = systemPrompt.trim();
  return [
    trimmed || 'Continue the conversation naturally.',
  ].join('\n');
}

function formatConversationMessage(message: WorkerMessage): string {
  return `${WORKER_ROLE_LABELS[message.role]}: ${message.content.trim()}`;
}

function buildBonsaiUserPrompt(prior: WorkerMessage[], latest: WorkerMessage): string {
  const conversation =
    prior.length > 0 ? prior.map(formatConversationMessage).join('\n\n') : NO_PRIOR_CONVERSATION;

  return [
    '--',
    'Continue the conversation.',
    '',
    'Conversation so far:',
    '',
    conversation,
    '',
    'User (most recent message):',
    latest.content.trim(),
    '',
  ].join('\n');
}

export function needsOpeningGreeting(turns: ConversationTurn[]): boolean {
  if (turns.some((turn) => turn.assistantText.trim())) return false;
  if (turns.length === 0) return true;
  return turns.every((turn) => !turn.userText.trim());
}

export function buildOpeningWorkerMessages(systemPrompt: string): WorkerMessage[] {
  return buildWorkerMessages(systemPrompt, [
    { id: 'opening', role: 'user', content: OPENING_USER_MESSAGE },
  ]);
}

export function buildWorkerMessages(systemPrompt: string, messages: ChatMessage[]): WorkerMessage[] {
  const chat = workerMessages(messages);
  const system: WorkerMessage = {
    role: 'system',
    content: buildBonsaiSystemPrompt(systemPrompt),
  };

  if (chat.length === 0) return [system];

  const last = chat[chat.length - 1];
  if (last.role !== 'user') {
    return [system, ...chat];
  }

  const prior = chat.slice(0, -1);
  return [
    system,
    {
      role: 'user',
      content: buildBonsaiUserPrompt(prior, last),
    },
  ];
}

/** Readable prompt with system, conversation history, and latest user message called out. */
export function formatWorkerMessagesForDisplay(messages: WorkerMessage[]): string {
  const parts: string[] = [];
  const remaining = [...messages];

  const systemContents: string[] = [];
  while (remaining[0]?.role === 'system') {
    systemContents.push(remaining.shift()!.content.trim());
  }

  if (systemContents.length > 0) {
    parts.push(`[System]\n${systemContents.join('\n\n')}`);
  }

  if (remaining.length === 0) {
    return parts.join('\n\n');
  }

  if (remaining.length === 1 && remaining[0].role === 'user') {
    return [...parts, remaining[0].content.trim()].join('\n\n');
  }

  const last = remaining[remaining.length - 1];
  const conversation = last.role === 'user' ? remaining.slice(0, -1) : remaining;
  const latestUserText = last.role === 'user' ? last.content.trim() : null;

  if (conversation.length > 0) {
    const conversationBody = conversation
      .map((message) => `[${WORKER_ROLE_LABELS[message.role]}]\n${message.content.trim()}`)
      .join('\n\n');
    parts.push(`[Conversation]\n${conversationBody}`);
  }

  if (latestUserText !== null) {
    parts.push(latestUserText);
  } else {
    parts.push(
      remaining
        .map((message) => `[${WORKER_ROLE_LABELS[message.role]}]\n${message.content.trim()}`)
        .join('\n\n'),
    );
  }

  return parts.join('\n\n');
}

const WORKER_ROLE_LABELS: Record<WorkerRole, string> = {
  system: 'System',
  user: 'User',
  assistant: 'You',
};

export function turnsToChatMessages(turns: ConversationTurn[]): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (const turn of turns) {
    messages.push({ id: `${turn.id}-user`, role: 'user', content: turn.userText });
    messages.push({ id: `${turn.id}-assistant`, role: 'assistant', content: turn.assistantText });
  }
  return messages;
}

/** History for the LLM — omits assistant slots that are not finished yet. */
export function turnsToWorkerMessages(turns: ConversationTurn[]): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (const turn of turns) {
    if (turn.userText.trim()) {
      messages.push({ id: `${turn.id}-user`, role: 'user', content: turn.userText });
    }
    if (turn.assistantText) {
      messages.push({ id: `${turn.id}-assistant`, role: 'assistant', content: turn.assistantText });
    }
  }
  return messages;
}

/** History sent to the LLM when generating a specific turn (prior turns complete, target turn user-only). */
export function turnsToWorkerMessagesForTurn(turns: ConversationTurn[], turnIndex: number): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (let i = 0; i <= turnIndex; i++) {
    const turn = turns[i];
    if (!turn) continue;
    if (i < turnIndex) {
      if (turn.userText.trim()) {
        messages.push({ id: `${turn.id}-user`, role: 'user', content: turn.userText });
      }
      if (turn.assistantText) {
        messages.push({ id: `${turn.id}-assistant`, role: 'assistant', content: turn.assistantText });
      }
      continue;
    }
    const userContent = turn.userText.trim() || OPENING_USER_MESSAGE;
    messages.push({ id: `${turn.id}-user`, role: 'user', content: userContent });
  }
  return messages;
}

/** Reconstruct the LLM input prompt for a turn (uses the current system prompt). */
export function buildLlmPromptForTurn(
  systemPrompt: string,
  turns: ConversationTurn[],
  turnId: string,
): string {
  const turnIndex = turns.findIndex((turn) => turn.id === turnId);
  if (turnIndex < 0) return '';
  return formatWorkerMessagesForDisplay(
    buildWorkerMessages(systemPrompt, turnsToWorkerMessagesForTurn(turns, turnIndex)),
  );
}

export function getLlmPromptForTurn(
  systemPrompt: string,
  turns: ConversationTurn[],
  turnId: string,
): string {
  const turn = turns.find((item) => item.id === turnId);
  if (turn?.llmPrompt) return turn.llmPrompt;
  return buildLlmPromptForTurn(systemPrompt, turns, turnId);
}

export function hasSupertonicExpressionTags(text: string): boolean {
  return EXPRESSION_TAG_RE.test(text);
}

/** Expression tags need a concrete language; `na` often vocalizes the tag names. */
export function ttsLangForText(text: string, lang: TtsLanguage): TtsLanguage {
  if (lang !== 'na' || !hasSupertonicExpressionTags(text)) return lang;
  return 'en';
}

/** Strip dialogue role prefixes the model sometimes echoes in replies. */
export function normaliseAssistantOutput(input: string): string {
  let text = input.trim();
  while (/^You:\s*/i.test(text)) {
    text = text.replace(/^You:\s*/i, '').trimStart();
  }
  return text;
}

export function normaliseForSupertonic(input: string): string {
  let text = normaliseAssistantOutput(input)
    .normalize('NFKC')

    // Pull block-style expression tags onto the same line as the text that follows.
    .replace(new RegExp(`(<(?:${EXPRESSION_TAG_PATTERN})>)[ \\t]*\\n\\s*\\n+`, 'gi'), '$1 ')
    .replace(new RegExp(`(<(?:${EXPRESSION_TAG_PATTERN})>)[ \\t]*\\n+`, 'gi'), '$1 ')

    // Remove markdown emphasis while keeping the words.
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/_([^_]+)_/g, '$1')

    // Curly quotes and apostrophes.
    .replace(/[“”]/g, '"')
    .replace(/[‘’´`]/g, "'")

    // Dashes: em dash often sounds better as a pause.
    .replace(/\s*[—–]\s*/g, '. ')

    // Remove bullets / separators that TTS may read oddly.
    .replace(/^[-•]\s*/gm, '')

    // Remove unsupported angle-bracket markup but preserve all Supertonic expression tags.
    .replace(/<([^>]+)>/g, (match, tagName: string) => {
      const normalizedTag = tagName.trim().toLowerCase();
      return ALLOWED_EXPRESSION_TAGS.has(normalizedTag) ? `<${normalizedTag}>` : ' ';
    })

    // Clean spacing around punctuation.
    .replace(/\s+([,.!?;:])/g, '$1')
    .replace(/([,.!?;:])(?=\S)/g, '$1 ')
    .replace(/\.\s+\.\s+\./g, '...')
    .replace(/\s+/g, ' ')
    .trim();

  text = text.replace(LAUGH_TAG_RE, '<laugh> <laugh> <laugh> ').replace(/\s+/g, ' ').trim();

  return text;
}

export function splitForSupertonic(
  input: string,
  options: {
    maxChars?: number;
    pauseMs?: number;
  } = {},
): TtsSegment[] {
  const maxChars = options.maxChars ?? 240;
  const pauseMs = options.pauseMs ?? 300;
  const text = normaliseForSupertonic(input);
  if (!text) return [];

  const sentences = text.match(/[^.!?]+[.!?]+["']?|[^.!?]+$/g) ?? [text];

  const segments: string[] = [];
  let current = '';

  for (const rawSentence of sentences) {
    const sentence = rawSentence.trim();
    if (!sentence) continue;

    if ((current + ' ' + sentence).trim().length <= maxChars) {
      current = (current + ' ' + sentence).trim();
    } else {
      if (current) segments.push(current);
      current = sentence;
    }
  }

  if (current) segments.push(current);

  return segments.map((segment, index) => ({
    text: segment,
    pauseAfterMs: index === segments.length - 1 ? 0 : pauseMs,
  }));
}

/** Split assistant text for sequential TTS synthesis. */
export function splitForSupertonicWithMode(
  input: string,
  mode: SpeechBreakMode,
  options: {
    maxChars?: number;
    pauseMs?: number;
  } = {},
): TtsSegment[] {
  if (mode === 'none') {
    return splitForSupertonic(input, { maxChars: 10_000, pauseMs: 0, ...options });
  }

  if (mode === 'paragraph') {
    const paragraphs = input
      .trim()
      .split(/\n\s*\n+/)
      .map((part) => part.trim())
      .filter(Boolean);
    if (paragraphs.length === 0) return [];

    const segments: TtsSegment[] = [];
    for (let i = 0; i < paragraphs.length; i++) {
      const partSegments = splitForSupertonic(paragraphs[i], options);
      for (let j = 0; j < partSegments.length; j++) {
        const isLastPart = i === paragraphs.length - 1;
        const isLastSegment = j === partSegments.length - 1;
        segments.push({
          text: partSegments[j].text,
          pauseAfterMs:
            isLastPart && isLastSegment
              ? 0
              : isLastSegment
                ? PAUSE_MS
                : partSegments[j].pauseAfterMs,
        });
      }
    }
    return segments;
  }

  return splitForSupertonic(input, options);
}

/** Post-processed text for display (segment boundaries shown as blank lines). */
export function formatPostProcessedTtsText(input: string, mode: SpeechBreakMode = 'sentence'): string {
  return splitForSupertonicWithMode(input, mode)
    .map((segment) => segment.text)
    .join('\n\n');
}

export function formatBytes(value: number | undefined) {
  if (!Number.isFinite(value)) return '';
  const bytes = value as number;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

export function modelName(id: ModelId) {
  return MODELS.find((m) => m.id === id)?.name ?? 'Model';
}

function normalizeSpeech(text: string) {
  return text.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
}

/** Ignore mic input that mostly matches what Supertonic just played. */
export function isLikelyEcho(utterance: string, assistantText: string) {
  const heard = normalizeSpeech(utterance);
  const spoken = normalizeSpeech(assistantText);
  if (!heard || !spoken || heard.length < 12) return false;
  if (spoken.includes(heard) || heard.includes(spoken)) return true;

  const heardWords = new Set(heard.split(' '));
  const spokenWords = spoken.split(' ').filter(Boolean);
  if (spokenWords.length < 4) return false;
  const overlap = spokenWords.filter((word) => heardWords.has(word)).length;
  return overlap / spokenWords.length >= 0.55;
}

export type SpeechRecognitionAlternative = {
  transcript: string;
};

export type SpeechRecognitionResult = {
  readonly isFinal: boolean;
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
};

export type SpeechRecognitionResultList = {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
};

export type SpeechRecognitionEvent = Event & {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
};

export type SpeechRecognitionErrorEvent = Event & {
  error?: string;
};

export type BrowserSpeechRecognition = EventTarget & {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onend: ((event: Event) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onstart: ((event: Event) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
};

type SpeechRecognitionConstructor = new () => BrowserSpeechRecognition;

export function getSpeechRecognition(): SpeechRecognitionConstructor | null {
  if (typeof window === 'undefined') return null;
  const win = window as typeof window & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return win.SpeechRecognition ?? win.webkitSpeechRecognition ?? null;
}
