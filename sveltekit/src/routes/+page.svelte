<script lang="ts">
  import { onMount } from 'svelte';
  import SimpleVoiceGraphic from '$lib/components/SimpleVoiceGraphic.svelte';
  import { createPlaybackMeter } from '$lib/audio-playback-meter';
  import { readViewMode, writeViewMode, type ViewMode } from '$lib/view-mode-store';
  import { clearSession, loadSession, saveSession } from '$lib/conversation-store';
  import { clearSystemPrompt, readSystemPrompt, writeSystemPrompt } from '$lib/system-prompt-store';
  import {
    BONSAI_WORKER_URL,
    SUPERTONIC_WORKER_URL,
    PAUSE_MS,
    POST_TTS_COOLDOWN_MS,
    MODELS,
    VOICES,
    SPEECH_BREAK_OPTIONS,
    TTS_LANGUAGE_GROUPS,
    buildWorkerMessages,
    needsOpeningGreeting,
    buildOpeningWorkerMessages,
    formatWorkerMessagesForDisplay,
    normaliseAssistantOutput,
    formatPostProcessedTtsText,
    getLlmPromptForTurn,
    splitForSupertonicWithMode,
    DEFAULT_SYSTEM_PROMPT,
    ttsLangForText,
    formatBytes,
    getSpeechRecognition,
    isLikelyEcho,
    messageId,
    modelName,
    turnsToWorkerMessages,
    type BonsaiWorkerMessage,
    type BrowserSpeechRecognition,
    type ConversationTurn,
    type ModelId,
    type SpeechBreakMode,
    type SupertonicWorkerMessage,
    type TtsSegment,
    type TtsLanguage,
    type VoiceId,
  } from '$lib/speech2speech';

  let modelId = $state<ModelId>('4b');
  let voice = $state<VoiceId>('F1');
  let ttsLanguage = $state<TtsLanguage>('na');
  let systemPrompt = $state(DEFAULT_SYSTEM_PROMPT);
  let speechBreak = $state<SpeechBreakMode>('paragraph');
  let turns = $state<ConversationTurn[]>([]);
  let loadedModelId = $state<ModelId | null>(null);
  let ttsReady = $state(false);
  let status = $state('Load models to begin listening.');
  let ttsStatus = $state('Supertonic loads with the language model.');
  let interim = $state('');
  let isListening = $state(false);
  let isLoadingModels = $state(false);
  let isThinking = $state(false);
  let isSpeaking = $state(false);
  let conversationActive = $state(false);
  let error = $state<string | null>(null);
  let tokensPerSecond = $state<number | null>(null);
  let speechSupported = $state(true);
  let speakingTurnId = $state<string | null>(null);
  let sessionHydrated = $state(false);
  let workersReady = $state(false);
  let viewMode = $state<ViewMode>('simple');

  const selectedModel = $derived(MODELS.find((m) => m.id === modelId) ?? MODELS[0]);
  const isReady = $derived(loadedModelId === modelId && ttsReady && !isLoadingModels);
  const simpleGraphicMode = $derived.by(() => {
    if (isLoadingModels) return 'loading' as const;
    if (isSpeaking) return 'speaking' as const;
    if (isThinking) return 'thinking' as const;
    return 'waiting' as const;
  });

  let bonsaiWorker: Worker | null = null;
  let ttsWorker: Worker | null = null;
  let recognition: BrowserSpeechRecognition | null = null;
  let pauseTimer: ReturnType<typeof setTimeout> | null = null;
  let pendingUtterance = '';
  let assistantDraft = '';
  let audioEl: HTMLAudioElement | null = null;
  const playbackMeter = createPlaybackMeter();
  const speechAudioLevel = {
    get current() {
      return playbackMeter.readLevel();
    },
  };
  let recognitionActive = false;
  let isThinkingFlag = false;
  let isSpeakingFlag = false;
  let isReadyFlag = false;
  let activeTurnId: string | null = null;
  let turnsSnapshot: ConversationTurn[] = [];
  let lastAssistantOutput = '';
  let systemPromptSnapshot = DEFAULT_SYSTEM_PROMPT;
  let speechBreakSnapshot: SpeechBreakMode = 'paragraph';
  let ttsLanguageSnapshot: TtsLanguage = 'na';
  let activeTtsLang: TtsLanguage = 'na';
  let speechSegments: TtsSegment[] = [];
  let speechSegmentIndex = 0;
  let segmentPauseTimer: ReturnType<typeof setTimeout> | null = null;
  let responseTextTab = $state<Record<string, 'original' | 'processed' | 'prompt'>>({});
  let resumeListeningAfterSpeak = false;
  let playbackGeneration = 0;
  let persistTimer: ReturnType<typeof setTimeout> | null = null;
  let persistInFlight = false;
  let persistQueued = false;

  $effect(() => {
    turnsSnapshot = turns;
  });

  $effect(() => {
    systemPromptSnapshot = systemPrompt;
  });

  $effect(() => {
    speechBreakSnapshot = speechBreak;
  });

  $effect(() => {
    ttsLanguageSnapshot = ttsLanguage;
  });

  $effect(() => {
    isReadyFlag = isReady;
  });

  $effect(() => {
    isThinkingFlag = isThinking;
  });

  $effect(() => {
    isSpeakingFlag = isSpeaking;
  });

  $effect(() => {
    if (!sessionHydrated || viewMode === 'simple') return;
    turns;
    speechBreak;
    modelId;
    voice;
    ttsLanguage;
    schedulePersist();
  });

  function handleSystemPromptInput(event: Event) {
    const value = (event.currentTarget as HTMLTextAreaElement).value;
    systemPrompt = value;
    writeSystemPrompt(value);
  }

  function resetSystemPrompt() {
    systemPrompt = DEFAULT_SYSTEM_PROMPT;
    writeSystemPrompt(DEFAULT_SYSTEM_PROMPT);
  }

  async function setViewMode(mode: ViewMode) {
    viewMode = mode;
    writeViewMode(mode);
    if (mode === 'simple') {
      await resetConversationState(true);
      loadedModelId = null;
      ttsReady = false;
      isLoadingModels = false;
      status = 'Load models to begin listening.';
      ttsStatus = 'Supertonic loads with the language model.';
    }
  }

  function schedulePersist() {
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = setTimeout(() => {
      void persistSession();
    }, 300);
  }

  function flushPersist() {
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = null;
    void persistSession();
  }

  async function hydrateSession() {
    if (viewMode === 'simple') {
      await clearSession();
      turns = [];
      conversationActive = false;
      interim = '';
      pendingUtterance = '';
      activeTurnId = null;
      speakingTurnId = null;
      assistantDraft = '';
      lastAssistantOutput = '';

      const storedPrompt = readSystemPrompt();
      if (storedPrompt !== null) {
        systemPrompt = storedPrompt;
      } else {
        writeSystemPrompt(systemPrompt);
      }
      sessionHydrated = true;
      return;
    }

    const saved = await loadSession();
    if (saved) {
      turns = saved.turns;
      speechBreak = saved.speechBreak;
      modelId = saved.modelId;
      voice = saved.voice;
      ttsLanguage = saved.ttsLanguage;

      const storedPrompt = readSystemPrompt();
      if (storedPrompt !== null) {
        systemPrompt = storedPrompt;
      } else {
        systemPrompt = saved.systemPrompt;
        writeSystemPrompt(saved.systemPrompt);
      }
    } else {
      const storedPrompt = readSystemPrompt();
      if (storedPrompt !== null) {
        systemPrompt = storedPrompt;
      } else {
        writeSystemPrompt(systemPrompt);
      }
    }
    sessionHydrated = true;
  }

  async function persistSession() {
    if (!sessionHydrated || viewMode === 'simple') return;
    if (persistInFlight) {
      persistQueued = true;
      return;
    }
    persistInFlight = true;
    do {
      persistQueued = false;
      await saveSession({
        turns,
        systemPrompt,
        speechBreak,
        modelId,
        voice,
        ttsLanguage,
        updatedAt: Date.now(),
      });
    } while (persistQueued);
    persistInFlight = false;
  }

  function finishLoadingIfReady() {
    if (!isLoadingModels || loadedModelId !== modelId || !ttsReady) return;
    isLoadingModels = false;
    ttsStatus = 'Supertonic ready.';
    if (speechSupported) {
      startConversation();
    } else {
      status = `${modelName(modelId)} + Supertonic ready.`;
    }
  }

  function clearPauseTimer() {
    if (pauseTimer) clearTimeout(pauseTimer);
    pauseTimer = null;
  }

  function startRecognition() {
    if (!recognition || !conversationActive || isThinkingFlag || isSpeakingFlag) return;
    if (recognitionActive) return;
    try {
      recognition.start();
      recognitionActive = true;
      isListening = true;
    } catch {
      recognitionActive = false;
      isListening = false;
    }
  }

  function stopRecognition() {
    clearPauseTimer();
    isListening = false;
    recognitionActive = false;
    try {
      recognition?.stop();
    } catch {
      /* ignore */
    }
  }

  function responseTextTabFor(turnId: string): 'original' | 'processed' | 'prompt' {
    return responseTextTab[turnId] ?? 'original';
  }

  function setResponseTextTab(turnId: string, tab: 'original' | 'processed' | 'prompt') {
    responseTextTab = { ...responseTextTab, [turnId]: tab };
  }

  function assistantTextForTab(turn: ConversationTurn): string {
    const tab = responseTextTabFor(turn.id);
    if (tab === 'processed') {
      return formatPostProcessedTtsText(turn.assistantText, speechBreak);
    }
    if (tab === 'prompt') {
      return getLlmPromptForTurn(systemPrompt, turns, turn.id);
    }
    return turn.assistantText;
  }

  function handleAssistantTextInput(turnId: string, event: Event) {
    const assistantText = (event.currentTarget as HTMLTextAreaElement).value;
    responseTextTab = { ...responseTextTab, [turnId]: 'original' };
    turns = turns.map((turn) => (turn.id === turnId ? { ...turn, assistantText } : turn));
  }

  function clearSegmentPauseTimer() {
    if (segmentPauseTimer) {
      clearTimeout(segmentPauseTimer);
      segmentPauseTimer = null;
    }
  }

  function finishSpeaking() {
    clearSegmentPauseTimer();
    playbackMeter.disconnect();
    isSpeakingFlag = false;
    isSpeaking = false;
    speechSegments = [];
    speechSegmentIndex = 0;
    speakingTurnId = null;
    status = conversationActive ? 'Listening...' : isReady ? `${selectedModel.name} + Supertonic ready.` : status;
    if (resumeListeningAfterSpeak && conversationActive) {
      setTimeout(startRecognition, POST_TTS_COOLDOWN_MS);
    }
  }

  function stopSpeaking() {
    playbackGeneration += 1;
    clearSegmentPauseTimer();
    audioEl?.pause();
    audioEl = null;
    finishSpeaking();
  }

  function synthesizeCurrentChunk() {
    const segment = speechSegments[speechSegmentIndex];
    if (!segment?.text.trim()) {
      finishSpeaking();
      return;
    }
    const total = speechSegments.length;
    ttsStatus =
      total > 1 ? `Synthesizing part ${speechSegmentIndex + 1} of ${total}...` : 'Synthesizing speech...';
    ttsWorker?.postMessage({
      type: 'speak',
      text: segment.text,
      voice,
      lang: activeTtsLang,
      speed: 1.05,
      steps: 8,
    });
  }

  function scheduleNextSegment(pauseMs: number) {
    clearSegmentPauseTimer();
    if (pauseMs > 0) {
      segmentPauseTimer = setTimeout(() => {
        segmentPauseTimer = null;
        synthesizeCurrentChunk();
      }, pauseMs);
      return;
    }
    synthesizeCurrentChunk();
  }

  function beginSpeaking(text: string, resumeListening: boolean, turnId: string | null = null) {
    clearPauseTimer();
    pendingUtterance = '';
    interim = '';
    stopRecognition();
    audioEl?.pause();
    playbackGeneration += 1;

    activeTtsLang = ttsLangForText(text, ttsLanguageSnapshot);
    speechSegments = splitForSupertonicWithMode(text, speechBreakSnapshot);
    if (speechSegments.length === 0 || !speechSegments.some((segment) => segment.text.trim())) {
      speakingTurnId = null;
      if (resumeListening && conversationActive) {
        setTimeout(startRecognition, POST_TTS_COOLDOWN_MS);
      }
      return;
    }

    lastAssistantOutput = text.trim();
    speechSegmentIndex = 0;
    resumeListeningAfterSpeak = resumeListening;
    speakingTurnId = turnId;
    isSpeakingFlag = true;
    isSpeaking = true;
    synthesizeCurrentChunk();
  }

  function speak(text: string, turnId: string | null) {
    beginSpeaking(text, conversationActive, turnId);
  }

  function replayTurn(turn: ConversationTurn) {
    if (!turn.assistantText.trim() || !isReadyFlag || isThinking || isSpeaking) return;
    status = 'Replaying response...';
    beginSpeaking(turn.assistantText, conversationActive, turn.id);
  }

  function sendOpeningMessage(): boolean {
    if (!isReady || isThinking || isSpeaking || !bonsaiWorker) return false;
    if (!needsOpeningGreeting(turns)) return false;

    clearPauseTimer();
    pendingUtterance = '';
    interim = '';
    isThinkingFlag = true;
    stopRecognition();
    error = null;
    isThinking = true;
    assistantDraft = '';
    status = 'Thinking...';

    const existingOpening = turns.find((turn) => !turn.userText.trim() && !turn.assistantText.trim());
    const turnId = existingOpening?.id ?? messageId('turn');
    activeTurnId = turnId;

    const workerPayload = buildOpeningWorkerMessages(systemPromptSnapshot);
    const llmPrompt = formatWorkerMessagesForDisplay(workerPayload);

    if (existingOpening) {
      turns = turns.map((turn) =>
        turn.id === turnId ? { ...turn, llmPrompt, assistantText: '' } : turn,
      );
    } else {
      turns = [
        ...turns,
        { id: turnId, userText: '', assistantText: '', llmPrompt },
      ];
    }

    bonsaiWorker.postMessage({ type: 'reset' });
    bonsaiWorker.postMessage({
      type: 'generate',
      data: workerPayload,
    });
    return true;
  }

  function sendUtterance(rawText: string) {
    const text = rawText.trim();
    if (!text || isThinkingFlag || isSpeakingFlag || !isReadyFlag) return;

    if (isLikelyEcho(text, lastAssistantOutput)) {
      pendingUtterance = '';
      interim = '';
      if (conversationActive) setTimeout(startRecognition, POST_TTS_COOLDOWN_MS);
      return;
    }

    clearPauseTimer();
    pendingUtterance = '';
    interim = '';
    isThinkingFlag = true;
    stopRecognition();
    error = null;
    isThinking = true;
    assistantDraft = '';
    status = 'Thinking...';

    const turnId = messageId('turn');
    activeTurnId = turnId;
    const pendingTurn: ConversationTurn = { id: turnId, userText: text, assistantText: '' };
    const nextTurns = [...turnsSnapshot, pendingTurn];
    const workerPayload = buildWorkerMessages(systemPromptSnapshot, turnsToWorkerMessages(nextTurns));
    pendingTurn.llmPrompt = formatWorkerMessagesForDisplay(workerPayload);
    turns = nextTurns;

    bonsaiWorker?.postMessage({ type: 'reset' });
    bonsaiWorker?.postMessage({
      type: 'generate',
      data: workerPayload,
    });
  }

  function updateActiveTurnAssistant(content: string) {
    if (!activeTurnId) return;
    turns = turns.map((turn) =>
      turn.id === activeTurnId ? { ...turn, assistantText: content } : turn,
    );
  }

  function handleBonsai(event: MessageEvent<BonsaiWorkerMessage>) {
    const message = event.data;
    switch (message.status) {
      case 'progress_total':
        status = `Downloading model ${formatBytes(message.loaded)} / ${formatBytes(message.total)}`;
        break;
      case 'loading':
        status = message.data ?? 'Loading model...';
        break;
      case 'ready':
        loadedModelId = modelId;
        if (ttsReady) {
          finishLoadingIfReady();
        } else {
          status = `${modelName(modelId)} ready. Waiting for Supertonic...`;
        }
        break;
      case 'start':
        assistantDraft = '';
        break;
      case 'update':
        if (message.tps != null) tokensPerSecond = message.tps;
        if (!message.output) break;
        assistantDraft += message.output;
        updateActiveTurnAssistant(normaliseAssistantOutput(assistantDraft));
        break;
      case 'complete': {
        const output = normaliseAssistantOutput(assistantDraft.trim() || message.output?.trim() || '');
        const completedTurnId = activeTurnId;
        updateActiveTurnAssistant(output);
        activeTurnId = null;
        isThinkingFlag = false;
        isThinking = false;
        status = 'Speaking...';
        void persistSession();
        speak(output, completedTurnId);
        break;
      }
      case 'error':
        isLoadingModels = false;
        loadedModelId = null;
        isThinkingFlag = false;
        isThinking = false;
        activeTurnId = null;
        error = message.data ?? 'The language model reported an error.';
        status = 'Error.';
        if (conversationActive) startRecognition();
        break;
    }
  }

  async function playAudioBlob(audio: ArrayBuffer, mimeType: string) {
    if (!isSpeaking) return;

    const generation = playbackGeneration;
    const blob = new Blob([audio], { type: mimeType || 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const audioNode = new Audio(url);
    audioEl = audioNode;
    await playbackMeter.connect(audioNode);
    audioNode.onended = () => {
      if (generation !== playbackGeneration) {
        URL.revokeObjectURL(url);
        return;
      }
      URL.revokeObjectURL(url);
      const pauseMs = speechSegments[speechSegmentIndex]?.pauseAfterMs ?? 0;
      speechSegmentIndex += 1;
      if (speechSegmentIndex < speechSegments.length) {
        ttsStatus = `Playing part ${speechSegmentIndex} of ${speechSegments.length}...`;
        scheduleNextSegment(pauseMs);
        return;
      }
      ttsStatus = 'Supertonic ready.';
      finishSpeaking();
    };
    audioNode.onerror = () => {
      URL.revokeObjectURL(url);
      if (generation !== playbackGeneration) return;
      error = 'Could not play the Supertonic audio.';
      finishSpeaking();
      if (resumeListeningAfterSpeak && conversationActive) {
        setTimeout(startRecognition, POST_TTS_COOLDOWN_MS);
      }
    };
    ttsStatus =
      speechSegments.length > 1
        ? `Playing part ${speechSegmentIndex + 1} of ${speechSegments.length}...`
        : 'Playing reply.';
    try {
      await audioNode.play();
    } catch {
      URL.revokeObjectURL(url);
      if (generation !== playbackGeneration) return;
      error = 'Could not play the Supertonic audio.';
      finishSpeaking();
      if (resumeListeningAfterSpeak && conversationActive) {
        setTimeout(startRecognition, POST_TTS_COOLDOWN_MS);
      }
    }
  }

  function handleTts(event: MessageEvent<SupertonicWorkerMessage>) {
    const message = event.data;
    switch (message.status) {
      case 'loading':
        if (isLoadingModels || !ttsReady) {
          ttsStatus = message.data ?? 'Loading Supertonic...';
        }
        break;
      case 'progress':
        if (isLoadingModels || !ttsReady) {
          if (message.file && message.loaded != null && message.total != null) {
            ttsStatus = `Supertonic: ${message.file} (${message.loaded}/${message.total})`;
          }
        }
        break;
      case 'ready':
        ttsReady = true;
        if (isLoadingModels) {
          if (loadedModelId === modelId) {
            finishLoadingIfReady();
          } else {
            ttsStatus = 'Supertonic ready.';
          }
        } else if (!isSpeaking) {
          ttsStatus = 'Supertonic ready.';
        }
        break;
      case 'speaking':
        if (speechSegments.length <= 1) {
          ttsStatus = 'Synthesizing speech...';
        }
        break;
      case 'audio':
        playAudioBlob(message.audio, message.mimeType || 'audio/wav');
        break;
      case 'error':
        if (isLoadingModels) {
          isLoadingModels = false;
          ttsReady = false;
        }
        ttsStatus = 'Supertonic error.';
        error = message.data ?? 'Supertonic reported an error.';
        finishSpeaking();
        if (resumeListeningAfterSpeak && conversationActive) {
          setTimeout(startRecognition, POST_TTS_COOLDOWN_MS);
        }
        break;
    }
  }

  async function resetConversationState(clearStorage = true) {
    conversationActive = false;
    resumeListeningAfterSpeak = false;
    stopRecognition();
    audioEl?.pause();
    audioEl = null;
    playbackMeter.disconnect();
    bonsaiWorker?.postMessage({ type: 'reset' });
    turns = [];
    interim = '';
    pendingUtterance = '';
    error = null;
    isThinkingFlag = false;
    isSpeaking = false;
    isSpeakingFlag = false;
    isThinking = false;
    tokensPerSecond = null;
    lastAssistantOutput = '';
    activeTurnId = null;
    speakingTurnId = null;
    assistantDraft = '';
    playbackGeneration += 1;
    speechSegments = [];
    speechSegmentIndex = 0;
    clearPauseTimer();
    clearSegmentPauseTimer();
    if (clearStorage) {
      await clearSession();
    }
  }

  async function loadModels() {
    if (!bonsaiWorker || !ttsWorker) {
      error = 'Workers are still starting. Try again in a moment.';
      return;
    }
    if (viewMode === 'simple') {
      await resetConversationState(true);
    }
    error = null;
    isLoadingModels = true;
    loadedModelId = null;
    ttsReady = false;
    status = `Loading ${selectedModel.name}...`;
    ttsStatus = 'Loading Supertonic...';
    bonsaiWorker.postMessage({ type: 'load', data: modelId });
    ttsWorker.postMessage({ type: 'load' });
  }

  function startConversation() {
    if (!speechSupported || !isReady) return;
    conversationActive = true;
    error = null;
    if (needsOpeningGreeting(turns)) {
      if (!sendOpeningMessage()) {
        status = 'Listening...';
        startRecognition();
      }
    } else {
      status = 'Listening...';
      startRecognition();
    }
  }

  function requestOpeningIfNeeded() {
    if (!conversationActive || !needsOpeningGreeting(turns)) return;
    if (isThinking || isSpeaking) return;
    sendOpeningMessage();
  }

  function stopConversation() {
    conversationActive = false;
    resumeListeningAfterSpeak = false;
    stopSpeaking();
    stopRecognition();
    status = isReady ? `${selectedModel.name} + Supertonic ready.` : 'Load models to begin listening.';
  }

  async function restartConversation() {
    await resetConversationState(true);
    if (viewMode !== 'simple') {
      await persistSession();
    }
    if (isReady && speechSupported) {
      startConversation();
    } else {
      status = isReady ? `${selectedModel.name} + Supertonic ready.` : 'Load models to begin listening.';
    }
    ttsStatus = ttsReady ? 'Supertonic ready.' : ttsStatus;
  }

  onMount(() => {
    const storedPrompt = readSystemPrompt();
    if (storedPrompt !== null) {
      systemPrompt = storedPrompt;
    }

    const storedViewMode = readViewMode();
    if (storedViewMode !== null) {
      viewMode = storedViewMode;
    }

    const handleBeforeUnload = () => {
      writeSystemPrompt(systemPrompt);
      flushPersist();
    };
    window.addEventListener('beforeunload', handleBeforeUnload);

    void (async () => {
      await hydrateSession();

      bonsaiWorker = new Worker(BONSAI_WORKER_URL, { type: 'module' });
      ttsWorker = new Worker(SUPERTONIC_WORKER_URL, { type: 'module' });
      bonsaiWorker.addEventListener('message', handleBonsai);
      ttsWorker.addEventListener('message', handleTts);
      workersReady = true;
      bonsaiWorker.postMessage({ type: 'check' });
      if (conversationActive) {
        requestOpeningIfNeeded();
      }
    })();

    const Recognition = getSpeechRecognition();
    speechSupported = Boolean(Recognition);
    if (Recognition) {
      recognition = new Recognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.onresult = (event) => {
        if (isThinkingFlag || isSpeakingFlag) return;

        let interimPart = '';
        for (let index = event.resultIndex; index < event.results.length; index += 1) {
          const result = event.results[index];
          const transcript = result[0]?.transcript ?? '';
          if (!transcript) continue;

          if (result.isFinal) {
            pendingUtterance = `${pendingUtterance} ${transcript}`.trim();
            clearPauseTimer();
            pauseTimer = setTimeout(() => {
              sendUtterance(pendingUtterance);
            }, PAUSE_MS);
          } else {
            interimPart += transcript;
          }
        }

        interim = `${pendingUtterance} ${interimPart}`.trim();
      };
      recognition.onerror = (event) => {
        if (event.error === 'no-speech') return;
        recognitionActive = false;
        isListening = false;
        error = event.error ? `Speech recognition: ${event.error}` : 'Speech recognition error.';
      };
      recognition.onstart = () => {
        recognitionActive = true;
        isListening = true;
      };
      recognition.onend = () => {
        recognitionActive = false;
        isListening = false;
        if (conversationActive && !isThinkingFlag && !isSpeakingFlag) {
          setTimeout(startRecognition, POST_TTS_COOLDOWN_MS);
        }
      };
    }

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      if (sessionHydrated) flushPersist();
      conversationActive = false;
      if (persistTimer) clearTimeout(persistTimer);
      clearPauseTimer();
      clearSegmentPauseTimer();
      recognitionActive = false;
      recognition?.abort();
      audioEl?.pause();
      playbackMeter.destroy();
      bonsaiWorker?.terminate();
      ttsWorker?.terminate();
      bonsaiWorker = null;
      ttsWorker = null;
    };
  });
</script>

<svelte:head>
  <title>Speech to speech</title>
</svelte:head>

<main class="app">
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </span>
        <div class="brand-copy">
          <span class="brand-name">Speech</span>
          <span class="brand-tagline">Speech to speech</span>
        </div>
      </div>
      <div class="view-toggle" role="tablist" aria-label="View mode">
        <button
          type="button"
          role="tab"
          class="view-toggle-btn"
          class:active={viewMode === 'simple'}
          aria-selected={viewMode === 'simple'}
          onclick={() => setViewMode('simple')}
        >
          Simple
        </button>
        <button
          type="button"
          role="tab"
          class="view-toggle-btn"
          class:active={viewMode === 'advanced'}
          aria-selected={viewMode === 'advanced'}
          onclick={() => setViewMode('advanced')}
        >
          Advanced
        </button>
      </div>
    </div>
  </header>

  <div class="app-body">
    {#if viewMode === 'simple'}
      <section class="simple-view" aria-label="Simple voice conversation">
        <div class="simple-stage">
          {#if !isReady && !isLoadingModels}
            <button
              type="button"
              class="btn primary simple-load"
              onclick={loadModels}
              disabled={!workersReady || !sessionHydrated || isThinking || isSpeaking}
            >
              Start
            </button>
          {:else if isLoadingModels}
            <p class="simple-caption">Loading models…</p>
          {/if}
          <SimpleVoiceGraphic mode={simpleGraphicMode} audioLevelSource={speechAudioLevel} />
        </div>
        {#if error}
          <p class="alert simple-alert">{error}</p>
        {/if}
        {#if !speechSupported}
          <p class="alert simple-alert">Chrome SpeechRecognition is not available in this browser.</p>
        {/if}
      </section>
    {:else}
    <section class="panel system-prompt">
      <div class="system-prompt-head">
        <span class="label">System prompt</span>
        <button
          type="button"
          class="btn text"
          onclick={resetSystemPrompt}
          disabled={!sessionHydrated || isThinking || isSpeaking}
        >
          Reset to default
        </button>
      </div>
      <textarea
        class="system-prompt-input"
        value={systemPrompt}
        oninput={handleSystemPromptInput}
        rows="5"
        disabled={!sessionHydrated || isThinking || isSpeaking}
        spellcheck="true"
      ></textarea>
      <p class="system-prompt-hint">
        {#if sessionHydrated}
          Sent as the first message on every turn. Saved automatically on every edit.
        {:else}
          Loading saved session…
        {/if}
      </p>
    </section>

    <section class="panel controls">
      <div class="control-row">
        <label class="field">
          <span class="label">Language model</span>
          <select
            bind:value={modelId}
            disabled={isLoadingModels || isThinking || isSpeaking || isListening}
            onchange={() => {
              loadedModelId = null;
            }}
          >
            {#each MODELS as model (model.id)}
              <option value={model.id}>{model.name} · {model.size}</option>
            {/each}
          </select>
        </label>

        <label class="field">
          <span class="label">Supertonic voice</span>
          <select bind:value={voice} disabled={isSpeaking}>
            {#each VOICES as item (item)}
              <option value={item}>{item}</option>
            {/each}
          </select>
        </label>

        <div class="actions">
          <button
            type="button"
            class="btn primary"
            onclick={loadModels}
            disabled={!workersReady || !sessionHydrated || isLoadingModels || isThinking || isSpeaking || isReady}
          >
            {#if isLoadingModels}
              <span class="spinner" aria-hidden="true"></span>
            {/if}
            {isReady ? 'Loaded' : 'Load'}
          </button>
        </div>
      </div>

      <div class="status-bar">
        <div class="status-main">
          <div class="mic-icon" class:active={isListening} aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" y1="19" x2="12" y2="23" />
              <line x1="8" y1="23" x2="16" y2="23" />
            </svg>
          </div>
          <div>
            <p class="status-text">{status}</p>
            <p class="status-sub">
              {ttsStatus}{#if tokensPerSecond != null} · {tokensPerSecond.toFixed(1)} tok/s{/if}
            </p>
          </div>
        </div>
        <div class="status-actions">
          {#if conversationActive}
            <button type="button" class="btn outline" onclick={stopConversation}>Stop</button>
          {:else}
            <button
              type="button"
              class="btn primary"
              onclick={startConversation}
              disabled={!speechSupported || !isReady || isThinking || isSpeaking}
            >
              Start
            </button>
          {/if}
        </div>
      </div>

      {#if !speechSupported}
        <p class="alert">Chrome SpeechRecognition is not available in this browser.</p>
      {/if}
      {#if error}
        <p class="alert">{error}</p>
      {/if}
    </section>

    <section class="grid two-col">
      <div class="panel">
        <span class="label">Heard</span>
        <p class="heard">{interim || 'Waiting for speech...'}</p>
      </div>
      <div class="panel">
        <span class="label">Audio output</span>
        <p class="audio-out" class:speaking={isSpeaking}>
          <span class="audio-icon" aria-hidden="true">
            {#if isSpeaking}
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M11 5L6 9H2v6h4l5 4V5zm8.5 3.5a9 9 0 0 1 0 7" stroke="currentColor" stroke-width="2" fill="none"/></svg>
            {:else}
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/></svg>
            {/if}
          </span>
          {isSpeaking ? 'Speaking reply…' : 'Audio plays automatically after each reply.'}
        </p>
      </div>
    </section>

    <div class="panel break-speech-bar">
      <label class="field break-field">
        <span class="label">Break speech at</span>
        <select bind:value={speechBreak} disabled={isSpeaking}>
          {#each SPEECH_BREAK_OPTIONS as option (option.value)}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>
      </label>
      <label class="field break-field">
        <span class="label">TTS language</span>
        <select bind:value={ttsLanguage} disabled={isSpeaking}>
          {#each TTS_LANGUAGE_GROUPS as group (group.label)}
            <optgroup label={group.label}>
              {#each group.languages as language (language.code)}
                <option value={language.code}>{language.label}</option>
              {/each}
            </optgroup>
          {/each}
        </select>
      </label>
    </div>

    <section class="panel conversation">
      <div class="conversation-head">
        <span class="label">Conversation</span>
        <div class="conversation-head-actions">
          <span class="turn-count" aria-hidden="true">{turns.length} turns</span>
          <button
            type="button"
            class="btn outline restart"
            onclick={restartConversation}
            disabled={isThinking || isSpeaking}
          >
            Restart
          </button>
        </div>
      </div>
      <div class="messages">
        {#if turns.length === 0}
          <p class="placeholder">
            Your assistant will greet you first, then listen after every 500ms pause in your speech.
          </p>
        {:else}
          {#each turns as turn (turn.id)}
            <div class="turn">
              {#if turn.userText.trim()}
                <div class="bubble user">{turn.userText}</div>
              {/if}
              <div class="bubble assistant">
                <div class="assistant-row">
                  <span class="avatar" aria-hidden="true">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="8" r="4" />
                      <path d="M4 20c0-4 4-6 8-6s8 2 8 6" />
                    </svg>
                  </span>
                  <div class="assistant-body">
                {#if turn.assistantText || responseTextTab[turn.id]}
                  <div class="response-tabs" role="tablist" aria-label="Response text view">
                    <button
                      type="button"
                      role="tab"
                      class="response-tab"
                      class:active={responseTextTabFor(turn.id) === 'original'}
                      aria-selected={responseTextTabFor(turn.id) === 'original'}
                      onclick={() => setResponseTextTab(turn.id, 'original')}
                    >
                      Original
                    </button>
                    <button
                      type="button"
                      role="tab"
                      class="response-tab"
                      class:active={responseTextTabFor(turn.id) === 'processed'}
                      aria-selected={responseTextTabFor(turn.id) === 'processed'}
                      onclick={() => setResponseTextTab(turn.id, 'processed')}
                    >
                      TTS processed
                    </button>
                    <button
                      type="button"
                      role="tab"
                      class="response-tab"
                      class:active={responseTextTabFor(turn.id) === 'prompt'}
                      aria-selected={responseTextTabFor(turn.id) === 'prompt'}
                      onclick={() => setResponseTextTab(turn.id, 'prompt')}
                    >
                      LLM prompt
                    </button>
                  </div>
                  {#if responseTextTabFor(turn.id) === 'original'}
                    <textarea
                      class="bubble-text response-editor"
                      value={turn.assistantText}
                      oninput={(event) => handleAssistantTextInput(turn.id, event)}
                      disabled={isThinking || isSpeaking}
                      aria-label="Edit original assistant response"
                    ></textarea>
                  {:else}
                    <p class="bubble-text" class:prompt-text={responseTextTabFor(turn.id) === 'prompt'}>
                      {assistantTextForTab(turn)}
                    </p>
                  {/if}
                {:else}
                  <p class="bubble-text thinking">Thinking…</p>
                {/if}
                {#if turn.assistantText}
                  {#if speakingTurnId === turn.id && isSpeaking}
                    <button
                      type="button"
                      class="btn play stop"
                      onclick={stopSpeaking}
                      aria-label="Stop playback"
                    >
                      Stop
                    </button>
                  {:else}
                    <button
                      type="button"
                      class="btn play"
                      onclick={() => replayTurn(turn)}
                      disabled={!isReady || isThinking || isSpeaking}
                      aria-label="Play assistant reply"
                    >
                      Play
                    </button>
                  {/if}
                {/if}
                  </div>
                </div>
              </div>
            </div>
          {/each}
        {/if}
      </div>
    </section>
    {/if}
  </div>
</main>

<style>
  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background:
      radial-gradient(ellipse 80% 50% at 50% -20%, rgba(16, 163, 127, 0.08), transparent),
      var(--bg);
  }

  .topbar {
    position: sticky;
    top: 0;
    z-index: 10;
    border-bottom: 1px solid var(--border);
    background: rgba(13, 13, 13, 0.82);
    backdrop-filter: blur(12px);
  }

  .topbar-inner {
    max-width: 52rem;
    margin: 0 auto;
    padding: 0.875rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    min-width: 0;
  }

  .brand-mark {
    width: 2rem;
    height: 2rem;
    display: grid;
    place-items: center;
    border-radius: var(--radius-md);
    background: linear-gradient(145deg, var(--accent), #0d8c6d);
    color: white;
    flex-shrink: 0;
  }

  .brand-mark svg {
    display: block;
  }

  .brand-copy {
    display: grid;
    gap: 0.05rem;
    min-width: 0;
  }

  .brand-name {
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
  }

  .brand-tagline {
    font-size: 0.75rem;
    color: var(--text-tertiary);
  }

  .view-toggle {
    display: inline-flex;
    padding: 0.2rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-full);
    background: var(--bg-muted);
    flex-shrink: 0;
  }

  .view-toggle-btn {
    min-height: 2rem;
    padding: 0 0.9rem;
    border: none;
    border-radius: var(--radius-full);
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.8125rem;
    font-weight: 500;
  }

  .view-toggle-btn.active {
    background: var(--bg-elevated);
    color: var(--text);
    box-shadow: var(--shadow-sm);
  }

  .app-body {
    flex: 1;
    width: 100%;
    max-width: 52rem;
    margin: 0 auto;
    padding: 1.5rem 1.25rem 3rem;
    display: grid;
    gap: 1rem;
    align-content: start;
  }

  .simple-view {
    display: grid;
    place-items: center;
    min-height: calc(100vh - 8rem);
    padding: 1rem 0 2rem;
  }

  .simple-stage {
    display: grid;
    gap: 1.5rem;
    justify-items: center;
  }

  .simple-load {
    min-width: 10rem;
    min-height: 2.75rem;
    padding: 0 1.75rem;
    font-size: 0.9375rem;
    font-weight: 600;
    border-radius: var(--radius-full);
    box-shadow: var(--shadow-md);
  }

  .simple-caption {
    margin: 0;
    font-size: 0.875rem;
    color: var(--text-secondary);
  }

  .simple-alert {
    margin-top: 1rem;
    max-width: 24rem;
    text-align: center;
  }

  .panel {
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    background: var(--bg-elevated);
    padding: 1rem 1.125rem;
  }

  .system-prompt-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.625rem;
  }

  .system-prompt-input {
    width: 100%;
    min-height: 7rem;
    padding: 0.875rem 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--bg-muted);
    color: var(--text);
    resize: vertical;
    line-height: 1.55;
    font-family: var(--font-mono);
    font-size: 0.8125rem;
  }

  .system-prompt-input:disabled {
    opacity: 0.55;
  }

  .system-prompt-hint {
    margin: 0.5rem 0 0;
    font-size: 0.75rem;
    color: var(--text-tertiary);
  }

  .btn.text {
    min-height: auto;
    padding: 0.35rem 0.5rem;
    background: transparent;
    border: none;
    color: var(--accent);
    font-size: 0.8125rem;
    font-weight: 500;
  }

  .btn.text:disabled {
    color: var(--text-tertiary);
  }

  .break-speech-bar {
    display: grid;
    gap: 0.75rem;
    padding: 0.875rem 1.125rem;
  }

  @media (min-width: 640px) {
    .break-speech-bar {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      align-items: end;
    }
  }

  .break-field {
    min-width: 0;
  }

  .btn.restart {
    flex-shrink: 0;
  }

  .control-row {
    display: grid;
    gap: 0.75rem;
  }

  @media (min-width: 768px) {
    .control-row {
      grid-template-columns: 1fr 1fr auto;
      align-items: end;
    }
  }

  .field {
    display: grid;
    gap: 0.4rem;
  }

  .label {
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-tertiary);
  }

  select {
    width: 100%;
    height: 2.5rem;
    padding: 0 0.75rem;
    border: 1px solid var(--border);
    background: var(--bg-muted);
    font-size: 0.875rem;
  }

  .actions {
    display: flex;
    gap: 0.5rem;
    align-items: end;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    min-height: 2.5rem;
    padding: 0 1rem;
    border: 1px solid transparent;
    font-size: 0.875rem;
    font-weight: 500;
    border-radius: var(--radius-md);
  }

  .btn.primary {
    background: var(--text);
    color: var(--bg);
  }

  .btn.primary:hover:not(:disabled) {
    background: #ffffff;
  }

  .btn.outline {
    background: transparent;
    border-color: var(--border-strong);
    color: var(--text);
  }

  .btn.outline:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .btn.play {
    margin-top: 0.625rem;
    min-height: 1.875rem;
    padding: 0 0.75rem;
    font-size: 0.75rem;
    background: var(--bg-muted);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    border-radius: var(--radius-full);
  }

  .btn.play:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text);
  }

  .btn.play.stop {
    background: var(--danger-soft);
    border-color: rgba(239, 68, 68, 0.2);
    color: var(--danger);
  }

  .spinner {
    width: 1rem;
    height: 1rem;
    border: 2px solid rgba(0, 0, 0, 0.15);
    border-top-color: var(--bg);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .status-bar {
    margin-top: 0.875rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.875rem 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--bg-muted);
  }

  .status-main {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    min-width: 0;
  }

  .mic-icon {
    width: 2.5rem;
    height: 2.5rem;
    display: grid;
    place-items: center;
    border-radius: var(--radius-md);
    background: var(--bg-hover);
    color: var(--text-secondary);
    flex-shrink: 0;
  }

  .mic-icon.active {
    background: var(--accent-soft);
    color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent-ring);
  }

  .status-text {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text);
  }

  .status-sub {
    margin: 0.1rem 0 0;
    font-size: 0.75rem;
    color: var(--text-tertiary);
  }

  .alert {
    margin: 0.75rem 0 0;
    padding: 0.75rem 0.875rem;
    border-radius: var(--radius-md);
    border: 1px solid rgba(239, 68, 68, 0.2);
    background: var(--danger-soft);
    color: #fca5a5;
    font-size: 0.8125rem;
  }

  .grid.two-col {
    display: grid;
    gap: 1rem;
  }

  @media (min-width: 768px) {
    .grid.two-col {
      grid-template-columns: 1fr 1fr;
    }
  }

  .heard {
    margin: 0.625rem 0 0;
    min-height: 4.5rem;
    font-size: 1.0625rem;
    line-height: 1.65;
    color: var(--text);
    font-weight: 400;
  }

  .audio-out {
    margin: 0.625rem 0 0;
    min-height: 4.5rem;
    display: flex;
    align-items: center;
    gap: 0.625rem;
    color: var(--text-secondary);
    font-size: 0.875rem;
  }

  .audio-icon {
    display: grid;
    place-items: center;
    width: 2rem;
    height: 2rem;
    border-radius: var(--radius-md);
    background: var(--bg-muted);
    color: var(--text-tertiary);
    flex-shrink: 0;
  }

  .audio-out.speaking {
    color: var(--text);
  }

  .audio-out.speaking .audio-icon {
    background: var(--accent-soft);
    color: var(--accent);
    animation: pulse 1.4s ease-in-out infinite;
  }

  @keyframes pulse {
    50% {
      opacity: 0.55;
    }
  }

  .conversation {
    padding-bottom: 1.25rem;
  }

  .conversation-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
  }

  .conversation-head-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .turn-count {
    font-size: 0.75rem;
    color: var(--text-tertiary);
  }

  .messages {
    display: grid;
    gap: 1.5rem;
  }

  .turn {
    display: grid;
    gap: 0.75rem;
  }

  .placeholder {
    margin: 0;
    padding: 1.25rem;
    border-radius: var(--radius-lg);
    background: var(--bg-muted);
    color: var(--text-secondary);
    font-size: 0.875rem;
    text-align: center;
    line-height: 1.6;
  }

  .bubble {
    max-width: min(85%, 28rem);
    padding: 0.75rem 1rem;
    border-radius: var(--radius-xl);
    font-size: 0.9375rem;
    line-height: 1.6;
  }

  .bubble-text {
    margin: 0;
    white-space: pre-wrap;
  }

  .bubble-text.thinking {
    color: var(--text-secondary);
  }

  .bubble-text.prompt-text {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    line-height: 1.5;
    color: var(--text-secondary);
  }

  .response-editor {
    display: block;
    width: 100%;
    box-sizing: border-box;
    min-height: 5rem;
    padding: 0;
    border: 0;
    outline: 0;
    resize: vertical;
    background: transparent;
    color: inherit;
    font: inherit;
    line-height: 1.6;
  }

  .response-editor:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .bubble.user {
    margin-left: auto;
    background: var(--user-bubble);
    color: var(--text);
    border-bottom-right-radius: var(--radius-sm);
  }

  .bubble.assistant {
    width: 100%;
    max-width: 100%;
    padding: 0;
    border: none;
    background: transparent;
    border-radius: 0;
  }

  .assistant-row {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.875rem;
    align-items: start;
    width: 100%;
  }

  .avatar {
    width: 1.75rem;
    height: 1.75rem;
    display: grid;
    place-items: center;
    border-radius: var(--radius-md);
    background: linear-gradient(145deg, var(--accent), #0d8c6d);
    color: white;
    flex-shrink: 0;
    margin-top: 0.1rem;
  }

  .avatar svg {
    display: block;
  }

  .assistant-body {
    min-width: 0;
    padding-top: 0.05rem;
  }

  .response-tabs {
    display: inline-flex;
    gap: 0.2rem;
    margin-bottom: 0.625rem;
    padding: 0.2rem;
    border-radius: var(--radius-md);
    background: var(--bg-muted);
    border: 1px solid var(--border);
  }

  .response-tab {
    padding: 0.35rem 0.65rem;
    border: 0;
    border-radius: calc(var(--radius-md) - 2px);
    background: transparent;
    color: var(--text-tertiary);
    font: inherit;
    font-size: 0.75rem;
    font-weight: 500;
    cursor: pointer;
  }

  .response-tab:hover {
    color: var(--text-secondary);
  }

  .response-tab.active {
    background: var(--bg-elevated);
    color: var(--text);
    box-shadow: var(--shadow-sm);
  }
</style>
