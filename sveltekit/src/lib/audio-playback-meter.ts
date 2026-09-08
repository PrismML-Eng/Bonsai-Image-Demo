export type PlaybackMeter = {
  connect: (audio: HTMLAudioElement) => Promise<void>;
  disconnect: () => void;
  readLevel: () => number;
  destroy: () => void;
};

export function createPlaybackMeter(): PlaybackMeter {
  let ctx: AudioContext | null = null;
  let analyser: AnalyserNode | null = null;
  let source: MediaElementAudioSourceNode | null = null;
  let connectedElement: HTMLAudioElement | null = null;
  let smoothed = 0;
  let freqData: Uint8Array<ArrayBuffer> | null = null;

  function ensureContext(): AudioContext {
    if (!ctx) ctx = new AudioContext();
    return ctx;
  }

  async function resumeContext() {
    const context = ensureContext();
    if (context.state === 'suspended') {
      await context.resume();
    }
  }

  async function connect(audio: HTMLAudioElement) {
    const context = ensureContext();
    if (connectedElement === audio && source) {
      await resumeContext();
      return;
    }

    disconnect();

    analyser = context.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.35;
    freqData = new Uint8Array(analyser.frequencyBinCount);

    source = context.createMediaElementSource(audio);
    source.connect(analyser);
    analyser.connect(context.destination);
    connectedElement = audio;

    await resumeContext();
  }

  function disconnect() {
    try {
      source?.disconnect();
      analyser?.disconnect();
    } catch {
      // Nodes may already be disconnected when playback is interrupted.
    }
    source = null;
    analyser = null;
    connectedElement = null;
    smoothed = 0;
  }

  function readLevel(): number {
    if (!analyser || !freqData) {
      smoothed *= 0.75;
      return smoothed;
    }

    analyser.getByteFrequencyData(freqData);

    // Emphasise speech band (~150Hz–4kHz at 44.1kHz sample rate).
    const start = 3;
    const end = 56;
    let sum = 0;
    let peak = 0;
    for (let i = start; i < end; i += 1) {
      const v = freqData[i] / 255;
      sum += v;
      if (v > peak) peak = v;
    }

    const avg = sum / (end - start);
    const mixed = peak * 0.7 + avg * 0.3;
    const normalized = Math.min(1, Math.pow(mixed * 3.2, 0.7));

    if (normalized > smoothed) {
      smoothed += (normalized - smoothed) * 0.62;
    } else {
      smoothed += (normalized - smoothed) * 0.2;
    }

    return smoothed;
  }

  function destroy() {
    disconnect();
    void ctx?.close();
    ctx = null;
    freqData = null;
  }

  return { connect, disconnect, readLevel, destroy };
}
