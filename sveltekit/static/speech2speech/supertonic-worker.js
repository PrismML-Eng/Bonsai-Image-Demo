import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.bundle.min.mjs";
import {
  AVAILABLE_LANGS,
  loadTextToSpeech,
  loadVoiceStyle,
  writeWavFile,
} from "./supertonic-helper.js";
import { getPreferWasm, setPreferWasm } from "./supertonic-model-cache.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.numThreads = 1;

const HF_BASE = "https://huggingface.co/Supertone/supertonic-3/resolve/main";
const ONNX_DIR = `${HF_BASE}/onnx`;
const VOICE_BASE = `${HF_BASE}/voice_styles`;
const WEBGPU_LOAD_TIMEOUT_MS = 4 * 60 * 1000;

const MODEL_HINTS = {
  "Duration Predictor": "~4 MB",
  "Text Encoder": "~36 MB",
  "Vector Estimator": "~257 MB — slowest step",
  Vocoder: "~101 MB",
  "Text processor": "tokenizer config",
  Configuration: "model config",
};

let loadPromise = null;
let textToSpeech = null;
let activeBackend = "wasm";
const styleCache = new Map();

function post(status, data = {}, transfer = []) {
  self.postMessage({ status, ...data }, transfer);
}

function normalizeLang(lang) {
  const code = typeof lang === "string" ? lang.trim().toLowerCase() : "na";
  return AVAILABLE_LANGS.includes(code) ? code : "na";
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)), ms);
    }),
  ]);
}

function reportLoadProgress(modelName, completed, total, phase) {
  const hint = MODEL_HINTS[modelName] ?? "";
  const step = Math.min(total, Math.max(1, phase === "start" ? completed + 1 : completed));
  const hintText = hint ? ` (${hint})` : "";

  if (phase === "start") {
    post("loading", {
      data: `Loading ${modelName}${hintText} — step ${step}/${total}. Models are cached locally after the first download.`,
    });
  }

  post("progress", {
    file: modelName,
    loaded: step,
    total,
    progress: total > 0 ? step / total : 0,
    phase,
  });
}

async function createTextToSpeech() {
  const sessionOptions = { graphOptimizationLevel: "all" };
  const progressCallback = reportLoadProgress;

  if (await getPreferWasm()) {
    post("loading", { data: "Loading Supertonic from browser cache (WASM)..." });
    const result = await loadTextToSpeech(
      ONNX_DIR,
      { ...sessionOptions, executionProviders: ["wasm"] },
      progressCallback,
    );
    activeBackend = "wasm";
    return result.textToSpeech;
  }

  try {
    post("loading", { data: "Trying WebGPU backend..." });
    const result = await withTimeout(
      loadTextToSpeech(
        ONNX_DIR,
        { ...sessionOptions, executionProviders: ["webgpu"] },
        progressCallback,
      ),
      WEBGPU_LOAD_TIMEOUT_MS,
      "WebGPU Supertonic load",
    );
    activeBackend = "webgpu";
    return result.textToSpeech;
  } catch (webgpuError) {
    console.warn("Supertonic WebGPU load failed; falling back to WASM.", webgpuError);
    await setPreferWasm();
    post("loading", { data: "WebGPU unavailable or too slow; loading from cache with WASM..." });
    const result = await loadTextToSpeech(
      ONNX_DIR,
      { ...sessionOptions, executionProviders: ["wasm"] },
      progressCallback,
    );
    activeBackend = "wasm";
    return result.textToSpeech;
  }
}

function getTextToSpeech() {
  if (!loadPromise) {
    post("loading", { data: "Loading Supertonic 3..." });

    loadPromise = createTextToSpeech()
      .then((engine) => {
        textToSpeech = engine;
        post("ready", { backend: activeBackend });
        return engine;
      })
      .catch((error) => {
        loadPromise = null;
        throw error;
      });
  }
  return loadPromise;
}

async function getVoiceStyle(voice) {
  const voiceId = typeof voice === "string" && voice ? voice : "F1";
  if (styleCache.has(voiceId)) return styleCache.get(voiceId);

  const style = await loadVoiceStyle([`${VOICE_BASE}/${voiceId}.json`]);
  styleCache.set(voiceId, style);
  return style;
}

self.addEventListener("message", async (event) => {
  const { type, text, voice = "F1", speed = 1.05, steps = 8, lang = "na" } = event.data ?? {};

  try {
    if (type === "load") {
      await getTextToSpeech();
      return;
    }

    if (type === "speak") {
      const trimmed = String(text ?? "").trim();
      if (!trimmed) return;

      const engine = await getTextToSpeech();
      const style = await getVoiceStyle(voice);
      const language = normalizeLang(lang);

      post("speaking");
      const { wav } = await engine.call(trimmed, language, style, steps, speed);
      const arrayBuffer = writeWavFile(wav, engine.sampleRate);
      post("audio", { audio: arrayBuffer, mimeType: "audio/wav" }, [arrayBuffer]);
    }
  } catch (error) {
    loadPromise = null;
    textToSpeech = null;
    styleCache.clear();
    post("error", { data: error instanceof Error ? error.message : String(error) });
  }
});
