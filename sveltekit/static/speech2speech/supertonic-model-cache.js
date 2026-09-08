const CACHE_NAME = "supertonic-3-v1";
const PREFER_WASM_KEY = "/__supertonic__/prefer-wasm";

/**
 * Fetch a URL with the Cache API so ~400 MB of ONNX assets survive page reloads.
 * Returns { buffer, cached } where cached=true means no network download.
 */
export async function fetchCachedArrayBuffer(url) {
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(url);
  if (hit) {
    return { buffer: await hit.arrayBuffer(), cached: true };
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url} (${response.status})`);
  }

  await cache.put(url, response.clone());
  return { buffer: await response.arrayBuffer(), cached: false };
}

export async function fetchCachedJson(url) {
  const { buffer } = await fetchCachedArrayBuffer(url);
  const text = new TextDecoder().decode(buffer);
  return JSON.parse(text);
}

export async function getPreferWasm() {
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(PREFER_WASM_KEY);
  if (!hit) return false;
  return (await hit.text()) === "1";
}

export async function setPreferWasm() {
  const cache = await caches.open(CACHE_NAME);
  await cache.put(PREFER_WASM_KEY, new Response("1"));
}
