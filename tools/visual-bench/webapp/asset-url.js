const appBase = new URL('./', import.meta.url);

export function assetUrl(path, base = appBase) {
  return new URL(String(path).replace(/^\/(?!\/)/, ''), base).href;
}
