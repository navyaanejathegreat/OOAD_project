// Thin client for the FastAPI server. Vite proxies /api to http://127.0.0.1:8000.
async function get(path, params) {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  const res = await fetch(`/api${path}${qs}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const fetchMeta = () => get("/demo/meta");
export const fetchState = ({ vehicle, minute, model, origin, dest }) =>
  get("/demo/state", { vehicle, minute, model, origin, dest });
