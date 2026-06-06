// src/utils/api.js
const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "API error");
  }
  return res.json();
}

export const api = {
  health:  ()           => request("/health"),
  models:  ()           => request("/models"),
  tasks:   ()           => request("/tasks"),
  history: (limit = 20) => request(`/history?limit=${limit}`),
  clearHistory: ()      => request("/history", { method: "DELETE" }),

  validate: (smiles) =>
    request("/validate", {
      method: "POST",
      body: JSON.stringify({ smiles }),
    }),

  predict: (smiles, model = "gin") =>
    request("/predict", {
      method: "POST",
      body: JSON.stringify({ smiles, model }),
    }),

  predictBatch: (smiles_list, model = "gin") =>
    request("/predict/batch", {
      method: "POST",
      body: JSON.stringify({ smiles_list, model }),
    }),

  predictCompare: (smiles) =>
    request("/predict/compare", {
      method: "POST",
      body: JSON.stringify({ smiles }),
    }),

  predictCSV: async (file, model = "gin") => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/predict/csv?model=${model}`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Upload failed");
    }
    return res.blob();
  },
};
