// src/App.jsx
import { useState, useEffect, useRef, useCallback } from "react";
import { Sun, Moon, FlaskConical, GitCompare, History, Upload, Download } from "lucide-react";
import toast, { Toaster } from "react-hot-toast";

import { useTheme } from "./context/ThemeContext";
import { useDebounce, useLocalStorage } from "./hooks/useDebounce";
import { api } from "./utils/api";

import ModelSelector    from "./components/ModelSelector";
import MoleculeViewer   from "./components/MoleculeViewer";
import ConfidenceGauge  from "./components/ConfidenceGauge";
import ToxicityRadar    from "./components/ToxicityRadar";
import TaskBars         from "./components/TaskBars";
import CompareMode      from "./components/CompareMode";
import PredictionHistory from "./components/PredictionHistory";
import BatchUpload      from "./components/BatchUpload";

import "./App.css";

const EXAMPLES = [
  { label: "Aspirin",      smiles: "CC(=O)Oc1ccccc1C(=O)O",         safe: true  },
  { label: "Caffeine",     smiles: "Cn1c(=O)c2c(ncn2C)n(c1=O)C",    safe: true  },
  { label: "Ethanol",      smiles: "CCO",                             safe: true  },
  { label: "Bisphenol A",  smiles: "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", safe: false },
  { label: "Anthracene",   smiles: "c1ccc2cc3ccccc3cc2c1",           safe: false },
  { label: "CCl₄",         smiles: "ClC(Cl)(Cl)Cl",                  safe: false },
];

const TABS = [
  { id: "predict",  label: "Predict",  icon: FlaskConical },
  { id: "compare",  label: "Compare",  icon: GitCompare   },
  { id: "batch",    label: "Batch",    icon: Upload       },
  { id: "history",  label: "History",  icon: History      },
];

export default function App() {
  const { theme, toggle } = useTheme();

  // State
  const [tab,          setTab]          = useState("predict");
  const [smiles,       setSmiles]       = useState("");
  const [model,        setModel]        = useState("gin");
  const [result,       setResult]       = useState(null);
  const [compareData,  setCompareData]  = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [error,        setError]        = useState("");
  const [apiStatus,    setApiStatus]    = useState("unknown");
  const [available,    setAvailable]    = useState([]);
  const [validation,   setValidation]   = useState(null);
  const [vizMode,      setVizMode]      = useState("bars"); // bars | radar
  const [history,      setHistory]      = useLocalStorage("tox21-history", []);

  const inputRef     = useRef(null);
  const debouncedSmi = useDebounce(smiles, 500);

  // API health check
  useEffect(() => {
    api.health()
      .then(data => {
        setApiStatus("ok");
        setAvailable(data.available || []);
      })
      .catch(() => setApiStatus("error"));
  }, []);

  // Live SMILES validation
  useEffect(() => {
    if (!debouncedSmi.trim()) { setValidation(null); return; }
    api.validate(debouncedSmi)
      .then(v => setValidation(v))
      .catch(() => setValidation(null));
  }, [debouncedSmi]);

  // Predict
  const handlePredict = useCallback(async (e) => {
    e?.preventDefault();
    if (!smiles.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const r = await api.predict(smiles.trim(), model);
      setResult(r);
      // Save to local history
      const newHistory = [r, ...history].slice(0, 50);
      setHistory(newHistory);
      if (r.toxic_tasks?.length > 0) {
        toast.error(`⚠ Toxic in ${r.toxic_tasks.length} assay${r.toxic_tasks.length > 1 ? "s" : ""}`, { duration: 3000 });
      } else {
        toast.success("✓ No toxicity detected", { duration: 2000 });
      }
    } catch (e) {
      setError(e.message);
      toast.error(e.message);
    } finally {
      setLoading(false);
    }
  }, [smiles, model, history, setHistory]);

  // Compare
  const handleCompare = useCallback(async () => {
    if (!smiles.trim()) return;
    setCompareLoading(true);
    setCompareData(null);
    try {
      const r = await api.predictCompare(smiles.trim());
      setCompareData(r);
      toast.success("Comparison complete");
    } catch (e) {
      toast.error(e.message);
    } finally {
      setCompareLoading(false);
    }
  }, [smiles]);

  // Keyboard shortcut
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Enter" && e.ctrlKey) handlePredict();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handlePredict]);

  // Export prediction as JSON
  const exportJSON = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = `tox21_${result.id || "prediction"}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Exported as JSON");
  };

  // Reload from history
  const handleReload = (item) => {
    setSmiles(item.smiles);
    setModel(item.model || "gin");
    setResult(item);
    setTab("predict");
    toast("Loaded from history", { icon: "↩" });
  };

  // Clear history
  const handleClearHistory = async () => {
    setHistory([]);
    try { await api.clearHistory(); } catch {}
    toast.success("History cleared");
  };

  // Validation indicator
  const validClass = validation
    ? validation.valid ? "input-valid" : "input-invalid"
    : "";

  return (
    <div className="app">
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "var(--surface2)",
            color: "var(--text-primary)",
            border: "1px solid var(--border)",
            fontFamily: "'Syne', sans-serif",
            fontSize: "13px",
          },
        }}
      />

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">⬡</div>
            <div>
              <div className="logo-name">Tox21 GNN</div>
              <div className="logo-sub">Molecular Toxicity Classifier</div>
            </div>
          </div>

          <nav className="header-nav">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={`nav-btn ${tab === id ? "nav-btn-active" : ""}`}
                onClick={() => setTab(id)}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </nav>

          <div className="header-right">
            <div className={`api-dot api-dot-${apiStatus}`} title={`API ${apiStatus}`} />
            <span className="api-label">{apiStatus === "ok" ? "Online" : apiStatus === "error" ? "Offline" : "…"}</span>
            <button className="theme-btn" onClick={toggle} title="Toggle theme">
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">

        {/* ════ PREDICT TAB ════ */}
        {tab === "predict" && (
          <div className="predict-layout">

            {/* Left column — input */}
            <div className="left-col">
              <div className="card">
                <h2 className="card-title">Input</h2>

                <form onSubmit={handlePredict}>
                  {/* SMILES input */}
                  <div className="field">
                    <label className="field-label">
                      SMILES String
                      {validation && (
                        <span className={`valid-badge ${validation.valid ? "valid-ok" : "valid-err"}`}>
                          {validation.valid ? `✓ ${validation.atoms} atoms` : "✗ Invalid"}
                        </span>
                      )}
                    </label>
                    <textarea
                      ref={inputRef}
                      className={`smiles-input ${validClass}`}
                      value={smiles}
                      onChange={e => { setSmiles(e.target.value); setResult(null); setError(""); }}
                      placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
                      rows={3}
                      spellCheck={false}
                      autoComplete="off"
                    />
                    <span className="input-hint">Ctrl+Enter to predict</span>
                  </div>

                  {/* Model selector */}
                  <ModelSelector
                    selected={model}
                    onChange={setModel}
                    available={available}
                  />

                  {/* Buttons */}
                  <div className="btn-row">
                    <button
                      type="submit"
                      className="btn-primary"
                      disabled={!smiles.trim() || loading}
                    >
                      {loading ? <><div className="spinner btn-spinner" /> Analyzing…</> : "⬡ Predict Toxicity"}
                    </button>

                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={handleCompare}
                      disabled={!smiles.trim() || compareLoading}
                    >
                      {compareLoading ? <><div className="spinner btn-spinner" /> Comparing…</> : <><GitCompare size={14} /> Compare All</>}
                    </button>
                  </div>
                </form>

                {/* Examples */}
                <div className="examples">
                  <span className="examples-label">Examples:</span>
                  {EXAMPLES.map(ex => (
                    <button
                      key={ex.label}
                      className={`example-chip ${ex.safe ? "example-safe" : "example-toxic"}`}
                      onClick={() => { setSmiles(ex.smiles); setResult(null); setError(""); }}
                    >
                      {ex.label}
                    </button>
                  ))}
                </div>

                {error && <div className="alert alert-error">{error}</div>}
              </div>

              {/* Molecule viewer */}
              <div className="card">
                <h2 className="card-title">Structure</h2>
                <MoleculeViewer smiles={smiles} width={300} height={220} theme={theme} />
                {smiles && (
                  <div className="smiles-display">{smiles}</div>
                )}
              </div>
            </div>

            {/* Right column — results */}
            <div className="right-col">
              {result ? (
                <div className="results-col">
                  {/* Verdict */}
                  <div className={`verdict verdict-${result.risk_level}`}>
                    <div className="verdict-icon">
                      {result.risk_level === "high" ? "☠" : result.risk_level === "moderate" ? "⚡" : "✓"}
                    </div>
                    <div className="verdict-body">
                      <div className="verdict-title">
                        {result.risk_level === "high" ? "High Toxicity Risk"
                          : result.risk_level === "moderate" ? "Moderate Risk"
                          : "Low Toxicity Risk"}
                      </div>
                      <div className="verdict-sub">
                        {result.toxic_tasks?.length > 0
                          ? `Triggered: ${result.toxic_tasks.join(", ")}`
                          : "No assay threshold exceeded"}
                      </div>
                      <div className="verdict-meta">
                        Model: {result.model?.toUpperCase()} · {result.latency_ms}ms
                      </div>
                    </div>
                    <ConfidenceGauge value={result.max_prob} riskLevel={result.risk_level} />
                  </div>

                  {/* Viz toggle + export */}
                  <div className="viz-toolbar">
                    <div className="viz-tabs">
                      <button
                        className={`viz-tab ${vizMode === "bars" ? "viz-tab-active" : ""}`}
                        onClick={() => setVizMode("bars")}
                      >Bars</button>
                      <button
                        className={`viz-tab ${vizMode === "radar" ? "viz-tab-active" : ""}`}
                        onClick={() => setVizMode("radar")}
                      >Radar</button>
                    </div>
                    <button className="btn-ghost btn-sm" onClick={exportJSON}>
                      <Download size={13} /> Export JSON
                    </button>
                  </div>

                  {/* Visualization */}
                  <div className="card">
                    <h2 className="card-title">Toxicity Profile</h2>
                    {vizMode === "bars"
                      ? <TaskBars predictions={result.predictions} />
                      : <ToxicityRadar predictions={result.predictions} />
                    }
                  </div>
                </div>
              ) : (
                <div className="results-empty">
                  <div className="results-empty-icon">⬡</div>
                  <p>Enter a SMILES string and click Predict</p>
                  <span>Results will appear here</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ════ COMPARE TAB ════ */}
        {tab === "compare" && (
          <div className="compare-tab">
            <div className="card">
              <h2 className="card-title">Compare All Models</h2>
              <p className="card-desc">Predict the same molecule with GCN, GAT, and GIN simultaneously.</p>

              <div className="field">
                <label className="field-label">SMILES String</label>
                <div className="input-row">
                  <textarea
                    className="smiles-input"
                    value={smiles}
                    onChange={e => { setSmiles(e.target.value); setCompareData(null); }}
                    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
                    rows={2}
                    spellCheck={false}
                  />
                  <button
                    className="btn-primary"
                    onClick={handleCompare}
                    disabled={!smiles.trim() || compareLoading}
                  >
                    {compareLoading
                      ? <><div className="spinner btn-spinner" /> Running…</>
                      : <><GitCompare size={14} /> Compare</>
                    }
                  </button>
                </div>
              </div>

              <div className="examples" style={{ marginTop: 12 }}>
                {EXAMPLES.map(ex => (
                  <button key={ex.label} className={`example-chip ${ex.safe ? "example-safe" : "example-toxic"}`}
                    onClick={() => { setSmiles(ex.smiles); setCompareData(null); }}>
                    {ex.label}
                  </button>
                ))}
              </div>
            </div>

            {compareData && (
              <div className="card">
                <h2 className="card-title">Comparison Results</h2>
                <CompareMode data={compareData.models} smiles={compareData.smiles} />
              </div>
            )}
          </div>
        )}

        {/* ════ BATCH TAB ════ */}
        {tab === "batch" && (
          <div className="batch-tab">
            <div className="card">
              <h2 className="card-title">Batch Prediction</h2>
              <p className="card-desc">Upload a CSV with a <code>smiles</code> column. Max 500 rows.</p>
              <div style={{ marginBottom: 16 }}>
                <ModelSelector selected={model} onChange={setModel} available={available} />
              </div>
              <BatchUpload model={model} />
            </div>
          </div>
        )}

        {/* ════ HISTORY TAB ════ */}
        {tab === "history" && (
          <div className="history-tab">
            <div className="card">
              <h2 className="card-title">Prediction History</h2>
              <p className="card-desc">Your recent predictions — stored locally in this browser.</p>
              <PredictionHistory
                history={history}
                onReload={handleReload}
                onClear={handleClearHistory}
              />
            </div>
          </div>
        )}

      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <p>
          GNN trained on <a href="https://tox21.gov" target="_blank" rel="noreferrer">Tox21</a>
          {" · "}12 assays · ~8,000 compounds · Scaffold split
        </p>
        <p className="footer-note">For research use only. Not for clinical decisions.</p>
      </footer>
    </div>
  );
}
