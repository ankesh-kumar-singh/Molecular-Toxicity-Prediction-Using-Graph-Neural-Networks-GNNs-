// src/components/MoleculeViewer.jsx
// Fetches molecule SVG from backend /mol/svg endpoint

import { useEffect, useState } from "react";

const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function MoleculeViewer({ smiles, width = 300, height = 220, theme = "dark" }) {
  const [svg,     setSvg]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    if (!smiles?.trim()) {
      setSvg(null);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    setSvg(null);

    fetch(`${BASE}/mol/svg`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: smiles.trim(), width, height, theme }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then((svgText) => {
        const cleaned = svgText
          .replace(/<\?xml[^?]*\?>/g, "")
          .replace(/<!DOCTYPE[^>]*>/g, "")
          .trim();
        setSvg(cleaned);
        setLoading(false);
      })
      .catch((err) => {
        setError("Could not render structure");
        setLoading(false);
      });
  }, [smiles, width, height, theme]);

  if (!smiles?.trim()) {
    return (
      <div className="mol-empty">
        <div className="mol-empty-icon">⬡</div>
        <p>Enter a SMILES string to view structure</p>
      </div>
    );
  }

  return (
    <div className="mol-viewer">
      {loading && (
        <div className="mol-loading">
          <div className="spinner" />
          <span>Rendering structure…</span>
        </div>
      )}
      {error && (
        <div className="mol-error">
          <span>⚠</span> {error}
        </div>
      )}
      {svg && !loading && (
        <div
          style={{
            width: "100%",
            borderRadius: "8px",
            overflow: "hidden",
            background: "transparent",
          }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      )}
    </div>
  );
}
