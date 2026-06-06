// src/components/MoleculeViewer.jsx
import { useEffect, useRef, useState } from "react";

let rdkitPromise = null;

function loadRDKit() {
  if (!rdkitPromise) {
    rdkitPromise = new Promise((resolve, reject) => {
      if (window.RDKit) { resolve(window.RDKit); return; }
      const script = document.createElement("script");
      script.src = "https://unpkg.com/@rdkit/rdkit@2023.9.4/Code/MinimalLib/dist/RDKit_minimal.js";
      script.onload = () => window.initRDKitModule().then(rdk => {
        window.RDKit = rdk; resolve(rdk);
      }).catch(reject);
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }
  return rdkitPromise;
}

export default function MoleculeViewer({ smiles, width = 300, height = 220, theme = "dark" }) {
  const canvasRef         = useRef(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [svg, setSvg]     = useState(null);

  useEffect(() => {
    if (!smiles) { setSvg(null); setError(null); return; }
    setLoading(true);
    setError(null);

    loadRDKit().then((RDKit) => {
      const mol = RDKit.get_mol(smiles);
      if (!mol || !mol.is_valid()) {
        setError("Invalid SMILES");
        setLoading(false);
        return;
      }
      const svgStr = mol.get_svg(width, height);
      mol.delete();

      // Adapt colors for theme
      const adapted = theme === "light"
        ? svgStr.replace(/fill:#[0-9a-fA-F]{6}/g, "fill:#1a1a2e")
        : svgStr;

      setSvg(adapted);
      setLoading(false);
    }).catch(() => {
      setError("RDKit unavailable");
      setLoading(false);
    });
  }, [smiles, width, height, theme]);

  if (!smiles) return (
    <div className="mol-empty">
      <div className="mol-empty-icon">⬡</div>
      <p>Enter a SMILES string to view structure</p>
    </div>
  );

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
          className="mol-svg-wrap"
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      )}
    </div>
  );
}
