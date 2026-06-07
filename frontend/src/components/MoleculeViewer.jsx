// src/components/MoleculeViewer.jsx
// Loads RDKit via npm package (bundled — no CDN dependency)

import { useEffect, useState } from "react";

let rdkitInstance = null;
let rdkitLoading  = false;
let rdkitCallbacks = [];

function getRDKit() {
  return new Promise((resolve, reject) => {
    // Already loaded
    if (rdkitInstance) { resolve(rdkitInstance); return; }

    // Queue callback if loading in progress
    rdkitCallbacks.push({ resolve, reject });
    if (rdkitLoading) return;

    rdkitLoading = true;

    // Dynamically import from npm package
    import("@rdkit/rdkit").then((mod) => {
      const initRDKit = mod.default || mod;
      return initRDKit();
    }).then((rdk) => {
      rdkitInstance = rdk;
      rdkitCallbacks.forEach(cb => cb.resolve(rdk));
      rdkitCallbacks = [];
    }).catch((err) => {
      rdkitLoading = false;
      rdkitCallbacks.forEach(cb => cb.reject(err));
      rdkitCallbacks = [];
    });
  });
}

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

    getRDKit()
      .then((RDKit) => {
        let mol;
        try {
          mol = RDKit.get_mol(smiles.trim());
        } catch {
          setError("Could not parse SMILES");
          setLoading(false);
          return;
        }

        if (!mol || !mol.is_valid()) {
          mol?.delete();
          setError("Invalid SMILES");
          setLoading(false);
          return;
        }

        let svgStr;
        try {
          svgStr = mol.get_svg(width, height);
        } finally {
          mol.delete();
        }

        if (!svgStr) {
          setError("Could not render molecule");
          setLoading(false);
          return;
        }

        // Make background transparent + adapt colors for theme
        let adapted = svgStr
          .replace(/style='background:\s*#ffffff[^']*'/gi, "style='background:transparent'")
          .replace(/style="background:\s*#ffffff[^"]*"/gi, 'style="background:transparent"');

        if (theme === "dark") {
          adapted = adapted
            .replace(/stroke:#000000/g, "stroke:#e8eaf0")
            .replace(/fill:#000000/g,   "fill:#e8eaf0");
        }

        setSvg(adapted);
        setLoading(false);
      })
      .catch(() => {
        setError("Could not load molecule renderer");
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
          className="mol-svg-wrap"
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      )}
    </div>
  );
}
