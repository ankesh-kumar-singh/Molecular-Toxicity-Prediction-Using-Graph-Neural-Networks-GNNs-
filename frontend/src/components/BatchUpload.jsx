// src/components/BatchUpload.jsx
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Download, AlertCircle } from "lucide-react";
import { api } from "../utils/api";

function riskColor(prob) {
  const p = parseFloat(prob);
  if (p >= 0.5) return "#ef4444";
  if (p >= 0.3) return "#f59e0b";
  return "#22c55e";
}

function parseCSV(text) {
  const lines  = text.trim().split("\n");
  const header = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
  return lines.slice(1).map(line => {
    const vals = line.split(",").map(v => v.trim().replace(/^"|"$/g, ""));
    return Object.fromEntries(header.map((h, i) => [h, vals[i] ?? ""]));
  });
}

export default function BatchUpload({ model }) {
  const [status,   setStatus]   = useState("idle");
  const [results,  setResults]  = useState([]);
  const [errMsg,   setErrMsg]   = useState("");
  const [fileName, setFileName] = useState("");
  const [stats,    setStats]    = useState(null);

  const onDrop = useCallback(async (accepted) => {
    const file = accepted[0];
    if (!file) return;
    setFileName(file.name);
    setStatus("loading");
    setResults([]);
    setErrMsg("");
    setStats(null);

    try {
      const blob = await api.predictCSV(file, model);
      const text = await blob.text();
      const rows = parseCSV(text);
      setResults(rows);

      const valid  = rows.filter(r => r.valid === "True" || r.valid === "true").length;
      const toxic  = rows.filter(r => parseFloat(r.max_prob) >= 0.5).length;
      setStats({ total: rows.length, valid, toxic });
      setStatus("done");
    } catch (e) {
      setErrMsg(e.message);
      setStatus("error");
    }
  }, [model]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
  });

  const download = () => {
    const header = Object.keys(results[0]).join(",");
    const rows   = results.map(r => Object.values(r).join(","));
    const csv    = [header, ...rows].join("\n");
    const url    = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a      = document.createElement("a");
    a.href = url; a.download = "tox21_predictions.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="batch-panel">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? "dropzone-active" : ""} ${status === "loading" ? "dropzone-loading" : ""}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-icon">
          {status === "loading" ? <div className="spinner large" /> : <Upload size={36} />}
        </div>
        <p className="dropzone-label">
          {status === "loading"
            ? `Processing ${fileName}…`
            : isDragActive
            ? "Drop your CSV here"
            : "Drag & drop a CSV file, or click to browse"}
        </p>
        <p className="dropzone-hint">
          CSV must have a <code>smiles</code> column · max 500 rows
        </p>
      </div>

      {/* Error */}
      {status === "error" && (
        <div className="alert alert-error">
          <AlertCircle size={16} /> {errMsg}
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="batch-stats">
          <div className="stat-chip">{stats.total} total</div>
          <div className="stat-chip stat-chip-green">{stats.valid} valid</div>
          <div className="stat-chip stat-chip-red">{stats.toxic} toxic</div>
          <div className="stat-chip stat-chip-safe">{stats.valid - stats.toxic} safe</div>
        </div>
      )}

      {/* Results */}
      {status === "done" && results.length > 0 && (
        <div className="batch-results">
          <div className="batch-results-header">
            <span>{results.length} molecules processed</span>
            <button className="btn-accent btn-sm" onClick={download}>
              <Download size={13} /> Download CSV
            </button>
          </div>
          <div className="batch-table-wrap">
            <table className="batch-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>SMILES</th>
                  <th>Valid</th>
                  <th>Risk</th>
                  <th>Max %</th>
                  <th>Toxic Assays</th>
                </tr>
              </thead>
              <tbody>
                {results.slice(0, 100).map((row, i) => {
                  const prob  = parseFloat(row.max_prob || 0);
                  const color = riskColor(prob);
                  return (
                    <tr key={i} className={row.valid === "false" ? "row-invalid" : ""}>
                      <td className="row-num">{i + 1}</td>
                      <td className="row-smiles" title={row.smiles}>
                        {row.smiles?.length > 30 ? row.smiles.slice(0, 30) + "…" : row.smiles}
                      </td>
                      <td>{row.valid === "True" || row.valid === "true" ? "✓" : "✗"}</td>
                      <td>
                        <span className="risk-pill" style={{ background: color + "22", color }}>
                          {row.risk_level || "—"}
                        </span>
                      </td>
                      <td style={{ color }}>{row.max_prob ? `${Math.round(prob * 100)}%` : "—"}</td>
                      <td className="row-toxic">{row.toxic_tasks || "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {results.length > 100 && (
              <p className="table-note">Showing first 100 rows. Download for all results.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
