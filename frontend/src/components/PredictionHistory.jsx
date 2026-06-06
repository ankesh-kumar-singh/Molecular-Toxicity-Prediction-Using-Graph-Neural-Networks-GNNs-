// src/components/PredictionHistory.jsx
import { Trash2, RotateCcw } from "lucide-react";

const RISK_COLORS = { high: "#ef4444", moderate: "#f59e0b", low: "#22c55e" };
const MODEL_LABELS = { gcn: "GCN", gat: "GAT", gin: "GIN" };

export default function PredictionHistory({ history, onReload, onClear }) {
  if (!history?.length) {
    return (
      <div className="history-empty">
        <p>No predictions yet</p>
        <span>Your recent predictions will appear here</span>
      </div>
    );
  }

  return (
    <div className="history-panel">
      <div className="history-header">
        <span>{history.length} recent predictions</span>
        <button className="btn-ghost btn-sm" onClick={onClear}>
          <Trash2 size={13} /> Clear
        </button>
      </div>

      <div className="history-list">
        {history.map((item, i) => {
          const color = RISK_COLORS[item.risk_level] || "#6b7280";
          const time  = item.timestamp
            ? new Date(item.timestamp * 1000).toLocaleTimeString()
            : "";

          return (
            <div key={item.id || i} className="history-item">
              <div className="history-item-left">
                <div className="history-smiles" title={item.smiles}>
                  {item.smiles?.length > 32
                    ? item.smiles.slice(0, 32) + "…"
                    : item.smiles}
                </div>
                <div className="history-meta">
                  <span className="history-model">
                    {MODEL_LABELS[item.model] || item.model}
                  </span>
                  <span className="history-time">{time}</span>
                  {item.toxic_tasks?.length > 0 && (
                    <span className="history-toxic" style={{ color }}>
                      {item.toxic_tasks.length} toxic
                    </span>
                  )}
                </div>
              </div>

              <div className="history-item-right">
                <span className="history-prob" style={{ color }}>
                  {Math.round(item.max_prob * 100)}%
                </span>
                <button
                  className="btn-ghost btn-xs"
                  onClick={() => onReload(item)}
                  title="Reload prediction"
                >
                  <RotateCcw size={12} />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
