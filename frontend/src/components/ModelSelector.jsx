// src/components/ModelSelector.jsx
import { Zap, Brain, GitBranch } from "lucide-react";

const ICONS = { gcn: GitBranch, gat: Brain, gin: Zap };

const MODEL_META = {
  gcn: { label: "GCN", full: "Graph Convolutional Network", auc: "0.8804", params: "319K", speed: "Fast", color: "#4C72B0" },
  gat: { label: "GAT", full: "Graph Attention Network",    auc: "0.8536", params: "3.7M", speed: "Medium", color: "#DD8452" },
  gin: { label: "GIN", full: "Graph Isomorphism Network",  auc: "0.8821", params: "881K", speed: "Fast", color: "#55A868" },
};

export default function ModelSelector({ selected, onChange, available = [] }) {
  return (
    <div className="model-selector">
      <label className="field-label">Model</label>
      <div className="model-cards">
        {Object.entries(MODEL_META).map(([key, meta]) => {
          const Icon      = ICONS[key];
          const isActive  = selected === key;
          const isAvail   = available.includes(key);

          return (
            <button
              key={key}
              className={`model-card ${isActive ? "model-card-active" : ""} ${!isAvail ? "model-card-disabled" : ""}`}
              onClick={() => isAvail && onChange(key)}
              title={meta.full}
              style={{ "--model-color": meta.color }}
            >
              <div className="model-card-header">
                <Icon size={14} />
                <span className="model-card-label">{meta.label}</span>
                {key === "gin" && <span className="model-badge">Best</span>}
              </div>
              <div className="model-card-auc">AUC {meta.auc}</div>
              <div className="model-card-meta">{meta.params} · {meta.speed}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
