// src/components/CompareMode.jsx
import ConfidenceGauge from "./ConfidenceGauge";

const MODEL_COLORS = { gcn: "#4C72B0", gat: "#DD8452", gin: "#55A868" };
const MODEL_LABELS = { gcn: "GCN", gat: "GAT", gin: "GIN" };

export default function CompareMode({ data, smiles }) {
  if (!data || !Object.keys(data).length) return null;

  const models   = Object.keys(data);
  const tasks    = Object.keys(data[models[0]]?.predictions || {});
  const bestModel = models.reduce((a, b) =>
    (data[a]?.max_prob || 0) > (data[b]?.max_prob || 0) ? a : b
  );

  return (
    <div className="compare-panel">
      <div className="compare-smiles">{smiles}</div>

      {/* Gauges row */}
      <div className="compare-gauges">
        {models.map(m => (
          <div key={m} className="compare-gauge-item">
            <div className="compare-model-label" style={{ color: MODEL_COLORS[m] }}>
              {MODEL_LABELS[m]}
              {m === bestModel && <span className="compare-best-badge">Best</span>}
            </div>
            <ConfidenceGauge
              value={data[m]?.max_prob || 0}
              riskLevel={data[m]?.risk_level || "low"}
            />
            <div className="compare-auc">
              {data[m]?.latency_ms}ms
            </div>
          </div>
        ))}
      </div>

      {/* Task comparison table */}
      <div className="compare-table-wrap">
        <table className="compare-table">
          <thead>
            <tr>
              <th>Task</th>
              {models.map(m => (
                <th key={m} style={{ color: MODEL_COLORS[m] }}>{MODEL_LABELS[m]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tasks.map(task => {
              const vals    = models.map(m => data[m]?.predictions?.[task] || 0);
              const maxVal  = Math.max(...vals);

              return (
                <tr key={task}>
                  <td className="compare-task-name">{task}</td>
                  {models.map((m, i) => {
                    const v       = data[m]?.predictions?.[task] || 0;
                    const isBest  = v === maxVal && maxVal > 0;
                    const color   = v >= 0.5 ? "#ef4444" : v >= 0.3 ? "#f59e0b" : "var(--text-secondary)";
                    return (
                      <td key={m} className={isBest ? "compare-best-val" : ""}>
                        <span style={{ color }}>{Math.round(v * 100)}%</span>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
