// src/components/TaskBars.jsx
import { useState } from "react";

const TASK_DESCRIPTIONS = {
  "NR-AR":        "Androgen Receptor — steroid hormone receptor",
  "NR-AR-LBD":    "Androgen Receptor Ligand Binding Domain",
  "NR-AhR":       "Aryl Hydrocarbon Receptor — dioxin toxicity",
  "NR-Aromatase": "Aromatase — estrogen biosynthesis enzyme",
  "NR-ER":        "Estrogen Receptor Alpha",
  "NR-ER-LBD":    "Estrogen Receptor Alpha LBD",
  "NR-PPAR-gamma":"PPAR-gamma — lipid metabolism",
  "SR-ARE":       "Antioxidant Response Element",
  "SR-ATAD5":     "ATAD5 — DNA damage / genome integrity",
  "SR-HSE":       "Heat Shock Factor Response Element",
  "SR-MMP":       "Mitochondrial Membrane Potential",
  "SR-p53":       "p53 Tumor Suppressor Pathway",
};

function riskColor(p) {
  if (p >= 0.5) return "#ef4444";
  if (p >= 0.3) return "#f59e0b";
  return "#22c55e";
}

function riskLabel(p) {
  if (p >= 0.5) return "TOXIC";
  if (p >= 0.3) return "WATCH";
  return "SAFE";
}

export default function TaskBars({ predictions = {} }) {
  const [selected, setSelected] = useState(null);

  const sorted = Object.entries(predictions).sort(([, a], [, b]) => b - a);

  return (
    <div className="task-bars">
      {sorted.map(([task, prob], i) => {
        const pct   = Math.round(prob * 100);
        const color = riskColor(prob);
        const label = riskLabel(prob);
        const isSelected = selected === task;

        return (
          <div
            key={task}
            className={`task-row ${isSelected ? "task-row-selected" : ""}`}
            onClick={() => setSelected(isSelected ? null : task)}
            style={{ animationDelay: `${i * 30}ms` }}
          >
            <div className="task-row-header">
              <span className="task-name">{task}</span>
              <span className="task-badge" style={{ color, borderColor: color }}>
                {label}
              </span>
            </div>

            <div className="task-track">
              <div
                className="task-fill"
                style={{
                  width: `${pct}%`,
                  background: color,
                  animationDelay: `${i * 30 + 100}ms`,
                }}
              />
              <span className="task-pct">{pct}%</span>
            </div>

            {isSelected && (
              <div className="task-detail">
                <p>{TASK_DESCRIPTIONS[task]}</p>
                <div className="task-detail-stats">
                  <span>Probability: <strong>{(prob * 100).toFixed(1)}%</strong></span>
                  <span>Threshold: <strong>50%</strong></span>
                  <span>Status: <strong style={{ color }}>{label}</strong></span>
                </div>
              </div>
            )}
          </div>
        );
      })}
      <p className="task-hint">Click any task to see details</p>
    </div>
  );
}
