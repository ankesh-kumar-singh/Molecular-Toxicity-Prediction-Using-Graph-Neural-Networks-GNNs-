// src/components/ToxicityRadar.jsx
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Tooltip,
} from "recharts";

const SHORT_LABELS = {
  "NR-AR":        "AR",
  "NR-AR-LBD":    "AR-LBD",
  "NR-AhR":       "AhR",
  "NR-Aromatase": "Arom",
  "NR-ER":        "ER",
  "NR-ER-LBD":    "ER-LBD",
  "NR-PPAR-gamma":"PPARγ",
  "SR-ARE":       "ARE",
  "SR-ATAD5":     "ATAD5",
  "SR-HSE":       "HSE",
  "SR-MMP":       "MMP",
  "SR-p53":       "p53",
};

export default function ToxicityRadar({ predictions = {} }) {
  if (!Object.keys(predictions).length) return null;

  const data = Object.entries(predictions).map(([task, val]) => ({
    task:  SHORT_LABELS[task] || task,
    value: Math.round(val * 100),
    full:  task,
  }));

  return (
    <div className="radar-wrap">
      <ResponsiveContainer width="100%" height={260}>
        <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis
            dataKey="task"
            tick={{ fill: "var(--text-secondary)", fontSize: 11, fontFamily: "'Space Mono', monospace" }}
          />
          <PolarRadiusAxis
            angle={30} domain={[0, 100]}
            tick={{ fill: "var(--text-secondary)", fontSize: 9 }}
            tickCount={4}
          />
          <Radar
            name="Toxicity"
            dataKey="value"
            stroke="#00e5ff"
            fill="#00e5ff"
            fillOpacity={0.18}
            strokeWidth={2}
            dot={{ fill: "#00e5ff", r: 3 }}
          />
          <Tooltip
            formatter={(v, n, p) => [`${v}%`, p.payload.full]}
            contentStyle={{
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: "8px",
              fontSize: "12px",
              fontFamily: "'Space Mono', monospace",
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
