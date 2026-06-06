// src/components/ConfidenceGauge.jsx
export default function ConfidenceGauge({ value = 0, riskLevel = "low" }) {
  const pct    = Math.round(value * 100);
  const radius = 54;
  const circ   = 2 * Math.PI * radius;
  const offset = circ - (pct / 100) * circ;

  const colors = {
    high:     "#ef4444",
    moderate: "#f59e0b",
    low:      "#22c55e",
    invalid:  "#6b7280",
  };
  const color = colors[riskLevel] || colors.low;

  const labels = {
    high:     "HIGH RISK",
    moderate: "MODERATE",
    low:      "LOW RISK",
    invalid:  "INVALID",
  };

  return (
    <div className="gauge-wrap">
      <svg width="140" height="140" viewBox="0 0 140 140">
        {/* Track */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke="var(--border)"
          strokeWidth="10"
        />
        {/* Progress */}
        <circle
          cx="70" cy="70" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          transform="rotate(-90 70 70)"
          style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.16,1,0.3,1), stroke 0.3s" }}
        />
        {/* Center text */}
        <text x="70" y="62" textAnchor="middle"
          fontSize="28" fontWeight="800"
          fontFamily="'Space Mono', monospace"
          fill={color}
        >
          {pct}%
        </text>
        <text x="70" y="82" textAnchor="middle"
          fontSize="9" fontWeight="700"
          fontFamily="'Syne', sans-serif"
          fill="var(--text-secondary)"
          letterSpacing="1.5"
        >
          {labels[riskLevel]}
        </text>
      </svg>
    </div>
  );
}
