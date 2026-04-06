import { useState, useEffect, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";

const BEARINGS = ["b1_ch1", "b2_ch1", "b3_ch1", "b4_ch1"];
const COLORS = { b1_ch1: "#ef4444", b2_ch1: "#f59e0b", b3_ch1: "#10b981", b4_ch1: "#3b82f6" };
const LABELS = { b1_ch1: "Bearing 1 (drive end)", b2_ch1: "Bearing 2 (inner 1)", b3_ch1: "Bearing 3 (inner 2)", b4_ch1: "Bearing 4 (free end)" };

const MOCK_ANOMALY = Array.from({ length: 984 }, (_, i) => {
  const t = i / 983;
  return {
    snap: i,
    b1_ch1: Math.sin(t * 3) * 0.05 + (t > 0.7 ? (t - 0.7) * 1.2 : 0) + (Math.random() - 0.5) * 0.04,
    b2_ch1: Math.sin(t * 2.5) * 0.03 + (t > 0.8 ? (t - 0.8) * 0.8 : 0) + (Math.random() - 0.5) * 0.03,
    b3_ch1: Math.sin(t * 2) * 0.02 + (t > 0.85 ? (t - 0.85) * 0.5 : 0) + (Math.random() - 0.5) * 0.02,
    b4_ch1: Math.sin(t * 4) * 0.04 + (t > 0.6 ? (t - 0.6) * 0.3 : 0) + (Math.random() - 0.5) * 0.03,
  };
});

const MOCK_RUL = Array.from({ length: 984 }, (_, i) => {
  const t = i / 983;
  return {
    snap: i,
    b1_ch1: Math.max(0, 1 - t * 1.1 + Math.sin(t * 5) * 0.02),
    b2_ch1: Math.max(0, 1 - t * 0.95 + Math.sin(t * 4) * 0.015),
    b3_ch1: Math.max(0, 1 - t * 0.85 + Math.sin(t * 3) * 0.01),
    b4_ch1: Math.max(0, 1 - t * 0.5 + Math.sin(t * 6) * 0.03),
  };
});

const SHAP_DATA = {
  950: [
    { feature: "BPFI band energy", bearing: "b2_ch1", value: -0.531, direction: "anomaly" },
    { feature: "Dominant frequency", bearing: "b2_ch1", value: -0.881, direction: "anomaly" },
    { feature: "Shape factor", bearing: "b2_ch1", value: -0.653, direction: "anomaly" },
    { feature: "Spectral centroid", bearing: "b2_ch1", value: -0.005, direction: "rul" },
    { feature: "RMS vibration", bearing: "b1_ch1", value: 0.005, direction: "rul" },
    { feature: "Kurtosis", bearing: "b1_ch1", value: 0.124, direction: "anomaly" },
  ],
};

const ALERTS = [
  { time: "snap 950", level: "high", msg: "INNER RACE FAULT on b2_ch1 — schedule within 48h", bearing: "b2_ch1" },
  { time: "snap 940", level: "medium", msg: "Elevated RUL decline rate on b1_ch1", bearing: "b1_ch1" },
  { time: "snap 920", level: "low", msg: "Weather MCP: 22°C, 52% humidity — LOW risk", bearing: "system" },
  { time: "snap 900", level: "high", msg: "BPFI band energy rising on b2_ch1 — inner race fault", bearing: "b2_ch1" },
  { time: "snap 850", level: "low", msg: "Routine monitoring — all bearings within baseline", bearing: "system" },
  { time: "snap 800", level: "medium", msg: "b4_ch1 temperature trend elevated — corrective WO issued", bearing: "b4_ch1" },
];

const Badge = ({ children, color }) => (
  <span style={{
    fontFamily: "'JetBrains Mono', monospace", fontSize: 10, fontWeight: 600,
    padding: "2px 8px", borderRadius: 4,
    color: color, background: color + "22",
  }}>{children}</span>
);

const StatusDot = () => (
  <span style={{
    width: 8, height: 8, borderRadius: "50%", background: "#10b981", display: "inline-block",
    animation: "pulse 2s ease-in-out infinite",
  }} />
);

const Card = ({ title, badge, badgeColor, children, style }) => (
  <div style={{
    background: "#111827", border: "1px solid #1e293b", borderRadius: 8,
    padding: 16, ...style,
  }}>
    {(title || badge) && (
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        {title && <span style={{ fontSize: 11, fontWeight: 500, textTransform: "uppercase", letterSpacing: 1, color: "#64748b" }}>{title}</span>}
        {badge && <Badge color={badgeColor || "#64748b"}>{badge}</Badge>}
      </div>
    )}
    {children}
  </div>
);

const StatCard = ({ label, value, sub, color }) => (
  <Card style={{ textAlign: "center", padding: "20px 16px" }}>
    <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 32, fontWeight: 700, color, lineHeight: 1, margin: "8px 0 4px" }}>{value}</div>
    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#94a3b8" }}>{sub}</div>
  </Card>
);

const BearingCard = ({ id, rul, color }) => {
  const pct = (rul * 100).toFixed(1);
  const status = rul < 0.15 ? "CRITICAL" : rul < 0.3 ? "WARNING" : "NORMAL";
  const sColor = rul < 0.15 ? "#ef4444" : rul < 0.3 ? "#f59e0b" : "#10b981";
  const bgColor = rul < 0.15 ? "rgba(239,68,68,0.12)" : rul < 0.3 ? "rgba(245,158,11,0.12)" : "rgba(16,185,129,0.12)";
  return (
    <div style={{
      background: bgColor, border: `1px solid ${sColor}33`, borderRadius: 6,
      padding: 12, textAlign: "center",
    }}>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, fontWeight: 600, color, marginBottom: 8 }}>{id.toUpperCase()}</div>
      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 24, fontWeight: 700, color: "#e2e8f0" }}>{pct}%</div>
      <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>RUL remaining</div>
      <div style={{
        fontFamily: "'JetBrains Mono', monospace", fontSize: 10, fontWeight: 600,
        color: sColor, background: sColor + "22", padding: "2px 6px",
        borderRadius: 3, display: "inline-block", marginTop: 8,
      }}>{status}</div>
    </div>
  );
};

const AlertItem = ({ time, level, msg }) => {
  const colors = { critical: "#ef4444", high: "#f59e0b", medium: "#3b82f6", low: "#10b981" };
  const c = colors[level] || "#64748b";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10, padding: "10px 12px",
      background: "#0a0e17", borderRadius: 6, borderLeft: `3px solid ${c}`, fontSize: 12,
    }}>
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#64748b", whiteSpace: "nowrap" }}>{time}</span>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace", fontSize: 9, fontWeight: 600,
        padding: "1px 6px", borderRadius: 3, whiteSpace: "nowrap",
        color: c, background: c + "22",
      }}>{level.toUpperCase()}</span>
      <span style={{ color: "#94a3b8", flex: 1 }}>{msg}</span>
    </div>
  );
};

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null;
  return (
    <div style={{
      background: "#1e293b", border: "1px solid #334155", borderRadius: 6,
      padding: "8px 12px", fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
    }}>
      <div style={{ color: "#64748b", marginBottom: 4 }}>Snapshot {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>{p.dataKey}: {p.value?.toFixed(4)}</div>
      ))}
    </div>
  );
};

export default function Dashboard() {
  const [snap, setSnap] = useState(950);

  const anomalyRow = MOCK_ANOMALY[snap] || {};
  const rulRow = MOCK_RUL[snap] || {};

  const stats = useMemo(() => {
    let worst = "", worstScore = -Infinity, minRul = Infinity, minRulB = "", anomalyCount = 0;
    BEARINGS.forEach(b => {
      const s = anomalyRow[b] || 0;
      const r = rulRow[b];
      if (s > worstScore) { worstScore = s; worst = b; }
      if (r < minRul) { minRul = r; minRulB = b; }
      if (s > 0.05) anomalyCount++;
    });
    const status = minRul < 0.15 ? "CRITICAL" : minRul < 0.3 ? "WARNING" : "NORMAL";
    const statusColor = minRul < 0.15 ? "#ef4444" : minRul < 0.3 ? "#f59e0b" : "#10b981";
    return { worst, worstScore, minRul, minRulB, anomalyCount, status, statusColor };
  }, [snap]);

  const shapFeatures = SHAP_DATA[950] || SHAP_DATA[Object.keys(SHAP_DATA)[0]] || [];

  return (
    <div style={{ background: "#0a0e17", color: "#e2e8f0", minHeight: "100vh", fontFamily: "'IBM Plex Sans', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a0e17; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }
      `}</style>

      {/* Topbar */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 24px", borderBottom: "1px solid #1e293b", background: "#111827",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 18, fontWeight: 700, color: "#06b6d4" }}>
            BearingMind <span style={{ color: "#64748b", fontWeight: 400 }}>/ condition monitoring</span>
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#94a3b8" }}>
            <StatusDot /> System online
          </div>
        </div>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#64748b" }}>
          NASA IMS Test Rig — 4 bearings @ 2000 RPM
        </span>
      </div>

      {/* Snapshot slider */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 24px 2px" }}>
        <span style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5 }}>Snapshot</span>
        <input type="range" min={0} max={983} value={snap} onChange={e => setSnap(+e.target.value)}
          style={{ flex: 1, accentColor: "#06b6d4" }} />
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 14, fontWeight: 600, color: "#06b6d4", minWidth: 48, textAlign: "right" }}>{snap}</span>
      </div>

      <div style={{ padding: "8px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
        {/* Stat cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
          <StatCard label="Worst bearing" value={stats.worst.toUpperCase()} sub={`score: ${stats.worstScore.toFixed(4)}`}
            color={stats.worstScore > 0.1 ? "#ef4444" : "#10b981"} />
          <StatCard label="Min RUL" value={`${(stats.minRul * 100).toFixed(1)}%`} sub={stats.minRulB}
            color={stats.minRul < 0.15 ? "#ef4444" : stats.minRul < 0.3 ? "#f59e0b" : "#10b981"} />
          <StatCard label="System status" value={stats.status} sub={`snapshot ${snap} / 983`}
            color={stats.statusColor} />
          <StatCard label="Anomalies" value={stats.anomalyCount} sub="of 4 bearings"
            color={stats.anomalyCount > 0 ? "#ef4444" : "#10b981"} />
        </div>

        {/* Bearing grid + Anomaly chart */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <Card title="Bearing health — RUL at current snapshot">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
              {BEARINGS.map(b => (
                <BearingCard key={b} id={b} rul={rulRow[b] || 0} color={COLORS[b]} />
              ))}
            </div>
          </Card>
          <Card title="Anomaly scores — time series" badge="Isolation Forest" badgeColor="#06b6d4">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={MOCK_ANOMALY}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="snap" stroke="#475569" tick={{ fontSize: 9, fontFamily: "'JetBrains Mono'" }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9, fontFamily: "'JetBrains Mono'" }} />
                <Tooltip content={<ChartTooltip />} />
                <ReferenceLine x={snap} stroke="#06b6d4" strokeDasharray="4 4" strokeOpacity={0.6} />
                {BEARINGS.map(b => <Line key={b} type="monotone" dataKey={b} stroke={COLORS[b]} dot={false} strokeWidth={1.5} />)}
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* RUL chart + SHAP table */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <Card title="RUL predictions — time series" badge="LSTM" badgeColor="#8b5cf6">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={MOCK_RUL}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="snap" stroke="#475569" tick={{ fontSize: 9, fontFamily: "'JetBrains Mono'" }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9, fontFamily: "'JetBrains Mono'" }} domain={[0, 1]} />
                <Tooltip content={<ChartTooltip />} />
                <ReferenceLine x={snap} stroke="#06b6d4" strokeDasharray="4 4" strokeOpacity={0.6} />
                <ReferenceLine y={0.15} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.4} label="" />
                <ReferenceLine y={0.3} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.3} label="" />
                {BEARINGS.map(b => <Line key={b} type="monotone" dataKey={b} stroke={COLORS[b]} dot={false} strokeWidth={1.5} />)}
              </LineChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Top SHAP features at snapshot" badge="Explainability" badgeColor="#10b981">
            <div style={{ fontSize: 12 }}>
              <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr", gap: 4, marginBottom: 8 }}>
                {["Feature", "Bearing", "Impact"].map(h => (
                  <span key={h} style={{ fontSize: 10, fontWeight: 500, textTransform: "uppercase", letterSpacing: 0.5, color: "#64748b", padding: "4px 0", borderBottom: "1px solid #1e293b" }}>{h}</span>
                ))}
              </div>
              {shapFeatures.map((f, i) => (
                <div key={i} style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr", gap: 4, padding: "5px 0", borderBottom: "1px solid #1e293b11" }}>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", color: "#94a3b8" }}>{f.feature}</span>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", color: COLORS[f.bearing] }}>{f.bearing}</span>
                  <div>
                    <div style={{
                      height: 6, borderRadius: 3, width: Math.min(Math.abs(f.value) * 100, 60),
                      background: f.value > 0 ? "#10b981" : "#ef4444",
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Alerts + Environment */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
          <Card title="Alert log" badge="Latest" badgeColor="#64748b">
            <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 240, overflowY: "auto" }}>
              {ALERTS.map((a, i) => <AlertItem key={i} {...a} />)}
            </div>
          </Card>
          <Card title="Environment" badge="Weather MCP" badgeColor="#3b82f6">
            {[
              ["Location", "Cincinnati, OH"],
              ["Temperature", "22.2°C"],
              ["Humidity", "52%"],
              ["Wind", "17.5 km/h"],
              ["Temp risk", "LOW", "#10b981"],
              ["Humidity risk", "LOW", "#10b981"],
              ["Combined risk", "LOW", "#10b981"],
            ].map(([label, value, color], i) => (
              <div key={i} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "8px 0", borderBottom: i < 6 ? "1px solid #1e293b" : "none",
              }}>
                <span style={{ fontSize: 12, color: "#94a3b8" }}>{label}</span>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 14, fontWeight: 500, color: color || "#e2e8f0" }}>{value}</span>
              </div>
            ))}
          </Card>
        </div>
      </div>

      {/* Footer */}
      <div style={{
        padding: "12px 24px", borderTop: "1px solid #1e293b", fontSize: 11, color: "#64748b",
        display: "flex", justifyContent: "space-between",
      }}>
        <span>BearingMind — Explainable AI & multi-agent predictive maintenance</span>
        <span>MCP: Equipment Manual (RAG) · CMMS (SQLite) · Weather (Open-Meteo)</span>
      </div>
    </div>
  );
}
