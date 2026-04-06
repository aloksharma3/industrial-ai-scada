import { useState, useEffect, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ── Constants ────────────────────────────────────────────────────────────────
const BEARINGS = ["b1_ch1", "b2_ch1", "b3_ch1", "b4_ch1"];
const COLORS = { b1_ch1: "#ef4444", b2_ch1: "#f59e0b", b3_ch1: "#10b981", b4_ch1: "#3b82f6" };
const BEARING_NAMES = {
  b1_ch1: "Bearing 1 — drive end",
  b2_ch1: "Bearing 2 — inner 1",
  b3_ch1: "Bearing 3 — inner 2",
  b4_ch1: "Bearing 4 — free end",
};

const mono = { fontFamily: "'JetBrains Mono', monospace" };

// ── Small reusable components ────────────────────────────────────────────────

function Badge({ children, color = "#64748b" }) {
  return (
    <span style={{
      ...mono, fontSize: 10, fontWeight: 600,
      padding: "2px 8px", borderRadius: 4,
      color, background: color + "22",
    }}>
      {children}
    </span>
  );
}

function Card({ title, badge, badgeColor, children, style }) {
  return (
    <div style={{
      background: "var(--bg-card)", border: "1px solid var(--border)",
      borderRadius: "var(--radius)", padding: 16, ...style,
    }}>
      {(title || badge) && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          {title && (
            <span style={{ fontSize: 11, fontWeight: 500, textTransform: "uppercase", letterSpacing: 1, color: "var(--text-muted)" }}>
              {title}
            </span>
          )}
          {badge && <Badge color={badgeColor}>{badge}</Badge>}
        </div>
      )}
      {children}
    </div>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <Card style={{ textAlign: "center", padding: "20px 16px" }}>
      <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ ...mono, fontSize: 32, fontWeight: 700, color, lineHeight: 1, margin: "8px 0 4px" }}>{value}</div>
      <div style={{ ...mono, fontSize: 11, color: "var(--text-secondary)" }}>{sub}</div>
    </Card>
  );
}

function BearingCard({ id, rul }) {
  const pct = (rul * 100).toFixed(1);
  const status = rul < 0.15 ? "CRITICAL" : rul < 0.3 ? "WARNING" : "NORMAL";
  const sColor = rul < 0.15 ? "var(--red)" : rul < 0.3 ? "var(--amber)" : "var(--green)";
  return (
    <div style={{
      background: sColor.replace("var(--", "").replace(")", "") === "red"
        ? "rgba(239,68,68,0.1)" : sColor.includes("amber")
        ? "rgba(245,158,11,0.1)" : "rgba(16,185,129,0.1)",
      border: `1px solid ${COLORS[id]}33`, borderRadius: 6,
      padding: 12, textAlign: "center",
    }}>
      <div style={{ ...mono, fontSize: 12, fontWeight: 600, color: COLORS[id], marginBottom: 6 }}>
        {id.toUpperCase()}
      </div>
      <div style={{ ...mono, fontSize: 28, fontWeight: 700 }}>{pct}%</div>
      <div style={{ fontSize: 10, color: "var(--text-muted)", margin: "2px 0 8px" }}>RUL remaining</div>
      <Badge color={rul < 0.15 ? "#ef4444" : rul < 0.3 ? "#f59e0b" : "#10b981"}>{status}</Badge>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload) return null;
  return (
    <div style={{
      background: "#1e293b", border: "1px solid #334155", borderRadius: 6,
      padding: "8px 12px", ...mono, fontSize: 11,
    }}>
      <div style={{ color: "#64748b", marginBottom: 4 }}>Snapshot {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>{p.dataKey}: {p.value?.toFixed(4)}</div>
      ))}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [snap, setSnap] = useState(950);
  const [anomalyData, setAnomalyData] = useState(null);
  const [rulData, setRulData] = useState(null);
  const [snapData, setSnapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [inferenceTime, setInferenceTime] = useState(null);

  // Fetch time series data once on mount
  useEffect(() => {
    async function fetchData() {
      try {
        const [aRes, rRes] = await Promise.all([
          fetch("/api/anomaly-scores"),
          fetch("/api/rul-predictions"),
        ]);
        if (!aRes.ok || !rRes.ok) throw new Error("API not available");
        setAnomalyData(await aRes.json());
        setRulData(await rRes.json());
        setLoading(false);
      } catch (e) {
        setError(e.message);
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Fetch snapshot-specific data when slider changes
  useEffect(() => {
    async function fetchSnap() {
      try {
        const res = await fetch(`/api/snapshot/${snap}`);
        if (res.ok) setSnapData(await res.json());
      } catch (e) { /* ignore */ }
    }
    fetchSnap();
  }, [snap]);

  // Compute stats from snapshot data
  const stats = useMemo(() => {
    if (!snapData) return { worst: "—", worstScore: 0, minRul: 1, minRulB: "", status: "—", statusColor: "#64748b", anomalyCount: 0, isLive: false };
    const s = snapData;
    const statusColor = s.status === "CRITICAL" ? "#ef4444" : s.status === "WARNING" ? "#f59e0b" : "#10b981";
    return {
      worst: s.worst_bearing || "—",
      worstScore: s.worst_score || 0,
      minRul: s.min_rul || 0,
      minRulB: s.min_rul_bearing || "",
      status: s.status || "—",
      statusColor,
      anomalyCount: s.anomaly_count || 0,
      isLive: s.live || false,
    };
  }, [snapData]);

  // Live analyze function
  async function analyzeLive() {
    setAnalyzing(true);
    setInferenceTime(null);
    try {
      const res = await fetch(`/api/analyze/${snap}`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setSnapData(data);
        setInferenceTime(data.inference_time_sec);
      }
    } catch (e) {
      console.error("Analyze failed:", e);
    }
    setAnalyzing(false);
  }

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", flexDirection: "column", gap: 16 }}>
        <div style={{ ...mono, fontSize: 20, color: "var(--cyan)" }}>BearingMind</div>
        <div style={{ fontSize: 14, color: "var(--text-muted)" }}>Loading data from API...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", flexDirection: "column", gap: 16, padding: 40 }}>
        <div style={{ ...mono, fontSize: 20, color: "var(--red)" }}>Connection error</div>
        <div style={{ fontSize: 14, color: "var(--text-secondary)", textAlign: "center", maxWidth: 500 }}>
          Could not reach the BearingMind API. Make sure the backend is running:
        </div>
        <code style={{ ...mono, fontSize: 13, color: "var(--cyan)", background: "#111827", padding: "8px 16px", borderRadius: 6 }}>
          cd bearingmind && python api/server.py
        </code>
        <div style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 8 }}>
          Then open <span style={{ color: "var(--cyan)" }}>http://localhost:3000</span>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* ── Topbar ──────────────────────────────────────────── */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 24px", borderBottom: "1px solid var(--border)", background: "var(--bg-card)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{ ...mono, fontSize: 18, fontWeight: 700, color: "var(--cyan)" }}>
            BearingMind <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>/ condition monitoring</span>
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--text-secondary)" }}>
            <span style={{
              width: 8, height: 8, borderRadius: "50%", background: "var(--green)",
              display: "inline-block", animation: "pulse 2s ease-in-out infinite",
            }} />
            System online
          </div>
        </div>
        <span style={{ ...mono, fontSize: 11, color: "var(--text-muted)" }}>
          NASA IMS Test Rig — 4 bearings @ 2000 RPM
        </span>
      </div>

      {/* ── Snapshot slider + Analyze button ──────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 24px 2px" }}>
        <span style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>
          Snapshot
        </span>
        <input
          type="range" min={0} max={983} value={snap}
          onChange={e => setSnap(+e.target.value)}
          style={{ flex: 1, accentColor: "#06b6d4" }}
        />
        <span style={{ ...mono, fontSize: 14, fontWeight: 600, color: "var(--cyan)", minWidth: 48, textAlign: "right" }}>
          {snap}
        </span>
        <button
          onClick={analyzeLive}
          disabled={analyzing}
          style={{
            ...mono, fontSize: 11, fontWeight: 600,
            padding: "6px 16px", borderRadius: 6, border: "none", cursor: analyzing ? "wait" : "pointer",
            background: analyzing ? "#1e293b" : "#06b6d4", color: analyzing ? "#64748b" : "#0a0e17",
            transition: "all 0.2s",
          }}
        >
          {analyzing ? "Analyzing..." : "Analyze Live"}
        </button>
        {stats.isLive && (
          <Badge color="#06b6d4">LIVE</Badge>
        )}
        {inferenceTime !== null && (
          <span style={{ ...mono, fontSize: 10, color: "var(--text-muted)" }}>
            {inferenceTime}s
          </span>
        )}
      </div>

      <div style={{ padding: "8px 24px 24px", display: "flex", flexDirection: "column", gap: 16 }}>

        {/* ── Stat cards ──────────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
          <StatCard label="Worst bearing" value={stats.worst.toUpperCase()}
            sub={`score: ${stats.worstScore.toFixed(4)}`}
            color={stats.worstScore > 0.1 ? "#ef4444" : "#10b981"} />
          <StatCard label="Min RUL" value={`${(stats.minRul * 100).toFixed(1)}%`}
            sub={stats.minRulB}
            color={stats.minRul < 0.15 ? "#ef4444" : stats.minRul < 0.3 ? "#f59e0b" : "#10b981"} />
          <StatCard label="System status" value={stats.status}
            sub={`snapshot ${snap} / 983`} color={stats.statusColor} />
          <StatCard label="Anomalies" value={stats.anomalyCount}
            sub="of 4 bearings"
            color={stats.anomalyCount > 0 ? "#ef4444" : "#10b981"} />
        </div>

        {/* ── Bearing grid + Anomaly chart ────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <Card title="Bearing health — RUL at current snapshot">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
              {BEARINGS.map(b => (
                <BearingCard key={b} id={b}
                  rul={snapData?.bearings?.[b]?.rul ?? 1} />
              ))}
            </div>
          </Card>

          <Card title="Anomaly scores — time series" badge="Isolation Forest" badgeColor="#06b6d4">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={anomalyData}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis dataKey="snap" stroke="#475569" tick={{ fontSize: 9, ...mono }} />
                <YAxis stroke="#475569" tick={{ fontSize: 9, ...mono }} />
                <Tooltip content={<ChartTooltip />} />
                <ReferenceLine x={snap} stroke="#06b6d4" strokeDasharray="4 4" strokeOpacity={0.6} />
                {BEARINGS.map(b => (
                  <Line key={b} type="monotone" dataKey={b} stroke={COLORS[b]} dot={false} strokeWidth={1.5} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* ── RUL chart (full width) ──────────────────────────────── */}
        <Card title="RUL predictions — time series" badge="LSTM" badgeColor="#8b5cf6">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={rulData}>
              <CartesianGrid stroke="#1e293b" />
              <XAxis dataKey="snap" stroke="#475569" tick={{ fontSize: 9, ...mono }} />
              <YAxis stroke="#475569" tick={{ fontSize: 9, ...mono }} domain={[0, 1]} />
              <Tooltip content={<ChartTooltip />} />
              <ReferenceLine x={snap} stroke="#06b6d4" strokeDasharray="4 4" strokeOpacity={0.6} />
              <ReferenceLine y={0.15} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.4} />
              <ReferenceLine y={0.3} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.3} />
              {BEARINGS.map(b => (
                <Line key={b} type="monotone" dataKey={b} stroke={COLORS[b]} dot={false} strokeWidth={1.5} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* ── RCA (left) + SHAP (right) ───────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

          {/* RCA Panel — always visible */}
          <Card title="Root cause analysis" badge={
            snapData?.rca && !snapData.rca.error ? snapData.rca.urgency : "—"
          } badgeColor={
            !snapData?.rca || snapData.rca.error ? "#64748b"
              : snapData.rca.urgency === "CRITICAL" ? "#ef4444"
              : snapData.rca.urgency === "HIGH" ? "#f59e0b" : "#10b981"
          }>
            {!snapData?.rca || snapData.rca.error ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 200, gap: 12 }}>
                <div style={{ width: 48, height: 48, borderRadius: "50%", border: "2px solid #334155", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <span style={{ ...mono, fontSize: 20, color: "#334155" }}>?</span>
                </div>
                <div style={{ ...mono, fontSize: 13, color: "var(--text-muted)" }}>No anomaly detected</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center", maxWidth: 250 }}>
                  Click "Analyze Live" on an anomalous snapshot to generate a root cause analysis report
                </div>
              </div>
            ) : (
              <div style={{ maxHeight: 500, overflowY: "auto" }}>
                {/* English Summary */}
                <div style={{
                  fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6,
                  padding: "10px 12px", background: "var(--bg-primary)", borderRadius: 6,
                  borderLeft: `3px solid ${snapData.rca.urgency === "CRITICAL" ? "#ef4444" : snapData.rca.urgency === "HIGH" ? "#f59e0b" : "#10b981"}`,
                  marginBottom: 14,
                }}>
                  {snapData.rca.summary || `${snapData.rca.diagnosis} detected. Click Analyze Live for details.`}
                </div>

                {/* Diagnosis */}
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 4 }}>Fault diagnosis</div>
                <div style={{ ...mono, fontSize: 14, color: "#f59e0b", marginBottom: 14 }}>
                  {snapData.rca.diagnosis || snapData.probable_fault}
                </div>

                {/* Actions */}
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 6 }}>Recommended actions</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 14 }}>
                  {(snapData.rca.recommended_actions || []).map((a, i) => (
                    <div key={i} style={{
                      fontSize: 11, color: "var(--text-secondary)", padding: "5px 8px",
                      background: "var(--bg-primary)", borderRadius: 4,
                      borderLeft: `2px solid ${i === 0 ? "#ef4444" : "#334155"}`,
                    }}>
                      <span style={{ ...mono, color: "var(--text-muted)", marginRight: 6 }}>{i + 1}.</span>{a}
                    </div>
                  ))}
                </div>

                {/* CMMS — Maintenance history */}
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 4 }}>
                  CMMS — Maintenance history
                </div>
                <div style={{ ...mono, fontSize: 12, color: "var(--text-primary)", marginBottom: 6 }}>
                  Last maintenance: <span style={{ color: snapData.rca.cmms_summary?.days_since_last_wo > 90 ? "#f59e0b" : "#10b981" }}>
                    {snapData.rca.cmms_summary?.days_since_last_wo ?? "N/A"} days ago
                  </span>
                </div>
                {(snapData.rca.cmms_summary?.work_orders || []).map((wo, i) => (
                  <div key={i} style={{
                    fontSize: 11, marginBottom: 6, padding: "6px 8px",
                    background: "var(--bg-primary)", borderRadius: 4, borderLeft: "2px solid #334155",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ ...mono, color: "#06b6d4" }}>{wo.wo_number}</span>
                      <span style={{ ...mono, color: "var(--text-muted)", fontSize: 10 }}>{wo.completed_date}</span>
                    </div>
                    <div style={{ color: "var(--text-secondary)", marginTop: 2 }}>{wo.description}</div>
                  </div>
                ))}

                {/* CMMS — Spare parts */}
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", marginTop: 12, marginBottom: 4 }}>
                  CMMS — Spare parts inventory
                </div>
                {(snapData.rca.cmms_summary?.spare_parts || []).map((p, i) => (
                  <div key={i} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    fontSize: 11, padding: "5px 0", borderBottom: "1px solid #1e293b33",
                  }}>
                    <div>
                      <span style={{ ...mono, color: "var(--text-secondary)" }}>{p.part_number}</span>
                      <span style={{ color: "var(--text-muted)", marginLeft: 8, fontSize: 10 }}>{p.description}</span>
                    </div>
                    <span style={{
                      ...mono, fontSize: 11, fontWeight: 600,
                      color: p.in_stock ? "#10b981" : "#ef4444",
                      background: p.in_stock ? "rgba(16,185,129,0.12)" : "rgba(239,68,68,0.12)",
                      padding: "2px 8px", borderRadius: 4,
                    }}>
                      {p.in_stock ? `${p.qty_available} in stock` : "OUT OF STOCK"}
                    </span>
                  </div>
                ))}

                {/* Equipment Manual RAG */}
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", marginTop: 12, marginBottom: 4 }}>
                  Equipment manual (RAG retrieval)
                </div>
                {(snapData.rca.manual_results || []).slice(0, 2).map((m, i) => (
                  <div key={i} style={{
                    fontSize: 11, marginBottom: 6, padding: "6px 8px",
                    background: "var(--bg-primary)", borderRadius: 4,
                  }}>
                    <div style={{ ...mono, fontSize: 10, color: "#06b6d4", marginBottom: 2 }}>{m.source}</div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 3 }}>{m.section}</div>
                    <div style={{ color: "var(--text-secondary)", lineHeight: 1.5 }}>{m.text}...</div>
                  </div>
                ))}

                {/* Weather */}
                {snapData.rca.weather_impact && (
                  <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", marginTop: 12, marginBottom: 4 }}>
                    Environmental risk (Weather MCP)
                  </div>
                )}
                {snapData.rca.weather_impact && (
                  <div style={{ ...mono, fontSize: 12, color: snapData.rca.weather_impact === "LOW" ? "#10b981" : "#f59e0b" }}>
                    Combined risk: {snapData.rca.weather_impact}
                  </div>
                )}

                {/* Alert routing */}
                {snapData.alert && (
                  <div style={{
                    marginTop: 12, padding: "8px 10px",
                    background: "var(--bg-primary)", borderRadius: 4, borderLeft: "2px solid #06b6d4",
                  }}>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 4 }}>Alert routing</div>
                    <div style={{ ...mono, fontSize: 11, color: "var(--text-secondary)" }}>{snapData.alert.summary}</div>
                    {snapData.alert.work_order && (
                      <div style={{ ...mono, fontSize: 11, color: "#06b6d4", marginTop: 3 }}>Work order created: {snapData.alert.work_order}</div>
                    )}
                  </div>
                )}
              </div>
            )}
          </Card>

          {/* SHAP Panel */}
          <Card title="SHAP feature importance" badge="Explainability" badgeColor="#10b981">
            <div>
              {snapData?.shap_snapshot !== undefined && snapData.shap_snapshot !== snap && (
                <div style={{ ...mono, fontSize: 10, color: "var(--text-muted)", marginBottom: 8 }}>
                  Nearest SHAP data: snapshot {snapData.shap_snapshot}
                </div>
              )}
              <div style={{
                display: "grid", gridTemplateColumns: "2fr 70px 100px", gap: 4,
                padding: "0 0 6px", borderBottom: "1px solid var(--border)",
              }}>
                {["Feature", "SHAP value", "Impact"].map(h => (
                  <span key={h} style={{ fontSize: 10, fontWeight: 500, textTransform: "uppercase", letterSpacing: 0.5, color: "var(--text-muted)" }}>
                    {h}
                  </span>
                ))}
              </div>
              <div style={{ maxHeight: 340, overflowY: "auto" }}>
                {(snapData?.shap_features || []).length === 0 ? (
                  <div style={{ padding: "12px 0", fontSize: 12, color: "var(--text-muted)" }}>
                    Click "Analyze Live" to compute SHAP values
                  </div>
                ) : (
                  (() => {
                    const features = snapData.shap_features;
                    const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_value || 0)), 0.001);
                    return features.slice(0, 16).map((f, i) => {
                      const val = f.shap_value || 0;
                      const barW = Math.max((Math.abs(val) / maxAbs) * 90, 2);
                      const barColor = val > 0 ? "#ef4444" : "#10b981";
                      return (
                        <div key={i} style={{
                          display: "grid", gridTemplateColumns: "2fr 70px 100px", gap: 4,
                          padding: "5px 0", borderBottom: "1px solid #1e293b22",
                        }}>
                          <span style={{ ...mono, fontSize: 11, color: i < 3 ? "var(--text-primary)" : "var(--text-secondary)" }}>
                            {f.feature}
                          </span>
                          <span style={{ ...mono, fontSize: 11, color: barColor, textAlign: "right" }}>
                            {val > 0 ? "+" : ""}{val.toFixed(4)}
                          </span>
                          <div style={{ display: "flex", alignItems: "center" }}>
                            <div style={{ height: 6, borderRadius: 3, width: barW, background: barColor, opacity: i < 3 ? 1 : 0.6 }} />
                          </div>
                        </div>
                      );
                    });
                  })()
                )}
              </div>
              {snapData?.probable_fault && (
                <div style={{ ...mono, fontSize: 11, color: "#f59e0b", marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
                  Probable fault: {snapData.probable_fault}
                </div>
              )}
              {snapData?.shap_features?.length > 0 && (
                <div style={{ ...mono, fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
                  Bearing: {snapData.shap_features[0].bearing} · Score: {snapData.shap_features[0].anomaly_score} · RUL: {(snapData.shap_features[0].rul_score * 100).toFixed(1)}%
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* ── Environment panel ────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
          <Card title="System information">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, fontSize: 12 }}>
              <div>
                <div style={{ color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", fontSize: 10 }}>Dataset</div>
                <div style={mono}>NASA IMS — 2nd test</div>
                <div style={{ color: "var(--text-muted)", marginTop: 2 }}>984 snapshots, 4 bearings</div>
              </div>
              <div>
                <div style={{ color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", fontSize: 10 }}>Models</div>
                <div style={mono}>Isolation Forest + LSTM</div>
                <div style={{ color: "var(--text-muted)", marginTop: 2 }}>SHAP explainability</div>
              </div>
              <div>
                <div style={{ color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", fontSize: 10 }}>MCP servers</div>
                <div style={mono}>Manual · CMMS · Weather</div>
                <div style={{ color: "var(--text-muted)", marginTop: 2 }}>3 tools connected</div>
              </div>
            </div>
          </Card>

          <Card title="Environment" badge="Weather MCP" badgeColor="#3b82f6">
            {[
              ["Location", "Cincinnati, OH"],
              ["Temperature", "22.2°C"],
              ["Humidity", "52%"],
              ["Temp risk", "LOW", "#10b981"],
              ["Combined risk", "LOW", "#10b981"],
            ].map(([label, value, color], i) => (
              <div key={i} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "6px 0", borderBottom: i < 4 ? "1px solid var(--border)" : "none",
              }}>
                <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{label}</span>
                <span style={{ ...mono, fontSize: 13, fontWeight: 500, color: color || "var(--text-primary)" }}>{value}</span>
              </div>
            ))}
          </Card>
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────── */}
      <div style={{
        padding: "12px 24px", borderTop: "1px solid var(--border)",
        fontSize: 11, color: "var(--text-muted)",
        display: "flex", justifyContent: "space-between",
      }}>
        <span>BearingMind — Explainable AI & multi-agent predictive maintenance</span>
        <span>MCP: Equipment Manual (RAG) · CMMS (SQLite) · Weather (Open-Meteo)</span>
      </div>
    </div>
  );
}
