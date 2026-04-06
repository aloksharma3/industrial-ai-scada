"""
server.py — FastAPI Backend for BearingMind Dashboard
Loads trained ML models at startup. Serves both historical CSV data
and LIVE inference with per-feature SHAP values.

Run:
    cd bearingmind
    pip install fastapi uvicorn
    python api/server.py

Endpoints:
    GET  /api/health           → system status + model info
    GET  /api/anomaly-scores   → historical IF scores (charts)
    GET  /api/rul-predictions  → historical RUL predictions (charts)
    GET  /api/snapshot/{n}     → basic data for one snapshot (fast)
    POST /api/analyze/{n}      → LIVE inference: IF + LSTM + SHAP (all 16 features)
    GET  /api/rul-metrics      → model accuracy per bearing
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS  = BASE_DIR / "results"
SRC_DIR  = BASE_DIR / "src"

# Add src to path for model imports
sys.path.insert(0, str(SRC_DIR))

# ── Pickle namespace fix ──────────────────────────────────────────────────────
from isolation_forest import BearingAnomalyDetector, SingleBearingDetector
from rul_lstm import BearingRULPredictor, SingleBearingRUL, LSTMRULModel
from mcp_equipment_manual import EquipmentManualMCP
from mcp_cmms import CMMSMCP
from rca_agent import RCAAgent
from alert_agent import AlertAgent

import __main__
for cls in [BearingAnomalyDetector, SingleBearingDetector,
            BearingRULPredictor, SingleBearingRUL, LSTMRULModel]:
    setattr(__main__, cls.__name__, cls)

# ── Feature labels ────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "rms": "RMS vibration", "peak_to_peak": "Peak-to-peak amplitude",
    "kurtosis": "Kurtosis", "crest_factor": "Crest factor",
    "skewness": "Skewness", "shape_factor": "Shape factor",
    "impulse_factor": "Impulse factor", "margin_factor": "Margin factor",
    "spectral_centroid": "Spectral centroid (Hz)",
    "spectral_bandwidth": "Spectral bandwidth (Hz)",
    "spectral_entropy": "Spectral entropy",
    "dominant_freq_hz": "Dominant frequency (Hz)",
    "hf_energy_ratio": "High-freq energy ratio",
    "bpfo_band_energy": "BPFO band energy (outer race)",
    "bpfi_band_energy": "BPFI band energy (inner race)",
    "bsf_band_energy": "BSF band energy (ball fault)",
}

FAULT_MAP = {
    "bpfo_band_energy": "outer race fault",
    "bpfi_band_energy": "inner race fault",
    "bsf_band_energy": "rolling element (ball) fault",
    "kurtosis": "impulsive fault (spalling/cracking)",
    "crest_factor": "impulsive fault (early stage)",
    "hf_energy_ratio": "surface degradation",
    "spectral_entropy": "distributed damage",
}

BEARINGS = ["b1_ch1", "b2_ch1", "b3_ch1", "b4_ch1"]

# ── Load CSV data (for historical charts) ─────────────────────────────────────
print("=" * 50)
print("BearingMind API — Loading data + models ...")
print("=" * 50)

anomaly_df = pd.read_csv(RESULTS / "if" / "anomaly_scores.csv")
rul_df     = pd.read_csv(RESULTS / "rul" / "rul_predictions.csv")
metrics_df = pd.read_csv(RESULTS / "rul" / "rul_metrics.csv", index_col=0)
feature_df = pd.read_csv(RESULTS / "feature_matrix.csv", index_col=0)

print(f"  Feature matrix: {feature_df.shape}")
print(f"  Anomaly scores: {len(anomaly_df)} snapshots")
print(f"  RUL predictions: {len(rul_df)} snapshots")

# ── Load ML models (for live inference) ───────────────────────────────────────
print("Loading ML models for live inference ...")

anomaly_det = BearingAnomalyDetector()
anomaly_det.fit_from_df(feature_df)
anomaly_det.load_models(str(RESULTS / "if" / "models"))
print(f"  Isolation Forest: {len(anomaly_det.bearing_ids_)} models loaded")

rul_pred = BearingRULPredictor()
rul_pred.feature_matrix_ = feature_df
rul_pred.load_models(str(RESULTS / "rul" / "models"))
rul_pred.bearing_ids_ = sorted(rul_pred.predictors_.keys())
print(f"  LSTM RUL: {len(rul_pred.bearing_ids_)} models loaded")

# ── Load SHAP explainers (for live SHAP values) ──────────────────────────────
print("Loading SHAP explainers (this may take a moment) ...")
shap_explainer = None
try:
    from shap_explainer import BearingShapExplainer
    shap_explainer = BearingShapExplainer(anomaly_det, rul_pred)
    shap_explainer.fit(df=feature_df, n_background=100)
    print("  SHAP explainers: ready for live inference")
except Exception as e:
    print(f"  SHAP explainers: failed ({e})")
    print("  Live SHAP will not be available")

# ── Load MCP servers + RCA agent ──────────────────────────────────────────────
print("Loading MCP servers ...")
manual_mcp = EquipmentManualMCP()
manual_mcp.load()

cmms_mcp = CMMSMCP(db_path=str(RESULTS / "rca" / "cmms.db"))
cmms_mcp.initialize()

weather_mcp = None
try:
    from mcp_weather import WeatherMCP
    weather_mcp = WeatherMCP()
    weather_mcp.fetch()
    print("  Weather MCP: connected")
except Exception as e:
    print(f"  Weather MCP: skipped ({e})")

rca_agent = RCAAgent(
    manual_mcp=manual_mcp, cmms_mcp=cmms_mcp,
    weather_mcp=weather_mcp)

alert_agent = AlertAgent(
    cmms_mcp=cmms_mcp,
    log_path=str(RESULTS / "rca" / "alert_log.json"))

print("  RCA Agent + Alert Agent: ready")

MODELS_LOADED = shap_explainer is not None
print(f"\n{'=' * 50}")
print(f"Models ready. Live inference: {'ENABLED' if MODELS_LOADED else 'DISABLED'}")
print(f"{'=' * 50}\n")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="BearingMind API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "total_snapshots": len(feature_df),
        "bearings": BEARINGS,
        "live_inference": MODELS_LOADED,
        "models": {
            "isolation_forest": f"{len(anomaly_det.bearing_ids_)} detectors",
            "lstm_rul": f"{len(rul_pred.bearing_ids_)} predictors",
            "shap": "ready" if shap_explainer else "unavailable",
        },
    }


@app.get("/api/anomaly-scores")
def get_anomaly_scores():
    """Historical anomaly scores for time series chart."""
    records = []
    for i, row in anomaly_df.iterrows():
        records.append({
            "snap": i,
            "b1_ch1": row.get("b1_ch1_score", 0),
            "b2_ch1": row.get("b2_ch1_score", 0),
            "b3_ch1": row.get("b3_ch1_score", 0),
            "b4_ch1": row.get("b4_ch1_score", 0),
        })
    return records


@app.get("/api/rul-predictions")
def get_rul_predictions():
    """Historical RUL predictions for time series chart."""
    records = []
    for i, row in rul_df.iterrows():
        records.append({
            "snap": i,
            "b1_ch1": row.get("b1_ch1_rul", 0),
            "b2_ch1": row.get("b2_ch1_rul", 0),
            "b3_ch1": row.get("b3_ch1_rul", 0),
            "b4_ch1": row.get("b4_ch1_rul", 0),
        })
    return records


@app.get("/api/rul-metrics")
def get_rul_metrics():
    return metrics_df.to_dict(orient="index")


@app.get("/api/snapshot/{snap}")
def get_snapshot(snap: int):
    """Quick snapshot data (from CSV, no live inference)."""
    if snap < 0 or snap >= len(anomaly_df):
        return {"error": f"Snapshot {snap} out of range"}

    a_row = anomaly_df.iloc[snap]
    r_row = rul_df.iloc[snap]
    bearings = {}
    worst_score, worst_bearing = -float("inf"), ""
    min_rul, min_rul_bearing = float("inf"), ""
    anomaly_count = 0

    for b in BEARINGS:
        score = float(a_row.get(f"{b}_score", 0))
        flag = int(a_row.get(f"{b}_flag", 0))
        rul = float(r_row.get(f"{b}_rul", 0))
        bearings[b] = {"anomaly_score": score, "flag": flag, "rul": rul}
        if score > worst_score:
            worst_score, worst_bearing = score, b
        if rul < min_rul:
            min_rul, min_rul_bearing = rul, b
        if flag == 1:
            anomaly_count += 1

    status = "CRITICAL" if min_rul < 0.15 else "WARNING" if min_rul < 0.3 else "NORMAL"

    return {
        "snapshot": snap, "bearings": bearings,
        "worst_bearing": worst_bearing, "worst_score": worst_score,
        "min_rul": min_rul, "min_rul_bearing": min_rul_bearing,
        "status": status, "anomaly_count": anomaly_count,
        "live": False,
    }


@app.post("/api/analyze/{snap}")
def analyze_live(snap: int):
    """
    LIVE inference — runs IF + LSTM + SHAP on the requested snapshot.
    Returns all 16 SHAP features with individual values.
    """
    if snap < 0 or snap >= len(feature_df):
        return {"error": f"Snapshot {snap} out of range"}

    if not MODELS_LOADED:
        return {"error": "Models not loaded — SHAP unavailable"}

    start = time.time()

    # ── Live anomaly scores ───────────────────────────────────────────
    scores_df = anomaly_det.score_all()
    bearings = {}
    worst_score, worst_bearing = -float("inf"), ""
    min_rul, min_rul_bearing = float("inf"), ""
    anomaly_count = 0

    for b in BEARINGS:
        score = float(scores_df[f"{b}_score"].iloc[snap])
        flag = int(scores_df[f"{b}_flag"].iloc[snap])
        preds = rul_pred.predictors_[b].predict(feature_df)
        rul = float(preds[snap])
        bearings[b] = {"anomaly_score": round(score, 4), "flag": flag, "rul": round(rul, 4)}
        if score > worst_score:
            worst_score, worst_bearing = score, b
        if rul < min_rul:
            min_rul, min_rul_bearing = rul, b
        if flag == 1:
            anomaly_count += 1

    status = "CRITICAL" if min_rul < 0.15 else "WARNING" if min_rul < 0.3 else "NORMAL"

    # ── Live SHAP (all 16 features per bearing) ───────────────────────
    shap_result = shap_explainer.explain_snapshot(snap)

    # Extract per-feature SHAP for worst bearing
    shap_features = []
    probable_fault = "undetermined"
    target = worst_bearing

    if target in shap_result:
        bdata = shap_result[target]
        anomaly_exp = bdata.get("anomaly", {})
        shap_vals = anomaly_exp.get("shap_values")
        feat_names = anomaly_exp.get("feature_names", [])
        anom_score = anomaly_exp.get("anomaly_score", 0)
        rul_score = bdata.get("rul", {}).get("rul_score", 0)
        probable_fault = bdata.get("probable_fault", "undetermined")

        if shap_vals is not None:
            # Build all 16 features with values
            pairs = []
            for i, fname in enumerate(feat_names):
                raw = fname.split("_", 2)[2] if fname.count("_") >= 2 else fname
                label = FEATURE_LABELS.get(raw, raw)
                # Negate: IF decision_function is positive-for-normal, so
                # its SHAP values are negative for anomaly-driving features.
                # Flip sign so positive SHAP = pushes toward anomaly (red in UI).
                val = -float(shap_vals[i])
                pairs.append((label, val, raw))

            # Sort by absolute value (most important first)
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            for label, val, raw in pairs:
                shap_features.append({
                    "feature": label,
                    "bearing": target,
                    "shap_value": round(val, 6),
                    "abs_value": round(abs(val), 6),
                    "anomaly_score": round(float(anom_score), 4),
                    "rul_score": round(float(rul_score), 4),
                })

    elapsed = round(time.time() - start, 2)

    # ── Run RCA agent (queries 3 MCP servers) ─────────────────────────
    # Only run when there's an actual problem: anomaly flagged OR RUL < 25%
    rca_result = None
    alert_result = None
    if shap_features and (anomaly_count > 0 or min_rul < 0.30):
        try:
            target_context = shap_result.get(target, shap_result)
            rca_report = rca_agent.analyze(target_context, bearing_id=target)
            rca_result = {
                "diagnosis": rca_report.get("fault_type", ""),
                "urgency": rca_report.get("urgency", "MEDIUM"),
                "recommended_actions": rca_report.get("recommended_actions", []),
                "generation_mode": rca_report.get("generation_mode", ""),
                "summary": (
                    f"{rca_report.get('fault_type', 'Fault').upper()} detected on "
                    f"{target}. SHAP analysis identified "
                    f"{shap_features[0]['feature'] if shap_features else 'unknown'} "
                    f"as the primary anomaly driver. "
                    f"Last maintenance was {rca_report.get('cmms_summary', {}).get('days_since_last_wo', '?')} "
                    f"days ago. "
                    f"{'Replacement parts are available in inventory.' if any(p.get('in_stock') for p in rca_report.get('cmms_summary', {}).get('spare_parts', []) if 'error' not in p) else 'Parts may need to be ordered.'} "
                    f"Urgency: {rca_report.get('urgency', 'MEDIUM')}."
                ),
                "manual_results": [
                    {"source": r.get("source", ""), "section": r.get("section", ""),
                     "text": r.get("text", "")[:200]}
                    for r in rca_report.get("manual_results", [])[:3]
                ],
                "cmms_summary": {
                    "days_since_last_wo": rca_report.get("cmms_summary", {}).get("days_since_last_wo"),
                    "work_orders": [
                        {"wo_number": wo.get("wo_number", ""),
                         "completed_date": wo.get("completed_date", ""),
                         "description": wo.get("description", "")[:100]}
                        for wo in rca_report.get("cmms_summary", {}).get("work_orders", [])[:3]
                    ],
                    "spare_parts": [
                        {"part_number": p.get("part_number", ""),
                         "description": p.get("description", ""),
                         "in_stock": p.get("in_stock", False),
                         "qty_available": p.get("qty_available", 0)}
                        for p in rca_report.get("cmms_summary", {}).get("spare_parts", [])
                        if "error" not in p
                    ],
                },
                "weather_impact": (rca_report.get("weather_impact") or {}).get("combined_risk"),
                "weather_conditions": (
                    {
                        "temperature_c": (rca_report["weather_impact"].get("conditions") or {}).get("temperature_c"),
                        "humidity_pct":  (rca_report["weather_impact"].get("conditions") or {}).get("humidity_pct"),
                    }
                    if rca_report.get("weather_impact") and isinstance(rca_report.get("weather_impact"), dict)
                    else None
                ),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            rca_result = {"error": str(e)}

    if rca_result and not rca_result.get("error"):
        try:
            alert_out = alert_agent.process(rca_report)
            alert_result = {
                "summary": alert_out.get("summary", ""),
                "notifications_count": len(alert_out.get("notifications", [])),
                "work_order": alert_out.get("work_order", {}).get("wo_number") if alert_out.get("work_order") else None,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            alert_result = {"error": str(e)}

    elapsed_total = round(time.time() - start, 2)

    return {
        "snapshot": snap,
        "bearings": bearings,
        "worst_bearing": worst_bearing,
        "worst_score": round(worst_score, 4),
        "min_rul": round(min_rul, 4),
        "min_rul_bearing": min_rul_bearing,
        "status": status,
        "anomaly_count": anomaly_count,
        "shap_features": shap_features,
        "probable_fault": probable_fault,
        "rca": rca_result,
        "alert": alert_result,
        "live": True,
        "inference_time_sec": elapsed_total,
    }


@app.get("/api/weather")
def get_weather():
    """Current weather conditions from Weather MCP (fetched at startup)."""
    if weather_mcp is None:
        return {"error": "Weather MCP not initialized", "combined_risk": "UNKNOWN", "conditions": {}}
    try:
        return weather_mcp.get_weather_impact()
    except Exception as e:
        return {"error": str(e), "combined_risk": "UNKNOWN", "conditions": {}}


# ── Serve dashboard ───────────────────────────────────────────────────────────
DASHBOARD_DIR = BASE_DIR / "dashboard" / "dist"
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True))

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"BearingMind API starting ...")
    print(f"  Dashboard: http://localhost:8000")
    print(f"  API docs:  http://localhost:8000/docs")
    print(f"  Live analyze: POST http://localhost:8000/api/analyze/950")
    uvicorn.run(app, host="0.0.0.0", port=8000)