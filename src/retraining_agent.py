"""
retraining_agent.py — Model Retraining Agent
Industrial AI Predictive Maintenance | BearingMind

Three-tier monitoring (inspired by Uber Michelangelo):
    Tier 1 — LOG     : every run, append lightweight metrics (near-zero cost)
    Tier 2 — CHECK   : every N runs, KS tests on feature distributions
    Tier 3 — RETRAIN : only when drift confirmed, retrain affected models

LangGraph path:
    ... → log_metrics → [counter ≥ N?] → check_drift → [drift?] → retrain → END
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

CHECK_EVERY_DEFAULT    = 50     # run drift check every N snapshots
KS_THRESHOLD           = 0.15   # KS statistic above this = drift
ANOMALY_RATE_THRESHOLD = 0.30   # if >30% of recent snapshots flagged, suspicious
RUL_BIAS_THRESHOLD     = 0.15   # mean RUL prediction off by this much = concept drift
MIN_SAMPLES_FOR_CHECK  = 30     # need at least this many logged metrics to check


# ── Drift detection functions ─────────────────────────────────────────────────

def ks_test_features(baseline: np.ndarray, recent: np.ndarray) -> dict:
    """
    KS test per feature — compares training vs recent distributions.
    Non-parametric, works on small samples. Standard in production ML monitoring.
    """
    from scipy.stats import ks_2samp

    n_features = baseline.shape[1]
    results = []
    drifted_features = []

    for i in range(n_features):
        stat, pvalue = ks_2samp(baseline[:, i], recent[:, i])
        results.append({
            "feature_index": i,
            "ks_statistic": float(stat),
            "p_value": float(pvalue),
            "drifted": stat > KS_THRESHOLD,
        })
        if stat > KS_THRESHOLD:
            drifted_features.append(i)

    return {
        "per_feature": results,
        "n_drifted": len(drifted_features),
        "n_total": n_features,
        "drift_ratio": len(drifted_features) / n_features if n_features > 0 else 0,
        "drifted_indices": drifted_features,
    }


def check_anomaly_rate_drift(metrics_log: list[dict]) -> dict:
    """If anomaly rate spikes above threshold, the healthy baseline has probably shifted."""
    if len(metrics_log) < MIN_SAMPLES_FOR_CHECK:
        return {"drifted": False, "reason": "insufficient data"}

    recent = metrics_log[-MIN_SAMPLES_FOR_CHECK:]
    anomaly_flags = [m.get("is_anomaly", False) for m in recent]
    anomaly_rate = sum(anomaly_flags) / len(anomaly_flags)

    return {
        "anomaly_rate": float(anomaly_rate),
        "threshold": ANOMALY_RATE_THRESHOLD,
        "drifted": anomaly_rate > ANOMALY_RATE_THRESHOLD,
        "reason": (f"Anomaly rate {anomaly_rate:.1%} exceeds "
                   f"{ANOMALY_RATE_THRESHOLD:.0%} threshold — possible "
                   f"baseline shift") if anomaly_rate > ANOMALY_RATE_THRESHOLD
                  else "Anomaly rate within normal range",
    }


def check_rul_bias(metrics_log: list[dict]) -> dict:
    """Concept drift: if RUL predictions are flat (low variance), the model is stale."""
    if len(metrics_log) < MIN_SAMPLES_FOR_CHECK:
        return {"drifted": False, "reason": "insufficient data"}

    recent = metrics_log[-MIN_SAMPLES_FOR_CHECK:]
    rul_means = [m.get("mean_rul", 0.5) for m in recent if "mean_rul" in m]

    if not rul_means:
        return {"drifted": False, "reason": "no RUL data in metrics"}

    overall_mean = float(np.mean(rul_means))

    # Check if predictions are biased toward extremes
    # A well-calibrated model on degrading data should show a downward trend
    # If mean is stuck near 0.5 on data that includes failures, model may be stale
    bias = abs(overall_mean - 0.5)
    # Also check if variance is suspiciously low (model not responding to changes)
    rul_std = float(np.std(rul_means))

    drifted = rul_std < 0.05 and len(rul_means) > 10  # flat predictions = stale model

    return {
        "mean_rul": overall_mean,
        "std_rul": rul_std,
        "drifted": drifted,
        "reason": (f"RUL predictions flat (std={rul_std:.3f}) — model may be "
                   f"stale") if drifted
                  else "RUL predictions showing expected variance",
    }


# ── Retraining Agent ──────────────────────────────────────────────────────────

class RetrainingAgent:
    """
    Three-tier model monitoring: log (every run) → check (periodic) → retrain (rare).
    """

    def __init__(self,
                 check_every: int = CHECK_EVERY_DEFAULT,
                 metrics_path: str = "results/rca/drift_metrics.json",
                 models_dir: str = "results"):
        self.check_every = check_every
        self.metrics_path = metrics_path
        self.models_dir = models_dir
        self._run_counter = 0
        self._metrics_log = self._load_metrics()

    # ── Tier 1: Log (every run) ───────────────────────────────────────────

    def log_metrics(self, state: dict) -> dict:
        """Tier 1: append lightweight metrics from the current run. Near-zero cost."""
        self._run_counter += 1

        entry = {
            "run_number": self._run_counter,
            "timestamp": datetime.now().isoformat(),
            "snapshot_index": state.get("snapshot_index"),
            "is_anomaly": state.get("is_anomaly", False),
            "urgency": state.get("urgency", "LOW"),
            "worst_bearing": state.get("worst_bearing"),
            "path_taken": state.get("path_taken", []),
        }

        # Log anomaly scores
        anomaly_scores = state.get("anomaly_scores", {})
        if anomaly_scores:
            entry["mean_anomaly_score"] = float(np.mean(
                list(anomaly_scores.values())))
            entry["max_anomaly_score"] = float(np.max(
                list(anomaly_scores.values())))

        # Log RUL scores
        rul_scores = state.get("rul_scores", {})
        if rul_scores:
            entry["mean_rul"] = float(np.mean(list(rul_scores.values())))
            entry["min_rul"] = float(np.min(list(rul_scores.values())))

        self._metrics_log.append(entry)
        self._save_metrics()

        check_due = (self._run_counter % self.check_every) == 0

        print(f"\n[log_metrics] Run #{self._run_counter} logged"
              f" | anomaly={entry['is_anomaly']}"
              f" | drift check {'DUE' if check_due else f'in {self.check_every - (self._run_counter % self.check_every)} runs'}")

        return {
            "run_number": self._run_counter,
            "check_due": check_due,
            "metrics_logged": len(self._metrics_log),
        }

    # ── Tier 2: Check (periodic) ──────────────────────────────────────────

    def should_check(self) -> bool:
        """Is it time to run the expensive drift detection?"""
        return (self._run_counter % self.check_every) == 0

    def check_drift(self, feature_df: pd.DataFrame,
                    n_normal: int = 500) -> dict:
        """Tier 2: run KS test + anomaly rate + RUL bias checks."""
        print(f"\n[check_drift] Running drift detection "
              f"(run #{self._run_counter}) ...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "run_number": self._run_counter,
            "checks": {},
        }

        # ── Check 1: Feature distribution drift (KS test) ─────────
        # Compare training data (first n_normal rows) vs recent data
        # (last n_normal rows or whatever is available)
        baseline = feature_df.iloc[:n_normal].values
        recent_start = max(n_normal, len(feature_df) - n_normal)
        recent = feature_df.iloc[recent_start:].values

        ks_result = ks_test_features(baseline, recent)
        results["checks"]["data_drift"] = {
            "method": "Kolmogorov-Smirnov test per feature",
            "n_drifted": ks_result["n_drifted"],
            "n_total": ks_result["n_total"],
            "drift_ratio": ks_result["drift_ratio"],
            "drifted": ks_result["drift_ratio"] > 0.25,  # >25% features drifted
            "detail": (f"{ks_result['n_drifted']}/{ks_result['n_total']} "
                       f"features drifted (KS > {KS_THRESHOLD})"),
        }
        print(f"  Data drift: {ks_result['n_drifted']}/{ks_result['n_total']} "
              f"features shifted")

        # ── Check 2: Anomaly rate spike ────────────────────────────
        anomaly_result = check_anomaly_rate_drift(self._metrics_log)
        results["checks"]["anomaly_rate"] = anomaly_result
        print(f"  Anomaly rate: {anomaly_result.get('anomaly_rate', 'N/A')}"
              f" — {anomaly_result['reason']}")

        # ── Check 3: RUL prediction bias ──────────────────────────
        rul_result = check_rul_bias(self._metrics_log)
        results["checks"]["rul_bias"] = rul_result
        print(f"  RUL bias: {rul_result['reason']}")

        # ── Overall verdict ────────────────────────────────────────
        drift_detected = any(
            check.get("drifted", False)
            for check in results["checks"].values()
        )
        results["drift_detected"] = drift_detected
        drift_reasons = [
            name for name, check in results["checks"].items()
            if check.get("drifted", False)
        ]
        results["drift_reasons"] = drift_reasons

        if drift_detected:
            print(f"  ⚠ DRIFT DETECTED — reasons: {', '.join(drift_reasons)}")
        else:
            print(f"  ✓ No drift detected")

        return results

    # ── Tier 3: Retrain (rare) ────────────────────────────────────────────

    def retrain(self, feature_df: pd.DataFrame,
                drift_result: dict,
                models_dir: str = None) -> dict:
        """
        Tier 3: retrain affected models. Backs up old models first.
        data_drift/anomaly_rate → retrain IF. rul_bias → retrain LSTM.
        """
        models_dir = models_dir or self.models_dir
        drift_reasons = drift_result.get("drift_reasons", [])

        print(f"\n[retrain] Retraining triggered — reasons: "
              f"{', '.join(drift_reasons)}")

        result = {
            "timestamp": datetime.now().isoformat(),
            "trigger_reasons": drift_reasons,
            "models_retrained": [],
        }

        retrain_if   = "data_drift" in drift_reasons or \
                        "anomaly_rate" in drift_reasons
        retrain_lstm = "rul_bias" in drift_reasons

        # ── Retrain Isolation Forest ──────────────────────────────
        if retrain_if:
            print("  Retraining Isolation Forest ...")
            try:
                from isolation_forest import BearingAnomalyDetector
                import __main__
                from isolation_forest import SingleBearingDetector
                setattr(__main__, "BearingAnomalyDetector",
                        BearingAnomalyDetector)
                setattr(__main__, "SingleBearingDetector",
                        SingleBearingDetector)

                # Retrain with expanded window — use more data as "normal"
                # if we suspect the baseline has shifted
                new_n_normal = min(600, len(feature_df) - 100)
                detector = BearingAnomalyDetector(n_normal=new_n_normal)
                detector.fit_from_df(feature_df)

                # Save retrained models with timestamp
                save_dir = os.path.join(models_dir, "if", "models")
                backup_dir = os.path.join(
                    models_dir, "if",
                    f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                # Backup old models
                if os.path.isdir(save_dir):
                    import shutil
                    shutil.copytree(save_dir, backup_dir)
                    print(f"    Old models backed up → {backup_dir}")

                detector.save_models(save_dir)
                result["models_retrained"].append({
                    "model": "IsolationForest",
                    "new_n_normal": new_n_normal,
                    "backup": backup_dir,
                    "save_dir": save_dir,
                })
                print(f"    ✓ Isolation Forest retrained "
                      f"(n_normal={new_n_normal})")

            except Exception as e:
                print(f"    ✗ IF retraining failed: {e}")
                result["models_retrained"].append({
                    "model": "IsolationForest",
                    "error": str(e),
                })

        # ── Retrain LSTM ──────────────────────────────────────────
        if retrain_lstm:
            print("  Retraining LSTM ...")
            try:
                from rul_lstm import BearingRULPredictor, SingleBearingRUL
                from rul_lstm import LSTMRULModel
                import __main__
                setattr(__main__, "BearingRULPredictor", BearingRULPredictor)
                setattr(__main__, "SingleBearingRUL", SingleBearingRUL)
                setattr(__main__, "LSTMRULModel", LSTMRULModel)

                predictor = BearingRULPredictor(
                    window_size=30, epochs=30)  # fewer epochs for retrain
                predictor.fit_from_df(feature_df)

                save_dir = os.path.join(models_dir, "rul", "models")
                backup_dir = os.path.join(
                    models_dir, "rul",
                    f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                if os.path.isdir(save_dir):
                    import shutil
                    shutil.copytree(save_dir, backup_dir)
                    print(f"    Old models backed up → {backup_dir}")

                predictor.save_models(save_dir)
                result["models_retrained"].append({
                    "model": "LSTM_RUL",
                    "epochs": 30,
                    "backup": backup_dir,
                    "save_dir": save_dir,
                })
                print(f"    ✓ LSTM retrained (30 epochs)")

            except Exception as e:
                print(f"    ✗ LSTM retraining failed: {e}")
                result["models_retrained"].append({
                    "model": "LSTM_RUL",
                    "error": str(e),
                })

        # ── Log retraining event ──────────────────────────────────
        self._metrics_log.append({
            "run_number": self._run_counter,
            "timestamp": datetime.now().isoformat(),
            "event": "retrain",
            "trigger_reasons": drift_reasons,
            "models_retrained": [
                m["model"] for m in result["models_retrained"]
                if "error" not in m
            ],
        })
        self._save_metrics()

        n_ok = sum(1 for m in result["models_retrained"] if "error" not in m)
        print(f"\n  Retraining complete: {n_ok} model(s) updated")

        return result

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_metrics(self) -> list:
        """Load existing metrics log from disk."""
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Restore run counter from last entry
                    if data:
                        last = data[-1]
                        self._run_counter = last.get("run_number", 0)
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_metrics(self) -> None:
        """Persist metrics log to disk."""
        Path(self.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self._metrics_log, f, indent=2, ensure_ascii=False)

    def get_summary(self) -> dict:
        """Return a summary of monitoring state for the dashboard."""
        n_runs = len([m for m in self._metrics_log
                      if m.get("event") != "retrain"])
        n_retrains = len([m for m in self._metrics_log
                          if m.get("event") == "retrain"])
        return {
            "total_runs": n_runs,
            "total_retrains": n_retrains,
            "check_every": self.check_every,
            "next_check_in": self.check_every - (self._run_counter % self.check_every),
            "last_run": self._metrics_log[-1] if self._metrics_log else None,
        }

