"""
features.py — Feature Engineering for NASA Bearing Dataset
Week 2 | Industrial AI Predictive Maintenance

Extends the RMS work from nasa_bearing_eda.ipynb to extract
16 features per bearing per snapshot:
  - 8 time-domain  : RMS, peak-to-peak, kurtosis, crest factor...
  - 8 freq-domain  : FFT spectral centroid, fault band energies...

Output: feature_matrix.csv  →  feeds isolation_forest.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft,fftfreq
import warnings
warnings.filterwarnings("ignore")


# ── Dataset constants ────────────────────────────────

SAMPLING_RATE_HZ = 20_480        # NASA IMS test rig

# 2nd test (4 bearings, 1 channel each) — the set you used for RMS analysis
COLUMNS_4CH = ["b1_ch1", "b2_ch1", "b3_ch1", "b4_ch1"]

# 1st test (4 bearings, 2 channels each)
COLUMNS_8CH = ["b1_ch1", "b1_ch2", "b2_ch1", "b2_ch2",
               "b3_ch1", "b3_ch2", "b4_ch1", "b4_ch2"]

# Bearing fault characteristic frequency multipliers (× shaft speed)
# These are physical properties of the IMS test rig bearings
BPFO = 3.585   # Ball Pass Frequency — Outer race fault
BPFI = 5.415   # Ball Pass Frequency — Inner race fault
BSF  = 2.357   # Ball Spin Frequency — Rolling element fault
SHAFT_RPM = 2000


# ── TIME-DOMAIN FEATURES ──────────────────────────────────────────────────────
# These operate on the raw vibration signal directly.
# You already used RMS in your notebook — these are the next level up.

def compute_rms(signal: np.ndarray) -> float:
    """
    Root Mean Square — overall vibration energy.
    This is exactly what your build_rms_dataframe() computed.
    """
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_peak_to_peak(signal: np.ndarray) -> float:
    """Max amplitude minus min amplitude. Sensitive to large transient spikes."""
    return float(signal.max() - signal.min())


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    4th statistical moment.
    Healthy bearing: ~3.0 (Gaussian noise).
    Bearing fault (spalling, cracking): spikes to 10-40.
    Most widely used bearing fault indicator in industry.
    """
    return float(stats.kurtosis(signal, fisher=False))


def compute_crest_factor(signal: np.ndarray) -> float:
    """
    Peak value / RMS.
    Normal: ~3-4. Impulsive fault (early spalling): can exceed 10.
    Best early-warning feature — rises before RMS does.
    """
    rms = compute_rms(signal)
    return float(np.max(np.abs(signal)) / rms) if rms > 1e-10 else 0.0


def compute_skewness(signal: np.ndarray) -> float:
    """Asymmetry of amplitude distribution. Asymmetric wear → nonzero skew."""
    return float(stats.skew(signal))


def compute_shape_factor(signal: np.ndarray) -> float:
    """RMS / mean(|signal|). Sensitive to waveform shape changes over time."""
    mean_abs = np.mean(np.abs(signal))
    return float(compute_rms(signal) / mean_abs) if mean_abs > 1e-10 else 0.0


def compute_impulse_factor(signal: np.ndarray) -> float:
    """Peak / mean(|signal|). Highlights impulsive fault events."""
    mean_abs = np.mean(np.abs(signal))
    return float(np.max(np.abs(signal)) / mean_abs) if mean_abs > 1e-10 else 0.0


def compute_margin_factor(signal: np.ndarray) -> float:
    """
    Peak / (sqrt_mean)^2.
    Most sensitive to very early-stage damage — rises before kurtosis.
    """
    sqrt_mean = np.mean(np.sqrt(np.abs(signal))) ** 2
    return float(np.max(np.abs(signal)) / sqrt_mean) if sqrt_mean > 1e-10 else 0.0


# ── FREQUENCY-DOMAIN FEATURES ─────────────────────────────────────────────────
# Apply FFT to the signal, then extract features from the frequency spectrum.
# Bearing faults create energy at specific known frequencies (BPFO, BPFI, BSF).

def compute_fft_features(signal: np.ndarray, fs: int = SAMPLING_RATE_HZ) -> dict:
    """
    FFT-based spectral features.

    Returns 8 features:
        spectral_centroid   — frequency centre of mass (Hz)
        spectral_bandwidth  — spread around centroid (Hz)
        spectral_entropy    — how spread out the energy is (0=pure tone, 1=noise)
        dominant_freq_hz    — frequency of peak magnitude
        hf_energy_ratio     — energy above 5kHz (grows as bearing degrades)
        bpfo_band_energy    — energy at outer race fault frequency
        bpfi_band_energy    — energy at inner race fault frequency
        bsf_band_energy     — energy at ball spin fault frequency
    """
    N           = len(signal)
    freqs       = fftfreq(N, d=1.0 / fs)[:N // 2]
    magnitudes  = np.abs(fft(signal))[:N // 2]
    power       = magnitudes ** 2
    total_power = power.sum() + 1e-10

    # Spectral centroid and bandwidth
    centroid  = float(np.sum(freqs * power) / total_power)
    variance  = np.sum(((freqs - centroid) ** 2) * power) / total_power
    bandwidth = float(np.sqrt(max(variance, 0)))

    # Spectral entropy — normalised
    prob    = power / total_power
    prob    = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)) / np.log2(len(freqs) + 1))

    # Dominant frequency
    dominant_freq = float(freqs[np.argmax(magnitudes)])

    # High-frequency energy ratio (>5kHz) — bearing damage shifts energy upward
    hf_mask         = freqs > 5000
    hf_energy_ratio = float(power[hf_mask].sum() / total_power)

    # Fault characteristic band energies
    def band_energy(freq_mult: float, window_hz: float = 50.0) -> float:
        """Energy fraction within ±window_hz of the fault frequency."""
        center_hz = freq_mult * (SHAFT_RPM / 60.0)
        mask = (freqs >= center_hz - window_hz) & (freqs <= center_hz + window_hz)
        return float(power[mask].sum() / total_power) if mask.any() else 0.0

    return {
        "spectral_centroid":  centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_entropy":   entropy,
        "dominant_freq_hz":   dominant_freq,
        "hf_energy_ratio":    hf_energy_ratio,
        "bpfo_band_energy":   band_energy(BPFO),
        "bpfi_band_energy":   band_energy(BPFI),
        "bsf_band_energy":    band_energy(BSF),
    }


# ── COMBINED FEATURE VECTOR ───────────────────────────────────────────────────

def extract_features(signal: np.ndarray, bearing_id: str,
                     fs: int = SAMPLING_RATE_HZ) -> dict:
    """
    Extract all 16 features from one bearing's signal.
    Returns a flat dict with bearing_id as prefix (e.g. 'b1_ch1_rms').

    Args:
        signal     : raw vibration array from one snapshot
        bearing_id : e.g. 'b1_ch1'
        fs         : sampling rate
    """
    p = bearing_id

    time_features = {
        f"{p}_rms":            compute_rms(signal),
        f"{p}_peak_to_peak":   compute_peak_to_peak(signal),
        f"{p}_kurtosis":       compute_kurtosis(signal),
        f"{p}_crest_factor":   compute_crest_factor(signal),
        f"{p}_skewness":       compute_skewness(signal),
        f"{p}_shape_factor":   compute_shape_factor(signal),
        f"{p}_impulse_factor": compute_impulse_factor(signal),
        f"{p}_margin_factor":  compute_margin_factor(signal),
    }

    freq_features = {
        f"{p}_{k}": v
        for k, v in compute_fft_features(signal, fs).items()
    }

    return {**time_features, **freq_features}


# ── FEATURE EXTRACTOR CLASS ───────────────────────────────────────────────────

class BearingFeatureExtractor:
    """
    Builds a time-indexed feature matrix from a NASA bearing test directory.

    Follows the same pattern as your build_rms_dataframe() in the notebook —
    loop over sorted files, compute features per snapshot, return a DataFrame.

    Args:
        data_path  : directory containing raw snapshot files
        n_channels : 4 for 2nd/3rd test, 8 for 1st test
        fs         : sampling rate (default 20480 Hz)

    Example:
        extractor = BearingFeatureExtractor("path/to/2nd_test")
        df = extractor.build_feature_matrix()
        df.to_csv("results/feature_matrix.csv")
        # Shape: (984 snapshots, 64 features)
    """

    def __init__(self, data_path: str, n_channels: int = 4,
                 fs: int = SAMPLING_RATE_HZ):
        self.data_path   = data_path
        self.columns     = COLUMNS_4CH if n_channels == 4 else COLUMNS_8CH
        self.bearing_ids = [c for c in self.columns if c.endswith("ch1")]
        self.fs          = fs

    def _load_snapshot(self, filepath: str) -> pd.DataFrame:
        """Load one snapshot file — same logic as your notebook."""
        return pd.read_csv(filepath, sep="\t", header=None, names=self.columns)

    def extract_snapshot(self, filepath: str) -> dict:
        """Extract features from all bearings in one snapshot file."""
        df  = self._load_snapshot(filepath)
        row = {}
        for bearing_id in self.bearing_ids:
            signal = df[bearing_id].values
            row.update(extract_features(signal, bearing_id, self.fs))
        return row

    def build_feature_matrix(self, verbose: bool = True) -> pd.DataFrame:
        """
        Process every snapshot file → one row of features each.

        Returns:
            pd.DataFrame shape (n_snapshots, 64)
            Index = snapshot filename (timestamp), same as your rms_df

        Features:
            16 features × 4 bearings = 64 columns total
        """
        files = sorted(
            f for f in os.listdir(self.data_path) if not f.startswith(".")
        )

        if not files:
            raise FileNotFoundError(f"No files found in: {self.data_path}")

        records = []
        for i, fname in enumerate(files):
            if verbose and i % 100 == 0:
                print(f"  [{i+1:4d}/{len(files)}] {fname}")
            try:
                row = self.extract_snapshot(os.path.join(self.data_path, fname))
                row["snapshot"] = fname
                records.append(row)
            except Exception as e:
                print(f"  Warning: skipped {fname} — {e}")

        df = pd.DataFrame(records).set_index("snapshot")

        if verbose:
            print(f"\n✓ Feature matrix built: "
                  f"{df.shape[0]} snapshots × {df.shape[1]} features")
            print(f"  Bearings : {self.bearing_ids}")
            print(f"  Features per bearing: {df.shape[1] // len(self.bearing_ids)}")

        return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python features.py <data_path> <output_csv> [n_channels]")
        print("  data_path  : directory with NASA bearing snapshot files")
        print("  output_csv : where to save feature_matrix.csv")
        print("  n_channels : 4 (2nd/3rd test) or 8 (1st test) — default 4")
        sys.exit(1)

    extractor = BearingFeatureExtractor(
        data_path=sys.argv[1],
        n_channels=int(sys.argv[3]) if len(sys.argv) > 3 else 4,
    )
    df = extractor.build_feature_matrix()
    df.to_csv(sys.argv[2])
    print(f"Saved → {sys.argv[2]}")
