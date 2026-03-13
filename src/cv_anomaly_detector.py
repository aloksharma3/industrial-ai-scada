"""
cv_anomaly_detector.py — CNN Autoencoder for Visual Anomaly Detection
| Industrial AI Predictive Maintenance

Implements the same approach ABB published in 2025:
  train a CNN autoencoder on NORMAL spectrogram images → healthy bearing
  patterns are reconstructed well → faulty patterns produce high MSE → anomaly.

Architecture (encoder → bottleneck → decoder):
  Input        64×64×3
  Conv2D 32    32×32×32   (stride 2)
  Conv2D 64    16×16×64   (stride 2)
  Conv2D 128    8× 8×128  (stride 2)
  Flatten      8192
  Dense 256    bottleneck (latent representation)
  Dense 8192   reconstruct
  Reshape       8× 8×128
  ConvT 64     16×16×64
  ConvT 32     32×32×32
  ConvT 3      64×64×3    sigmoid output

Anomaly score = per-image MSE between input and reconstruction.
Threshold = 99th percentile of training set reconstruction errors.

Pipeline position:
    signal_to_image.py → [images/normal/, images/all/]
                       → cv_anomaly_detector.py
                       → [cv_anomaly_scores.csv]

Usage:
    from src.cv_anomaly_detector import CVAnomalyDetector
    detector = CVAnomalyDetector()
    detector.fit("data/images/normal")
    scores = detector.score_dataset("data/images/all")
    scores.to_csv("results/cv_anomaly_scores.csv")
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── TensorFlow / Keras import (graceful fallback) ─────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Install with: pip install tensorflow")


IMAGE_SIZE      = (64, 64, 3)
LATENT_DIM      = 256
BATCH_SIZE      = 32
EPOCHS_DEFAULT  = 50
LEARNING_RATE   = 1e-3
THRESHOLD_PCT   = 99      # anomaly threshold: 99th pct of train reconstruction errors
RANDOM_SEED     = 42


# ── Model Builder ─────────────────────────────────────────────────────────────

def build_cnn_autoencoder(input_shape: tuple = IMAGE_SIZE,
                           latent_dim: int = LATENT_DIM) -> "Model":
    """
    Build CNN autoencoder.

    Encoder compresses 64×64×3 → latent_dim vector.
    Decoder mirrors encoder to reconstruct 64×64×3.
    Trained with MSE loss on normal bearing spectrograms.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required. pip install tensorflow")

    # ── Encoder ──────────────────────────────────────────────────────────────
    inp = keras.Input(shape=input_shape, name="input")

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu",
                      name="enc_conv1")(inp)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu",
                      name="enc_conv2")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu",
                      name="enc_conv3")(x)
    x = layers.BatchNormalization()(x)

    shape_before_flatten = x.shape[1:]   # (8, 8, 128)
    x = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, name="bottleneck")(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = layers.Dense(np.prod(shape_before_flatten), activation="relu")(encoded)
    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu",
                               name="dec_conv1")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu",
                               name="dec_conv2")(x)
    x = layers.BatchNormalization()(x)

    decoded = layers.Conv2DTranspose(input_shape[-1], 3, strides=2, padding="same",
                                     activation="sigmoid", name="output")(x)

    autoencoder = Model(inp, decoded, name="CNN_Autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
    )
    return autoencoder


# ── CV Anomaly Detector ───────────────────────────────────────────────────────

class CVAnomalyDetector:
    """
    CNN Autoencoder-based anomaly detector for bearing spectrogram images.

    Trains exclusively on healthy bearing images. At inference, high
    reconstruction error = pattern the model hasn't seen before = anomaly.

    Args:
        latent_dim  : size of bottleneck layer (default 256)
        epochs      : training epochs (default 50, use 20 for quick test)
        batch_size  : training batch size
    """

    def __init__(self, latent_dim: int = LATENT_DIM, epochs: int = EPOCHS_DEFAULT,
                 batch_size: int = BATCH_SIZE):
        self.latent_dim  = latent_dim
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.model_      = None
        self.threshold_  = None
        self.history_    = None
        self.is_fitted_  = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, normal_image_dir: str,
            validation_split: float = 0.1) -> "CVAnomalyDetector":
        """
        Train autoencoder on normal (healthy) bearing images.

        Args:
            normal_image_dir  : directory of .npy images (from signal_to_image.py)
            validation_split  : fraction held out for validation loss curve
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required.")

        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        print(f"Loading normal images from: {normal_image_dir}")
        X = self._load_images(normal_image_dir)
        print(f"  Loaded {X.shape[0]} images, shape {X.shape[1:]}")

        self.model_ = build_cnn_autoencoder(
            input_shape=X.shape[1:], latent_dim=self.latent_dim)
        self.model_.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ]

        print(f"\nTraining for up to {self.epochs} epochs "
              f"(early stopping enabled)...")
        self.history_ = self.model_.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        # Threshold: 99th percentile of training reconstruction errors
        recon  = self.model_.predict(X, verbose=0)
        errors = np.mean((X - recon) ** 2, axis=(1, 2, 3))
        self.threshold_ = float(np.percentile(errors, THRESHOLD_PCT))
        self.is_fitted_ = True

        print(f"\n✓ Training complete.")
        print(f"  Final val_loss : {self.history_.history['val_loss'][-1]:.6f}")
        print(f"  Anomaly threshold (99th pct): {self.threshold_:.6f}")

        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_image(self, img: np.ndarray) -> float:
        """Reconstruction MSE for a single image. Returns anomaly score."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        x = img[np.newaxis, ...]
        recon = self.model_.predict(x, verbose=0)
        return float(np.mean((x - recon) ** 2))

    def score_dataset(self, image_dir: str,
                      batch_size: int = 128) -> pd.DataFrame:
        """
        Score all images in a directory.

        Returns:
            DataFrame with columns: filename, bearing_id, snapshot,
                                    cv_score, cv_flag
        Sorted by snapshot order (time sequence).
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")

        paths  = sorted(Path(image_dir).glob("*.npy"))
        names  = [p.stem for p in paths]
        X      = np.stack([np.load(str(p)) for p in paths], axis=0)

        print(f"Scoring {len(X)} images ...")
        recon  = self.model_.predict(X, batch_size=batch_size, verbose=1)
        errors = np.mean((X - recon) ** 2, axis=(1, 2, 3))
        flags  = (errors > self.threshold_).astype(int)

        # Parse snapshot + bearing_id from filename (e.g. "2004.02.12.10.32.39_b1_ch1")
        records = []
        for name, score, flag in zip(names, errors, flags):
            parts  = name.rsplit("_", 2)
            snap   = parts[0] if len(parts) >= 2 else name
            bid    = "_".join(parts[1:]) if len(parts) >= 2 else "unknown"
            records.append({
                "filename":   name,
                "snapshot":   snap,
                "bearing_id": bid,
                "cv_score":   float(score),
                "cv_flag":    int(flag),
            })

        df = pd.DataFrame(records)
        n_flagged = df["cv_flag"].sum()
        print(f"\n✓ {n_flagged} / {len(df)} images flagged as anomalous "
              f"({100*n_flagged/len(df):.1f}%)")
        return df

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_training(self, output_path: str = None) -> None:
        """Plot train/validation loss curves."""
        if self.history_ is None:
            print("No training history. Call fit() first.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.history_.history["loss"],     label="Train loss")
        ax.plot(self.history_.history["val_loss"], label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title("CNN Autoencoder — Training Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Training curve saved → {output_path}")
        plt.close()

    def plot_scores(self, scores_df: pd.DataFrame, bearing_id: str = "b1_ch1",
                    output_path: str = None) -> None:
        """Plot CV anomaly score over time for one bearing."""
        df = scores_df[scores_df["bearing_id"] == bearing_id].copy()
        df = df.reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df.index, df["cv_score"], color="steelblue", linewidth=0.8)
        ax.axhline(self.threshold_, color="orange", linestyle="--",
                   label=f"Threshold ({self.threshold_:.5f})")

        flagged = df[df["cv_flag"] == 1]
        if not flagged.empty:
            ax.scatter(flagged.index, flagged["cv_score"], color="red",
                       s=8, alpha=0.6, label="Anomaly flagged", zorder=5)

        ax.set_xlabel("Snapshot index")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title(f"CNN Autoencoder Anomaly Scores — {bearing_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Score plot saved → {output_path}")
        plt.close()

    def plot_reconstructions(self, image_dir: str, n_samples: int = 8,
                              output_path: str = None) -> None:
        """
        Side-by-side: original spectrogram vs reconstruction.
        Useful for visually confirming the autoencoder is working.
        """
        paths = sorted(Path(image_dir).glob("*.npy"))[:n_samples]
        X = np.stack([np.load(str(p)) for p in paths])
        R = self.model_.predict(X, verbose=0)

        fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))
        for i in range(n_samples):
            axes[0, i].imshow(X[i])
            axes[0, i].axis("off")
            if i == 0: axes[0, i].set_ylabel("Original", fontsize=8)

            axes[1, i].imshow(R[i])
            axes[1, i].axis("off")
            if i == 0: axes[1, i].set_ylabel("Reconstructed", fontsize=8)

        fig.suptitle("CNN Autoencoder: Input vs Reconstruction", fontsize=10)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Reconstruction grid saved → {output_path}")
        plt.close()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save model weights + threshold."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model_.save(os.path.join(directory, "autoencoder.keras"))
        meta = {"threshold": self.threshold_, "latent_dim": self.latent_dim}
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print(f"✓ Model saved → {directory}")

    def load(self, directory: str) -> None:
        """Load saved model weights + threshold."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required.")
        self.model_ = keras.models.load_model(
            os.path.join(directory, "autoencoder.keras"))
        with open(os.path.join(directory, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.threshold_ = meta["threshold"]
        self.latent_dim  = meta["latent_dim"]
        self.is_fitted_  = True
        print(f"✓ Model loaded from {directory} (threshold={self.threshold_:.6f})")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_images(directory: str) -> np.ndarray:
        paths  = sorted(Path(directory).glob("*.npy"))
        if not paths:
            raise FileNotFoundError(f"No .npy files in {directory}")
        return np.stack([np.load(str(p)) for p in paths], axis=0)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python cv_anomaly_detector.py <normal_dir> <all_dir> "
              "<output_dir> [epochs]")
        print("  normal_dir : path to normal/ images (from signal_to_image.py)")
        print("  all_dir    : path to all/ images")
        print("  output_dir : where to save scores, plots, model")
        print("  epochs     : training epochs (default 50)")
        sys.exit(1)

    normal_dir = sys.argv[1]
    all_dir    = sys.argv[2]
    output_dir = sys.argv[3]
    epochs     = int(sys.argv[4]) if len(sys.argv) > 4 else EPOCHS_DEFAULT

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    detector = CVAnomalyDetector(epochs=epochs)
    detector.fit(normal_dir)
    detector.plot_training(os.path.join(output_dir, "training_curve.png"))
    detector.plot_reconstructions(normal_dir,
                                  output_path=os.path.join(output_dir,
                                                           "reconstructions.png"))

    scores = detector.score_dataset(all_dir)
    scores.to_csv(os.path.join(output_dir, "cv_anomaly_scores.csv"), index=False)

    for bid in scores["bearing_id"].unique():
        detector.plot_scores(scores, bearing_id=bid,
                             output_path=os.path.join(output_dir,
                                                      f"cv_scores_{bid}.png"))

    detector.save(os.path.join(output_dir, "model"))
    print(f"\n✓ All outputs saved → {output_dir}")
