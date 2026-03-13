# Industrial AI Predictive Maintenance
**NASA Bearing Dataset | Deep Learning Anomaly Detection**

## Overview

This project builds a predictive maintenance system for rotating machinery using vibration sensor data. Using the **NASA IMS Bearing Dataset**, the project demonstrates how Industrial AI systems detect early signs of mechanical failure through signal processing, feature engineering, and anomaly detection models.

The goal is to build an **end-to-end machine learning workflow** similar to real industrial monitoring systems used in energy, manufacturing, and heavy equipment industries.

---

## Project Objectives

- Analyze vibration signals from rotating bearings
- Extract statistical features from time-series data
- Detect abnormal machine behavior using machine learning
- Visualize degradation patterns over time

---

## Dataset

**NASA IMS Bearing Dataset**

This dataset contains vibration measurements collected from bearings running until failure under controlled conditions.

### Dataset Characteristics

- High-frequency vibration signals
- Multiple bearings monitored simultaneously
- Progressive degradation leading to failure

---

## Repository Structure

```text
industrial-ai-scada
│
├── notebooks
│   ├── nasa_bearing_eda.ipynb
│   └── nasa_bearing_eda.py
│
├── src
│   ├── features.py
│   ├── cv_anomaly_detector.py
│   └── __init__.py
│
├── results
│   ├── rms_over_time.png
│   ├── rms_individual_bearings.png
│   └── first_vs_last.png
│
└── README.md
```

---

## Exploratory Data Analysis

The **EDA stage** investigates vibration signal behavior over time.

### Key analyses performed

- RMS vibration trends over time
- Comparison across multiple bearings
- Distribution differences between early and late stage signals

### Visualizations generated

- RMS vibration over time
- Individual bearing RMS comparisons
- Early vs late signal distributions

All plots are stored in the **`results/`** directory.

---

## Feature Engineering

Feature extraction is implemented in:

```python
src/features.py
```

### Extracted Features

Statistical features derived from vibration signals:

- Mean
- Standard deviation
- Root Mean Square (RMS)
- Peak values

These features summarize vibration behavior and help machine learning models detect abnormal conditions.

---

## Anomaly Detection

Implemented in:

```python
src/cv_anomaly_detector.py
```

The project uses a **CNN autoencoder-based anomaly detection model**.

### Model Workflow

1. Train the model on normal vibration patterns
2. Reconstruct signals using the autoencoder
3. Compute reconstruction error
4. Detect anomalies when reconstruction error exceeds a threshold

This approach is commonly used in **industrial predictive maintenance systems**.

---

## Results

Key outputs include vibration analysis plots that show machine degradation behavior.

### Example Insights

- **RMS trend over time** showing increasing vibration amplitude as failure approaches
- **Bearing comparison plots** highlighting abnormal behavior
- **Signal distribution comparison** between early and late stages

These visualizations confirm that vibration characteristics change as mechanical failure approaches.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/industrial-ai-scada.git
```

Navigate to the project directory:

```bash
cd industrial-ai-scada
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Typical Dependencies

```text
numpy
pandas
matplotlib
scikit-learn
scipy
tensorflow or pytorch
```

---

## Running the Project

### Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/nasa_bearing_eda.ipynb
```

### Run Feature Engineering

```bash
python src/features.py
```

### Run Anomaly Detection

```bash
python src/cv_anomaly_detector.py
```

---

## Applications

This system demonstrates techniques used in real industrial AI systems such as:

- Predictive maintenance for rotating machinery
- Fault detection in motors and turbines
- Industrial IoT monitoring systems
- Smart manufacturing analytics

### Industries Using These Systems

- Energy
- Manufacturing
- Oil & Gas
- Aerospace
- Utilities

---

## Future Improvements

Planned improvements include:

- Time-series cross-validation
- Remaining Useful Life (RUL) prediction using LSTM
- Model explainability using SHAP
- Real-time anomaly detection pipeline
- Drift detection and monitoring

---

## Author

**Alok**

Graduate student focusing on **Industrial AI, Machine Learning, and Energy Systems**.
