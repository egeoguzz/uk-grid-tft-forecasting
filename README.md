# UK National Grid Demand Forecasting (TFT)

This repository implements a deep learning approach to forecast UK electricity demand using the **Temporal Fusion Transformer (TFT)** architecture.

The project focuses on **probabilistic forecasting** and **interpretability**, addressing the challenges of grid stability in the context of renewable energy integration.

## Project Scope & Context

Traditional forecasting models often struggle with the non-linear dependencies introduced by variable renewable energy sources. This project utilizes the TFT architecture to:

1.  **Quantify Uncertainty:** Instead of single-point predictions, the model outputs prediction intervals (quantiles), allowing for better risk assessment in grid management.
2.  **Explain Decisions:** The model leverages attention mechanisms to identify which variables (e.g., wind generation vs. time of day) are driving the forecast at any given step.

## Methodology

### Data Pipeline
* **Source:** UK National Grid historical data (merged from raw settlement CSVs).
* **Preprocessing:** Time-alignment, missing value handling via interpolation, and feature engineering (cyclic time features).
* **Target:** National Demand (ND).

### Model Configuration
* **Architecture:** Temporal Fusion Transformer (Google Research).
* **Framework:** PyTorch Forecasting / PyTorch Lightning.
* **Loss Function:** Quantile Loss (predicting P10, P50, P90).

### Experiment Design Note
In this experimental setup, historical `Wind` and `Solar` generation data are treated as *known inputs* for the forecast horizon. This **ex-post analysis** approach is chosen to isolate the direct correlation between renewable generation and grid demand, assuming ideal weather forecasting conditions. In a production environment, these inputs would be replaced by Numerical Weather Prediction (NWP) feeds.

## Results

### 1. Forecast Performance
The model successfully captures daily seasonality and intra-day volatility. The shaded regions in the plot below represent the confidence intervals, providing a range of probable outcomes rather than a static guess.

![Forecast Visualization](results/forecast_result.png)

### 2. Variable Importance (Interpretability)
Using the TFT's variable selection network, we can rank the features that influence the model's output.

* **Encoder (History):** `Wind Generation` and `Solar Generation` are identified as significant historical drivers.
* **Decoder (Future):** `Hour of Day` remains the dominant predictor for short-term horizons, reflecting human behavioral patterns.

| Historical Drivers (Encoder) | Future Drivers (Decoder) |
| :---: | :---: |
| ![Encoder Importance](results/feature_importance_encoder_variables.png) | ![Decoder Importance](results/feature_importance_decoder_variables.png) |

*(Additional analysis artifacts, including attention weights and static variable importance, can be found in the `results/` directory.)*

## Installation & Usage

To reproduce the experiments locally:

```bash
# Clone the repository
git clone [https://github.com/egeoguzz/uk-grid-tft-forecasting.git](https://github.com/egeoguzz/uk-grid-tft-forecasting.git)
cd uk-grid-tft-forecasting

# Install dependencies
pip install -r requirements.txt
