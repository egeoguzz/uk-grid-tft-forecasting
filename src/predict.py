import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- COMPATIBILITY PATCH ---
# Resolves 'np.float' deprecation in newer numpy versions used by pytorch-forecasting
if not hasattr(np, "float"):
    np.float = float
# ---------------------------

# --- PATH CONFIGURATION ---
# Dynamically resolve project root to ensure resources are found
# regardless of execution context (IDE vs Terminal).
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
# --------------------------

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from data_loader import UKEnergyLoader


def predict_and_plot():
    """
    Main inference pipeline:
    1. Loads the trained TFT model.
    2. Prepares the validation dataset (unseen data).
    3. Generates probabilistic forecasts.
    4. Renders a publication-ready visualization.
    """
    print(f">>> WORKING DIRECTORY SET: {os.getcwd()}")
    print(">>> INFERENCE PIPELINE INITIATED")

    # 1. Define Model Artifact Path
    model_path = os.path.join(PROJECT_ROOT, "models", "final_model.pth")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model artifact not found at: {model_path}")
        print("Please ensure training is complete.")
        return

    print(f"[INFO] Loading model weights from: {model_path}")

    try:
        # 2. Initialize Data Loader
        loader = UKEnergyLoader(data_dir="data/raw")
        dataset = loader.get_dataset()

        # 3. Initialize Model Architecture
        # NOTE: These hyperparameters must match the training configuration exactly.
        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.001,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.3,
            hidden_continuous_size=16,
            log_interval=0
        )

        # 4. Load Trained Weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("[SUCCESS] Model topology and weights loaded.")

    except Exception as e:
        print(f"[FATAL] Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Prepare Inference Data
    print("[INFO] Selecting validation samples (Hold-out set)...")
    validation_dataset = TimeSeriesDataSet.from_dataset(
        dataset, loader.data, predict=True, stop_randomization=True
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    # 6. Generate Forecasts
    print("[INFO] Computing probabilistic forecasts...")

    # STRATEGY: Manual Batch Processing
    # We fetch input (x) and prediction (y_hat) separately to ensure
    # type safety and avoid dictionary attribute errors in newer library versions.

    # A. Fetch Input Data for Visualization
    x, y = next(iter(val_dataloader))

    # B. Generate Prediction Tensor
    # 'return_x=False' ensures we get the raw output tensor, not a complex object
    y_hat = model.predict(val_dataloader, mode="raw", return_x=False)

    # 7. Visualization
    print("[INFO] Rendering publication-quality plot...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plotting Logic
    model.plot_prediction(
        x,  # Input data (History)
        y_hat,  # Model Output (Forecast)
        idx=0,  # visualize the first sample in batch
        add_loss_to_title=False,  # Disable auto-title to prevent overlap
        ax=ax,
        plot_attention=False  # Disable attention weights for a cleaner look
    )

    # Styling
    plt.title("UK National Grid Demand Forecast: AI Prediction (2024-2025 Data)",
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Time Steps (History -> Prediction Horizon)", fontsize=12)
    plt.ylabel("Electricity Demand (MW)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add clear legend
    plt.legend(["Observed (Actual)", "AI Forecast (Predicted)", "Prediction Interval"],
               loc="upper left")

    # Final Layout Adjustments
    plt.tight_layout()

    # Save Output
    output_file = "forecast_result.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    print(f"\n>>> COMPLETED. Visualization saved to: '{output_file}'")
    print(">>> Ready for GitHub submission.")


if __name__ == "__main__":
    predict_and_plot()
