import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- COMPATIBILITY PATCH ---
if not hasattr(np, "float"):
    np.float = float
# ---------------------------

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
# --------------------------

from pytorch_forecasting import TemporalFusionTransformer
from data_loader import UKEnergyLoader


def analyze_feature_importance():
    """
    Explainable AI (XAI) Module:
    Loads the trained model and calculates 'Feature Importance'.
    Saves multiple plots (Encoder, Decoder, Attention) to visualize model decisions.
    """
    print(f">>> XAI INTERPRETABILITY MODULE INITIATED")
    print(f">>> Working Directory: {os.getcwd()}")

    # 1. Define Model Artifact Path
    model_path = os.path.join(PROJECT_ROOT, "models", "final_model.pth")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model artifact not found at: {model_path}")
        return

    try:
        # 2. Load Dataset
        print("[INFO] Loading dataset to establish feature mappings...")
        loader = UKEnergyLoader(data_dir="data/raw")
        dataset = loader.get_dataset()

        # 3. Initialize Model Architecture
        print("[INFO] Initializing model architecture...")
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
        model.eval()
        print("[SUCCESS] Model topology and weights loaded.")

        # 5. Prepare Analysis Data
        val_dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        # 6. Perform Interpretation (DIRECT FORWARD PASS)
        print("[INFO] Calculating variable importance weights...")

        # Fetch a single batch manually
        x, y = next(iter(val_dataloader))

        # Direct model call to get raw output
        with torch.no_grad():
            raw_output = model(x)

        # Calculate interpretation metrics
        interpretation = model.interpret_output(raw_output, reduction="sum")

        # 7. Visualization (UPDATED TO HANDLE DICTIONARY OUTPUT)
        print("[INFO] Generating interpretability plots...")

        # This returns a DICTIONARY of figures (e.g., {'attention': fig1, 'static_variables': fig2...})
        figures = model.plot_interpretation(interpretation)

        # Loop through the dictionary and save each figure separately
        for name, fig in figures.items():
            output_file = f"feature_importance_{name}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   > [SAVED] {output_file}")

        print(f"\n>>> ANALYSIS COMPLETE. Multiple charts saved.")
        print(">>> Check the 'feature_importance_*.png' files in your folder.")

    except Exception as e:
        print(f"[FATAL] An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_feature_importance()