import numpy as np
import os
import sys
import warnings

# --- COMPATIBILITY PATCH ---
if not hasattr(np, "float"):
    np.float = float
# ---------------------------

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from src.data_loader import UKEnergyLoader
    from src.model import TFTModelBuilder
    from src.trainer import ModelTrainer
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)


def main():
    print(">>> UK Grid Intelligence System: Pipeline Initiated")

    # 1. DATA INGESTION (MULTI-YEAR SUPPORT)
    print("\n[STEP 1] Executing Data Pipeline...")
    # Pointing to the directory, not a single file
    loader = UKEnergyLoader(data_dir="data/raw")

    full_data = loader.process_data()

    # Validation Cutoff: Last 4 weeks of the ENTIRE dataset
    validation_cutoff = full_data["time_idx"].max() - (48 * 7 * 4)
    print(f"   > Validation Cutoff Index: {validation_cutoff}")

    train_df = full_data[full_data['time_idx'] < validation_cutoff]
    print(f"   > Training Samples: {len(train_df)}")

    loader.data = train_df

    training_dataset = loader.get_dataset()

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        full_data,
        predict=True,
        stop_randomization=True
    )

    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    print(f"[SUCCESS] Data split complete. Train batches: {len(train_dataloader)}")

    # 2. MODEL ARCHITECTURE
    print("\n[STEP 2] Building TFT Architecture...")
    builder = TFTModelBuilder(training_dataset=training_dataset)
    tft_model = builder.build_model()

    # 3. TRAINING LOOP
    print("\n[STEP 3] Starting Training Sequence...")
    trainer_wrapper = ModelTrainer(model=tft_model)

    try:
        trainer_wrapper.fit(train_loader=train_dataloader, val_loader=val_dataloader)

        print("\n[STEP 4] Saving Model Artifacts...")
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(tft_model.state_dict(), "models/final_model.pth")
        print("[SUCCESS] Model weights saved to 'models/final_model.pth'.")

    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n>>> SYSTEM STATUS: OPERATION COMPLETED SUCCESSFULLY.")


if __name__ == '__main__':
    main()
