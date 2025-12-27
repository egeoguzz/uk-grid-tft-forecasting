import numpy as np
import os
import sys
import warnings

# --- COMPATIBILITY  ---
# Fixes 'np.float' deprecation issue in pytorch-forecasting library
if not hasattr(np, "float"):
    np.float = float
# ---------------------------

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

    # 1. DATA INGESTION & PREPROCESSING
    print("\n[STEP 1] Executing Data Pipeline...")
    loader = UKEnergyLoader(file_path="data/raw/demanddata_2025.csv")

    # Process full dataset
    full_data = loader.process_data()

    # Define Validation Cutoff (Last 4 weeks)
    validation_cutoff = full_data["time_idx"].max() - (48 * 7 * 4)
    print(f"   > Validation Cutoff Index: {validation_cutoff}")

    # Manual Train/Validation Split (Pandas Level)
    train_df = full_data[full_data['time_idx'] < validation_cutoff]
    print(f"   > Training Samples: {len(train_df)}")

    # Assign training data to loader
    loader.data = train_df

    # Create Training Dataset
    training_dataset = loader.get_dataset()

    # Create Validation Dataset (Preserving context)
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        full_data,
        predict=True,
        stop_randomization=True
    )

    # Create DataLoaders
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
        # Execute Training
        trainer_wrapper.fit(train_loader=train_dataloader, val_loader=val_dataloader)

        # Save Artifacts
        print("\n[STEP 4] Saving Model Artifacts...")
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(tft_model.state_dict(), "models/final_model.pth")
        print("[SUCCESS] Model weights saved to 'models/final_model.pth'.")

    except Exception as exception:
        print(f"[FATAL] Training failed: {exception}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n>>> SYSTEM STATUS: OPERATION COMPLETED SUCCESSFULLY.")


if __name__ == '__main__':
    main()
