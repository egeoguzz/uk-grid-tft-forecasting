import os
import sys
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure src directory is in python path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from src.data_loader import UKEnergyLoader
    from src.model import TFTModelBuilder
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def main():
    print("--- UK Grid Forecasting System: Execution Started ---")

    # PHASE 1: Data Ingestion & Processing
    print("\n[1] Initializing Data Pipeline...")
    try:
        loader = UKEnergyLoader(file_path="data/raw/demanddata_2025.csv")
        dataset = loader.get_dataset()
        print(f"Status: Data loaded successfully.")
        print(f"Input Data Shape: {loader.data.shape}")
    except Exception as e:
        print(f"Critical Error in Data Pipeline: {e}")
        sys.exit(1)

    # PHASE 2: Model Architecture Initialization
    print("\n[2] Building TFT Model Architecture...")
    try:
        # Initialize builder with the processed dataset
        builder = TFTModelBuilder(training_dataset=dataset)

        # Build the model
        model = builder.build_model()

        print("Status: Model built successfully.")
        print("Ready for training phase.")

    except Exception as e:
        print(f"Critical Error in Model Construction: {e}")
        # Debug hint for common errors
        if "device" in str(e):
            print("Hint: Check PyTorch/CUDA compatibility.")
        sys.exit(1)

    print("\n--- System Status: OK. All modules functional. ---")


if __name__ == '__main__':
    main()