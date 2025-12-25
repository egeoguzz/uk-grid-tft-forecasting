import sys

from src.data_loader import UKEnergyLoader


def main():
    print(">>> Initializing UK Grid Intelligence System...")
    print("-" * 60)

    # 1. Initialize Data Ingestion
    print("[INFO] Triggering Data Ingestion Pipeline...")

    try:
        loader = UKEnergyLoader(file_path="data/raw/demanddata_2025.csv")

        # Create the TimeSeriesDataSet
        dataset = loader.get_dataset()

        print("-" * 60)
        print("SUCCESS: Data pipeline executed successfully.")
        print(f"[INFO] Dataset Shape: {loader.data.shape}")
        print(f"[INFO] Target Variable: National Demand (ND)")
        print(
            f"[INFO] Time Series Context: {dataset.max_encoder_length} hours history -> {dataset.max_prediction_length} hours prediction")
        print("-" * 60)
        print("System Ready: Proceeding to Model Training Initialization...")

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
