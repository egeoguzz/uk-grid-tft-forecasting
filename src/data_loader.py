import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import os


class UKEnergyLoader:
    def __init__(self, file_path="data/raw/demanddata_2025.csv"):
        # Resolve path relative to the current working directory
        self.file_path = os.path.join(os.getcwd(), file_path)
        self.data = None

    def process_data(self):
        """
        Loads, cleans, and performs feature engineering for the TFT model.
        Ensures time-series integrity and extracts seasonal features from timestamps.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}. Please check the 'data/raw' directory.")

        print(f"Loading data from {self.file_path}...")
        df = pd.read_csv(self.file_path)

        # 1. Standardize Date Format
        df['date'] = pd.to_datetime(df['SETTLEMENT_DATE'])

        # 2. Create Time Index (Critical step for TFT architecture)
        # Ensures a continuous integer index for the time series
        df = df.sort_values(['date', 'SETTLEMENT_PERIOD']).reset_index(drop=True)
        df['time_idx'] = df.index

        # 3. Feature Engineering
        # Extract categorical variables from datetime objects to capture seasonality
        df['month'] = df['date'].dt.month.astype(str).astype("category")
        df['day_of_week'] = df['date'].dt.dayofweek.astype(str).astype("category")
        df['hour'] = ((df['SETTLEMENT_PERIOD'] - 1) / 2).astype(int).astype(str).astype("category")

        # Log transformation (Stabilize variance and handle outliers in energy demand)
        df['log_ND'] = np.log1p(df['ND'])

        # Static Group ID (Required for handling single or multiple time series)
        df['group_id'] = 'UK'

        # Optimize data types for memory efficiency
        df['ND'] = df['ND'].astype(float)

        self.data = df
        print(f"SUCCESS: Data processed. Shape: {df.shape}")
        print("Engineered Features: Month, Day, Hour, Log_ND created.")
        return df

    def get_dataset(self, max_prediction_length=48):
        """
        Creates a PyTorch Forecasting TimeSeriesDataSet object.
        Defines the encoder/decoder lengths and variable treatments for the model.
        """
        if self.data is None:
            self.process_data()

        # TFT Dataset Definition
        training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx < x.time_idx.max() - max_prediction_length],
            time_idx="time_idx",
            target="ND",  # Target: National Demand
            group_ids=["group_id"],
            min_encoder_length=48,  # Min lookback: 24 hours
            max_encoder_length=168,  # Max lookback: 1 week (context window)
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,  # Prediction horizon: 24 hours (48 periods)

            static_categoricals=["group_id"],
            time_varying_known_categoricals=["month", "day_of_week", "hour"],
            time_varying_known_reals=["time_idx"],

            # Variables known only in the past (Demand, Wind, Solar, Interconnectors)
            time_varying_unknown_reals=[
                "ND",
                "EMBEDDED_WIND_GENERATION",
                "EMBEDDED_SOLAR_GENERATION",
                "IFA_FLOW"
            ],

            target_normalizer=GroupNormalizer(
                groups=["group_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        return training


# Execution Block for Testing
if __name__ == "__main__":
    try:
        loader = UKEnergyLoader(file_path="data/raw/demanddata_2025.csv")
        dataset = loader.get_dataset()
        print("\n--- TEST RESULT ---")
        print("TFT Dataset object created successfully!")
        print("Model ready for training.")
    except Exception as e:
        print(f"ERROR: {e}")