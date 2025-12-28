import pandas as pd
import numpy as np
import glob
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import os


class UKEnergyLoader:
    """
    Data ingestion and preprocessing module for UK Energy Demand forecasting.
    Capable of merging multiple CSV files (e.g., yearly data) into a single dataset.
    """

    def __init__(self, data_dir="data/raw"):
        # We look for the directory, not a specific file
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.data = None

    def process_data(self) -> pd.DataFrame:
        """
        Loads ALL CSV files from the raw directory, merges them, and performs feature engineering.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        print(f"[INFO] Found {len(csv_files)} data files. Loading and merging...")

        # Load and concat all files
        df_list = []
        for file in csv_files:
            print(f"   > Loading: {os.path.basename(file)}")
            try:
                temp_df = pd.read_csv(file)
                df_list.append(temp_df)
            except Exception as e:
                print(f"   [WARNING] Could not read {file}: {e}")

        if not df_list:
            raise ValueError("No valid data loaded.")

        df = pd.concat(df_list, ignore_index=True)
        print(f"[INFO] Merged Raw Data Shape: {df.shape}")

        # 1. Standardize Date Format
        df['date'] = pd.to_datetime(df['SETTLEMENT_DATE'])

        # 2. Sort and Deduplicate
        # Critical: When merging files, ensure time is sorted and no overlaps exist
        df = df.sort_values(['date', 'SETTLEMENT_PERIOD']).drop_duplicates(
            subset=['date', 'SETTLEMENT_PERIOD']).reset_index(drop=True)

        # 3. Create Continuous Time Index
        df['time_idx'] = df.index

        # 4. Feature Engineering
        df['month'] = df['date'].dt.month.astype(str).astype("category")
        df['day_of_week'] = df['date'].dt.dayofweek.astype(str).astype("category")
        df['hour'] = ((df['SETTLEMENT_PERIOD'] - 1) / 2).astype(int).astype(str).astype("category")

        df['log_ND'] = np.log1p(df['ND'])
        df['group_id'] = 'UK'
        df['ND'] = df['ND'].astype(float)

        self.data = df
        print(f"[INFO] Final Processed Data Shape: {df.shape}")
        return df

    def get_dataset(self, max_prediction_length=48) -> TimeSeriesDataSet:
        if self.data is None:
            self.process_data()

        training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx < x.time_idx.max() - max_prediction_length],
            time_idx="time_idx",
            target="ND",
            group_ids=["group_id"],
            min_encoder_length=48,
            max_encoder_length=168,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,

            static_categoricals=["group_id"],
            time_varying_known_categoricals=["month", "day_of_week", "hour"],
            time_varying_known_reals=["time_idx"],

            time_varying_unknown_reals=[
                "ND",
                "EMBEDDED_WIND_GENERATION",
                "EMBEDDED_SOLAR_GENERATION",
                "IFA_FLOW"
            ],

            categorical_encoders={
                "month": NaNLabelEncoder(add_nan=True),
                "day_of_week": NaNLabelEncoder(add_nan=True),
                "hour": NaNLabelEncoder(add_nan=True),
                "group_id": NaNLabelEncoder(add_nan=True)
            },

            target_normalizer=GroupNormalizer(
                groups=["group_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        return training
