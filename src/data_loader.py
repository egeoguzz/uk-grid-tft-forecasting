import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import os


class UKEnergyLoader:
    """
    Data ingestion and preprocessing module for UK Energy Demand forecasting.
    Handles data loading, feature engineering, and TimeSeriesDataSet creation.
    """

    def __init__(self, file_path="data/raw/demanddata_2025.csv"):
        self.file_path = os.path.join(os.getcwd(), file_path)
        self.data = None

    def process_data(self) -> pd.DataFrame:
        """
        Loads raw CSV data and performs feature engineering.

        Returns:
            pd.DataFrame: Processed dataframe with time index and engineered features.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        print(f"[INFO] Loading data from {self.file_path}...")
        df = pd.read_csv(self.file_path)

        # 1. Standardize Date Format
        df['date'] = pd.to_datetime(df['SETTLEMENT_DATE'])

        # 2. Time Index Creation
        # Sort by date and period to ensure correct time sequence
        df = df.sort_values(['date', 'SETTLEMENT_PERIOD']).reset_index(drop=True)
        df['time_idx'] = df.index

        # 3. Feature Engineering
        # Extract temporal features to capture seasonality
        df['month'] = df['date'].dt.month.astype(str).astype("category")
        df['day_of_week'] = df['date'].dt.dayofweek.astype(str).astype("category")
        df['hour'] = ((df['SETTLEMENT_PERIOD'] - 1) / 2).astype(int).astype(str).astype("category")

        # Log transformation to stabilize variance in demand
        df['log_ND'] = np.log1p(df['ND'])
        # Static Group ID for single-series handling
        df['group_id'] = 'UK'
        # Ensure target variable is float
        df['ND'] = df['ND'].astype(float)

        self.data = df
        print(f"[INFO] Data processed successfully. Shape: {df.shape}")
        return df

    def get_dataset(self, max_prediction_length=48) -> TimeSeriesDataSet:
        """
        Converts the pandas DataFrame into a PyTorch Forecasting TimeSeriesDataSet.

        Args:
            max_prediction_length (int): Forecast horizon (default: 48 periods / 24 hours).

        Returns:
            TimeSeriesDataSet: The configured dataset for training.
        """
        if self.data is None:
            self.process_data()

        # Define the TimeSeriesDataSet
        # We use NaNLabelEncoder to handle potentially unseen categories during validation/inference
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

            # Robust encoder configuration for production stability
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
