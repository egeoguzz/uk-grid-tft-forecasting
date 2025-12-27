import torch
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


class TFTModelBuilder:
    """
    Builder class for the Temporal Fusion Transformer (TFT) architecture.
    Implements state-of-the-art time series forecasting with attention mechanisms.
    """

    def __init__(self, training_dataset):
        self.training_dataset = training_dataset
        self.model = None

    def build_model(self, hidden_size: int = 16, attention_heads: int = 4, dropout: float = 0.1):
        """
        Configures and initializes the TFT model.

        Args:
            hidden_size (int): Size of the internal neural network layers.
            attention_heads (int): Number of attention heads for multi-head attention.
            dropout (float): Dropout rate for regularization.

        Returns:
            TemporalFusionTransformer: The initialized model.
        """
        # Hardware acceleration check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Initializing TFT architecture on device: {device.upper()}")

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=0.03,
            hidden_size=hidden_size,
            attention_head_size=attention_heads,
            dropout=dropout,
            hidden_continuous_size=8,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=0,
            reduce_on_plateau_patience=4,
        )

        self._log_model_summary()
        return self.model

    def _log_model_summary(self):
        """Logs the architectural details and parameter count."""
        if self.model:
            params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("--- Model Architecture Summary ---")
            print(f" --> Hidden Size: {self.model.hparams.hidden_size}")
            print(f" --> Attention Heads: {self.model.hparams.attention_head_size}")
            print(f" --> Trainable Parameters: {params:,}")
            print(f" --> Loss Function: QuantileLoss (Probabilistic)")
