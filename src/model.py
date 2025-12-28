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

    def build_model(self, hidden_size: int = 32, attention_heads: int = 4, dropout: float = 0.3):
        """
        Configures and initializes the TFT model.

        Args:
            hidden_size (int): Size of the internal neural network layers. Increased to 32 for better capacity.
            attention_heads (int): Number of attention heads.
            dropout (float): Dropout rate (0.3) to prevent overfitting and improve generalization.

        Returns:
            TemporalFusionTransformer: The initialized model.
        """
        # Hardware acceleration check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Initializing TFT architecture on device: {device.upper()}")

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            # OPTIMIZATION PARAMETERS:
            # Lower learning rate (0.001) ensures smoother convergence and prevents
            # the model from getting stuck in local minima too early.
            learning_rate=0.001,

            hidden_size=hidden_size,
            attention_head_size=attention_heads,
            dropout=dropout,
            hidden_continuous_size=16,
            output_size=7,  # Predicting 7 quantiles for uncertainty estimation
            loss=QuantileLoss(),

            # Disable internal logging to avoid 'add_figure' compatibility issues
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
            print(f" > Hidden Size: {self.model.hparams.hidden_size}")
            print(f" > Attention Heads: {self.model.hparams.attention_head_size}")
            print(f" > Learning Rate: {self.model.hparams.learning_rate}")
            print(f" > Dropout Rate: {self.model.hparams.dropout}")
            print(f" > Trainable Parameters: {params:,}")
            print(f" > Loss Function: QuantileLoss (Probabilistic)")
