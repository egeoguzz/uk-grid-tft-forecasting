import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


class TFTModelBuilder:
    """
    Handles the initialization and configuration of the Temporal Fusion Transformer (TFT) architecture.
    References: ArXiv:1912.09363
    """

    def __init__(self, training_dataset):
        self.training_dataset = training_dataset
        self.model = None

    def build_model(self, hidden_size: int = 16, attention_heads: int = 4,
                    dropout: float = 0.1) -> TemporalFusionTransformer:
        """
        Configures the TFT model parameters based on the dataset properties.

        Args:
            hidden_size: Hidden dimension for the LSTM and attention layers.
            attention_heads: Number of attention heads for the multi-head attention block.
            dropout: Dropout rate for regularization.

        Returns:
            Initialized TemporalFusionTransformer model.
        """
        # Determine the available hardware accelerator
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Initializing TFT architecture on device: {device.upper()}")

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=0.03,
            hidden_size=hidden_size,
            attention_head_size=attention_heads,
            dropout=dropout,
            hidden_continuous_size=8,
            output_size=7,  # Quantiles: 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        self._log_model_summary()
        return self.model

    def _log_model_summary(self):
        """Internal method to log architectural details."""
        if self.model:
            params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Model Architecture Summary:")
            print(f" - Hidden Size: {self.model.hparams.hidden_size}")
            print(f" - Attention Heads: {self.model.hparams.attention_head_size}")
            print(f" - Trainable Parameters: {params:,}")
            print(f" - Loss Function: QuantileLoss")