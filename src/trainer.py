import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


class ModelTrainer:
    """
    Manages the training lifecycle of the TFT model.
    Utilizes PyTorch Lightning for efficient training loops and callbacks.
    """

    def __init__(self, model):
        self.model = model
        self.trainer = None

    def build_trainer(self, max_epochs: int = 30) -> pl.Trainer:
        """
        Configures the PyTorch Lightning Trainer with robust callbacks.

        Callbacks:
            - ModelCheckpoint: Saves the best performing model.
            - EarlyStopping: Prevents overfitting by stopping when loss plateaus.
            - LearningRateMonitor: Logs learning rate schedules.
        """
        # 1. Checkpoint Callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="models/checkpoints",
            filename="tft-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min"
        )

        # 2. Early Stopping Callback
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=5,
            verbose=True,
            mode="min"
        )

        # 3. Learning Rate Monitor
        lr_logger = LearningRateMonitor()

        # Hardware Accelerator (Auto-detect)
        accelerator = "auto"
        print(f"[INFO] Initializing Trainer with Accelerator: {accelerator.upper()}")

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            gradient_clip_val=0.1,
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            limit_train_batches=1.0,
        )

        return self.trainer

    def fit(self, train_loader, val_loader):
        """
        Executes the training process.
        """
        if self.trainer is None:
            self.build_trainer()

        print("[INFO] Starting Training Loop...")
        self.trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        print("[INFO] Training Completed Successfully.")
