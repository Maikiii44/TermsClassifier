import os
import mlflow
from typing import List, Union, Optional
from datetime import datetime

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import MLFlowLogger, Logger
from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from terms.logs import get_logger


logger = get_logger(__name__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TermsTrainer:
    def __init__(
        self,
        pl_model: LightningModule,
        pl_datamodule: LightningDataModule,
        logger: Optional[Logger] = None,
        callbacks: Optional[Callback] = None,
        accelerator: Union[str, Accelerator] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Optional[_PRECISION_INPUT] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        max_epochs: int | None = None,
        model_dir: str = "./model_dir",
    ):
        self.pl_model = pl_model
        self.pl_datamodule = pl_datamodule
        self.pretrained_model_name = pl_model.pretrained_model_name
        self.model_dir = model_dir

        self._logger = logger or self._default_logger()
        self._callbacks = callbacks or self._default_callbacks()

        self.trainer = Trainer(
            logger=self._logger,
            callbacks=self._callbacks,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            check_val_every_n_epoch=check_val_every_n_epoch,
            log_every_n_steps=log_every_n_steps,
            num_sanity_val_steps=num_sanity_val_steps,
            max_epochs=max_epochs,
        )

    def train(self):
        """Fit the model on the training set and validate periodically.

        Returns:
            dict: The *final* metrics dictionary produced by Lightning after the
            training loop completes – typically containing the best validation
            score, elapsed time, etc.
        """

        self.set_seed()

        logger.info("Start training the model...")
        self.trainer.fit(model=self.pl_model, datamodule=self.pl_datamodule)
        logger.info("Training finished!")

        return self.trainer.logged_metrics

    def test(self):
        """Run evaluation on the test set using the *best* checkpoint.

        Returns:
            list[dict[str, Any]]: One metrics dictionary per dataloader, as produced by
            :pymeth:`~pytorch_lightning.Trainer.test`.
        """

        logger.info("Start testing the model...")
        results = self.trainer.test(datamodule=self.pl_datamodule, ckpt_path="best")
        logger.info("Testing finished!")

        return results

    @classmethod
    def set_seed(cls, seed: int = 42):
        """Seed Python, NumPy, PyTorch, and Lightning PRNGs for reproducibility.

        Args:
            seed (int): Any non-negative integer. The same seed guarantees bit-wise
                identical results given the same hardware and deterministic
                operations.
        """
        logger.info("Setting the seed value to {seed}")
        seed_everything(seed=seed)

    def _default_logger(self):
        """Instantiate an :class:`MLFlowLogger` with a timestamped run-name.

        Returns:
            MLFlowLogger: Configured MLflow logger instance.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        name_prefix = (
            self.pretrained_model_name.replace("/", "_")
            if self.pretrained_model_name
            else "model"
        )
        run_name = f"{name_prefix}_{timestamp}_run"
        tracking_uri = "file:./mlruns"
        mlflow.set_tracking_uri(tracking_uri)

        return MLFlowLogger(
            experiment_name="terms_logs",
            run_name=run_name,
            tracking_uri=tracking_uri,
        )

    def _default_callbacks(self):
        """Return early-stopping + best-checkpoint callbacks with sensible defaults.

        Returns:
            list[Callback]: List of PyTorch Lightning callbacks.
        """
        name_prefix = (
            self.pretrained_model_name.replace("/", "_")
            if self.pretrained_model_name
            else "model"
        )
        ckpt_name = f"{name_prefix}_{{epoch}}_{{val_loss:.2f}}"

        return [
            EarlyStopping(patience=6, mode="min", monitor="val_loss", min_delta=0.001),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"{self.model_dir}/checkpoints",
                filename=ckpt_name,
                mode="min",
                save_top_k=1,
            ),
        ]

    @property
    def logger(self) -> MLFlowLogger:
        """Returns the logger instance used by the trainer."""
        return self._logger

    @property
    def callbacks(self) -> List[Callback]:
        """Returns the list of callbacks used by the trainer."""
        return self._callbacks
