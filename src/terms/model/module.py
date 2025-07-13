import os
from typing import Any, Dict, Tuple, Optional

import torch
from torch.optim import AdamW
from torchmetrics import MetricCollection
from pytorch_lightning import LightningModule

from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    BitsAndBytesConfig,
)

from terms.utils import get_base_model_name, get_peft_config_dict
from terms.logs import get_logger

LOGGER = get_logger(name=__name__)


class TermsModule(LightningModule):
    def __init__(
        self,
        model: PeftModel,
        *args,
        metrics: Optional[MetricCollection] = None,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 250,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model = model

        if metrics:
            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")

        self.hparams.update(
            model_name=get_base_model_name(self.model),
            lora_config=get_peft_config_dict(self.model),
        )
        self.save_hyperparameters(ignore=["model", "metrics"])

        LOGGER.info(
            f"Initialized {self.__class__.__name__} with model: {self.hparams['model_name']}"
        )

    @classmethod
    def from_peft_config(
        cls,
        lora_config: dict | LoraConfig,
        pretrained_model_name: str,
        num_classes: int,
        metrics: MetricCollection,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[str] = "auto",
        **kwargs: Any,
    ):
        LOGGER.info(
            f"Loading base model '{pretrained_model_name}' with {num_classes} classes."
        )

        base_model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                num_labels=num_classes,
                device_map=device,
                quantization_config=quantization_config,
            )
        )

        if isinstance(lora_config, dict):
            lora_config = LoraConfig(**lora_config)

        model = get_peft_model(model=base_model, peft_config=lora_config, **kwargs)

        LOGGER.info("Peft model created successfully from config.")

        return cls(
            model=model,
            metrics=metrics,
        )

    @classmethod
    def from_peft_adapter(
        cls,
        adapter_dir: str,
        pretrained_model_name: str,
        num_classes: int,
        device_map: str | Dict[str, int] | None = None,
        **kwargs,
    ):

        LOGGER.info(
            f"Loading base model '{pretrained_model_name}' from adapter '{adapter_dir}'."
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name,
            num_labels=num_classes,
            device_map=device_map,
        )

        model = PeftModel.from_pretrained(
            model=base_model, model_id=adapter_dir, device_map=device_map
        )

        LOGGER.info("Peft model loaded successfully from adapter.")

        return cls(model=model)

    @property
    def pretrained_model_name(self):
        try:
            return self.model.base_model.model.config._name_or_path
        except AttributeError:
            return "unknown"

    @property
    def peft_config(self) -> dict:

        try:
            return self.model.peft_config.get("default").__dict__
        except Exception:
            return {}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def _shared_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared logic for training/validation/test steps."""
        labels = batch.pop("labels")
        outputs = self(**batch, labels=labels)

        loss: torch.Tensor = outputs.loss
        logits: torch.Tensor = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        return loss, probs, preds, labels

    def _safe_compute(self, metrics: MetricCollection, stage: str) -> dict:
        """Computes metrics safely, catching and logging exceptions."""
        try:
            return metrics.compute()
        except Exception as e:
            LOGGER.warning(f"Skipping {stage} metric computation due to: {e}")
            return {}

    def on_train_start(self):
        if not self.train_metrics:
            raise RuntimeError("You started `fit()` but `metrics` was None.")

        LOGGER.info("Training started.")

    def training_step(self, batch, batch_idx):
        loss, probs, _preds, labels = self._shared_step(batch=batch)
        self.train_metrics.update(probs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, _preds, labels = self._shared_step(batch=batch)
        self.val_metrics.update(probs, labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss, probs, preds, labels = self._shared_step(batch=batch)
        self.test_metrics.update(probs, labels)
        self.log(name="test_loss", value=loss)

    def predict_step(self, batch, batch_idx):
        _, _probs, preds, _labels = self._shared_step(batch)
        return preds.detach().cpu()

    def on_train_epoch_end(self) -> None:
        self.log_dict(
            dictionary=self._safe_compute(self.train_metrics, "train"), sync_dist=True
        )
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(
            dictionary=self._safe_compute(self.val_metrics, "val"), sync_dist=True
        )
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(
            dictionary=self._safe_compute(self.test_metrics, "test"), sync_dist=True
        )
        self.test_metrics.reset()

    def configure_optimizers(self):
        # requires_grad is the flag PyTorch uses to decide whether a tensor needs
        # a gradient. Adds new “adapter” weights (tiny rank-decomposition matrices).
        # Freezes every original parameter of the backbone by setting
        # Give the optimiser only the parameters that actually need gradients – i.e. the LoRA adapters (and optionally the classification head).
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        LOGGER.info("Optimizer configured.")
        return [optimizer]

    def on_save_checkpoint(self, checkpoint: dict):
        """Save LoRA adapter weights inside the checkpoint folder."""
        adapter_dir = os.path.join(
            self.trainer.checkpoint_callback.dirpath, "peft_adapter"
        )
        self.model.save_pretrained(adapter_dir)  # <-- LoRA + classifier
        checkpoint["adapter_save_dir"] = adapter_dir
