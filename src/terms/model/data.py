import os

from pathlib import Path
from typing import Optional, Dict, Any
import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, DataCollatorWithPadding
from datasets import Dataset

from terms.schemas import TermsDataModel
from terms.utils import save_dict_to_json
from terms.constants import COL_LABELS, DEFAULT_TOKENIZER_CONFIG

from terms.logs import get_logger

LOGGER = get_logger(name=__name__)


class TermsDataModule(LightningDataModule):
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        pretrained_model_name: str,
        *args,
        batch_size: int = 32,
        num_workers: int = os.cpu_count() or 0,  # Understand CPU / GPU
        persistent_workers: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        set_pad_token: bool = True,
        pad_token_value: Optional[str] = None,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        pin_memory_device: Optional[str] = None,
        model_dir: str = "./model_dir",
    ):

        super().__init__()
        LOGGER.info("Initialising TermsDataModule …")
        LOGGER.info(
            f"Batch size: {batch_size}, workers: {num_workers}, persistent: {persistent_workers}"
        )
        LOGGER.info(f"Tokenizer kwargs: {tokenizer_kwargs}")

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self._pretrained_model = pretrained_model_name

        self._label2id = None
        self._id2label = None

        # Tokeniser setup – created on every process.
        self._tokenizer = self._build_tokenizer(
            model_name=pretrained_model_name,
            set_pad_token=set_pad_token,
            pad_token_value=pad_token_value,
        )

        # Hugging Face collator for dynamic padding.
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self._tokenizer,
            return_tensors="pt",
        )

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.pin_memory = pin_memory
        self.pin_memory_device = (
            pin_memory_device
            if pin_memory_device is not None
            else (
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else ""
            )
        )
        self.tokenizer_call_kwargs = tokenizer_kwargs or DEFAULT_TOKENIZER_CONFIG

        # Disk cache for Arrow files
        self.cache_dir = Path(model_dir, "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        save_dict_to_json(
            data=self.tokenizer_call_kwargs, path=f"./{model_dir}/tokenizer_kwargs.json"
        )

    @property
    def label2id(self):
        return self._label2id

    @property
    def id2label(self):
        return self._id2label

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The *HF* tokenizer used for all splits."""
        return self._tokenizer

    @property
    def pretrained_model_name(self) -> str:
        """Name or path of the underlying base model."""
        return self._pretrained_model

    @staticmethod
    def _build_tokenizer(
        model_name: str,
        set_pad_token: bool,
        pad_token_value: Optional[str],
    ) -> PreTrainedTokenizer:
        LOGGER.info(f"Loading tokenizer: {model_name}")
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token_id is None and set_pad_token:
            if pad_token_value:
                tokenizer.add_special_tokens({"pad_token": pad_token_value})
            else:
                tokenizer.pad_token = (
                    tokenizer.eos_token
                )  # fallback for GPT‑like models
            LOGGER.info(
                f"PAD token set to: {tokenizer.pad_token} (id={tokenizer.pad_token_id})"
            )

        return tokenizer

    def _tokenize_fn(self, batch):
        return self.tokenizer(batch[TermsDataModel.Terms], **self.tokenizer_call_kwargs)

    def prepare_data(self) -> None:

        classes = sorted(self.df_train[TermsDataModel.NiceClass].unique())
        self._label2id = {label: idx for idx, label in enumerate(classes)}
        self._id2label = {idx: label for label, idx in self._label2id.items()}

        LOGGER.info(f"Label mapping: {self.label2id}")

        for split_name, df in {
            "train": self.df_train,
            "val": self.df_val,
            "test": self.df_test,
        }.items():
            cache_path = self.cache_dir / split_name
            if cache_path.exists():
                continue

            LOGGER.info(
                f"Tokenising {split_name} split (rows={len(df)}) → {cache_path}"
            )

            df = df.copy()
            df[COL_LABELS] = df[TermsDataModel.NiceClass].map(self._label2id)
            raw_ds = Dataset.from_pandas(df=df, preserve_index=False)
            remove_cols = [col for col in raw_ds.column_names if col != COL_LABELS]
            tokenised_ds = raw_ds.map(
                function=self._tokenize_fn,
                batched=True,
                remove_columns=remove_cols,
                desc=f"Tokenising {split_name}",
            )
            tokenised_ds.save_to_disk(cache_path)

    def setup(self, stage: str | None = None) -> None:
        """Load the cached datasets (run in every process)."""

        def _load(split: str):
            ds = Dataset.load_from_disk(self.cache_dir / split)
            return ds.with_format(
                type="torch", columns=["input_ids", "attention_mask", COL_LABELS]
            )

        if stage in ("fit", None):
            self.train_ds = _load("train")
            self.val_ds = _load("val")
            LOGGER.info(
                f"Datasets loaded → train: {len(self.train_ds)}, val: {len(self.val_ds)}"
            )
        if stage in ("test", None):
            self.test_ds = _load("test")
            LOGGER.info(f"Test dataset loaded → n: {len(self.test_ds)}")

    def train_dataloader(self):  # noqa: D401
        LOGGER.info("Creating training DataLoader …")
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            pin_memory_device=self.pin_memory_device,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self.data_collator,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            shuffle=True,
        )

    def val_dataloader(self):  # noqa: D401
        LOGGER.info("Creating validation DataLoader …")
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            pin_memory_device=self.pin_memory_device,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self.data_collator,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            shuffle=False,
        )

    def test_dataloader(self):  # noqa: D401
        LOGGER.info("Creating test DataLoader …")
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            pin_memory_device=self.pin_memory_device,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self.data_collator,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            shuffle=False,
        )
