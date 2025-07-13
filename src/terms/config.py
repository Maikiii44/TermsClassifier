from typing import List, Union
from pathlib import Path

import yaml
import torch
from peft import LoraConfig, TaskType
from pydantic import BaseModel, Field, field_validator
from transformers import BitsAndBytesConfig


class BaseLoraConfig(BaseModel):
    task_type: TaskType
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    # string for the “all-linear” flag OR a list of real module names
    target_modules: Union[str, List[str]] = Field(default="all-linear")

    # convert single *non-flag* strings to a 1-element list
    @field_validator("target_modules", mode="before")
    def _coerce_modules(cls, v):
        if isinstance(v, str) and v != "all-linear":
            return [v]
        return v

    def to_lora_obj(self) -> LoraConfig:
        return LoraConfig(**self.model_dump())

    @classmethod
    def load_from_yaml(cls, path: Path | str) -> "BaseLoraConfig":
        with Path(path).open(encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


class BaseQuantisationConfig(BaseModel):
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: torch.dtype

    def to_bits_and_bytes_obj(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(**self.model_dump())

    @classmethod
    def load_from_yaml(cls, path: Path | str):
        with Path(path).open(encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))

    class Config:
        arbitrary_types_allowed = True
