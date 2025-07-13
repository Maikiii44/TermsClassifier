from typing import Optional, List, Sequence, Tuple

import typer
import json
import torch
import pandas as pd
from typing import Any, Dict
from peft import PeftModel
from datasets import Dataset
from sklearn.model_selection import train_test_split

from terms.schemas import TermsDataModel


def get_base_model_name(peft_model: PeftModel) -> str:
    """
    Extract the base model name from a PEFT model.

    Args:
        peft_model (PeftModel): The PEFT model object from which to extract the base model name.

    Returns:
        str: The name or path of the original pretrained model.
             Returns "unknown" if the attribute is not accessible.
    """
    try:
        return peft_model.base_model.model.config._name_or_path
    except AttributeError:
        return "unknown"


def get_peft_config_dict(peft_model: PeftModel) -> Dict[str, Any]:
    """
    Retrieve the PEFT configuration (e.g., LoRA) as a dictionary.

    Args:
        peft_model (PeftModel): The PEFT model object.

    Returns:
        Dict[str, Any]: A JSON-serializable dictionary of the PEFT configuration.
                        Returns an empty dictionary if retrieval fails.
    """
    try:
        return peft_model.peft_config["default"].to_dict()
    except Exception:
        return {}


def save_dict_to_json(data: Dict[str, Any], path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (Dict[str, Any]): The data dictionary to be saved.
        path (str): The file path where the JSON file will be written.
    """
    with open(path, "w") as f:
        json.dump(data, f)


def load_json_to_dict(path: str) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.

    Args:
        path (str): The file path to read the JSON data from.

    Returns:
        Dict[str, Any]: The dictionary loaded from the JSON file.
    """
    with open(path, "r") as f:
        loaded_dict = json.load(f)

    return loaded_dict


def split_dataframe(
    df: pd.DataFrame,
    *,
    train_size: float = 0.7,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
    stratify_by: Optional[str | Sequence[str]] = None,
    random_state: int | None = 42,
    shuffle: bool = True,
    reset_index: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train/validation/test DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Full data set.
    train_size : float, default 0.7
        Fraction of rows for training.
    val_size, test_size : float or None
        Fractions for validation and test.  If **one** (or both) is
        None, the remainder after *train_size* is split equally.
    stratify_by : str | Sequence[str] | None, default None
        Column(s) whose class proportions should be preserved.
    random_state : int | None, default 42
        Seeds the underlying RNG for reproducible splits.
    shuffle : bool, default True
        Forwarded to ``sklearn.model_selection.train_test_split``.
    reset_index : bool, default True
        Give each resulting frame a fresh RangeIndex.

    Returns
    -------
    (train_df, val_df, test_df)

    Raises
    ------
    ValueError
        If requested fractions are out of (0, 1) or fail to add up to 1.
    """

    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1 (exclusive)")

    remainder = 1.0 - train_size
    if val_size is None and test_size is None:
        val_size = test_size = remainder / 2
    elif val_size is None:
        val_size = remainder - test_size
    elif test_size is None:
        test_size = remainder - val_size

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train_size + val_size + test_size must equal 1.0, got {total:.3f}"
        )

    # 1) train vs. remainder
    train_df, remainder_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[stratify_by] if stratify_by else None,
        random_state=random_state,
        shuffle=shuffle,
    )

    # 2) validation vs. test inside the remainder
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        remainder_df,
        test_size=relative_test_size,
        stratify=remainder_df[stratify_by] if stratify_by else None,
        random_state=random_state,
        shuffle=shuffle,
    )

    if reset_index:
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def resolve_device(device: str = "auto") -> str:
    device = device.lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise typer.BadParameter("CUDA was requested but is not available.")
        return "cuda"
    elif device == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise typer.BadParameter(
                "MPS (Apple GPU) was requested but is not available."
            )
        return "mps"
    elif device == "cpu":
        return "cpu"
    else:
        raise typer.BadParameter(f"Unknown device: {device}")
