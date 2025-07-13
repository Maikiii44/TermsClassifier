import typer
import pandas as pd
from typing import Optional, Annotated, List
from pathlib import Path

from terms.model.data import TermsDataModule
from terms.model.metrics import get_metrics
from terms.model.train import TermsTrainer
from terms.model.module import TermsModule
from terms.config import BaseLoraConfig, BaseQuantisationConfig
from terms.schemas import TermsDataModel
from terms.preprocess import preprocess
from terms.utils import split_dataframe
from terms.logs import get_logger
from terms.constants import (
    PRETRAINED_MODEL_NAME,
    DEFAULT_TOKENIZER_CONFIG,
)

logger = get_logger(name=__name__)

app = typer.Typer(name="terms")


@app.command(name="train")
def train(
    data_filepath: Annotated[
        str, typer.Option(help="Path to the input dataset file (.parquet).")
    ],
    pretrained_model_name: Annotated[
        str, typer.Option(help="Name or path of the pretrained model to use.")
    ] = PRETRAINED_MODEL_NAME,
    lora_config_path: Annotated[
        Optional[str],
        typer.Option(help="Optional path to the LoRA configuration YAML file."),
    ] = None,
    quantisation_path: Annotated[
        Optional[str],
        typer.Option(help="Optional path to quantisation configuration or weights."),
    ] = None,
    epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = 100,
    top_k: Annotated[
        List[int],
        typer.Option(
            help="Top-k values to compute classification accuracy (e.g., --top-k 1 --top-k 3)."
        ),
    ] = [1, 3],
    train_size: Annotated[
        float,
        typer.Option(
            help="Proportion of data to use for training. Must be between 0.0 and 1.0."
        ),
    ] = 0.7,
    precision: Annotated[
        Optional[str],
        typer.Option(
            help="Precision setting for training (e.g., '16-mixed', 'bf16-mixed')."
        ),
    ] = None,
    model_dir: Annotated[
        str, typer.Option(help="Directory to save the trained model.")
    ] = None,
):
    """
    Train a trademark classification model using LoRA and optional quantisation.
    """
    if not (0.0 < train_size < 1.0):
        raise typer.BadParameter("train_size must be between 0.0 and 1.0")

    if model_dir is None:
        model_dir = f"model_{pretrained_model_name.replace("/", "_")}"

    logger.info("Loading dataset...")
    df_base = pd.read_parquet(path=data_filepath)

    logger.info("Preprocessing the data...")
    df_preprocess = preprocess(data=df_base)

    logger.info("Splitting dataset...")
    df_train, df_val, df_test = split_dataframe(
        df=df_preprocess,
        train_size=train_size,
        shuffle=True,
        stratify_by=TermsDataModel.NiceClass,
    )
    num_classes = len(df_train.NiceClass.unique())

    logger.info("Loading tokenizer and preparing datamodule...")
    pl_datamodule = TermsDataModule(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        pretrained_model_name=pretrained_model_name,
        tokenizer_kwargs=DEFAULT_TOKENIZER_CONFIG,
        pin_memory=False,
        model_dir=model_dir,
    )

    logger.info("Loading metrics...")
    metrics = get_metrics(num_classes=num_classes, top_k=top_k)

    logger.info("Loading LoRA configuration...")
    if lora_config_path is not None:
        lora_config_obj = BaseLoraConfig.load_from_yaml(path=lora_config_path)
        lora_config = lora_config_obj.to_lora_obj()
    else:
        quantisation_config_obj = BaseLoraConfig().to_lora_obj()

    logger.info("Loading quantisation config...")
    if quantisation_path is not None:
        quantisation_config_obj = BaseQuantisationConfig.load_from_yaml(
            path=quantisation_path
        )
        quantisation_config = quantisation_config_obj.to_bits_and_bytes_obj()
    else:
        quantisation_config = None

    logger.info("Initializing model...")
    pl_model = TermsModule.from_peft_config(
        pretrained_model_name=pretrained_model_name,
        num_classes=num_classes,
        metrics=metrics,
        lora_config=lora_config,
        quantization_config=quantisation_config,
    )

    logger.info("Initializing trainer...")
    trainer = TermsTrainer(
        pl_datamodule=pl_datamodule,
        pl_model=pl_model,
        max_epochs=epochs,
        model_dir=model_dir,
        precision=precision,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    trainer.test()

    raise typer.Exit(code=0)
