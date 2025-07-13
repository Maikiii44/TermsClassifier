from typing import Dict, Union, List
from torchmetrics import Recall, Precision, Accuracy, MetricCollection, Metric


def get_metrics(num_classes: int, top_k: Union[int, List[int]] = 1) -> MetricCollection:
    """
    Generate a MetricCollection containing accuracy, precision, and recall metrics
    for specified top-k values.

    Args:
        num_classes (int): The number of classes for the classification task.
        top_k (Union[int, List[int]]): Single value or list of values for top-k evaluation.

    Returns:
        MetricCollection: A collection of torchmetrics including accuracy, precision,
        and recall for each specified top-k.
    """

    if isinstance(top_k, int):
        top_k = [top_k]

    metrics: Dict[str, Metric] = {}

    for k in top_k:

        metrics.update(get_accuracy(num_classes=num_classes, top_k=k))
        metrics.update(get_precision(num_classes=num_classes, top_k=k))
        metrics.update(get_recall(num_classes=num_classes, top_k=k))

    return MetricCollection(metrics=metrics)


def get_accuracy(num_classes: int, top_k: int = 1) -> Dict[str, Metric]:
    """
    Generate accuracy metrics (macro, micro, weighted) for a given top-k.

    Args:
        num_classes (int): Number of classes.
        top_k (int): Top-k value for accuracy computation.

    Returns:
        Dict[str, Metric]: Dictionary of accuracy metrics keyed by name.
    """

    return {
        f"top_{top_k}_macro_accuracy": Accuracy(
            num_classes=num_classes,
            task="multiclass",
            average="macro",
            top_k=top_k,
        ),
        f"top_{top_k}_micro_accuracy": Accuracy(
            num_classes=num_classes,
            task="multiclass",
            average="micro",
            top_k=top_k,
        ),
        f"top_{top_k}_weighted_accuracy": Accuracy(
            num_classes=num_classes,
            task="multiclass",
            average="weighted",
            top_k=top_k,
        ),
    }


def get_recall(num_classes: int, top_k: int = 1) -> Dict[str, Metric]:
    """
    Generate recall metrics (macro, micro, weighted) for a given top-k.

    Args:
        num_classes (int): Number of classes.
        top_k (int): Top-k value for recall computation.

    Returns:
        Dict[str, Metric]: Dictionary of recall metrics keyed by name.
    """
    return {
        f"top_{top_k}_macro_recall": Recall(
            num_classes=num_classes,
            task="multiclass",
            average="macro",
            top_k=top_k,
        ),
        f"top_{top_k}_micro_recall": Recall(
            num_classes=num_classes,
            task="multiclass",
            average="micro",
            top_k=top_k,
        ),
        f"top_{top_k}_weighted_recall": Recall(
            num_classes=num_classes,
            task="multiclass",
            average="weighted",
            top_k=top_k,
        ),
    }


def get_precision(num_classes: int, top_k: int = 1) -> Dict[str, Metric]:
    """
    Generate precision metrics (macro, micro, weighted) for a given top-k.

    Args:
        num_classes (int): Number of classes.
        top_k (int): Top-k value for precision computation.

    Returns:
        Dict[str, Metric]: Dictionary of precision metrics keyed by name.
    """

    return {
        f"top_{top_k}_macro_precision": Precision(
            num_classes=num_classes,
            task="multiclass",
            average="macro",
            top_k=top_k,
        ),
        f"top_{top_k}_micro_precision": Precision(
            num_classes=num_classes,
            task="multiclass",
            average="micro",
            top_k=top_k,
        ),
        f"top_{top_k}_weighted_precision": Precision(
            num_classes=num_classes,
            task="multiclass",
            average="weighted",
            top_k=top_k,
        ),
    }
