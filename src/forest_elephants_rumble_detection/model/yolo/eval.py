"""
Evaluation of YOLO models.
"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics


def load_trained_model(weights_path: Path) -> YOLO:
    """Loads the trained `model` weights."""
    return YOLO(weights_path)


def evaluate(
    model: YOLO,
    split: str = "test",
    save_json: bool = False,
    save_hybrid: bool = False,
) -> DetMetrics:
    """Evaluates the model on the split (train, val.

    or test) and returns a DetMetrics object.
    """
    assert split in ["train", "val", "test"], "split should be in {train, val, test}"
    return model.val(
        split=split,
        save_json=save_json,
        save_hybrid=save_hybrid,
    )
