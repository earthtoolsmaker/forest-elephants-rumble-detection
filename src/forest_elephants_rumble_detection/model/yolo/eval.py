from pathlib import Path

from ultralytics import YOLO


def load_trained_model(weights_path: Path) -> YOLO:
    """Loads the trained `model` weights."""
    return YOLO(weights_path)


def evaluate(model: YOLO, split: str = "test") -> None:
    assert split in ["train", "val", "test"]
    return model.val(split=split)


# REPL
# weights_path = Path("./data/04_models/yolov8/baseline_small_dataset/weights/best.pt")
# weights_path.exists()

# model = load_trained_model(weights_path)
# model.info()

# evaluate(model, split="val")
# evaluate(model, split="test")
