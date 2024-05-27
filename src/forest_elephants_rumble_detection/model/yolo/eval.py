from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics


def load_trained_model(weights_path: Path) -> YOLO:
    """Loads the trained `model` weights."""
    return YOLO(weights_path)


def evaluate(model: YOLO, split: str = "test") -> DetMetrics:
    """Evaluates the model on the split (train, val.

    or test) and returns a DetMetrics object.
    """
    assert split in ["train", "val", "test"], "split should be in {train, val, test}"
    return model.val(split=split)


# REPL

# from forest_elephants_rumble_detection.utils import yaml_write

# weights_path = Path("./data/04_models/yolov8/baseline_small_dataset/weights/best.pt")
# weights_path = Path("./data/04_models/yolov8/dumb_small_dataset/weights/best.pt")
# weights_path.exists()

# load_trained_model(weights_path)
# model = load_trained_model(weights_path)
# model.info()

# evaluate(model, split="val")
# results = evaluate(model, split="val")
# # evaluate(model, split="test")
# results

# output_dir = Path("./data/06_reporting/yolov8/dumb_small_dataset/")
# output_dir.mkdir(exist_ok=True, parents=True)

# import json
# import shutil

# with open(output_dir / "results.json", "w") as fp:
#   json.dump(results.results_dict, fp)

# yaml_write(to=output_dir / "results.yaml", data=results.results_dict)

# def write_json(to: Path, data: dict) -> None:
#     with open(to, "w") as fp:
#       json.dump(data, fp)

# write_json(to=output_dir / "results.json", data=results.results_dict)
# write_json(to=output_dir / "speed.json", data=results.speed)

# results.save_dir

# dst = output_dir / "artifacts"
# dst.rmdir()
# shutil.copytree(src=results.save_dir, dst=dst)
