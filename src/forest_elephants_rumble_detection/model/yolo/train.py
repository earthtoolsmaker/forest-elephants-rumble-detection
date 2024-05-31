from pathlib import Path

from ultralytics import YOLO


def load_pretrained_model(model_str: str) -> YOLO:
    """Loads the pretrained `model`"""
    return YOLO(model_str)


def train(
    model: YOLO,
    data_yaml_path: Path,
    params: dict,
    project: str = "data/04_models/yolov8/",
    experiment_name: str = "train",
):
    """Main function for running a train run."""
    assert data_yaml_path.exists(), f"data_yaml_path does not exist, {data_yaml_path}"
    default_params = {
        "batch": 16,
        "epochs": 10,
        "patience": 100,
        "imgsz": 640,
        "lr0": 0.01,
        "lrf": 0.01,
        "optimizer": "auto",
        # data augmentation
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        # "hsv_h": 0.015,
        # "hsv_s": 0.7,
        # "hsv_v": 0.4,
        "mixup": 0.5,
        "scale": 0.0,
        "close_mosaic": 10,
        "degrees": 0.0,
        "translate": 0.5,
        "flipud": 0.0,
        "fliplr": 0.0,
    }
    params = {**default_params, **params}
    model.train(
        project=project,
        name=experiment_name,
        data=data_yaml_path.absolute(),
        # data=data_yaml_path,
        batch=params["batch"],
        epochs=params["epochs"],
        lr0=params["lr0"],
        lrf=params["lrf"],
        optimizer=params["optimizer"],
        imgsz=params["imgsz"],
        close_mosaic=params["close_mosaic"],
        # Data Augmentation parameters
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        scale=params["scale"],
        mixup=params["mixup"],
        degrees=params["degrees"],
        flipud=params["flipud"],
        translate=params["translate"],
    )
