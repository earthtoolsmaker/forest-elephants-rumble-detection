"""Script to train a YOLOv8 model input for the object detection task of fire
smokes."""

import argparse
import logging
import shutil
from pathlib import Path

from ultralytics import settings

from forest_elephants_rumble_detection.model.yolo.train import (
    load_pretrained_model,
    train,
)
from forest_elephants_rumble_detection.utils import yaml_read


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="filepath to the data_yaml config file for the dataset",
        default="./data/03_model_input/yolov8/small/datasets/data.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model_artifacts",
        default="./data/04_models/yolov8/",
        type=Path,
    )
    parser.add_argument(
        "--experiment-name",
        help="experiment name",
        default="my_experiment",
        type=str,
    )
    parser.add_argument(
        "--config",
        help="Yaml configuration file to train the model on",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not args["data"].exists():
        logging.error("Invalid --data filepath does not exist")
        return False
    elif not args["config"].exists():
        logging.error("Invalid --config filepath does not exist")
        return False
    else:
        return True


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)
    else:
        logging.info(args)
        params = yaml_read(args["config"])
        logging.info(f"Parsed run params: {params}")
        model_type = params["model_type"]
        logging.info(f"Loading model: {model_type}")
        model = load_pretrained_model(model_type)
        # Cleaning the train run directory
        shutil.rmtree(args["output_dir"] / args["experiment_name"], ignore_errors=True)

        # Update ultralytics settings to log with MLFlow
        settings.update({"mlflow": True})

        train(
            model=model,
            data_yaml_path=args["data"],
            params=params,
            project=str(args["output_dir"]),
            experiment_name=args["experiment_name"],
        )
        exit(0)
