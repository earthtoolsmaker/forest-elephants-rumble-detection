"""Script to eval a YOLOv8 model input for the object detection task of
elephant rumbles."""

import argparse
import logging
import shutil
from pathlib import Path

from ultralytics import settings

from forest_elephants_rumble_detection.model.yolo.eval import (
    evaluate,
    load_trained_model,
)
from forest_elephants_rumble_detection.utils import yaml_read, yaml_write


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-filepath",
        help="filepath of the model weights",
        default="./data/04_models/yolov8/baseline_small_dataset/weights/best.pt",
        type=Path,
    )
    parser.add_argument(
        "--split",
        help="value in {train, val, test}",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model evaluation reports",
        default="./data/06_reporting/yolov8/baseline_small_dataset/",
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
    if not args["weights_filepath"].exists():
        logging.error("Invalid --weights-filepath filepath does not exist")
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
        logging.info(f"Loading model from: {args['weights_filepath']}")
        model = load_trained_model(args["weights_filepath"])
        model.info()
        # TODO: persist the report in the right output_dir
        results = evaluate(model, split=args["split"])
        # TODO: do something with the result
        logging.info(results)
        output_dir = args["output_dir"]
