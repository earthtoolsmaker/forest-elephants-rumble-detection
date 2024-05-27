"""Script to eval a YOLOv8 model input for the object detection task of
elephant rumbles."""

import argparse
import logging
import shutil
from pathlib import Path

from forest_elephants_rumble_detection.model.yolo.eval import (
    evaluate,
    load_trained_model,
)
from forest_elephants_rumble_detection.utils import write_json, yaml_read, yaml_write


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
    elif not args["split"] in ["train", "val", "test"]:
        logging.error("Invalid --split value, should be in {train, val, test}")
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
        results = evaluate(model, split=args["split"])
        logging.info(results)
        output_dir = args["output_dir"] / args["split"]
        logging.info(f"output_dir: {output_dir}")
        output_dir.mkdir(exist_ok=True, parents=True)
        write_json(to=output_dir / "results.json", data=results.results_dict)
        write_json(to=output_dir / "speed.json", data=results.speed)
        shutil.copytree(src=results.save_dir, dst=output_dir / "artifacts")
