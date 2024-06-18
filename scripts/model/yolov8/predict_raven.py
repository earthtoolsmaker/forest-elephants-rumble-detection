"""Script to run predictions with a YOLOv8 in a Raven Pro compatible format."""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir-audio-filepaths",
        help="directory containing the audio filepaths to analyze",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="directory to save the results",
        default="./data/05_model_output/yolov8/predict/",
        type=Path,
    )
    parser.add_argument(
        "--model-weights-filepath",
        help="path to the model weights",
        default="./data/08_artifacts/model/rumbles/yolov8/weights/best.pt",
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
    if not args["input_dir_audio_filepaths"].exists():
        logging.error("Invalid --input-dir-audio-filepaths dir does not exist")
        return False
    elif not args["model_weights_filepath"].exists():
        logging.error("Invalid --model_weights-filepath filepath does not exist")
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
        model = YOLO(args["model_weights_filepath"])
        model.info()

        input_dir = args["input_dir_audio_filepaths"]
        logging.info(input_dir)

        # TODO

        exit(0)
