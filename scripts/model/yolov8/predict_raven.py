"""Script to run predictions with a YOLOv8 in a Raven Pro compatible format."""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

from forest_elephants_rumble_detection.model.yolo.predict import pipeline
from forest_elephants_rumble_detection.utils import yaml_read, yaml_write


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
        "--verbose",
        help="Should it be verbose? Can take significantly longer as it will save some intermediate spectrograms and predictions",
        action='store_true',
    )
    parser.add_argument(
        "--overlap",
        help="Overlap in seconds between two subsequent spectrograms.",
        default=10.0,
        type=float,
    )
    parser.add_argument(
        "--batch-size",
        help="batch size for running inference. Higher value means running inference faster but using more CPU/GPU",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--model-weights-filepath",
        help="path to the model weights",
        default="./data/08_artifacts/model/rumbles/yolov8/weights/best.pt",
        type=Path,
    )
    parser.add_argument(
        "--model-config",
        help="path to the model weights",
        default="./data/08_artifacts/model/rumbles/yolov8/config.yaml",
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
        logging.error("Invalid --model-weights-filepath filepath does not exist")
        return False
    elif not args["model_config"].exists():
        logging.error("Invalid --model-config filepath does not exist")
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
        logging.info(f"Loading the model from weights {args['model_weights_filepath']}")
        model = YOLO(args["model_weights_filepath"])
        model.info()

        input_dir = args["input_dir_audio_filepaths"]
        output_dir = args["output_dir"]
        logging.info(input_dir)
        logging.info(output_dir)

        audio_filepaths = [fp for fp in input_dir.iterdir() if fp.is_file()]

        config = yaml_read(args["model_config"])
        logging.info(f"Loaded config {config}")

        overlap = args["overlap"]
        batch_size = args["batch_size"]

        output_dir.mkdir(parents=True, exist_ok=True)

        df_pipeline = pipeline(
            model=model,
            audio_filepaths=audio_filepaths,
            duration=config["duration"],
            overlap=overlap,
            width=config["width"],
            height=config["height"],
            freq_min=config["freq_min"],
            freq_max=config["freq_max"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            batch_size=batch_size,
            output_dir=output_dir if args["verbose"] else None,
        )

        logging.info(df_pipeline.head())
        df_pipeline.to_csv(output_dir / "results.csv")
        yaml_write(
            output_dir / "args.yaml",
            {
                "config": {**config},
                "args": {
                    "batch_size": batch_size,
                    "overlap": overlap,
                    "model_weights_filepath": str(args["model_weights_filepath"]),
                    "input_dir_audio_filepaths": [str(fp) for fp in audio_filepaths],
                },
            },
        )

        exit(0)
