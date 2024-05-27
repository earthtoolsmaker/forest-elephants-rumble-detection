"""Script to generate model input (yolov8 format) from the spectrograms."""

import argparse
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Tuple

from forest_elephants_rumble_detection.utils import yaml_write


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-features",
        help="dir containing the spectrograms, usually generated with build_features.py.",
        type=Path,
        default=Path("./data/02_features/rumbles/spectrograms/"),
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model input for yolov8 object detector.",
        type=Path,
        default=Path("./data/03_model_input/yolov8/"),
    )
    parser.add_argument(
        "--random-seed",
        help="random seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ratio",
        help="ratio to sample from the original dataset",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ratio-train-val",
        help="train_val split ratio.",
        type=float,
        default=0.8,
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
    if not args["input_features"].exists():
        logging.error(f"invalid --input_features, dir does not exist")
        return False
    else:
        return True


# FIXME: split by audio recordings - the current implementation leads to train/val leakage
def train_val_split(
    X: list[Path],
    split_ratio: float = 0.8,
    random_seed: int = 0,
) -> Tuple[list[Path], list[Path]]:
    """Splits the list of spectrograms filepaths into train and val."""
    X_copy = X.copy()
    random.Random(random_seed).shuffle(X_copy)
    N = len(X)
    n_train = int(split_ratio * N)
    return X_copy[0:n_train], X_copy[n_train:]


def make_model_input(
    input_features: Path,
    output_dir: Path,
    ratio: float,
    ratio_train_val: float = 0.8,
    random_seed: int = 0,
) -> None:
    train_features = input_features / "training"
    test_features = input_features / "testing"

    train_and_val_spectrograms = sample_spectrograms(
        input_dir=train_features,
        ratio=ratio,
        random_seed=random_seed,
    )
    train_spectrograms, val_spectrograms = train_val_split(
        train_and_val_spectrograms,
        split_ratio=ratio_train_val,
        random_seed=random_seed,
    )
    test_spectrograms = sample_spectrograms(
        input_dir=test_features,
        ratio=1.0,
        random_seed=random_seed,
    )

    train_annotations = get_annotation_filepaths(train_spectrograms)
    val_annotations = get_annotation_filepaths(val_spectrograms)
    test_annotations = get_annotation_filepaths(test_spectrograms)

    logging.info(f"Scaffolding {output_dir}")
    output_train_dir = output_dir / "train"
    output_val_dir = output_dir / "val"
    output_test_dir = output_dir / "test"

    output_train_images_dir = output_train_dir / "images"
    output_train_labels_dir = output_train_dir / "labels"
    output_train_images_dir.mkdir(exist_ok=True, parents=True)
    output_train_labels_dir.mkdir(exist_ok=True, parents=True)

    output_val_images_dir = output_val_dir / "images"
    output_val_labels_dir = output_val_dir / "labels"
    output_val_images_dir.mkdir(exist_ok=True, parents=True)
    output_val_labels_dir.mkdir(exist_ok=True, parents=True)

    output_test_images_dir = output_test_dir / "images"
    output_test_labels_dir = output_test_dir / "labels"
    output_test_images_dir.mkdir(exist_ok=True, parents=True)
    output_test_labels_dir.mkdir(exist_ok=True, parents=True)

    # Train
    for filepath in train_spectrograms:
        shutil.copyfile(src=filepath, dst=output_train_images_dir / filepath.name)
    for filepath in train_annotations:
        shutil.copyfile(src=filepath, dst=output_train_labels_dir / filepath.name)

    # Val
    for filepath in val_spectrograms:
        shutil.copyfile(src=filepath, dst=output_val_images_dir / filepath.name)
    for filepath in val_annotations:
        shutil.copyfile(src=filepath, dst=output_val_labels_dir / filepath.name)

    # Test
    for filepath in test_spectrograms:
        shutil.copyfile(src=filepath, dst=output_test_images_dir / filepath.name)
    for filepath in test_annotations:
        shutil.copyfile(src=filepath, dst=output_test_labels_dir / filepath.name)


def sample_spectrograms(
    input_dir: Path,
    ratio: float,
    random_seed: int = 0,
) -> list[Path]:
    """Returns a downsample selection of spectrograms based on the ratio and
    the random_seed."""
    assert 0.0 <= ratio <= 1.0, "ratio should be between 0 and 1"
    result = []
    subdirs = [input_dir / fn for fn in os.listdir(input_dir)]
    for subdir in subdirs:
        spectrograms = list(subdir.glob("*.png"))
        N = len(spectrograms)
        k = int(ratio * N)
        sample = random.Random(random_seed).sample(spectrograms, k)
        result.extend(sample)
    return result


def get_annotation_filepaths(spectrograms: list[Path]) -> list[Path]:
    """Returns all annotation filepaths associated with the spectrograms."""
    result = []
    for spectrogram_filepath in spectrograms:
        stem = spectrogram_filepath.stem
        annotation_filename = f"{stem}.txt"
        annotation_filepath = spectrogram_filepath.parent / annotation_filename
        if annotation_filepath.exists():
            result.append(annotation_filepath)
    return result


def write_data_yaml(yaml_filepath: Path) -> None:
    """Writes the data.yaml file used by the yolov8 model."""
    content = {
        "train": "./train/images",
        "val": "./val/images",
        "test": "./test/images",
        "nc": 1,
        "names": ["rumble"],
    }
    yaml_write(to=yaml_filepath, data=content)


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        output_dir = args["output_dir"]
        logging.info(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        input_features = args["input_features"]
        random_seed = args["random_seed"]
        ratio_train_val = args["ratio_train_val"]
        ratio = args["ratio"]
        yaml_write(
            to=output_dir / "config.yaml",
            data={
                **args,
                "output_dir": str(output_dir),
                "input_features": str(input_features),
            },
        )
        write_data_yaml(output_dir / "data.yaml")
        make_model_input(
            input_features=input_features,
            output_dir=output_dir,
            ratio=ratio,
            ratio_train_val=ratio_train_val,
            random_seed=random_seed,
        )
