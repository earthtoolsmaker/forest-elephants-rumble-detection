"""Script to generate model input (yolov8 format) from the spectrograms."""

import argparse
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd

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
        "--testing-features-only",
        help="Should we use the testing features only? They are better annotated for instance.",
        type=bool,
        default=False,
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


def get_metadata_df(split_features_dir: Path) -> pd.DataFrame:
    """Returns a dataframe that contains all concatenated metadata.csv file for
    a split_features_dir.

    It also adds the following columns:
    - subdir: str - name of the subdir
    - spectrogram_filepath: Path
    - annotation_filepath: Path - can be None if there is no rumbles
    """
    subdirs = os.listdir(split_features_dir)
    xs = []
    for subdir in subdirs:
        filepath_metadata = split_features_dir / subdir / "metadata.csv"
        df_metadata = pd.read_csv(filepath_metadata)
        df_metadata["subdir"] = subdir
        df_metadata["spectrogram_filepath"] = df_metadata["filename_stem"].map(
            lambda stem: filepath_metadata.parent / f"{stem}.png"
        )
        df_metadata["annotation_filepath"] = df_metadata["filename_stem"].map(
            lambda stem: (
                filepath_metadata.parent / f"{stem}.txt"
                if (filepath_metadata.parent / f"{stem}.txt").exists()
                else None
            )
        )
        xs.append(df_metadata)
    return pd.concat(xs)


def train_val_increasing_offsets_split(
    train_features_dir: Path,
    split_ratio: float = 0.8,
) -> Tuple[list[Path], list[Path]]:
    """Splits the list of spectrograms filepaths into train and val.

    Prevent data leakage by splitting by increasing offsets. It is an
    entirely deterministic function.
    """
    subdirs = os.listdir(train_features_dir)
    df_metadata = get_metadata_df(train_features_dir)
    logging.info(df_metadata.info())

    X_train, X_val = [], []
    for subdir in subdirs:
        df_subdir_metadata = df_metadata[df_metadata["subdir"] == subdir].sort_values(
            by="offset"
        )
        N = len(df_subdir_metadata)
        k = int(N * split_ratio)
        X_train.extend(df_subdir_metadata["spectrogram_filepath"][:k].tolist())
        X_val.extend(df_subdir_metadata["spectrogram_filepath"][k:].tolist())

    return (X_train, X_val)


# import random

# random_seed = 0
# rng = random.Random(random_seed)
# rng
# rng.choice([True, False])

# import os
# from pathlib import Path

# test_features_dir = Path("./data/02_features/rumbles/spectrograms/testing")
# test_features_dir.exists()
# random_seed = 0
# rng = random.Random(random_seed)
# subdirs = os.listdir(test_features_dir)
# df_metadata = get_metadata_df(test_features_dir)
# df_metadata.info()
# train_val_ratio = 0.8
# val_test_ratio = 0.5
# X_train, X_val, X_test = [], [], []

# for subdir in subdirs:
#     # Randomly sort the offsets by ascending or descending order
#     ascending = rng.choice([False, True])
#     df_subdir_metadata = df_metadata[df_metadata["subdir"] == subdir].sort_values(
#         by="offset",
#         ascending=ascending,
#     )
#     N = len(df_subdir_metadata)
#     k = int(N * train_val_ratio)
#     j = k + int((N - k) * val_test_ratio)
#     X_train.extend(df_subdir_metadata["spectrogram_filepath"][:k].tolist())
#     X_val.extend(df_subdir_metadata["spectrogram_filepath"][k:j].tolist())
#     X_test.extend(df_subdir_metadata["spectrogram_filepath"][j:].tolist())

# len(X_train), len(X_val), len(X_test)
# X_val_set = set(X_val)
# X_val_set
# X_test_set = set(X_test)
# set(X_train).intersection(set(X_val))
# set(X_train).intersection(set(X_test))
# set(X_val).intersection(set(X_test))

# not X_val_set.intersection(X_test_set)

# X_train2, X_val2, X_test2 = train_val_test_increasing_offsets_split(test_features_dir=test_features_dir)
# train_val_test_increasing_offsets_split(test_features_dir=test_features_dir)
# len(X_train2), len(X_val2), len(X_test2)


def train_val_test_increasing_offsets_split(
    testing_features_dir: Path,
    ratio_train_val: float = 0.8,
    ratio_val_test: float = 0.5,
    random_seed: int = 0,
) -> Tuple[list[Path], list[Path], list[Path]]:
    """Splits the list of spectrograms filepaths into train, val and test.

    Prevent data leakage by splitting by increasing offsets.
    """
    rng = random.Random(random_seed)
    subdirs = os.listdir(testing_features_dir)
    df_metadata = get_metadata_df(testing_features_dir)
    logging.info(df_metadata.info())

    X_train, X_val, X_test = [], [], []
    for subdir in subdirs:
        # Randomly sort the offsets by ascending or descending order
        ascending = rng.choice([False, True])
        df_subdir_metadata = df_metadata[df_metadata["subdir"] == subdir].sort_values(
            by="offset",
            ascending=ascending,
        )
        N = len(df_subdir_metadata)
        k = int(N * ratio_train_val)
        j = k + int((N - k) * ratio_val_test)
        X_train.extend(df_subdir_metadata["spectrogram_filepath"][:k].tolist())
        X_val.extend(df_subdir_metadata["spectrogram_filepath"][k:j].tolist())
        X_test.extend(df_subdir_metadata["spectrogram_filepath"][j:].tolist())

    # Checking that all the sets are disjoint
    assert not set(X_train).intersection(
        set(X_val)
    ), "X_train and X_val should be distinct"
    assert not set(X_val).intersection(
        set(X_test)
    ), "X_val and X_test should be distinct"
    assert not set(X_train).intersection(
        set(X_test)
    ), "X_train and X_test should be distinct"

    return (X_train, X_val, X_test)


def make_model_input_from_testing_features_only(
    input_features: Path,
    output_dir: Path,
    ratio: float,
    ratio_train_val: float = 0.8,
    ratio_val_test: float = 0.5,
    random_seed: int = 0,
) -> None:
    """Main entry point to organize spectrograms and annotations into a yolov8
    compatible structure and format.

    It only uses the testing folder to create the splits.
    """
    testing_features_dir = input_features / "testing"

    train_spectrograms_full, val_spectrograms_full, test_spectrograms_full = (
        train_val_test_increasing_offsets_split(
            testing_features_dir=testing_features_dir,
            ratio_train_val=ratio_train_val,
            ratio_val_test=ratio_val_test,
            random_seed=random_seed,
        )
    )

    rng = random.Random(random_seed)
    N = len(train_spectrograms_full)
    k = int(ratio * N)
    train_spectrograms = rng.sample(train_spectrograms_full, k)

    N = len(val_spectrograms_full)
    k = int(ratio * N)
    val_spectrograms = rng.sample(val_spectrograms_full, k)

    N = len(test_spectrograms_full)
    k = int(ratio * N)
    test_spectrograms = rng.sample(test_spectrograms_full, k)

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


def make_model_input(
    input_features: Path,
    output_dir: Path,
    ratio: float,
    ratio_train_val: float = 0.8,
    random_seed: int = 0,
) -> None:
    """Main entry point to organize spectrograms and annotations into a yolov8
    compatible structure and format."""
    assert 0.0 <= ratio <= 1.0, "ratio should be between 0 and 1"
    train_features = input_features / "training"
    test_features = input_features / "testing"

    train_spectrograms_full, val_spectrograms_full = train_val_increasing_offsets_split(
        train_features_dir=train_features,
        split_ratio=ratio_train_val,
    )
    N = len(train_spectrograms_full)
    k = int(ratio * N)
    train_spectrograms = random.Random(random_seed).sample(train_spectrograms_full, k)

    N = len(val_spectrograms_full)
    k = int(ratio * N)
    val_spectrograms = random.Random(random_seed).sample(val_spectrograms_full, k)

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
        if not args["testing_features_only"]:
            logging.info(f"Building model input with training and testing features")
            make_model_input(
                input_features=input_features,
                output_dir=output_dir,
                ratio=ratio,
                ratio_train_val=ratio_train_val,
                random_seed=random_seed,
            )
        else:
            logging.info(f"Building model input with only testing features")
            make_model_input_from_testing_features_only(
                input_features=input_features,
                output_dir=output_dir,
                ratio=ratio,
                ratio_train_val=ratio_train_val,
                ratio_val_test=0.5,
                random_seed=random_seed,
            )
# TODO: Run the script and generate the new model inputs folders
