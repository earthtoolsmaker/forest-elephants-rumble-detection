"""Script to generate spectrograms from the raw audio files and the provided
raven pro txt files containing the annotated rumbles."""

import argparse
import logging
import multiprocessing
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import forest_elephants_rumble_detection.data.features.testing as features_testing
import forest_elephants_rumble_detection.data.features.training as features_training
from forest_elephants_rumble_detection.data.audio import load_audio
from forest_elephants_rumble_detection.data.offsets import get_offsets
from forest_elephants_rumble_detection.data.spectrogram import (
    df_rumbles_to_all_spectrogram_yolov8_bboxes,
    make_spectrogram,
    make_spectrogram2,
    select_rumbles_at,
)
from forest_elephants_rumble_detection.data.yolov8 import bboxes_to_yolov8_txt_format
from forest_elephants_rumble_detection.utils import yaml_write


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-rumbles-dir",
        help="dir containing the rumbles.",
        type=Path,
        default=Path("./data/01_raw/cornell_data/Rumble/"),
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the generated spectrograms.",
        type=Path,
        default=Path("./data/02_features/rumbles/spectrograms/"),
    )
    parser.add_argument(
        "--duration",
        help="duration in seconds of the generated spectrograms.",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--freq-min",
        help="min frequency (Hz) of the generated spectrograms.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--freq-max",
        help="max frequency (Hz) of the generated spectrograms.",
        type=float,
        default=250.0,
    )
    parser.add_argument(
        "--random-seed",
        help="Random seed for generating the random offsets in audio files.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ratio-random-offsets",
        help="ratio for the random offsets to be generated - float between 0. and 1.",
        type=float,
        default=0.2,
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
    if not args["input_rumbles_dir"].exists():
        logging.error(f"invalid --input_rumbles_dir, dir does not exist")
        return False
    else:
        return True


def spectrogram_stem(audio_filepath: Path, index: int) -> str:
    """Returns the stem for the file where the spectrogram is saved."""
    return f"{audio_filepath.stem}_spectrogram_{index}"


def generate_and_save_annotated_spectogram2(
    audio_filepath: Path,
    filename: str,
    output_dir: Path,
    df: pd.DataFrame,
    offset: float,
    duration: float,
    freq_min: float,
    freq_max: float,
    dpi: int,
    n_fft: int,
    top_db: float,
    hop_length: int,
    width: int,
    height: int,
) -> None:
    """Saves a spectrogram alongside its annotation bboxes of the rumbles."""
    # Non interactive mode for matplotlib
    matplotlib.use("Agg")

    df_rumbles = select_rumbles_at(df=df, offset=offset, duration=duration)

    audio_filepaths = df_rumbles["audio_filepath"].unique()
    assert (
        len(audio_filepaths) <= 1
    ), "Assumption that only one Begin File per df_rumbles"
    audio, sr = load_audio(audio_filepath, duration=duration, offset=offset)
    fig = make_spectrogram2(
        audio=audio,
        sr=sr,
        n_fft=n_fft,
        top_db=top_db,
        fmin=freq_min,
        fmax=freq_max,
        dpi=dpi,
        hop_length=hop_length,
        width=width,
        height=height,
    )
    ax = fig.get_axes()[0]

    if len(df_rumbles) > 0:
        bboxes = df_rumbles_to_all_spectrogram_yolov8_bboxes(
            df_rumbles=df_rumbles,
            offset=offset,
            duration=duration,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        labels = bboxes_to_yolov8_txt_format(bboxes)

        if labels:
            with open(output_dir / f"{filename}.txt", "w") as f:
                f.write(labels)

    plt.savefig(output_dir / f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
    plt.close("all")

    # Clearing memory
    del fig
    del ax
    del audio


def generate_and_save_annotated_spectogram(
    audio_filepath: Path,
    filename: str,
    output_dir: Path,
    df: pd.DataFrame,
    offset: float,
    duration: float,
    freq_min: float = 0.0,
    freq_max: float = 250.0,
) -> None:
    """Saves a spectrogram alongside its annotation bboxes of the rumbles."""
    # Non interactive mode for matplotlib
    matplotlib.use("Agg")

    df_rumbles = select_rumbles_at(df=df, offset=offset, duration=duration)

    audio_filepaths = df_rumbles["audio_filepath"].unique()
    assert (
        len(audio_filepaths) <= 1
    ), "Assumption that only one Begin File per df_rumbles"
    audio, sr = load_audio(audio_filepath, duration=duration, offset=offset)
    fig = make_spectrogram(
        audio,
        sr,
        duration=duration,
        offset=offset,
        fmin=freq_min,
        fmax=freq_max,
        verbose=False,
        n_fft=2048,
    )
    ax = fig.get_axes()[0]

    if len(df_rumbles) > 0:
        bboxes = df_rumbles_to_all_spectrogram_yolov8_bboxes(
            df_rumbles=df_rumbles,
            offset=offset,
            duration=duration,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        labels = bboxes_to_yolov8_txt_format(bboxes)

        if labels:
            with open(output_dir / f"{filename}.txt", "w") as f:
                f.write(labels)

    plt.savefig(output_dir / f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
    plt.close("all")

    # Clearing memory
    del fig
    del ax
    del audio


def task_generate_spectrograms_for(params: dict):
    """Generates the spectograms for the provided audio_filepath in params.

    It is usually run inside a process to parallelize the creation of
    spectograms.
    """
    duration = params["duration"]
    output_dir = params["output_dir"]
    audio_filepath = params["audio_filepath"]
    df_prepared = params["df_prepared"]
    random_seed = params["random_seed"]
    ratio_random_offsets = params["ratio_random_offsets"]
    n_fft = params["n_fft"]
    top_db = params["top_db"]
    freq_min = params["freq_min"]
    freq_max = params["freq_max"]
    dpi = params["dpi"]
    hop_length = params["hop_length"]
    spectrogram_width = params["spectrogram_width"]
    spectrogram_height = params["spectrogram_height"]

    process_id = os.getpid()
    logging.info(f"[{process_id}] Processing audio_filepath {audio_filepath}")
    output_audio_filepath_dir = output_dir / audio_filepath.stem
    output_audio_filepath_dir.mkdir(exist_ok=True, parents=True)
    df_audio_filemane = df_prepared[df_prepared["audio_filepath"] == audio_filepath]
    offsets = get_offsets(
        df=df_audio_filemane,
        random_seed=random_seed,
        ratio_random=ratio_random_offsets,
    )
    filepath_metadata = output_audio_filepath_dir / "metadata.csv"
    logging.info(f"Saving metadata in {filepath_metadata}")
    metadata = [
        {
            "offset": offset,
            "duration": duration,
            "audio_filepath": audio_filepath,
            "filename_stem": spectrogram_stem(audio_filepath, idx),
        }
        for idx, offset in enumerate(offsets)
    ]
    df_metadata = pd.DataFrame(metadata)
    logging.info(f"[{process_id}] {df_metadata.head()}")
    df_metadata.to_csv(output_audio_filepath_dir / "metadata.csv")
    logging.info(f"[{process_id}] Number of offsets: {len(offsets)}")
    for idx, offset in enumerate(tqdm(offsets)):
        filename = spectrogram_stem(audio_filepath, idx)
        generate_and_save_annotated_spectogram2(
            audio_filepath=audio_filepath,
            filename=filename,
            output_dir=output_audio_filepath_dir,
            df=df_audio_filemane,
            offset=offset,
            duration=duration,
            freq_min=freq_min,
            freq_max=freq_max,
            dpi=dpi,
            n_fft=n_fft,
            top_db=top_db,
            hop_length=hop_length,
            width=spectrogram_width,
            height=spectrogram_height,
        )
    return audio_filepath


def build_testing_dataset_parallel(
    test_dir: Path,
    train_dir: Path,
    output_dir: Path,
    duration: float,
    n_fft: int,
    top_db: float,
    freq_min: float,
    freq_max: float,
    dpi: int,
    hop_length: int,
    spectrogram_width: int,
    spectrogram_height: int,
    random_seed: int = 0,
    ratio_random_offsets: float = 0.20,
) -> None:
    """Main entry point to generate the spectrogram from the testing data
    files."""
    logging.info("Loading and parsing txt files")
    df = features_testing.parse_all_testing_txt_files(test_dir)
    logging.info(df.info())

    logging.info("Preparing dataframe")
    df_prepared = features_testing.prepare_df(
        df,
        train_dir=train_dir,
        test_dir=test_dir,
    )

    audio_filepaths = [
        fp for fp in df_prepared["audio_filepath"].unique() if fp.exists()
    ]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        task_args = [
            {
                "audio_filepath": fp,
                "output_dir": output_dir,
                "df_prepared": df_prepared,
                "random_seed": random_seed,
                "ratio_random_offsets": ratio_random_offsets,
                "duration": duration,
                "n_fft": n_fft,
                "top_db": top_db,
                "freq_min": freq_min,
                "freq_max": freq_max,
                "dpi": dpi,
                "hop_length": hop_length,
                "spectrogram_width": spectrogram_width,
                "spectrogram_height": spectrogram_height,
            }
            for fp in audio_filepaths
        ]
        pool.map(task_generate_spectrograms_for, task_args)
        return None


def build_testing_dataset(
    test_dir: Path,
    train_dir: Path,
    output_dir: Path,
    duration: float,
    n_fft: int,
    top_db: float,
    freq_min: float,
    freq_max: float,
    dpi: int,
    hop_length: int,
    spectrogram_width: int,
    spectrogram_height: int,
    random_seed: int = 0,
    ratio_random_offsets: float = 0.20,
) -> None:
    """Main entry point to generate the spectrogram from the testing data
    files."""
    logging.info("Loading and parsing txt files")
    df = features_testing.parse_all_testing_txt_files(test_dir)
    logging.info(df.info())

    logging.info("Preparing dataframe")
    df_prepared = features_testing.prepare_df(
        df,
        train_dir=train_dir,
        test_dir=test_dir,
    )

    audio_filepaths = [
        fp for fp in df_prepared["audio_filepath"].unique() if fp.exists()
    ]

    for audio_filepath in tqdm(audio_filepaths):
        logging.info(f"audio_filepath: {audio_filepath}")
        output_audio_filepath_dir = output_dir / audio_filepath.stem
        output_audio_filepath_dir.mkdir(exist_ok=True, parents=True)
        df_audio_filemane = df_prepared[df_prepared["audio_filepath"] == audio_filepath]
        offsets = get_offsets(
            df=df_audio_filemane,
            random_seed=random_seed,
            ratio_random=ratio_random_offsets,
        )
        filepath_metadata = output_audio_filepath_dir / "metadata.csv"
        logging.info(f"Saving metadata in {filepath_metadata}")
        metadata = [
            {
                "offset": offset,
                "duration": duration,
                "audio_filepath": audio_filepath,
                "filename_stem": spectrogram_stem(audio_filepath, idx),
            }
            for idx, offset in enumerate(offsets)
        ]
        df_metadata = pd.DataFrame(metadata)
        logging.info(df_metadata.head())
        df_metadata.to_csv(output_audio_filepath_dir / "metadata.csv")
        logging.info(f"number of offsets: {len(offsets)}")
        for idx, offset in enumerate(tqdm(offsets)):
            filename = spectrogram_stem(audio_filepath, idx)
            generate_and_save_annotated_spectogram2(
                audio_filepath=audio_filepath,
                filename=filename,
                output_dir=output_audio_filepath_dir,
                df=df_audio_filemane,
                offset=offset,
                duration=duration,
                freq_min=freq_min,
                freq_max=freq_max,
                dpi=dpi,
                n_fft=n_fft,
                top_db=top_db,
                hop_length=hop_length,
                width=spectrogram_width,
                height=spectrogram_height,
            )


def build_training_dataset(
    filepath_rumble_clearings: Path,
    output_dir: Path,
    duration: float,
    n_fft: int,
    top_db: float,
    freq_min: float,
    freq_max: float,
    dpi: int,
    hop_length: int,
    spectrogram_width: int,
    spectrogram_height: int,
    random_seed: int = 0,
    ratio_random_offsets: float = 0.20,
) -> None:
    """Main entry point to generate the spectrogram from the training data
    file."""

    logging.info(f"Loading and parsing {filepath_rumble_clearings}")
    df_rumble_clearings = features_training.parse_text_file(filepath_rumble_clearings)
    logging.info(f"Preparing dataframe")
    df = features_training.prepare_df(df_rumble_clearings)
    audio_filepaths = [fp for fp in df["audio_filepath"].unique() if fp.exists()]

    for audio_filepath in tqdm(audio_filepaths):
        logging.info(f"audio_filepath: {audio_filepath}")
        output_audio_filepath_dir = output_dir / audio_filepath.stem
        output_audio_filepath_dir.mkdir(exist_ok=True, parents=True)
        df_audio_filemane = features_training.filter_audio_filename(
            df, audio_filename=audio_filepath.name
        )
        offsets = get_offsets(
            df_audio_filemane,
            random_seed=random_seed,
            ratio_random=ratio_random_offsets,
        )
        filepath_metadata = output_audio_filepath_dir / "metadata.csv"
        logging.info(f"Saving metadata in {filepath_metadata}")
        metadata = [
            {
                "offset": offset,
                "duration": duration,
                "audio_filepath": audio_filepath,
                "filename_stem": spectrogram_stem(audio_filepath, idx),
            }
            for idx, offset in enumerate(offsets)
        ]
        df_metadata = pd.DataFrame(metadata)
        logging.info(df_metadata.head())
        df_metadata.to_csv(output_audio_filepath_dir / "metadata.csv")

        logging.info(f"number of offsets: {len(offsets)}")
        for idx, offset in enumerate(tqdm(offsets)):
            filename = spectrogram_stem(audio_filepath, idx)
            generate_and_save_annotated_spectogram2(
                audio_filepath=audio_filepath,
                filename=filename,
                output_dir=output_audio_filepath_dir,
                df=df_audio_filemane,
                offset=offset,
                duration=duration,
                freq_min=freq_min,
                freq_max=freq_max,
                dpi=dpi,
                n_fft=n_fft,
                top_db=top_db,
                hop_length=hop_length,
                width=spectrogram_width,
                height=spectrogram_height,
            )


def build_training_dataset_parallel(
    filepath_rumble_clearings: Path,
    output_dir: Path,
    duration: float,
    n_fft: int,
    top_db: float,
    freq_min: float,
    freq_max: float,
    dpi: int,
    hop_length: int,
    spectrogram_width: int,
    spectrogram_height: int,
    random_seed: int = 0,
    ratio_random_offsets: float = 0.20,
) -> None:
    """Main entry point to generate the spectrogram from the training data
    file."""

    logging.info(f"Loading and parsing {filepath_rumble_clearings}")
    df_rumble_clearings = features_training.parse_text_file(filepath_rumble_clearings)
    logging.info(f"Preparing dataframe")
    df_prepared = features_training.prepare_df(df_rumble_clearings)
    audio_filepaths = [
        fp for fp in df_prepared["audio_filepath"].unique() if fp.exists()
    ]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        task_args = [
            {
                "audio_filepath": fp,
                "output_dir": output_dir,
                "df_prepared": df_prepared,
                "random_seed": random_seed,
                "ratio_random_offsets": ratio_random_offsets,
                "duration": duration,
                "n_fft": n_fft,
                "top_db": top_db,
                "freq_min": freq_min,
                "freq_max": freq_max,
                "dpi": dpi,
                "hop_length": hop_length,
                "spectrogram_width": spectrogram_width,
                "spectrogram_height": spectrogram_height,
            }
            for fp in audio_filepaths
        ]
        pool.map(task_generate_spectrograms_for, task_args)
        return None


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
        rumbles_dir = args["input_rumbles_dir"]
        test_dir = rumbles_dir / "Testing"
        train_dir = rumbles_dir / "Training"
        filepath_rumble_clearings = (
            train_dir / "Clearings/rumble_clearing_00-24hr_56days.txt"
        )

        logging.info("Persisting parameters used to generate the dataset")
        duration = args["duration"]
        freq_min = args["freq_min"]
        freq_max = args["freq_max"]
        random_seed = args["random_seed"]
        ratio_random_offsets = args["ratio_random_offsets"]

        # Default parameters used to generate the spectrograms
        dpi = 96
        n_fft = 1048 * 8
        top_db = 70
        hop_length = 512
        spectrogram_width = 640
        spectrogram_height = 640

        config_data = {
            "duration": duration,
            "freq_min": freq_min,
            "freq_max": freq_max,
            "dpi": dpi,
            "n_fft": n_fft,
            "top_db": top_db,
            "hop_length": hop_length,
            "width": spectrogram_width,
            "height": spectrogram_height,
            "random_seed": random_seed,
            "ratio_random_offsets": ratio_random_offsets,
        }

        yaml_write(
            to=output_dir / "config.yaml",
            data=config_data,
        )
        logging.info("Building the testing dataset")
        build_testing_dataset_parallel(
            test_dir=test_dir,
            train_dir=train_dir,
            output_dir=output_dir / "testing",
            duration=duration,
            n_fft=n_fft,
            top_db=top_db,
            freq_min=freq_min,
            freq_max=freq_max,
            dpi=dpi,
            hop_length=hop_length,
            spectrogram_width=spectrogram_width,
            spectrogram_height=spectrogram_height,
            random_seed=random_seed,
            ratio_random_offsets=ratio_random_offsets,
        )

        logging.info("Building the training dataset")
        build_training_dataset_parallel(
            filepath_rumble_clearings=filepath_rumble_clearings,
            output_dir=output_dir / "training",
            duration=duration,
            n_fft=n_fft,
            top_db=top_db,
            freq_min=freq_min,
            freq_max=freq_max,
            dpi=dpi,
            hop_length=hop_length,
            spectrogram_width=spectrogram_width,
            spectrogram_height=spectrogram_height,
            random_seed=random_seed,
            ratio_random_offsets=ratio_random_offsets,
        )
