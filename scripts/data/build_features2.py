"""Script to generate spectrograms from the raw audio files and the provided
raven pro txt files containing the annotated rumbles."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm

import forest_elephants_rumble_detection.data.features.testing as features_testing
from forest_elephants_rumble_detection.data.offsets import get_offsets
from forest_elephants_rumble_detection.data.rumbles import (
    df_rumbles_to_all_spectrogram_yolov8_bboxes,
    select_rumbles_at,
)
from forest_elephants_rumble_detection.data.spectrogram.torchaudio import (
    chunk,
    clip,
    waveform_to_np_image,
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
        default=Path("./data/02_features/rumbles/spectrograms_torchaudio/"),
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
        logging.error(
            f"invalid --input_rumbles_dir, dir {args['input_rumbles_dir']} does not exist"
        )
        return False
    else:
        return True


def spectrogram_stem(audio_filepath: Path, index: int) -> str:
    """Returns the stem for the file where the spectrogram is saved."""
    return f"{audio_filepath.stem}_spectrogram_{index}"


def generate_and_save_annotated_spectogram(
    waveform: torch.Tensor,
    sample_rate: int,
    duration: float,
    filename: str,
    output_dir: Path,
    df: pd.DataFrame,
    offset: float,
    freq_min: float,
    freq_max: float,
    n_fft: int,
    hop_length: int,
    width: int,
    height: int,
) -> None:
    arr = waveform_to_np_image(
        waveform=waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        freq_max=freq_max,
        width=width,
        height=height,
    )
    img = Image.fromarray(arr)
    img.save(output_dir / f"{filename}.png")
    df_rumbles = select_rumbles_at(df=df, offset=offset, duration=duration)
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


def build_testing_dataset(
    test_dir: Path,
    train_dir: Path,
    output_dir: Path,
    duration: float,
    n_fft: int,
    freq_min: float,
    freq_max: float,
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
        df_audio_filename = df_prepared[df_prepared["audio_filepath"] == audio_filepath]
        offsets = get_offsets(
            df=df_audio_filename,
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
        logging.info(f"Loading waveform {audio_filepath} signal into memory")
        waveform_full, sample_rate = torchaudio.load(audio_filepath)

        for idx, offset in enumerate(tqdm(offsets)):
            filename = spectrogram_stem(audio_filepath, idx)
            waveform = clip(
                waveform_full, offset=offset, duration=duration, sample_rate=sample_rate
            )

            generate_and_save_annotated_spectogram(
                waveform=waveform,
                sample_rate=sample_rate,
                duration=duration,
                filename=filename,
                output_dir=output_audio_filepath_dir,
                df=df_audio_filename,
                offset=offset,
                freq_min=freq_min,
                freq_max=freq_max,
                n_fft=n_fft,
                hop_length=hop_length,
                width=spectrogram_width,
                height=spectrogram_height,
            )


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
        n_fft = 4096
        hop_length = 1024
        spectrogram_width = 640
        spectrogram_height = 256

        config_data = {
            "duration": duration,
            "freq_min": freq_min,
            "freq_max": freq_max,
            "n_fft": n_fft,
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
        build_testing_dataset(
            test_dir=test_dir,
            train_dir=train_dir,
            output_dir=output_dir / "testing",
            duration=duration,
            n_fft=n_fft,
            freq_min=freq_min,
            freq_max=freq_max,
            hop_length=hop_length,
            spectrogram_width=spectrogram_width,
            spectrogram_height=spectrogram_height,
            random_seed=random_seed,
            ratio_random_offsets=ratio_random_offsets,
        )
