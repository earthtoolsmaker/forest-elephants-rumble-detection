"""Script to generate spectrograms from the raw audio files and the provided
raven pro txt files containing the annotated rumbles."""

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from forest_elephants_rumble_detection.data.spectrogram import (
    bboxes_to_yolov8_txt_format,
    df_rumbles_to_all_spectrogram_yolov8_bboxes,
    filter_audio_filename,
    get_offsets,
    load_audio,
    make_spectrogram,
    parse_text_file,
    prepare_df,
    select_rumbles_at,
)


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath-rumble-clearings",
        help="filepath pointing to the rumble clearings txt file exported from raven pro",
        type=Path,
        default=Path(
            "./data/01_raw/cornell_data/Rumble/Training/Clearings/rumble_clearing_00-24hr_56days.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the generated spectrograms.",
        type=Path,
        default=Path("./data/02_features/rumbles/training/spectrograms2"),
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
    if not args["filepath_rumble_clearings"].exists():
        logging.error(f"invalid --filepath-rumble-clearings, filepath does not exist")
        return False
    else:
        return True


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


def spectrogram_stem(audio_filepath: Path, index: int) -> str:
    """Returns the stem for the file where the spectrogram is saved."""
    return f"{audio_filepath.stem}_spectrogram_{index}"


def build_training_dataset(
    filepath_rumble_clearings: Path,
    output_dir: Path,
    duration: float,
    freq_min: float,
    freq_max: float,
    random_seed: int = 0,
    ratio_random_offsets: float = 0.20,
) -> None:
    """Main entry point to generate the spectrogram from the training data
    file."""

    logging.info(f"Loading and parsing {filepath_rumble_clearings}")
    df_rumble_clearings = parse_text_file(filepath_rumble_clearings)
    logging.info(f"Preparing dataframe")
    df = prepare_df(df_rumble_clearings)
    audio_filepaths = [fp for fp in df["audio_filepath"].unique() if fp.exists()]

    for audio_filepath in tqdm(audio_filepaths[0:1]):
        logging.info(f"audio_filepath: {audio_filepath}")
        output_audio_filepath_dir = output_dir / audio_filepath.stem
        output_audio_filepath_dir.mkdir(exist_ok=True, parents=True)
        df_audio_filemane = filter_audio_filename(
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
            generate_and_save_annotated_spectogram(
                audio_filepath=audio_filepath,
                filename=filename,
                output_dir=output_audio_filepath_dir,
                df=df_audio_filemane,
                offset=offset,
                duration=duration,
                freq_min=freq_min,
                freq_max=freq_max,
            )


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        exit(1)
    else:
        logging.info(args)
        build_training_dataset(
            filepath_rumble_clearings=args["filepath_rumble_clearings"],
            output_dir=args["output_dir"],
            duration=args["duration"],
            freq_min=args["freq_min"],
            freq_max=args["freq_max"],
            random_seed=args["random_seed"],
            ratio_random_offsets=args["ratio_random_offsets"],
        )
