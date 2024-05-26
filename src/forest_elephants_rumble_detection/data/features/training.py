"""Loading, Preparing and generating training data for the provided dataset."""

from pathlib import Path

import pandas as pd


def parse_text_file(path: Path) -> pd.DataFrame:
    """Returns a pandas dataframe of the parsed `path`."""
    return pd.read_csv(path, sep="\t")


def begin_path_to_audio_filepath(prefix_dir: Path, begin_path: str) -> Path:
    """Given a prefix_path and the begin_path from the provided raven data, it
    returns a UNIX readable path that can be directly used for reading
    sounds."""
    suffix_path = "/".join(begin_path.replace("\\", "/").split("/")[7:])
    return prefix_dir / suffix_path


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new dataframe that contains extra columns:
    - duration: duration in second of the rumble
    - t_start, t_end: time in second when the rumble starts and ends
    - freq_low, freq_hig: frequency in hertz of the rumbles

    Note: t_start, t_end, freq_low and freq_high make it possible to localize rumbles on the spectrograms.
    """
    df_result = df.copy()
    df_result["duration"] = df_result["End Time (s)"] - df_result["Begin Time (s)"]
    df_result["t_start"] = df_result["File Offset (s)"]
    df_result["t_end"] = df_result["t_start"] + df_result["duration"]
    df_result["freq_low"] = df_result["Low Freq (Hz)"]
    df_result["freq_high"] = df_result["High Freq (Hz)"]

    prefix_dir = Path("./data/01_raw/cornell_data/Rumble")
    df_result["audio_filepath"] = df_result["Begin Path"].map(
        lambda begin_path: begin_path_to_audio_filepath(
            prefix_dir=prefix_dir, begin_path=begin_path
        )
    )
    return df_result


def filter_audio_filename(df: pd.DataFrame, audio_filename: str) -> pd.DataFrame:
    """Keeps rumbles for the provided `audio_filename`."""
    return df[df["Begin File"] == audio_filename]
