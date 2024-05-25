# Import necessary libraries.
import argparse
import logging
import os
import random
import shutil
import time
from pathlib import Path

import IPython.display as display
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm import tqdm


def parse_text_file(path: Path) -> pd.DataFrame:
    """Returns a pandas dataframe of the parsed `path`."""
    return pd.read_csv(path, sep="\t")


def load_audio(sound_path: Path, duration: float = 10.0, offset: float = 0.0):
    """Load audio path and clip it to duration with the provided offset."""
    return librosa.load(sound_path, sr=None, duration=duration, offset=offset)


def make_spectrogram(
    audio,
    sr: float,
    duration: float,
    offset: float,
    n_fft: int = 2048,
    fmin: float = 0.0,
    fmax: float = 250.0,
    verbose: bool = False,
):

    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(audio, n_fft=n_fft)

    # Convert the amplitude to decibels
    S_DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plotting the Spectrogram
    figsize = (int(duration / 6), 1)
    fig = plt.figure(figsize=figsize)
    color_mesh = librosa.display.specshow(S_DB, sr=sr, x_axis="time", y_axis="log")
    ax = fig.get_axes()[0]
    plt.ylim(fmin, fmax)

    if verbose:
        # Display labels and axes with useful information
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectogram - duration: {duration}s - offset: {offset}s")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
    else:
        # Raw spectrogram data without labels or axis
        plt.axis("off")
        ax.set_axis_off()
        plt.margins(x=0)

    # Release allocated memory to generate the spectrogram
    del color_mesh

    return fig


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


def select_rumbles_at(df: pd.DataFrame, offset: float, duration: float) -> pd.DataFrame:
    """Filters our rows in the dataframe `df` that are outside the offset and
    duration range.

    Returns a dataframe.
    """
    return df[
        (df["t_end"] > offset) & (df["t_start"] < offset + duration)
    ].reset_index()


def filter_audio_filename(df: pd.DataFrame, audio_filename: str) -> pd.DataFrame:
    """Keeps rumbles for the provided `audio_filename`."""
    return df[df["Begin File"] == audio_filename]


def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))


def raven_data_to_spectrogram_yolov8_bbox2(
    raven_data: dict,
    offset: float,
    duration: float,
    freq_min: float = 0.0,
    freq_max: float = 250.0,
):
    """
    Returns a normalized yolov8 bbox TXT format: https://roboflow.com/formats/yolov8-pytorch-txt
    Input:
      raven_data is a row in the dataframes that are loaded via pandas.

    Output:
      dictionnary with the following keys: center_x, center_y, width, height - all these values are normalized values.
    """
    t_min, t_max = 0.0, duration
    t_start, t_end = raven_data["t_start"], raven_data["t_end"]
    freq_low, freq_high = raven_data["freq_low"], raven_data["freq_high"]
    duration = raven_data["duration"]

    x1 = clamp(t_min, t_start - offset, t_max)
    x2 = clamp(t_min, t_end - offset, t_max)

    # Make sure that center_y is properly taken from the top left corner
    y1 = clamp(freq_min, freq_high, freq_max)
    y2 = clamp(freq_min, freq_high, freq_max)

    y1 = clamp(freq_min, freq_max - freq_high, freq_max)
    y2 = clamp(freq_min, freq_max - freq_low, freq_max)

    assert 0.0 <= x1 <= t_max, "x1 should be in (0, t_max)"
    assert 0.0 <= x2 <= t_max, "x2 should be in (0, t_max)"
    assert 0.0 <= y1 <= freq_max, "y1 should be in (freq_min, freq_max)"
    assert 0.0 <= y2 <= freq_max, "y2 should be in (freq_min, freq_max)"

    center_x = (x1 + ((x2 - x1) / 2)) / (t_max - t_min)
    # Make sure that center_y is properly taken from the top left corner
    center_y = (y1 + ((y2 - y1) / 2)) / (freq_max - freq_min)
    # center_y = (y2 + ((y2 - y1) / 2)) / (freq_max - freq_min)
    width = (x2 - x1) / (t_max - t_min)
    height = (y2 - y1) / (freq_max - freq_min)

    return {
        "center_x": clamp(0.0, center_x, 1.0),
        "center_y": clamp(0.0, center_y, 1.0),
        "width": clamp(0.0, width, 1.0),
        "height": clamp(0.0, height, 1.0),
    }


def draw_yolov8_bbox(
    ax, bbox: dict, duration: float, freq_min: float = 0.0, freq_max: float = 250.0
) -> None:
    """Adds a rectangle patch onto ax, which usually contains the spectogram.

    specs: https://roboflow.com/formats/yolov8-pytorch-txt
    """
    x = (bbox["center_x"] - bbox["width"] / 2) * duration
    y = (1 - bbox["center_y"] - bbox["height"] / 2) * (freq_max - freq_min)
    width = bbox["width"] * duration
    height = bbox["height"] * (freq_max - freq_min)
    ax.add_patch(
        Rectangle(
            xy=(x, y),
            width=width,
            height=height,
            color="lime",
            fc="none",
            linewidth=1.0,
        )
    )


def df_rumbles_to_all_spectrogram_yolov8_bboxes(
    df_rumbles: pd.DataFrame,
    offset: float,
    duration: float,
    freq_min: float,
    freq_max: float,
):
    """Returns all rumbles as bboxes using the df_rumbles as source of truth
    and offset, duration for normalizing the coordinates."""
    bboxes = []
    for idx, raven_data in df_rumbles.iterrows():
        bbox = raven_data_to_spectrogram_yolov8_bbox2(
            raven_data=raven_data,
            offset=offset,
            duration=duration,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        bboxes.append(bbox)
    return bboxes


def plot_rumbles(
    df_rumbles: pd.DataFrame,
    offset: float,
    duration: float,
    freq_min: float,
    freq_max: float,
    dir_sounds: Path,
    verbose: bool = True,
) -> None:
    """Plots all rumbles contained in the df_rumbles dataframe.

    Which should have been filtered out to contain rumbles of only one
    audio file.
    """
    begin_files = df_rumbles["Begin File"].unique()
    assert len(begin_files) == 1, "Assumption that only one Begin File per df_rumbles"

    audio_filename = begin_files[0]
    audio_filepath = dir_sounds / audio_filename

    audio, sr = load_audio(audio_filepath, duration=duration, offset=offset)
    fig = make_spectrogram(
        audio,
        sr,
        duration=duration,
        offset=offset,
        fmin=freq_min,
        fmax=freq_max,
        verbose=verbose,
    )
    ax = fig.get_axes()[0]

    bboxes = df_rumbles_to_all_spectrogram_yolov8_bboxes(
        df_rumbles=df_rumbles,
        offset=offset,
        duration=duration,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    for bbox in bboxes:
        draw_yolov8_bbox(
            ax, bbox=bbox, duration=duration, freq_min=freq_min, freq_max=freq_max
        )


def get_random_offsets(n: int, df: pd.DataFrame, random_seed: int = 42) -> list[float]:
    """Returns a list of random offsets to consider for plotting.

    random_seed is used to make this function deterministic.
    """
    random.seed(random_seed)
    t_start_xs = list(df["t_start"].unique())
    t0, tmax = min(*t_start_xs), max(*t_start_xs)
    return [random.randrange(int(t0), int(tmax)) for i in range(n)]


def get_rumble_offsets(df: pd.DataFrame, epsilon: float = 1.5) -> list[float]:
    """Returns a list of rumble offsets to consider for plotting.

    Epsilon should be small compared to duration
    """
    return [max(0, e - epsilon) for e in list(df["t_start"].unique())]


def get_offsets(
    df: pd.DataFrame,
    epsilon: float = 1.5,
    ratio_random: float = 0.20,
    random_seed: int = 42,
) -> list[float]:
    """Returns a list of offsets for the audio file.

    It is a mix of rumble offsets and random offsets provided the `ratio_random`
    """
    assert 0.0 <= ratio_random <= 1.0, "ratio_random should be between 0 and 1"

    rumble_offsets = get_rumble_offsets(df=df, epsilon=epsilon)
    n = int(len(rumble_offsets) * ratio_random)
    random_offsets = get_random_offsets(n=n, df=df, random_seed=random_seed)
    return [*random_offsets, *rumble_offsets]


def bbox_to_yolov8_txt_format(bbox: dict, rumble_class: int = 0) -> str:
    return f"{rumble_class} {bbox['center_x']} {bbox['center_y']} {bbox['width']} {bbox['height']}"


def bboxes_to_yolov8_txt_format(
    bboxes: list[dict],
    rumble_class: int = 0,
) -> str | None:
    if not bboxes:
        return None
    else:
        return "\n".join(
            [
                bbox_to_yolov8_txt_format(bbox, rumble_class=rumble_class)
                for bbox in bboxes
            ]
        )
