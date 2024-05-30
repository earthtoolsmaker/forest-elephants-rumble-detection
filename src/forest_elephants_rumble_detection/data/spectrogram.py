"""Create and save spectrograms from raw audio inputs."""

from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from forest_elephants_rumble_detection.data.audio import load_audio

from .math import clamp
from .yolov8 import bboxes_to_yolov8_txt_format


def make_spectrogram(
    audio,
    sr: float,
    duration: float,
    offset: float,
    n_fft: int = 2048,
    fmin: float = 0.0,
    fmax: float = 250.0,
    verbose: bool = False,
) -> Figure:
    """Makes a spectrogram figure that using the provided parameters.

    With verbose=True, it adds additional labels and axis onto the
    figure.
    """

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

    # Release allocated memory to generate the spectrogram - if not released, a memory leak occurs.
    del color_mesh

    return fig


def make_spectrogram2(
    audio,
    sr: float,
    n_fft: float,
    top_db: float,
    fmin: float,
    fmax: float,
    dpi: int,
    hop_length: int,
    width: int,
    height: int,
):

    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Convert the amplitude to decibels
    S_DB = librosa.amplitude_to_db(np.abs(D), ref=np.max, top_db=top_db)

    # Matplotlib specific: by default it counts some inches for the padding -
    # we subtract it to match our target width and height
    pad_inches = 0.31
    fig = plt.figure(
        figsize=(width / dpi + pad_inches, height / dpi + pad_inches),
        dpi=dpi,
    )
    fig.tight_layout()
    color_mesh = librosa.display.specshow(
        S_DB,
        sr=sr,
        x_axis="time",
        y_axis="log",
        n_fft=n_fft,
    )
    ax = fig.get_axes()[0]
    # Raw spectrogram data without labels or axis
    plt.ylim(fmin, fmax)
    plt.axis("off")
    ax.set_axis_off()
    plt.margins(x=0)

    # Prevents memory leakage
    del color_mesh

    return fig


def select_rumbles_at(df: pd.DataFrame, offset: float, duration: float) -> pd.DataFrame:
    """Filters our rows in the dataframe `df` that are outside the offset and
    duration range.

    Returns a dataframe.
    """
    return df[
        (df["t_end"] > offset) & (df["t_start"] < offset + duration)
    ].reset_index()


def raven_data_to_spectrogram_yolov8_bbox(
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
    center_y = (y1 + ((y2 - y1) / 2)) / (freq_max - freq_min)
    width = (x2 - x1) / (t_max - t_min)
    height = (y2 - y1) / (freq_max - freq_min)

    return {
        "center_x": clamp(0.0, center_x, 1.0),
        "center_y": clamp(0.0, center_y, 1.0),
        "width": clamp(0.0, width, 1.0),
        "height": clamp(0.0, height, 1.0),
    }


def draw_yolov8_bbox(
    ax,
    bbox: dict,
    duration: float,
    freq_min: float = 0.0,
    freq_max: float = 250.0,
) -> None:
    """Adds a rectangle patch onto ax, which usually contains the spectrogram.

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
    for _idx, raven_data in df_rumbles.iterrows():
        bbox = raven_data_to_spectrogram_yolov8_bbox(
            raven_data=raven_data.to_dict(),
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

    # Clearing out memory
    del fig
    del ax
    del audio
