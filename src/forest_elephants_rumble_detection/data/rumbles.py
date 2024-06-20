"""Rumbles utils

This module provides functions to work with raven_data files and extracting
bounding boxes for the rumbles."""

import pandas as pd

from .yolov8 import raven_data_to_spectrogram_yolov8_bbox


def select_rumbles_at(df: pd.DataFrame, offset: float, duration: float) -> pd.DataFrame:
    """Filters our rows in the dataframe `df` that are outside the offset and
    duration range.

    Returns a dataframe.
    """
    return df[
        (df["t_end"] > offset) & (df["t_start"] < offset + duration)
    ].reset_index()


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
