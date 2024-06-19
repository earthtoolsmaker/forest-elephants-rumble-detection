import logging
import math
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from forest_elephants_rumble_detection.data.spectrogram.torchaudio import (
    waveform_to_np_image,
)


def batch_sequence(xs: list, batch_size: int):
    """
    Yields successive n-sized batches from xs.
    """
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def clip(
    waveform: torch.Tensor,
    offset: float,
    duration: float,
    sample_rate: int,
) -> torch.Tensor:
    """
    Returns a clipped waveform of `duration` seconds at `offset` in seconds.
    """
    offset_frames_start = int(offset * sample_rate)
    offset_frames_end = offset_frames_start + int(duration * sample_rate)
    return waveform[:, offset_frames_start:offset_frames_end]


def chunk(
    waveform: torch.Tensor,
    sample_rate: int,
    duration: float,
    overlap: float,
) -> list[torch.Tensor]:
    """
    Returns a list of waveforms as torch.Tensor. Each of these waveforms have the specified
    duration and the specified overlap in seconds.
    """
    total_seconds = waveform.shape[1] / sample_rate
    number_spectrograms = total_seconds / (duration - overlap)
    offsets = [
        idx * (duration - overlap) for idx in range(0, math.floor(number_spectrograms))
    ]
    return [
        clip(
            waveform=waveform,
            offset=offset,
            duration=duration,
            sample_rate=sample_rate,
        )
        for offset in offsets
    ]


def load_audio(audio_filepath: Path) -> Tuple[torch.Tensor, int]:
    """
    Loads an audio_filepath and returns the waveform and sample_rate of the file.
    """
    start_time = time.time()
    waveform, sample_rate = torchaudio.load(audio_filepath)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(
        f"Elapsed time to load audio file {audio_filepath.name}: {elapsed_time:.2f}s"
    )
    return waveform, sample_rate


def inference(
    model: YOLO,
    audio_filepath: Path,
    duration: float,
    overlap: float,
    width: int,
    height: int,
    freq_max: float,
    n_fft: int,
    hop_length: int,
    batch_size: int,
    output_dir: Path,
    save_spectrograms: bool,
    save_predictions: bool,
    verbose: bool,
) -> list:
    """
    Inference entry point for running on an entire audio_filepath sound file.
    """
    logging.info(f"Loading audio filepath {audio_filepath}")
    # waveform, sample_rate = torchaudio.load(audio_filepath)
    waveform, sample_rate = load_audio(audio_filepath)
    waveforms = chunk(
        waveform=waveform,
        sample_rate=sample_rate,
        duration=duration,
        overlap=overlap,
    )
    logging.info(f"Chunking the waveform into {len(waveforms)} overlapping clips")
    logging.info(f"Generating {len(waveforms)} spectrograms")
    images = [
        Image.fromarray(
            waveform_to_np_image(
                waveform=y,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                freq_max=freq_max,
                width=width,
                height=height,
            )
        )
        for y in tqdm(waveforms)
    ]
    if save_spectrograms:
        save_dir = output_dir / "spectrograms"
        logging.info(f"Saving spectrograms in {save_dir}")
        save_dir.mkdir(exist_ok=True, parents=True)
        for i, image in tqdm(enumerate(images), total=len(images)):
            image.save(save_dir / f"spectrogram_{i}.png")

    results = []

    batches = list(batch_sequence(images, batch_size=batch_size))
    logging.info(f"Running inference on the spectrograms, {len(batches)} batches")
    for batch in tqdm(batches):
        results.extend(model.predict(batch, verbose=verbose))

    if save_predictions:
        save_dir = output_dir / "predictions"
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving predictions in {save_dir}")
        for i, yolov8_prediction in tqdm(enumerate(results), total=len(results)):
            yolov8_prediction.save(str(save_dir / f"prediction_{i}.png"))

    return results


def index_to_relative_offset(idx: int, duration: float, overlap: float) -> float:
    """
    Returns the relative offset in seconds based on the provided spectrogram index, the duration and the overlap.
    """
    return idx * (duration - overlap)


def from_yolov8_prediction(
    yolov8_prediction,
    idx: int,
    duration: float,
    overlap: float,
    freq_min: float,
    freq_max: float,
) -> list[dict]:
    results = []
    for k, box_xyxyn in enumerate(yolov8_prediction.boxes.xyxyn):
        conf = yolov8_prediction.boxes.conf[k].item()
        x1, y1, x2, y2 = box_xyxyn.numpy()
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        freq_start = ymin * (freq_max - freq_min)
        freq_end = ymax * (freq_max - freq_min)
        t_start = xmin * duration + index_to_relative_offset(
            idx=idx, duration=duration, overlap=overlap
        )
        t_end = xmax * duration + index_to_relative_offset(
            idx=idx, duration=duration, overlap=overlap
        )
        data = {
            "probability": conf,
            "freq_start": freq_start,
            "freq_end": freq_end,
            "t_start": t_start,
            "t_end": t_end,
        }
        results.append(data)
    return results


def to_dataframe(
    yolov8_predictions,
    duration: float,
    overlap: float,
    freq_min: float,
    freq_max: float,
) -> pd.DataFrame:
    """
    Turns the yolov8 predictions into a pandas dataframe, taking into account the relative offset of each prediction.
    The dataframes contains the following columns
      probability (float): float in 0-1 that represents the probability that this is an actual rumble
      freq_start (float): Hz - where the box starts on the frequency axis
      freq_end (float): Hz - where the box ends on the frequency axis
      t_start (float): Hz - where the box starts on the time axis
      t_end (float): Hz - where the box ends on the time axis
    """
    results = []
    for idx, yolov8_prediction in enumerate(yolov8_predictions):
        results.extend(
            from_yolov8_prediction(
                yolov8_prediction,
                idx=idx,
                duration=duration,
                overlap=overlap,
                freq_min=freq_min,
                freq_max=freq_max,
            )
        )
    return pd.DataFrame(results)


def pipeline(
    model: YOLO,
    audio_filepaths: list[Path],
    duration: float,
    overlap: float,
    width: int,
    height: int,
    freq_min: float,
    freq_max: float,
    n_fft: int,
    hop_length: int,
    batch_size: int,
    output_dir: Path,
    save_spectrograms: bool,
    save_predictions: bool,
    verbose: bool,
) -> pd.DataFrame:
    """
    Main entrypoint to generate the predictions on a set of audio_filepaths
    """
    dfs = []
    for audio_filepath in audio_filepaths:
        start_time = time.time()
        sub_output_dir = output_dir / audio_filepath.stem
        sub_output_dir.mkdir(exist_ok=True, parents=True)
        yolov8_predictions = inference(
            model=model,
            audio_filepath=audio_filepath,
            duration=duration,
            overlap=overlap,
            width=width,
            height=height,
            freq_max=freq_max,
            n_fft=n_fft,
            hop_length=hop_length,
            batch_size=batch_size,
            output_dir=sub_output_dir,
            save_spectrograms=save_spectrograms,
            save_predictions=save_predictions,
            verbose=verbose,
        )
        df = to_dataframe(
            yolov8_predictions=yolov8_predictions,
            duration=duration,
            overlap=overlap,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        df["audio_filepath"] = str(audio_filepath)
        df["instance_class"] = "rumble"
        df.to_csv(sub_output_dir / "results.csv")
        dfs.append(df)
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time = end_time - start_time
        logging.info(
            f"Elapsed time to analyze {audio_filepath.name}: {elapsed_time:.2f}s"
        )
    return pd.concat(dfs)
