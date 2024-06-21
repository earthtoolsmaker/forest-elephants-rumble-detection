# import logging
# import math
# import time
# from pathlib import Path
# from typing import Tuple, List

# import pandas as pd
# import torch
# import torchaudio
# from PIL import Image
# from tqdm import tqdm
# from ultralytics import YOLO
# import numpy as np

# from forest_elephants_rumble_detection.data.spectrogram.torchaudio_torch import (
#     waveform_to_image,
# )


# def batch_sequence(xs: list, batch_size: int):
#     """
#     Yields successive n-sized batches from xs.
#     """
#     for i in range(0, len(xs), batch_size):
#         yield xs[i : i + batch_size]


# def clip(
#     waveform: torch.Tensor,
#     offset: float,
#     duration: float,
#     sample_rate: int,
# ) -> torch.Tensor:
#     """
#     Returns a clipped waveform of `duration` seconds at `offset` in seconds.
#     """
#     offset_frames_start = int(offset * sample_rate)
#     offset_frames_end = offset_frames_start + int(duration * sample_rate)
#     return waveform[:, offset_frames_start:offset_frames_end]


# def chunk(
#     waveform: torch.Tensor,
#     sample_rate: int,
#     duration: float,
#     overlap: float,
# ) -> list[torch.Tensor]:
#     """
#     Returns a list of waveforms as torch.Tensor. Each of these waveforms have the specified
#     duration and the specified overlap in seconds.
#     """
#     total_seconds = waveform.shape[1] / sample_rate
#     number_spectrograms = total_seconds / (duration - overlap)
#     offsets = [
#         idx * (duration - overlap) for idx in range(0, math.floor(number_spectrograms))
#     ]
#     return [
#         clip(
#             waveform=waveform,
#             offset=offset,
#             duration=duration,
#             sample_rate=sample_rate,
#         )
#         for offset in offsets
#     ]


# def load_audio(audio_filepath: Path) -> Tuple[torch.Tensor, int]:
#     """
#     Loads an audio_filepath and returns the waveform and sample_rate of the file.
#     """
#     start_time = time.time()
#     waveform, sample_rate = torchaudio.load(audio_filepath)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     logging.info(
#         f"Elapsed time to load audio file {audio_filepath.name}: {elapsed_time:.2f}s"
#     )
#     return waveform, sample_rate


# class SpectrogramPipeline:
#     '''
#     Spectrogram pipeline that transforms the audio to a spectrogram and then resizes it to a given size.
#     '''

#     def __init__(self, n_fft=4096, hop_length=1024, size=(640, 640), freq_max=250.00):
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.size = size
#         #self.sample_rate = sample_rate
#         self.flip = True  # Flip due to origin conflicts (image vs spectrogram origins)
#         self.freq_max = freq_max  # Maximum frequency to include in the spectrogram

#         self.spectrogram_transform = torchaudio.transforms.Spectrogram(
#             n_fft=self.n_fft, power=2, hop_length=self.hop_length)

#         self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

#     def _resize_spectrogram(self, spectrograms):
#         '''
#         Resizes the spectrogram to a given size.
#         '''
#         if spectrograms.shape[2] < self.size[0] or spectrograms.shape[3] < self.size[1]:
#             padding = (
#                 (self.size[0] - spectrograms.shape[2]) // 2,
#                 (self.size[1] - spectrograms.shape[3]) // 2,
#             )
#             spectrograms = torch.nn.functional.pad(spectrograms, (padding[1], padding[1], padding[0], padding[0]))
#         return torch.nn.functional.interpolate(spectrograms, size=self.size, mode='bicubic')

#     def __call__(self, audio: torch.Tensor, sample_rate: int):
#         '''
#         Transforms the given audio to a spectrogram.
#         '''
#         # Apply a lowpass filter based on freq_max
#         filtered_audio = torchaudio.functional.lowpass_biquad(
#             waveform=audio.unsqueeze(0), sample_rate=sample_rate, cutoff_freq=self.freq_max)

#         # Generate the spectrogram
#         spectrograms = self.amp_to_db(self.spectrogram_transform(filtered_audio) + 1e-6)
#         resized_spectrogram = self._resize_spectrogram(spectrograms)

#         # Manipulations for image alignment and normalization
#         resized_spectrogram = resized_spectrogram.squeeze(0).unsqueeze(1)
#         if self.flip:
#             resized_spectrogram = torch.flip(resized_spectrogram, [2])
        
#         # Normalize the spectrogram to be between 0 and 1
#         resized_spectrogram -= resized_spectrogram.min()
#         resized_spectrogram /= resized_spectrogram.max()

#         # Add rgb channels
#         resized_spectrogram = resized_spectrogram.repeat(1, 3, 1, 1)

#         return resized_spectrogram

# def inference(
#     model: YOLO,
#     audio_filepath: Path,
#     duration: float,
#     overlap: float,
#     width: int,
#     height: int,
#     freq_max: float,
#     n_fft: int,
#     hop_length: int,
#     batch_size: int,
#     output_dir: Path,
#     save_spectrograms: bool,
#     save_predictions: bool,
#     verbose: bool,
# ) -> List:
#     """
#     Inference entry point for running on an entire audio_filepath sound file.
#     """

#     logging.info(f"Loading audio filepath {audio_filepath}")
#     waveform, sample_rate = load_audio(audio_filepath)
#     waveforms = chunk(
#         waveform=waveform,
#         sample_rate=sample_rate,
#         duration=duration,
#         overlap=overlap,
#     )
#     logging.info(f"Chunking the waveform into {len(waveforms)} overlapping clips")
#     logging.info(f"Generating {len(waveforms)} spectrograms")

#     spectrograms = [waveform_to_image(
#                 waveform=y,
#                 sample_rate=sample_rate,
#                 n_fft=n_fft,
#                 hop_length=hop_length,
#                 freq_max=freq_max,
#                 width=width,
#                 height=height,
#             ).float()
#         for y in tqdm(waveforms)]

#     # Convert spectrograms to 3-channel tensors
#     spectrograms = [torch.stack([spect, spect, spect], dim=0) for spect in spectrograms]

#     #      Ensure the spectrograms are of shape (3, 640, 640)
#     spectrograms = [torch.nn.functional.interpolate(spect.unsqueeze(0), size=(640, 640)).squeeze(0) for spect in spectrograms]

#     if save_spectrograms:
#         save_dir = output_dir / "spectrograms"
#         logging.info(f"Saving spectrograms in {save_dir}")
#         save_dir.mkdir(exist_ok=True, parents=True)
#         for i, spectrogram in tqdm(enumerate(spectrograms), total=len(spectrograms)):
#             image = Image.fromarray(spectrogram.squeeze().numpy().transpose(1, 2, 0))
#             image.save(save_dir / f"spectrogram_{i}.png")

#     results = []

#     device = model.device  # Ensure the model's device is used for tensors
    
#     batches = list(batch_sequence(spectrograms, batch_size=batch_size))
#     print(len(batches[0]))
#     logging.info(f"Running inference on the spectrograms, {len(batches)} batches")

#     for batch in tqdm(batches):
#         batch_tensor = torch.stack(batch).to(device)
#         if batch_tensor.shape[1] == 1:
#             batch_tensor = batch_tensor.squeeze(1)  # Remove the second dimension if it is 1
#         results.extend(model(batch_tensor, verbose=verbose))
    
#     if save_predictions:
#         save_dir = output_dir / "predictions"
#         save_dir.mkdir(parents=True, exist_ok=True)
#         logging.info(f"Saving predictions in {save_dir}")
#         for i, yolov8_prediction in tqdm(enumerate(results), total=len(results)):
#             yolov8_prediction.save(str(save_dir / f"prediction_{i}.png"))

#     return results


# def index_to_relative_offset(idx: int, duration: float, overlap: float) -> float:
#     """
#     Returns the relative offset in seconds based on the provided spectrogram index, the duration and the overlap.
#     """
#     return idx * (duration - overlap)


# def from_yolov8_prediction(
#     yolov8_prediction,
#     idx: int,
#     duration: float,
#     overlap: float,
#     freq_min: float,
#     freq_max: float,
# ) -> List[dict]:
#     results = []
#     for k, box_xyxyn in enumerate(yolov8_prediction.boxes.xyxyn):
#         conf = yolov8_prediction.boxes.conf[k].item()
#         x1, y1, x2, y2 = box_xyxyn.numpy()
#         xmin = min(x1, x2)
#         xmax = max(x1, x2)
#         ymin = min(y1, y2)
#         ymax = max(y1, y2)
#         freq_start = ymin * (freq_max - freq_min)
#         freq_end = ymax * (freq_max - freq_min)
#         t_start = xmin * duration + index_to_relative_offset(
#             idx=idx, duration=duration, overlap=overlap
#         )
#         t_end = xmax * duration + index_to_relative_offset(
#             idx=idx, duration=duration, overlap=overlap
#         )
#         data = {
#             "probability": conf,
#             "freq_start": freq_start,
#             "freq_end": freq_end,
#             "t_start": t_start,
#             "t_end": t_end,
#         }
#         results.append(data)
#     return results


# def to_dataframe(
#     yolov8_predictions,
#     duration: float,
#     overlap: float,
#     freq_min: float,
#     freq_max: float,
# ) -> pd.DataFrame:
#     """
#     Turns the yolov8 predictions into a pandas dataframe, taking into account the relative offset of each prediction.
#     The dataframes contains the following columns
#       probability (float): float in 0-1 that represents the probability that this is an actual rumble
#       freq_start (float): Hz - where the box starts on the frequency axis
#       freq_end (float): Hz - where the box ends on the frequency axis
#       t_start (float): Hz - where the box starts on the time axis
#       t_end (float): Hz - where the box ends on the time axis
#     """
#     results = []
#     for idx, yolov8_prediction in enumerate(yolov8_predictions):
#         results.extend(
#             from_yolov8_prediction(
#                 yolov8_prediction,
#                 idx=idx,
#                 duration=duration,
#                 overlap=overlap,
#                 freq_min=freq_min,
#                 freq_max=freq_max,
#             )
#         )
#     return pd.DataFrame(results)


# def pipeline(
#     model: YOLO,
#     audio_filepaths: list[Path],
#     duration: float,
#     overlap: float,
#     width: int,
#     height: int,
#     freq_min: float,
#     freq_max: float,
#     n_fft: int,
#     hop_length: int,
#     batch_size: int,
#     output_dir: Path,
#     save_spectrograms: bool,
#     save_predictions: bool,
#     verbose: bool,
# ) -> pd.DataFrame:
#     """
#     Main entrypoint to generate the predictions on a set of audio_filepaths
#     """
#     dfs = []
#     for audio_filepath in audio_filepaths:
#         start_time = time.time()
#         sub_output_dir = output_dir / audio_filepath.stem
#         sub_output_dir.mkdir(exist_ok=True, parents=True)
#         yolov8_predictions = inference(
#             model=model,
#             audio_filepath=audio_filepath,
#             duration=duration,
#             overlap=overlap,
#             width=width,
#             height=height,
#             freq_max=freq_max,
#             n_fft=n_fft,
#             hop_length=hop_length,
#             batch_size=batch_size,
#             output_dir=sub_output_dir,
#             save_spectrograms=save_spectrograms,
#             save_predictions=save_predictions,
#             verbose=verbose,
#         )
#         df = to_dataframe(
#             yolov8_predictions=yolov8_predictions,
#             duration=duration,
#             overlap=overlap,
#             freq_min=freq_min,
#             freq_max=freq_max,
#         )
#         df["audio_filepath"] = str(audio_filepath)
#         df["instance_class"] = "rumble"
#         df.to_csv(sub_output_dir / "results.csv")
#         dfs.append(df)
#         end_time = time.time()
#         # Calculate elapsed time
#         elapsed_time = end_time - start_time
#         logging.info(
#             f"Elapsed time to analyze {audio_filepath.name}: {elapsed_time:.2f}s"
#         )
#     return pd.concat(dfs)

import logging
import math
import time
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

from forest_elephants_rumble_detection.data.spectrogram.torchaudio_torch import (
    waveform_to_image,
)

def batch_sequence(xs: list, batch_size: int):
    """
    Yields successive n-sized batches from xs.
    """
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]

def clip(waveform: torch.Tensor, offset: float, duration: float, sample_rate: int) -> torch.Tensor:
    """
    Returns a clipped waveform of `duration` seconds at `offset` in seconds.
    """
    offset_frames_start = int(offset * sample_rate)
    offset_frames_end = offset_frames_start + int(duration * sample_rate)
    return waveform[:, offset_frames_start:offset_frames_end]

def chunk(waveform: torch.Tensor, sample_rate: int, duration: float, overlap: float) -> list[torch.Tensor]:
    """
    Returns a list of waveforms as torch.Tensor. Each of these waveforms have the specified
    duration and the specified overlap in seconds.
    """
    total_seconds = waveform.shape[1] / sample_rate
    number_spectrograms = total_seconds / (duration - overlap)
    offsets = [idx * (duration - overlap) for idx in range(0, math.floor(number_spectrograms))]
    return [clip(waveform=waveform, offset=offset, duration=duration, sample_rate=sample_rate) for offset in offsets]

def load_audio(audio_filepath: Path) -> Tuple[torch.Tensor, int]:
    """
    Loads an audio_filepath and returns the waveform and sample_rate of the file.
    """
    start_time = time.time()
    waveform, sample_rate = torchaudio.load(audio_filepath)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time to load audio file {audio_filepath.name}: {elapsed_time:.2f}s")
    return waveform, sample_rate

class SpectrogramPipeline:
    '''
    Spectrogram pipeline that transforms the audio to a spectrogram and then resizes it to a given size.
    '''

    def __init__(self, n_fft=4096, hop_length=1024, size=(640, 640), freq_max=250.00):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.size = size
        self.flip = True  # Flip due to origin conflicts (image vs spectrogram origins)
        self.freq_max = freq_max  # Maximum frequency to include in the spectrogram

        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, power=2, hop_length=self.hop_length)

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def _resize_spectrogram(self, spectrograms):
        '''
        Resizes the spectrogram to a given size.
        '''
        if spectrograms.shape[2] < self.size[0] or spectrograms.shape[3] < self.size[1]:
            padding = (
                (self.size[0] - spectrograms.shape[2]) // 2,
                (self.size[1] - spectrograms.shape[3]) // 2,
            )
            spectrograms = torch.nn.functional.pad(spectrograms, (padding[1], padding[1], padding[0], padding[0]))
        return torch.nn.functional.interpolate(spectrograms, size=self.size, mode='bicubic')

    def __call__(self, audio: torch.Tensor, sample_rate: int):
        '''
        Transforms the given audio to a spectrogram.
        '''
        # Apply a lowpass filter based on freq_max
        filtered_audio = torchaudio.functional.lowpass_biquad(
            waveform=audio.unsqueeze(0), sample_rate=sample_rate, cutoff_freq=self.freq_max)

        # Generate the spectrogram
        spectrograms = self.amp_to_db(self.spectrogram_transform(filtered_audio) + 1e-6)
        resized_spectrogram = self._resize_spectrogram(spectrograms)

        # Manipulations for image alignment and normalization
        resized_spectrogram = resized_spectrogram.squeeze(0).unsqueeze(1)
        if self.flip:
            resized_spectrogram = torch.flip(resized_spectrogram, [2])
        
        # Normalize the spectrogram to be between 0 and 1
        resized_spectrogram -= resized_spectrogram.min()
        resized_spectrogram /= resized_spectrogram.max()

        # Add rgb channels
        resized_spectrogram = resized_spectrogram.repeat(1, 3, 1, 1)

        return resized_spectrogram

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
) -> List:
    """
    Inference entry point for running on an entire audio_filepath sound file.
    """

    logging.info(f"Loading audio filepath {audio_filepath}")
    waveform, sample_rate = load_audio(audio_filepath)
    waveforms = chunk(
        waveform=waveform,
        sample_rate=sample_rate,
        duration=duration,
        overlap=overlap,
    )
    logging.info(f"Chunking the waveform into {len(waveforms)} overlapping clips")
    logging.info(f"Generating {len(waveforms)} spectrograms")

    spectrograms = [waveform_to_image(
                waveform=y,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                freq_max=freq_max,
                width=width,
                height=height,
            ).float()
        for y in tqdm(waveforms)]

    # Convert spectrograms to 3-channel tensors
    spectrograms = [torch.stack([spect, spect, spect], dim=0) for spect in spectrograms]

    # Ensure the spectrograms are of shape (3, 640, 640)
    spectrograms = [torch.nn.functional.interpolate(spect.unsqueeze(0), size=(640, 640)).squeeze(0) for spect in spectrograms]

    if save_spectrograms:
        save_dir = output_dir / "spectrograms"
        logging.info(f"Saving spectrograms in {save_dir}")
        save_dir.mkdir(exist_ok=True, parents=True)
        for i, spectrogram in tqdm(enumerate(spectrograms), total=len(spectrograms)):
            image = Image.fromarray(spectrogram.squeeze().numpy().transpose(1, 2, 0))
            image.save(save_dir / f"spectrogram_{i}.png")

    results = []

    device = model.device  # Ensure the model's device is used for tensors
    
    batches = list(batch_sequence(spectrograms, batch_size=batch_size))
    logging.info(f"Running inference on the spectrograms, {len(batches)} batches")

    for batch in tqdm(batches):
        batch_tensor = torch.stack(batch).to(device)
        if batch_tensor.shape[1] == 1:
            batch_tensor = batch_tensor.squeeze(1)  # Remove the second dimension if it is 1
        results.extend(model(batch_tensor, verbose=verbose))
    
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
) -> List[dict]:
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
        t_start = xmin * duration + index_to_relative_offset(idx=idx, duration=duration, overlap=overlap)
        t_end = xmax * duration + index_to_relative_offset(idx=idx, duration=duration, overlap=overlap)
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
        logging.info(f"Elapsed time to analyze {audio_filepath.name}: {elapsed_time:.2f}s")
    return pd.concat(dfs)
