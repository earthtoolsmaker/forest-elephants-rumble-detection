"""
Generates spectrograms using torchaudio without transforming to numpy.
"""

import math
import numpy as np
import cv2

import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms.functional as TF


def clip(
    waveform: torch.Tensor, offset: float, duration: float, sample_rate: int
) -> torch.Tensor:
    """
    Returns a clipped waveform of `duration` seconds at `offset` in seconds.
    """
    offset_frames_start = int(offset * sample_rate)
    offset_frames_end = offset_frames_start + int(duration * sample_rate)
    return waveform[:, offset_frames_start:offset_frames_end]


def waveform_to_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    freq_max: float,
) -> torch.Tensor:
    """
    Returns a spectrogram as a torch.Tensor given the provided arguments.
    See torchaudio.transforms.Spectrogram for more details about the parameters.

    Args:
      waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
      sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
      n_fft (int): Size of FFT
      hop_length (int): Length of hop between STFT windows.
      freq_max (float): cutoff frequency (Hz)
    """
    filtered_waveform = torchaudio.functional.lowpass_biquad(
        waveform=waveform, sample_rate=sample_rate, cutoff_freq=freq_max
    )
    transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
    spectrogram = transform(filtered_waveform)
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    frequencies = torch.linspace(0, sample_rate // 2, spectrogram_db.size(1))
    max_freq_bin = torch.searchsorted(frequencies, freq_max).item()
    filtered_spectrogram_db = spectrogram_db[:, :max_freq_bin, :]
    return filtered_spectrogram_db

def normalize(x: np.ndarray, max_value: int = 255) -> np.ndarray:
    """
    Returns the normalized array, value in [0 - max_value]
    Useful for image conversion.
    """
    _min, _max = x.min(), x.max()
    x_normalized = max_value * (x - _min) / (_max - _min)
    return x_normalized.astype(np.uint8)

def spectrogram_tensor_to_torch_image(
    spectrogram: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """
    Returns a torch tensor of shape (1, height, width) that represents the spectrogram tensor as an image.
    """
    spectrogram_db_np = spectrogram[0].numpy()
    # Normalize to [0, 255] for image conversion

    spectrogram_db_normalized = normalize(spectrogram_db_np, max_value=255)
    resized_spectrogram_array = cv2.resize(
        spectrogram_db_normalized, (width, height), interpolation=cv2.INTER_LINEAR
    )
    
    # Horizontal flip to make it show the low frequency range at the bottom left of the image instead of the top left
    flipped_resized_spectrogram_array = np.flipud(resized_spectrogram_array)
    
    # Convert the numpy array back to a torch tensor
    flipped_resized_spectrogram_tensor = torch.tensor(flipped_resized_spectrogram_array)
    
    # Add a channel dimension to match the shape (1, height, width)
    flipped_resized_spectrogram_tensor = flipped_resized_spectrogram_tensor.unsqueeze(0)
    
    return flipped_resized_spectrogram_tensor
 

def waveform_to_image(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    freq_max: float,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Returns a tensor image of shape (height, width) that represents the waveform tensor as an image of its spectrogram.

    Args:
      waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
      sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
      duration (float): time in seconds of the waveform
      n_fft (int): Size of FFT
      hop_length (int): Length of hop between STFT windows.
      freq_max (float): cutoff frequency (Hz)
      width (int): width of the generated image
      height (int): height of the generated image
    """
    spectrogram = waveform_to_spectrogram(
        waveform=waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        freq_max=freq_max,
    )
    return spectrogram_tensor_to_torch_image(
        spectrogram=spectrogram,
        width=width,
        height=height,
    )


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