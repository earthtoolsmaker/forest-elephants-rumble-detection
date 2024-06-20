"""Deal with audio data."""

from pathlib import Path

import librosa


def load_audio(sound_path: Path, duration: float = 10.0, offset: float = 0.0):
    """Load audio path and clip it to duration with the provided offset using
    librosa."""
    return librosa.load(sound_path, sr=None, duration=duration, offset=offset)
