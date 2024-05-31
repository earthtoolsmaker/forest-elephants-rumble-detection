"""Loading, Preparing and generating testing data for the provided dataset."""

from pathlib import Path

import pandas as pd


def parse_text_file(path: Path) -> pd.DataFrame:
    """Returns a pandas dataframe of the parsed `path`."""
    return pd.read_csv(path, sep="\t")


def get_all_audio_filepaths(train_dir: Path, test_dir: Path) -> list[Path]:
    train_sounds_dir = train_dir / "Sounds"
    test_dzanga_sounds_dir = test_dir / "Dzanga" / "Sounds"
    test_pnnn_sounds_dir = test_dir / "PNNN" / "Sounds"
    return [
        *train_sounds_dir.glob("*.wav"),
        *test_dzanga_sounds_dir.glob("*.wav"),
        *test_pnnn_sounds_dir.glob("*.wav"),
    ]


def get_testing_txt_files(test_dir: Path) -> list[Path]:
    return list(test_dir.rglob("*.txt"))


def replace_audio_path(audio_filepath: Path) -> Path:
    stem = audio_filepath.stem
    new_stem = stem.replace("dzan", "dz")
    return audio_filepath.parent / f"{new_stem}{audio_filepath.suffix}"


def replace_audio_filename(audio_filename: str) -> str:
    return audio_filename.replace("dz", "dzan")


def make_audio_filename_to_audio_filepath(
    train_dir: Path, test_dir: Path
) -> dict[str, Path]:
    """Returns a mapping from audio_filename (as a string) to the path pointing
    the the audio file."""
    audio_filepaths = get_all_audio_filepaths(train_dir=train_dir, test_dir=test_dir)
    return {
        replace_audio_filename(audio_filepath.name): replace_audio_path(audio_filepath)
        for audio_filepath in audio_filepaths
    }


def parse_all_testing_txt_files(test_dir: Path) -> pd.DataFrame:
    """Parses all txt files of the testing dataset and returns it as a
    dataframe."""
    txt_files = get_testing_txt_files(test_dir)
    return pd.concat([parse_text_file(txt_file) for txt_file in txt_files])


def prepare_df(df: pd.DataFrame, train_dir: Path, test_dir: Path) -> pd.DataFrame:
    """
    Returns a new dataframe that contains extra columns:
    - duration: duration in second of the rumble
    - t_start, t_end: time in second when the rumble starts and ends
    - freq_low, freq_hig: frequency in hertz of the rumbles

    Note: t_start, t_end, freq_low and freq_high make it possible to localize rumbles on the spectrograms.
    """
    audio_filename_to_audio_filepath = make_audio_filename_to_audio_filepath(
        train_dir=train_dir,
        test_dir=test_dir,
    )
    df_result = df.copy()
    df_result["duration"] = df_result["End Time (s)"] - df_result["Begin Time (s)"]
    df_result["t_start"] = df_result["File Offset (s)"]
    df_result["t_end"] = df_result["t_start"] + df_result["duration"]
    df_result["freq_low"] = df_result["Low Freq (Hz)"]
    df_result["freq_high"] = df_result["High Freq (Hz)"]
    df_result["audio_filepath"] = df_result["Begin File"].map(
        lambda begin_file: audio_filename_to_audio_filepath.get(begin_file)
    )
    # Dropping where there is no audio_filepath
    df_result = df_result[~df_result["audio_filepath"].isnull()]
    df_result["audio_filepath_exists"] = df_result["audio_filepath"].map(
        lambda p: p.exists()
    )
    return df_result
