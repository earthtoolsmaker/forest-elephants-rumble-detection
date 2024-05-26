"""Module to work with audio offsets."""

import random

import pandas as pd


def get_random_offsets(n: int, df: pd.DataFrame, random_seed: int = 0) -> list[float]:
    """Returns a list of random offsets to consider for plotting.

    random_seed is used to make this function deterministic.
    """
    random.seed(random_seed)
    t_start_xs = list(df["t_start"].unique())
    t0, tmax = min(*t_start_xs), max(*t_start_xs)
    return [random.randrange(int(t0), int(tmax)) for _ in range(n)]


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
