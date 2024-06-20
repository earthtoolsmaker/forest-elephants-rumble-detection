"""Math module."""


def clamp(minimum, x, maximum):
    """
    Clamp a number within the inclusive range specified by the given minimum and maximum values.

    Parameters:
    minimum (float or int): The lower bound of the range.
    x (float or int): The value to be clamped.
    maximum (float or int): The upper bound of the range.

    Returns:
    float or int: The clamped value, which will be:
                  - 'minimum' if 'x' is less than 'minimum'.
                  - 'maximum' if 'x' is greater than 'maximum'.
                  - 'x' if 'x' is between 'minimum' and 'maximum'.

    Examples:
    >>> clamp(0, 5, 10)
    5
    >>> clamp(0, -5, 10)
    0
    >>> clamp(0, 15, 10)
    10
    """
    return max(minimum, min(x, maximum))
