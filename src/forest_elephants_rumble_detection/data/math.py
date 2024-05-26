"""Math module."""


def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))
