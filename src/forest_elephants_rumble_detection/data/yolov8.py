"""Util functions to work with yolov8 format."""

from pathlib import Path

from .math import clamp


def bbox_to_yolov8_txt_format(
    bbox: dict,
    rumble_class: int = 0,
) -> str:
    """Turns a `bbox` into a yolov8 string."""
    return f"{rumble_class} {bbox['center_x']} {bbox['center_y']} {bbox['width']} {bbox['height']}"


def bboxes_to_yolov8_txt_format(
    bboxes: list[dict],
    rumble_class: int = 0,
) -> str | None:
    """Turns a sequence of bboxes into a yolov8 str."""
    if not bboxes:
        return None
    else:
        return "\n".join(
            [
                bbox_to_yolov8_txt_format(bbox, rumble_class=rumble_class)
                for bbox in bboxes
            ]
        )


def parse_yolov8_txt(filepath: Path) -> list[dict]:
    """Parses a YOLOv8 txt file. Returns a list of bboxes.

    A bbox contains the following keys:
    - center_x: float - (0., 1.)
    - center_y: float - (0., 1.)
    - width: float - (0., 1.)
    - height: float - (0., 1.)
    - class_inst: int
    """
    with open(filepath, "r") as fp:
        bboxes = []
        content = fp.read()
        lines = content.split("\n")
        for line in lines:
            xs = line.split(" ")
            class_inst, center_x, center_y, width, height = (
                xs[0],
                xs[1],
                xs[2],
                xs[3],
                xs[4],
            )
            bbox = {
                "class_inst": int(class_inst),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "width": float(width),
                "height": float(height),
            }
            bboxes.append(bbox)
        return bboxes


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
