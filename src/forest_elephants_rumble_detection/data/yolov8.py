"""Util functions to work with yolov8 format."""

from pathlib import Path


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
