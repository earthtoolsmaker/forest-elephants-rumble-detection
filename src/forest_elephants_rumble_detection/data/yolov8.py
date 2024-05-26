"""Util functions to work with yolov8 format."""


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
