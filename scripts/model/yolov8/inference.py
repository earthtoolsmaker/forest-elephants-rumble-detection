"""Script to run inferences with a YOLOv8 model, it makes it possible to
visually inspect predictions and compare them with the ground truth labels."""

import argparse
import logging
import random
from pathlib import Path

import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from ultralytics import YOLO

from forest_elephants_rumble_detection.data.yolov8 import parse_yolov8_txt
from forest_elephants_rumble_detection.model.yolo.eval import load_trained_model


def make_cli_parser() -> argparse.ArgumentParser:
    """Makes the CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir-yolov8-dataset",
        help="path to save the model_artifacts",
        default="./data/03_model_input/yolov8/full/",
        type=Path,
    )
    parser.add_argument(
        "--split",
        help="value in {train, val, test}",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--random-seed",
        help="Random seed",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--k",
        help="Number of predictions to generate and persist.",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--output-dir",
        help="path to save the model predictions",
        default="./data/05_model_output/yolov8/best_full_dataset/",
        type=Path,
    )
    parser.add_argument(
        "--weights-filepath",
        help="path to the model weights",
        default="./data/04_models/yolov8/best_full_dataset/weights/best.pt",
        type=Path,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    """Returns whether the parsed args are valid."""
    if not args["input_dir_yolov8_dataset"].exists():
        logging.error("Invalid --input-dir-yolov8-dataset dir does not exist")
        return False
    elif not args["weights_filepath"].exists():
        logging.error("Invalid --weights-filepath filepath does not exist")
        return False
    elif args["split"] not in ["train", "val", "test"]:
        logging.error("Invalid --split value")
        return False
    else:
        return True


def draw_yolov8_bbox_from(ax, bbox, width_pixels: int, height_pixels: int) -> None:
    x = (bbox["center_x"] - bbox["width"] / 2) * width_pixels
    y = (bbox["center_y"] - bbox["height"] / 2) * height_pixels
    width = bbox["width"] * width_pixels
    height = bbox["height"] * height_pixels
    ax.add_patch(
        Rectangle(
            xy=(x, y),
            width=width,
            height=height,
            color="lime",
            fc="none",
            linewidth=1.0,
        )
    )


def save_detailed_prediction(
    model: YOLO,
    output_dir: Path,
    image_filepath: Path,
    label_filepath: Path,
) -> None:
    predictions = model.predict(image_filepath)
    bboxes_predictions = []
    if predictions:
        bboxes_predictions = [
            {"center_x": x[0], "center_y": x[1], "width": x[2], "height": x[3]}
            for x in predictions[0].boxes.xywhn.cpu().numpy()
        ]
    fig, axis = plt.subplots(3)
    fig.suptitle(f"Spectogram - duration: 60s")
    fig.supylabel("Frequency (Hz)")
    fig.supxlabel("Time (s)")

    axis[0].axis("off")
    axis[1].axis("off")
    axis[2].axis("off")

    axis[0].set_title("Original Spectrogram")
    axis[1].set_title("Ground Truth")
    axis[2].set_title("Predictions")

    spectrogram = img.imread(image_filepath)
    H, W, _ = spectrogram.shape
    bboxes = parse_yolov8_txt(label_filepath)

    axis[0].imshow(spectrogram)
    axis[1].imshow(spectrogram)
    axis[2].imshow(spectrogram)

    for bbox in bboxes:
        draw_yolov8_bbox_from(ax=axis[1], bbox=bbox, width_pixels=W, height_pixels=H)

    for bbox in bboxes_predictions:
        draw_yolov8_bbox_from(ax=axis[2], bbox=bbox, width_pixels=W, height_pixels=H)

    output_filepath = output_dir / image_filepath.name
    plt.savefig(output_filepath, bbox_inches="tight")
    plt.close("all")


def persist_random_predictions(
    model: YOLO,
    input_dir_yolov8_dataset: Path,
    split: str,
    output_dir: Path,
    k: int = 10,
    random_seed: int = 0,
) -> None:
    """Persists random predictions with its associated ground truth.

    It makes it easy to assess how well the model is doing. Also it
    makes it possible to spot false positives and false negatives.
    """
    split_dir = input_dir_yolov8_dataset / split
    assert split_dir.exists(), f"The directory {split_dir} does not exist."

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    labels_filepaths = list(labels_dir.glob("*.txt"))
    labels_sample_filepaths = random.Random(random_seed).sample(labels_filepaths, k)
    images_sample_filepaths = [
        images_dir / f"{fp.stem}.png" for fp in labels_sample_filepaths
    ]

    save_dir = output_dir / split
    save_dir.mkdir(exist_ok=True, parents=True)

    for idx, label_filepath in enumerate(tqdm(labels_sample_filepaths)):
        image_filepath = images_sample_filepaths[idx]
        save_detailed_prediction(
            model=model,
            output_dir=save_dir,
            label_filepath=label_filepath,
            image_filepath=image_filepath,
        )


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logging.basicConfig(level=args["loglevel"].upper())
    if not validate_parsed_args(args):
        logging.error(f"Could not validate the parsed args: {args}")
        exit(1)
    else:
        logging.info(args)
        model = load_trained_model(args["weights_filepath"])
        model.info()

        persist_random_predictions(
            model=model,
            input_dir_yolov8_dataset=args["input_dir_yolov8_dataset"],
            split=args["split"],
            output_dir=args["output_dir"],
            k=args["k"],
            random_seed=args["random_seed"],
        )

        exit(0)
