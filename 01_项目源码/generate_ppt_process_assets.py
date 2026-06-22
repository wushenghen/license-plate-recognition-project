from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from license_plate_app import (
    build_clean_binary,
    extract_characters,
    locate_blue_plate,
    order_points,
)


PROJECT = Path(r"E:\pycharm code\pycharm python study\final program")
ASSETS = PROJECT / "final_delivery" / "ppt_assets"


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".jpg" if path.suffix.lower() in {".jpg", ".jpeg"} else ".png"
    cv2.imencode(suffix, image)[1].tofile(str(path))


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def locate_with_debug(image: np.ndarray) -> tuple[np.ndarray | None, dict[str, object]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 90, 45], dtype=np.uint8)
    upper_blue = np.array([132, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)), iterations=2)
    mask_closed = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug = {
        "mask": mask,
        "mask_closed": mask_closed,
        "candidate_count": len(contours),
        "best_box": None,
        "best_score": 0.0,
    }

    best_plate = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, center_y), (width, height), _ = rect
        long_side = max(width, height)
        short_side = min(width, height)
        if short_side <= 0:
            continue

        ratio = long_side / short_side
        area = long_side * short_side
        if not (1.5 <= ratio <= 5.8 and area >= 500):
            continue
        if center_y < image.shape[0] * 0.30:
            continue

        target_width = max(int(long_side), 1)
        target_height = max(int(short_side), 1)
        box = cv2.boxPoints(rect)
        src = order_points(np.array(box, dtype="float32"))
        dst = np.array(
            [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
            dtype="float32",
        )
        transform = cv2.getPerspectiveTransform(src, dst)
        plate = cv2.warpPerspective(image, transform, (target_width, target_height))
        if plate.shape[0] > plate.shape[1]:
            plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)

        plate_hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        blue_ratio = float(np.mean(cv2.inRange(plate_hsv, lower_blue, upper_blue) > 0))
        white_ratio = float(np.mean((plate_hsv[:, :, 1] < 95) & (plate_hsv[:, :, 2] > 135)))
        vertical_preference = 0.75 + 0.5 * min(max(center_y / image.shape[0], 0.0), 1.0)
        area_preference = min(area / max(float(image.shape[0] * image.shape[1]) * 0.035, 1.0), 1.0)
        if blue_ratio < 0.22 or white_ratio < 0.018:
            continue

        score = ratio * blue_ratio * (1.0 + min(white_ratio, 0.22) * 8.0) * vertical_preference * (0.45 + area_preference)
        if score > debug["best_score"]:
            debug["best_score"] = score
            debug["best_box"] = box.astype(int).tolist()
            best_plate = plate

    return best_plate, debug


def save_process_assets() -> dict[str, object]:
    source = PROJECT / "test_images" / "2.jpg"
    image = read_image(source)
    plate, debug = locate_with_debug(image)
    if plate is None:
        plate = locate_blue_plate(image)
    if plate is None:
        raise RuntimeError("sample plate not found")

    candidate = image.copy()
    if debug["best_box"] is not None:
        box = np.array(debug["best_box"], dtype=np.int32)
        cv2.polylines(candidate, [box], True, (0, 255, 0), 3)

    clean_binary = build_clean_binary(plate)
    characters, segmentation = extract_characters(clean_binary)
    gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    write_image(ASSETS / "01_original.jpg", image)
    write_image(ASSETS / "02_hsv_mask.png", debug["mask"])
    write_image(ASSETS / "03_morph_mask.png", debug["mask_closed"])
    write_image(ASSETS / "04_candidate_box.jpg", candidate)
    write_image(ASSETS / "05_plate_corrected.png", plate)
    write_image(ASSETS / "06_gray_plate.png", gray_plate)
    write_image(ASSETS / "07_binary_clean.png", clean_binary)
    write_image(ASSETS / "08_segmentation.png", segmentation)

    char_paths = []
    for index, char_image in enumerate(characters, start=1):
        char_path = ASSETS / f"char_{index:02d}.png"
        write_image(char_path, char_image)
        char_paths.append(str(char_path))

    training_examples: dict[str, list[str]] = {}
    for label in ["京", "浙", "川", "A", "B", "1", "8", "9"]:
        label_dir = PROJECT / "train_chars_expanded_final" / label
        files = sorted(label_dir.glob("*.png"))[:6]
        training_examples[label] = [str(path) for path in files]

    stats = {
        "sample": str(source),
        "candidate_count": debug["candidate_count"],
        "best_score": debug["best_score"],
        "characters": len(characters),
        "char_paths": char_paths,
        "training_examples": training_examples,
        "pipeline_assets": {
            "original": str(ASSETS / "01_original.jpg"),
            "hsv_mask": str(ASSETS / "02_hsv_mask.png"),
            "morph_mask": str(ASSETS / "03_morph_mask.png"),
            "candidate": str(ASSETS / "04_candidate_box.jpg"),
            "plate": str(ASSETS / "05_plate_corrected.png"),
            "gray": str(ASSETS / "06_gray_plate.png"),
            "binary": str(ASSETS / "07_binary_clean.png"),
            "segmentation": str(ASSETS / "08_segmentation.png"),
        },
    }
    (ASSETS / "asset_manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    info = save_process_assets()
    print(json.dumps(info, ensure_ascii=False, indent=2))
