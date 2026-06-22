from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print("OpenCV is required. Install it with: pip install opencv-python numpy")
    raise SystemExit(1) from exc

from license_plate_app import extract_characters, build_clean_binary, order_points


PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
]

ALPHANUMS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_plate_text(image_path: Path) -> str | None:
    parts = image_path.stem.split("-")
    if len(parts) < 5:
        return None

    indexes = [int(item) for item in parts[4].split("_")]
    if len(indexes) != 7:
        return None

    try:
        return PROVINCES[indexes[0]] + "".join(ALPHANUMS[index] for index in indexes[1:])
    except IndexError:
        return None


def parse_vertices(image_path: Path) -> np.ndarray | None:
    parts = image_path.stem.split("-")
    if len(parts) < 4:
        return None

    try:
        points = []
        for point_text in parts[3].split("_"):
            x_text, y_text = point_text.split("&")
            points.append([float(x_text), float(y_text)])
        if len(points) != 4:
            return None
        return np.array(points, dtype="float32")
    except ValueError:
        return None


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def crop_plate_by_vertices(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    ordered = order_points(vertices)
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    height_left = np.linalg.norm(ordered[3] - ordered[0])

    target_width = max(int(max(width_top, width_bottom)), 1)
    target_height = max(int(max(height_right, height_left)), 1)
    target = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, target)
    plate = cv2.warpPerspective(image, matrix, (target_width, target_height))
    if plate.shape[0] > plate.shape[1]:
        plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)
    return plate


def safe_label(label: str) -> str:
    return "".join(char if char not in '\\/:*?"<>|' else "_" for char in label)


def iter_images(dataset_root: Path, subsets: list[str]) -> list[Path]:
    images: list[Path] = []
    for subset in subsets:
        subset_dir = dataset_root / subset
        if not subset_dir.exists():
            print(f"Skip missing subset: {subset_dir}")
            continue
        images.extend(
            path for path in subset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(images)


def save_characters(
    char_images: list[np.ndarray],
    plate_text: str,
    image_path: Path,
    output_dir: Path,
    counters: Counter[str],
    max_per_class: int,
) -> int:
    saved = 0
    for index, (char_image, label) in enumerate(zip(char_images, plate_text), start=1):
        if counters[label] >= max_per_class:
            continue
        label_dir = output_dir / safe_label(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        target = label_dir / f"{image_path.parent.name}_{image_path.stem[:18]}_{index:02d}.png"
        cv2.imencode(".png", char_image)[1].tofile(str(target))
        counters[label] += 1
        saved += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CCPD2019 images into train_chars character folders.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=Path(r"F:\BaiduNetdiskDownload\CCPD2019\CCPD2019"),
        help="CCPD2019 root folder.",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("train_chars_ccpd"), help="Output train_chars folder.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum source images to scan.")
    parser.add_argument("--max-per-class", type=int, default=800, help="Maximum saved samples per character class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for source image sampling.")
    parser.add_argument("--no-shuffle", action="store_true", help="Keep source image order instead of random sampling.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["ccpd_base"],
        help="CCPD subsets to use, for example: ccpd_base ccpd_blur ccpd_rotate.",
    )
    parser.add_argument("--report", type=Path, default=Path("ccpd_convert_report.csv"), help="Conversion report CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.exists():
        print(f"Dataset folder not found: {args.dataset}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.dataset, args.subsets)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(image_paths)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    counters: Counter[str] = Counter()
    scanned = 0
    converted = 0
    saved_chars = 0
    rows: list[dict[str, str]] = []

    for image_path in image_paths:
        scanned += 1
        plate_text = parse_plate_text(image_path)
        vertices = parse_vertices(image_path)
        image = read_image(image_path)
        status = "ok"

        if plate_text is None:
            status = "bad_label"
        elif vertices is None:
            status = "bad_vertices"
        elif image is None:
            status = "read_failed"
        else:
            plate = crop_plate_by_vertices(image, vertices)
            clean_binary = build_clean_binary(plate)
            char_images, _ = extract_characters(clean_binary)
            if len(char_images) != len(plate_text):
                status = f"char_count_{len(char_images)}"
            else:
                saved = save_characters(char_images, plate_text, image_path, args.output, counters, args.max_per_class)
                saved_chars += saved
                converted += 1

        rows.append({"file": str(image_path), "text": plate_text or "", "status": status})
        if scanned % 500 == 0:
            print(f"Scanned {scanned}, converted {converted}, saved chars {saved_chars}")

    with args.report.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file", "text", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scanned images: {scanned}")
    print(f"Converted plates: {converted}")
    print(f"Saved characters: {saved_chars}")
    print(f"Classes: {len(counters)}")
    print(f"Output: {args.output.resolve()}")
    print(f"Report: {args.report.resolve()}")
    return 0 if converted > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
