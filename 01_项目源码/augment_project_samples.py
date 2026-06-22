from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print("OpenCV is required. Install it with: pip install opencv-python numpy")
    raise SystemExit(1) from exc


def safe_label(label: str) -> str:
    return "".join(char if char not in '\\/:*?"<>|' else "_" for char in label)


def read_gray(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    result = image.copy()
    if rng.random() < 0.45:
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1) if rng.random() < 0.5 else cv2.erode(result, kernel, iterations=1)

    dx = rng.randint(-1, 1)
    dy = rng.randint(-2, 2)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    result = cv2.warpAffine(result, matrix, (result.shape[1], result.shape[0]), borderValue=0)

    if rng.random() < 0.25:
        result = cv2.GaussianBlur(result, (3, 3), 0)
        _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add repeated augmented project samples into train_chars.")
    parser.add_argument("-s", "--source", type=Path, default=Path("output_final_check"), help="Segmented output folder.")
    parser.add_argument("-l", "--labels", type=Path, default=Path("project_plate_labels.csv"), help="CSV file: folder,text")
    parser.add_argument("-o", "--output", type=Path, default=Path("train_chars_final"), help="Training dataset folder.")
    parser.add_argument("--copies", type=int, default=30, help="Augmented copies per project character.")
    parser.add_argument("--seed", type=int, default=2027, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    copied = 0
    skipped = 0

    with args.labels.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            folder = (row.get("folder") or "").strip()
            text = (row.get("text") or "").strip().replace(" ", "")
            if not folder or not text:
                skipped += 1
                continue

            char_files = sorted((args.source / folder).glob("char_*.png"))
            if len(char_files) != len(text):
                print(f"Skip {folder}: chars={len(char_files)}, text_len={len(text)}")
                skipped += 1
                continue

            for index, (char_file, label) in enumerate(zip(char_files, text), start=1):
                image = read_gray(char_file)
                if image is None:
                    skipped += 1
                    continue
                label_dir = args.output / safe_label(label)
                label_dir.mkdir(parents=True, exist_ok=True)
                for copy_index in range(args.copies):
                    generated = augment(image, rng)
                    target = label_dir / f"project_{folder}_{index:02d}_{copy_index:03d}.png"
                    cv2.imencode(".png", generated)[1].tofile(str(target))
                    copied += 1

    print(f"Augmented project samples: {copied}")
    print(f"Skipped rows/items: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
