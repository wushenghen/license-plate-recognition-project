from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


def safe_label(label: str) -> str:
    return "".join(char if char not in '\\/:*?"<>|' else "_" for char in label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train_chars from segmented character images and a label CSV.")
    parser.add_argument("-s", "--source", type=Path, default=Path("output_train_ready"), help="Segmentation output folder.")
    parser.add_argument("-l", "--labels", type=Path, default=Path("plate_labels.csv"), help="CSV file: folder,text")
    parser.add_argument("-o", "--output", type=Path, default=Path("train_chars"), help="Training dataset output folder.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        print(f"Source folder not found: {args.source}")
        return 1
    if not args.labels.exists():
        print(f"Label file not found: {args.labels}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
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
                print(f"Skip {folder}: chars={len(char_files)}, text={text}, text_len={len(text)}")
                skipped += 1
                continue

            for index, (char_file, label) in enumerate(zip(char_files, text), start=1):
                if label in {"?", "？", "_"}:
                    continue
                label_dir = args.output / safe_label(label)
                label_dir.mkdir(parents=True, exist_ok=True)
                target = label_dir / f"{folder}_{index:02d}.png"
                shutil.copy2(char_file, target)
                copied += 1

    print(f"Copied samples: {copied}")
    print(f"Skipped rows: {skipped}")
    print(f"Training folder: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
