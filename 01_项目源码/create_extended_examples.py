from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from convert_ccpd_to_train_chars import IMAGE_EXTENSIONS, parse_plate_text


DEFAULT_SUBSETS = [
    "ccpd_base",
    "ccpd_blur",
    "ccpd_challenge",
    "ccpd_rotate",
    "ccpd_tilt",
    "ccpd_weather",
]


def iter_images(dataset_root: Path, subsets: list[str]) -> list[Path]:
    images: list[Path] = []
    for subset in subsets:
        subset_dir = dataset_root / subset
        if not subset_dir.exists():
            print(f"Skip missing subset: {subset_dir}")
            continue
        images.extend(
            path
            for path in subset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return images


def load_existing_labels(label_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not label_path.exists():
        return labels

    with label_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            folder = (row.get("folder") or "").strip()
            text = (row.get("text") or "").strip()
            if folder and text:
                labels[folder] = text
    return labels


def copy_original_examples(source_dir: Path, output_dir: Path, labels: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_path in sorted(source_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target_name = f"original_{image_path.name}"
        shutil.copy2(image_path, output_dir / target_name)
        key = image_path.stem
        rows.append(
            {
                "file": target_name,
                "text": labels.get(key, ""),
                "source": "project_test_images",
                "source_file": str(image_path),
            }
        )
    return rows


def copy_ccpd_examples(
    dataset_root: Path,
    output_dir: Path,
    count: int,
    seed: int,
    subsets: list[str],
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    image_paths = iter_images(dataset_root, subsets)
    rng.shuffle(image_paths)

    rows: list[dict[str, str]] = []
    used_names: set[str] = set()

    for image_path in image_paths:
        plate_text = parse_plate_text(image_path)
        if not plate_text:
            continue

        prefix = f"ccpd_{len(rows) + 1:03d}_{image_path.parent.name}"
        target_name = f"{prefix}_{image_path.name}"
        if target_name in used_names:
            continue
        used_names.add(target_name)
        shutil.copy2(image_path, output_dir / target_name)
        rows.append(
            {
                "file": target_name,
                "text": plate_text,
                "source": image_path.parent.name,
                "source_file": str(image_path),
            }
        )
        if len(rows) >= count:
            break

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an extended example image set for the license plate project.")
    parser.add_argument("--dataset", type=Path, default=Path(r"F:\BaiduNetdiskDownload\CCPD2019\CCPD2019"))
    parser.add_argument("--project-test-images", type=Path, default=Path("test_images"))
    parser.add_argument("--project-labels", type=Path, default=Path("project_plate_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("test_images_extended"))
    parser.add_argument("--labels-output", type=Path, default=Path("extended_plate_labels.csv"))
    parser.add_argument("--ccpd-count", type=int, default=82)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.project_test_images.exists():
        print(f"Project test image folder not found: {args.project_test_images}")
        return 1
    if not args.dataset.exists():
        print(f"Dataset folder not found: {args.dataset}")
        return 1

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    labels = load_existing_labels(args.project_labels)
    rows = copy_original_examples(args.project_test_images, args.output, labels)
    rows.extend(copy_ccpd_examples(args.dataset, args.output, args.ccpd_count, args.seed, args.subsets))

    with args.labels_output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file", "text", "source", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Original examples: {sum(row['source'] == 'project_test_images' for row in rows)}")
    print(f"CCPD examples: {sum(row['source'] != 'project_test_images' for row in rows)}")
    print(f"Total examples: {len(rows)}")
    print(f"Output folder: {args.output.resolve()}")
    print(f"Label file: {args.labels_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
