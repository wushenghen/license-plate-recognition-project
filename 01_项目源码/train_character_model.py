from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print("OpenCV is required. Install it with: pip install opencv-python numpy")
    raise SystemExit(1) from exc

from license_plate_app import IMAGE_EXTENSIONS, extract_hog_feature


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, str], list[tuple[Path, str]]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    label_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not label_dirs:
        raise ValueError(f"No label folders found in: {dataset_dir}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    label_names: dict[int, str] = {}
    skipped: list[tuple[Path, str]] = []

    for label_id, label_dir in enumerate(label_dirs):
        label_names[label_id] = label_dir.name
        image_paths = sorted(
            path for path in label_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            image = read_image(image_path)
            if image is None:
                skipped.append((image_path, "read failed"))
                continue
            features.append(extract_hog_feature(image))
            labels.append(label_id)

    if not features:
        raise ValueError("No training images were loaded.")

    return (
        np.array(features, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        label_names,
        skipped,
    )


def split_indices(labels: np.ndarray, validation_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_label.setdefault(label, []).append(index)

    train_indices: list[int] = []
    valid_indices: list[int] = []
    for indices in by_label.values():
        rng.shuffle(indices)
        valid_count = int(round(len(indices) * validation_ratio))
        if len(indices) >= 5:
            valid_count = max(1, valid_count)
        valid_indices.extend(indices[:valid_count])
        train_indices.extend(indices[valid_count:])

    return train_indices, valid_indices


def train_svm(features: np.ndarray, labels: np.ndarray) -> cv2.ml_SVM:
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(2.5)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
    svm.train(features, cv2.ml.ROW_SAMPLE, labels)
    return svm


def evaluate(svm: cv2.ml_SVM, features: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    _, predictions = svm.predict(features)
    predictions = predictions.reshape(-1).astype(np.int32)
    return float(np.mean(predictions == labels))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a license plate character recognition model.")
    parser.add_argument("-d", "--dataset", type=Path, default=Path("train_chars"), help="Training dataset folder.")
    parser.add_argument("-m", "--model", type=Path, default=Path("models/plate_char_svm.xml"), help="Output model path.")
    parser.add_argument("-l", "--labels", type=Path, default=Path("models/plate_char_labels.json"), help="Output labels path.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for validation split.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features, labels, label_names, skipped = load_dataset(args.dataset)

    train_indices, valid_indices = split_indices(labels, args.validation_ratio, args.seed)
    if not train_indices:
        print("Not enough training samples.")
        return 1

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    svm = train_svm(train_features, train_labels)

    train_accuracy = evaluate(svm, train_features, train_labels)
    valid_accuracy = evaluate(svm, features[valid_indices], labels[valid_indices]) if valid_indices else 0.0

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(args.model))
    args.labels.write_text(
        json.dumps({"labels": label_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Classes: {len(label_names)}")
    print(f"Samples: {len(labels)}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(valid_indices)}")
    print(f"Train accuracy: {train_accuracy:.3f}")
    if valid_indices:
        print(f"Validation accuracy: {valid_accuracy:.3f}")
    if skipped:
        print(f"Skipped images: {len(skipped)}")
    print(f"Model saved to: {args.model.resolve()}")
    print(f"Labels saved to: {args.labels.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
