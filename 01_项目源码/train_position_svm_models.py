from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from license_plate_app import ALPHANUMS, LETTERS, PROVINCES
from train_character_model import load_dataset, split_indices


GROUPS = {
    "province": PROVINCES,
    "letter": LETTERS,
    "alphanum": ALPHANUMS,
}


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


def train_group(
    group_name: str,
    allowed_chars: set[str],
    features: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str],
    output_dir: Path,
    validation_ratio: float,
    seed: int,
) -> dict[str, object]:
    selected_ids = [label_id for label_id, label in label_names.items() if label in allowed_chars]
    selected_set = set(selected_ids)
    selected_indices = [index for index, label in enumerate(labels.tolist()) if label in selected_set]
    if not selected_indices:
        raise ValueError(f"No samples for group: {group_name}")

    group_features = features[selected_indices]
    original_labels = labels[selected_indices]
    local_labels = {original_id: local_id for local_id, original_id in enumerate(selected_ids)}
    reverse_labels = {local_id: label_names[original_id] for original_id, local_id in local_labels.items()}
    group_labels = np.array([local_labels[int(label)] for label in original_labels], dtype=np.int32)

    train_indices, valid_indices = split_indices(group_labels, validation_ratio, seed)
    svm = train_svm(group_features[train_indices], group_labels[train_indices])
    train_accuracy = evaluate(svm, group_features[train_indices], group_labels[train_indices])
    valid_accuracy = evaluate(svm, group_features[valid_indices], group_labels[valid_indices]) if valid_indices else 0.0

    model_path = output_dir / f"plate_{group_name}_svm.xml"
    svm.save(str(model_path))

    return {
        "model": str(model_path),
        "labels": {str(key): value for key, value in reverse_labels.items()},
        "classes": len(reverse_labels),
        "samples": int(len(group_labels)),
        "train_accuracy": train_accuracy,
        "validation_accuracy": valid_accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train position-specific SVM models for license plate recognition.")
    parser.add_argument("-d", "--dataset", type=Path, default=Path("train_chars_augmented"), help="Training dataset folder.")
    parser.add_argument("-o", "--output", type=Path, default=Path("models"), help="Output model folder.")
    parser.add_argument("-m", "--manifest", type=Path, default=Path("models/plate_position_models.json"), help="Output manifest path.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features, labels, label_names, skipped = load_dataset(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    groups = {}
    for group_name, allowed_chars in GROUPS.items():
        info = train_group(group_name, allowed_chars, features, labels, label_names, args.output, args.validation_ratio, args.seed)
        groups[group_name] = info
        print(
            f"{group_name}: classes={info['classes']}, samples={info['samples']}, "
            f"train={info['train_accuracy']:.3f}, valid={info['validation_accuracy']:.3f}"
        )

    args.manifest.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "skipped": len(skipped),
                "groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Manifest saved to: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
