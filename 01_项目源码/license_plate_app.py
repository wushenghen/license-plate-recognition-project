from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print("OpenCV is required. Install it with: pip install opencv-python numpy")
    raise SystemExit(1) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHAR_SIZE = (20, 40)
PROVINCES = set("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新")
LETTERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
ALPHANUMS = LETTERS | set("0123456789")


@dataclass
class PlateResult:
    image_path: Path
    plate_image: np.ndarray | None
    clean_binary: np.ndarray | None
    segmentation_preview: np.ndarray | None
    characters: list[np.ndarray]
    text: str
    message: str


class CharacterRecognizer:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.mode = "nearest" if model_path.suffix.lower() == ".npz" else "svm"
        self.model = None
        self.position_models: dict[str, cv2.ml_SVM] = {}
        self.position_labels: dict[str, dict[int, str]] = {}
        self.features: np.ndarray | None = None
        self.targets: np.ndarray | None = None

        if model_path.suffix.lower() == ".json":
            self.mode = "position"
            manifest = json.loads(model_path.read_text(encoding="utf-8"))
            for group_name, info in manifest["groups"].items():
                group_model = Path(info["model"])
                if not group_model.exists():
                    group_model = model_path.parent / group_model.name
                self.position_models[group_name] = cv2.ml.SVM_load(str(group_model))
                self.position_labels[group_name] = {int(key): value for key, value in info["labels"].items()}
            self.labels = {}
        elif self.mode == "nearest":
            data = np.load(model_path, allow_pickle=True)
            self.features = data["features"].astype(np.float32)
            self.targets = data["labels"].astype(np.int32)
            label_names = [str(label) for label in data["label_names"].tolist()]
            self.labels = {index: label for index, label in enumerate(label_names)}
        else:
            self.model = cv2.ml.SVM_load(str(model_path))
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            self.labels = {int(key): value for key, value in data["labels"].items()}

    @classmethod
    def load_if_available(cls, model_path: Path, labels_path: Path) -> CharacterRecognizer | None:
        position_manifest = Path("models/plate_position_models.json")
        if model_path == position_manifest and model_path.exists():
            return cls(model_path, labels_path)

        if model_path.exists() and (model_path.suffix.lower() == ".npz" or labels_path.exists()):
            return cls(model_path, labels_path)

        if position_manifest.exists():
            return cls(position_manifest, labels_path)

        nearest_model = Path("models/plate_char_nearest.npz")
        if nearest_model.exists():
            return cls(nearest_model, labels_path)

        fallback_model = Path("models/plate_char_svm.xml")
        fallback_labels = Path("models/plate_char_labels.json")
        if fallback_model.exists() and fallback_labels.exists():
            return cls(fallback_model, fallback_labels)
        return None

    def predict(self, character: np.ndarray, position: int | None = None) -> str:
        feature = extract_hog_feature(character).reshape(1, -1)

        if self.mode == "position":
            label = self.predict_position_svm(feature, position)
            return refine_position_prediction(label, character, position)
        elif self.mode == "nearest":
            label_id = self.predict_nearest(feature.reshape(-1), position)
        else:
            assert self.model is not None
            _, result = self.model.predict(feature)
            label_id = int(result[0, 0])
        return refine_position_prediction(self.labels.get(label_id, "?"), character, position)

    def predict_position_svm(self, feature: np.ndarray, position: int | None) -> str:
        if position == 0:
            group_name = "province"
        elif position == 1:
            group_name = "letter"
        else:
            group_name = "alphanum"

        model = self.position_models[group_name]
        labels = self.position_labels[group_name]
        _, result = model.predict(feature)
        return labels.get(int(result[0, 0]), "?")

    def predict_nearest(self, feature: np.ndarray, position: int | None) -> int:
        assert self.features is not None
        assert self.targets is not None

        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = (feature / norm).astype(np.float32)

        allowed = self.allowed_label_ids(position)
        if allowed:
            mask = np.isin(self.targets, np.array(list(allowed), dtype=np.int32))
            features = self.features[mask]
            targets = self.targets[mask]
        else:
            features = self.features
            targets = self.targets

        distances = 2.0 - 2.0 * np.dot(features, feature)
        return int(targets[int(np.argmin(distances))])

    def allowed_label_ids(self, position: int | None) -> set[int]:
        if position is None:
            return set()
        if position == 0:
            allowed_chars = PROVINCES
        elif position == 1:
            allowed_chars = LETTERS
        else:
            allowed_chars = ALPHANUMS

        return {label_id for label_id, label in self.labels.items() if label in allowed_chars}

    def predict_all(self, characters: list[np.ndarray]) -> str:
        return normalize_plate_prediction("".join(self.predict(character, index) for index, character in enumerate(characters)))


def make_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=CHAR_SIZE,
        _blockSize=(10, 10),
        _blockStride=(5, 5),
        _cellSize=(5, 5),
        _nbins=9,
    )


def normalize_character(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border) > 127:
        binary = 255 - binary

    points = np.column_stack(np.where(binary > 0))
    if len(points) == 0:
        return np.zeros((CHAR_SIZE[1], CHAR_SIZE[0]), dtype=np.uint8)

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)
    character = binary[y_min:y_max + 1, x_min:x_max + 1]

    height, width = character.shape
    scale = min((CHAR_SIZE[0] - 4) / max(width, 1), (CHAR_SIZE[1] - 4) / max(height, 1))
    resized = cv2.resize(
        character,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((CHAR_SIZE[1], CHAR_SIZE[0]), dtype=np.uint8)
    y = (CHAR_SIZE[1] - resized.shape[0]) // 2
    x = (CHAR_SIZE[0] - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def extract_hog_feature(image: np.ndarray) -> np.ndarray:
    normalized = normalize_character(image)
    feature = make_hog().compute(normalized)
    return feature.reshape(-1).astype(np.float32)


def normalize_plate_prediction(text: str) -> str:
    if len(text) <= 7:
        return text

    candidates: list[tuple[int, str]] = []
    for size in (7, 8):
        if len(text) < size:
            continue
        for start in range(0, len(text) - size + 1):
            candidate = text[start:start + size]
            score = 0
            if candidate[0] in PROVINCES:
                score += 6
            if len(candidate) > 1 and candidate[1] in LETTERS:
                score += 5
            score += sum(1 for char in candidate[2:] if char in ALPHANUMS)
            score -= sum(1 for char in candidate if char == "?")
            if size == 7:
                score += 2
            candidates.append((score, candidate))

    if not candidates:
        return text
    return max(candidates, key=lambda item: item[0])[1]


def refine_position_prediction(label: str, character: np.ndarray, position: int | None) -> str:
    if position is None or position < 2 or label != "L":
        return label

    image = normalize_character(character)
    points = np.column_stack(np.where(image > 0))
    if len(points) == 0:
        return label

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)
    width = int(x_max - x_min + 1)
    height = int(y_max - y_min + 1)
    fill_ratio = float(np.mean(image > 0))

    if width <= 6 and height >= 24 and fill_ratio < 0.18:
        return "1"
    return label


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def choose_default_input() -> Path:
    local_images = Path("test_images")
    if local_images.exists():
        return local_images

    project_images = Path(r"E:\pycharm code\pycharm python study\final program\test_images")
    if project_images.exists():
        return project_images

    return local_images


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        return []
    return sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def locate_blue_plate(image: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 90, 45], dtype=np.uint8)
    upper_blue = np.array([132, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_plate = None
    best_score = 0.0

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
        if score > best_score:
            best_score = score
            best_plate = plate

    return best_plate


def build_clean_binary(plate: np.ndarray) -> np.ndarray:
    if plate.shape[1] < 220:
        scale = 220.0 / plate.shape[1]
        plate = cv2.resize(plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    height, width = binary.shape
    return binary[int(height * 0.12):int(height * 0.88), int(width * 0.03):int(width * 0.97)]


def merge_close_segments(segments: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def merge_leading_province_fragments(segments: list[tuple[int, int]], width: int) -> list[tuple[int, int]]:
    if len(segments) < 3:
        return segments

    tiny_width = max(8, int(width * 0.055))
    province_limit = int(width * 0.26)
    leading: list[tuple[int, int]] = []
    index = 0

    while index < len(segments):
        start, end = segments[index]
        if start > province_limit or end - start > tiny_width:
            break
        leading.append((start, end))
        index += 1

    if len(leading) < 2:
        return segments

    return [(leading[0][0], leading[-1][1]), *segments[index:]]


def split_wide_segments(segments: list[tuple[int, int]], binary: np.ndarray, expected_count: int = 8) -> list[tuple[int, int]]:
    if len(segments) >= expected_count or not segments:
        return segments

    split_segments = segments[:]
    while len(split_segments) < expected_count:
        widths = [end - start for start, end in split_segments]
        median_width = float(np.median(widths))
        wide_index = max(range(len(split_segments)), key=lambda index: widths[index])
        start, end = split_segments[wide_index]
        width = end - start

        if width < max(12, median_width * 1.25):
            break

        region = binary[:, start:end]
        projection = np.sum(region == 255, axis=0)
        mid = width // 2
        search_radius = max(3, width // 5)
        left = max(2, mid - search_radius)
        right = min(width - 2, mid + search_radius)
        if right <= left:
            break

        split_at = left + int(np.argmin(projection[left:right]))
        if split_at < 4 or width - split_at < 4:
            split_at = mid
        split_segments[wide_index:wide_index + 1] = [(start, start + split_at), (start + split_at, end)]

    return split_segments


def trim_to_expected_plate_count(candidates: list[tuple[np.ndarray, tuple[int, int, int, int], float]]) -> list[tuple[np.ndarray, tuple[int, int, int, int], float]]:
    if len(candidates) <= 7:
        return candidates

    trimmed = candidates[:]
    while len(trimmed) > 7:
        edge_indices = [0, len(trimmed) - 1]
        weak_edges = [index for index in edge_indices if trimmed[index][2] < 0.40]
        if weak_edges:
            drop_index = min(weak_edges, key=lambda index: trimmed[index][2])
        else:
            drop_index = min(range(len(trimmed)), key=lambda index: trimmed[index][2])
        trimmed.pop(drop_index)
    return trimmed


def extract_characters(clean_binary: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    bridged = cv2.morphologyEx(
        clean_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
    )

    height, width = bridged.shape
    projection = np.sum(bridged == 255, axis=0)
    threshold = max(float(np.mean(projection) * 0.35), height * 0.08)

    segments: list[tuple[int, int]] = []
    in_char = False
    start = 0
    for index, count in enumerate(projection):
        if count > threshold and not in_char:
            start = index
            in_char = True
        elif count <= threshold and in_char:
            segments.append((start, index))
            in_char = False

    if in_char:
        segments.append((start, width))

    segments = merge_close_segments(segments, max(2, int(width * 0.012)))
    segments = merge_leading_province_fragments(segments, width)
    segments = split_wide_segments(segments, clean_binary, expected_count=8)

    preview = cv2.cvtColor(clean_binary, cv2.COLOR_GRAY2BGR)
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int], float]] = []

    for start, end in segments:
        if end - start < max(3, int(width * 0.01)):
            continue

        char_region = clean_binary[:, start:end]
        foreground = np.where(char_region == 255)
        if len(foreground[0]) == 0:
            continue

        y_min, y_max = int(foreground[0].min()), int(foreground[0].max())
        x_min, x_max = int(foreground[1].min()), int(foreground[1].max())
        char_width = x_max - x_min + 1
        char_height = y_max - y_min + 1

        if char_height < height * 0.42 or char_width < 3:
            continue
        if start > width * 0.18 and char_width < max(3, int(width * 0.018)):
            continue
        if char_width > width * 0.28:
            continue

        char_image = char_region[y_min:y_max + 1, x_min:x_max + 1]
        char_image = normalize_character(char_image)
        fill_ratio = float(np.mean(char_image > 0))
        width_score = min(char_width / max(width * 0.055, 1), 1.0)
        height_score = min(char_height / max(height * 0.70, 1), 1.0)
        density_score = 1.0 - min(abs(fill_ratio - 0.28) / 0.28, 1.0)
        score = width_score * 0.35 + height_score * 0.45 + density_score * 0.20
        candidates.append((char_image, (start + x_min, y_min, start + x_max, y_max), score))

    candidates = trim_to_expected_plate_count(candidates)
    characters = [item[0] for item in candidates]
    for _, (x_min, y_min, x_max, y_max), _ in candidates:
        cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    return characters, preview


def process_image(image_path: Path, recognizer: CharacterRecognizer | None) -> PlateResult:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return PlateResult(image_path, None, None, None, [], "", "image read failed")

    plate = locate_blue_plate(image)
    if plate is None:
        return PlateResult(image_path, None, None, None, [], "", "blue plate not found")

    clean_binary = build_clean_binary(plate)
    characters, preview = extract_characters(clean_binary)
    text = recognizer.predict_all(characters) if recognizer is not None and characters else ""
    return PlateResult(image_path, plate, clean_binary, preview, characters, text, "ok")


def safe_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)


def save_result(result: PlateResult, output_dir: Path) -> None:
    image_output = output_dir / safe_stem(result.image_path)
    image_output.mkdir(parents=True, exist_ok=True)

    if result.plate_image is not None:
        cv2.imencode(".png", result.plate_image)[1].tofile(str(image_output / "plate.png"))
    if result.clean_binary is not None:
        cv2.imencode(".png", result.clean_binary)[1].tofile(str(image_output / "binary.png"))
    if result.segmentation_preview is not None:
        cv2.imencode(".png", result.segmentation_preview)[1].tofile(str(image_output / "segmentation.png"))

    for index, char_image in enumerate(result.characters, start=1):
        cv2.imencode(".png", char_image)[1].tofile(str(image_output / f"char_{index:02d}.png"))

    if result.text:
        (image_output / "result.txt").write_text(f"text={result.text}\n", encoding="utf-8")


def show_result(result: PlateResult) -> None:
    if result.plate_image is not None:
        cv2.imshow("Plate", result.plate_image)
    if result.clean_binary is not None:
        cv2.imshow("Binary", result.clean_binary)
    if result.segmentation_preview is not None:
        cv2.imshow("Segmentation", result.segmentation_preview)
    for index, char_image in enumerate(result.characters, start=1):
        cv2.imshow(f"Char {index}", char_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="License plate localization, segmentation, and trained-model recognition.")
    parser.add_argument("-i", "--input", type=Path, default=choose_default_input(), help="Image file or image folder.")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="Folder for processed results.")
    parser.add_argument("--model", type=Path, default=Path("models/plate_position_models.json"), help="Trained model path.")
    parser.add_argument("--labels", type=Path, default=Path("models/plate_char_labels.json"), help="Model label file path.")
    parser.add_argument("--show", action="store_true", help="Show OpenCV windows for each processed image.")
    parser.add_argument("--no-save", action="store_true", help="Do not save processed images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = list_images(args.input)
    if not image_paths:
        print(f"No images found: {args.input}")
        return 1

    recognizer = CharacterRecognizer.load_if_available(args.model, args.labels)
    if recognizer is None:
        print("No trained model found. The program will segment characters only.")
        print(f"Expected model: {args.model}")
        print(f"Expected labels: {args.labels}")

    print(f"Processing {len(image_paths)} image(s).")
    ok_count = 0

    for image_path in image_paths:
        result = process_image(image_path, recognizer)
        if result.message == "ok":
            ok_count += 1

        text = f", text={result.text}" if result.text else ""
        print(f"{image_path.name}: {result.message}, chars={len(result.characters)}{text}")

        if not args.no_save:
            save_result(result, args.output)
        if args.show:
            show_result(result)

    print(f"Done. Plates found: {ok_count}/{len(image_paths)}")
    if not args.no_save:
        print(f"Results saved to: {args.output.resolve()}")
    return 0 if ok_count > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
