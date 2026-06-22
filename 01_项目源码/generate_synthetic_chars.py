from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


PROVINCES = list("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新")
ALPHANUMS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
DEFAULT_FONTS = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simsunb.ttf",
]
LATIN_FONTS = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyhbd.ttc",
]


def safe_label(label: str) -> str:
    return "".join(char if char not in '\\/:*?"<>|' else "_" for char in label)


def load_fonts(font_paths: list[str]) -> list[str]:
    return [path for path in font_paths if Path(path).exists()]


def render_character(label: str, font_path: str, rng: random.Random) -> Image.Image:
    canvas = Image.new("L", (48, 80), 0)
    draw = ImageDraw.Draw(canvas)
    font_size = rng.randint(35, 54) if label in PROVINCES else rng.randint(42, 62)
    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (canvas.width - text_width) // 2 - bbox[0] + rng.randint(-3, 3)
    y = (canvas.height - text_height) // 2 - bbox[1] + rng.randint(-4, 4)
    draw.text((x, y), label, fill=255, font=font)

    angle = rng.uniform(-4.0, 4.0)
    canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=0)

    if rng.random() < 0.35:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.7)))

    bbox = canvas.getbbox()
    if bbox is None:
        return Image.new("L", (20, 40), 0)

    char = canvas.crop(bbox)
    char.thumbnail((18, 36), Image.Resampling.LANCZOS)
    output = Image.new("L", (20, 40), 0)
    x = (20 - char.width) // 2 + rng.randint(-1, 1)
    y = (40 - char.height) // 2 + rng.randint(-1, 1)
    output.paste(char, (max(0, x), max(0, y)))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic license plate character samples.")
    parser.add_argument("-o", "--output", type=Path, default=Path("train_chars_ccpd_aug"), help="Output train_chars folder.")
    parser.add_argument("--per-class", type=int, default=300, help="Synthetic samples per class.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--fonts", nargs="*", default=DEFAULT_FONTS, help="Font files to use.")
    parser.add_argument("--latin-fonts", nargs="*", default=LATIN_FONTS, help="Font files to use for letters and numbers.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chinese_fonts = load_fonts(args.fonts)
    latin_fonts = load_fonts(args.latin_fonts)
    if not chinese_fonts or not latin_fonts:
        print("No usable fonts found.")
        return 1

    labels = PROVINCES + ALPHANUMS
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    saved = 0
    for label in labels:
        label_dir = args.output / safe_label(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        usable_fonts = chinese_fonts if label in PROVINCES else latin_fonts
        for index in range(args.per_class):
            font_path = rng.choice(usable_fonts)
            image = render_character(label, font_path, rng)
            image.save(label_dir / f"synthetic_{index:04d}.png")
            saved += 1

    print(f"Chinese fonts: {len(chinese_fonts)}")
    print(f"Latin fonts: {len(latin_fonts)}")
    print(f"Classes: {len(labels)}")
    print(f"Synthetic samples: {saved}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
