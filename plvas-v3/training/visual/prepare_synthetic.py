from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from training.synth_screen import (
    SECRET_TEMPLATE_LABELS,
    SENSITIVE_IMAGE_KINDS,
    generate_records,
)
from training.visual.prepare_webpii import DETECTOR_CLASSES, yolo_line


VISUAL_CLASS_MAP = {
    "NAME": "NAME",
    "ADDRESS": "ADDRESS",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "CARD_NUMBER": "CARD_NUMBER",
    "CVC": "CVC",
    "PASSWORD": "SECRET",
    "API_KEY": "SECRET",
    "AUTH_TOKEN": "SECRET",
    "PRIVATE_KEY": "SECRET",
    "DOB": "SENSITIVE_FIELD",
    "GOV_ID": "SENSITIVE_FIELD",
    "BANK_ACCOUNT": "SENSITIVE_FIELD",
}

MINIMUM_SUPPORT_RECORDS = 80


def _render_sensitive_image(
    draw: ImageDraw.ImageDraw,
    record: dict[str, Any],
    element: dict[str, str],
) -> list[float]:
    """Draw a deterministic, entirely synthetic sensitive-image stand-in."""

    kind = element["kind"]
    if kind not in SENSITIVE_IMAGE_KINDS:
        raise ValueError(f"unknown synthetic sensitive-image kind: {kind}")
    rng = random.Random(int(record["seed"]) ^ 0x51A6E)
    split_shift = {"train": 0, "validation": 34, "holdout": 68}[record["split"]]
    widths = {
        "avatar": 190,
        "map": 330,
        "id_photo": 300,
        "payment_card_image": 310,
    }
    width = widths[kind]
    height = 174
    x1 = min(1190 + split_shift + rng.randint(-18, 18), 1564 - width)
    y1 = 52 + rng.randint(-8, 8)
    x2, y2 = x1 + width, y1 + height
    palette = (
        ("#dbeafe", "#2563eb", "#1e3a8a"),
        ("#dcfce7", "#16a34a", "#14532d"),
        ("#fef3c7", "#d97706", "#78350f"),
        ("#f3e8ff", "#9333ea", "#581c87"),
    )[int(record["seed"]) % 4]
    pale, accent, dark = palette

    draw.rounded_rectangle((x1, y1, x2, y2), 18, fill=pale, outline=dark, width=3)
    if kind == "avatar":
        center_x = (x1 + x2) / 2
        draw.ellipse(
            (center_x - 35, y1 + 25, center_x + 35, y1 + 95),
            fill=accent,
            outline=dark,
            width=3,
        )
        draw.ellipse(
            (center_x - 68, y1 + 92, center_x + 68, y2 - 10),
            fill=accent,
            outline=dark,
            width=3,
        )
    elif kind == "map":
        for offset in (35, 76, 117):
            draw.line(
                (x1 + 14, y1 + offset, x2 - 14, y1 + offset - 18),
                fill="#ffffff",
                width=10,
            )
        route = [
            (x1 + 35, y2 - 30),
            (x1 + width * 0.38, y1 + 52),
            (x1 + width * 0.68, y2 - 50),
            (x2 - 30, y1 + 35),
        ]
        draw.line(route, fill=accent, width=7, joint="curve")
        for point in (route[0], route[-1]):
            draw.ellipse(
                (point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10),
                fill=dark,
            )
    elif kind == "id_photo":
        photo = (x1 + 18, y1 + 18, x1 + 118, y2 - 18)
        draw.rounded_rectangle(photo, 10, fill="#ffffff", outline=dark, width=2)
        draw.ellipse(
            (photo[0] + 27, photo[1] + 18, photo[2] - 27, photo[1] + 64),
            fill=accent,
        )
        draw.ellipse(
            (photo[0] + 17, photo[1] + 60, photo[2] - 17, photo[3] - 2),
            fill=accent,
        )
        for offset, length in ((34, 124), (68, 145), (102, 105), (132, 135)):
            draw.rounded_rectangle(
                (x1 + 138, y1 + offset, x1 + 138 + length, y1 + offset + 8),
                4,
                fill=dark,
            )
    else:  # payment_card_image
        draw.rounded_rectangle(
            (x1 + 24, y1 + 35, x1 + 82, y1 + 78),
            8,
            fill="#ffffff",
            outline=dark,
            width=2,
        )
        draw.line((x1 + 24, y1 + 113, x2 - 24, y1 + 113), fill=dark, width=8)
        for offset in range(4):
            dot_x = x1 + 38 + offset * 28
            draw.ellipse((dot_x, y1 + 132, dot_x + 8, y1 + 140), fill=dark)
        draw.rounded_rectangle(
            (x2 - 82, y2 - 42, x2 - 24, y2 - 20), 8, fill=accent
        )
    return [float(x1), float(y1), float(x2), float(y2)]


def render_record(
    record: dict[str, Any],
    image_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int]]:
    width, height = 1600, 280
    styles = {
        "train": ("#f7f8fa", "#ffffff", "#101828", "#667085", 48, 112),
        "validation": ("#eef2ff", "#ffffff", "#172554", "#4f46e5", 70, 104),
        "holdout": ("#f0fdf4", "#ffffff", "#14532d", "#15803d", 92, 120),
    }
    background, card, text_color, muted, base_x, base_y = styles[record["split"]]
    rng = random.Random(int(record["seed"]) ^ 0xA11CE)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    draw.rounded_rectangle((24, 20, width - 24, height - 20), 18, fill=card)
    draw.text((base_x, 44), "PLVA SYNTHETIC UI", fill=muted, font=small)
    x, y = base_x + rng.randint(-8, 8), base_y + rng.randint(-5, 5)
    draw.text((x, y), record["text"], fill=text_color, font=font)
    _left, top, _right, bottom = draw.textbbox((x, y), record["text"], font=font)

    annotations: list[dict[str, Any]] = []
    for span in record["spans"]:
        visual_class = VISUAL_CLASS_MAP[span["label"]]
        left = x + float(draw.textlength(record["text"][: span["start"]], font=font))
        right = x + float(draw.textlength(record["text"][: span["end"]], font=font))
        box = [left, float(top), right, float(bottom)]
        annotations.append(
            {
                "class": visual_class,
                "source_class": span["label"],
                "bbox_xyxy": box,
            }
        )
    for element in record.get("visual_elements", []):
        if element.get("class") != "SENSITIVE_IMAGE":
            raise ValueError(
                f"unsupported visual element class: {element.get('class')}"
            )
        annotations.append(
            {
                "class": "SENSITIVE_IMAGE",
                "source_class": "SENSITIVE_IMAGE",
                "synthetic_kind": element["kind"],
                "bbox_xyxy": _render_sensitive_image(draw, record, element),
            }
        )
    full_box = list(map(float, draw.textbbox((x, y), record["text"], font=font)))
    hard_negatives = (
        [
            {
                "bbox_xyxy": full_box,
                "reason": (
                    "synthetic-sensitive-image-caption"
                    if record.get("visual_elements")
                    else "synthetic-negative-ui-text"
                ),
            }
        ]
        if not record["spans"]
        else []
    )
    image.save(image_path, format="PNG", optimize=False)
    return annotations, hard_negatives, (width, height)


def prepare_split(
    root: Path,
    split: str,
    count: int,
    *,
    seed: int,
) -> dict[str, Any]:
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    records_path = root / "records" / f"{split}.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    source_classes: Counter[str] = Counter()
    secret_source_classes: Counter[str] = Counter()
    sensitive_image_kinds: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    digest = hashlib.sha256()

    with records_path.open("w", encoding="utf-8") as output:
        for record in generate_records(
            split,
            count,
            seed=seed,
            noise_fraction=0.0,
        ):
            stem = record["id"].replace(":", "-")
            image_relative = Path("images") / split / f"{stem}.png"
            label_relative = Path("labels") / split / f"{stem}.txt"
            annotations, hard_negatives, (width, height) = render_record(
                record,
                root / image_relative,
            )
            label_lines = [
                yolo_line(item["class"], item["bbox_xyxy"], width, height)
                for item in annotations
            ]
            (root / label_relative).write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            visual_record = {
                "id": stem,
                "synthetic": True,
                "split": split,
                "template_id": record["template_id"],
                "seed": record["seed"],
                "width": width,
                "height": height,
                "image": str(image_relative),
                "image_sha256": hashlib.sha256(
                    (root / image_relative).read_bytes()
                ).hexdigest(),
                "label_file": str(label_relative),
                "annotations": annotations,
                "hard_negative_boxes": hard_negatives,
            }
            encoded = (
                json.dumps(visual_record, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            output.write(encoded.decode())
            digest.update(encoded)
            counts["records"] += 1
            counts["annotations"] += len(annotations)
            counts["hard_negative_boxes"] += len(hard_negatives)
            if not annotations:
                counts["pure_negative_records"] += 1
            templates[record["template_id"]] += 1
            for annotation in annotations:
                classes[annotation["class"]] += 1
                source_classes[annotation["source_class"]] += 1
                if annotation["class"] == "SECRET":
                    secret_source_classes[annotation["source_class"]] += 1
                if annotation["class"] == "SENSITIVE_IMAGE":
                    sensitive_image_kinds[annotation["synthetic_kind"]] += 1
    return {
        **counts,
        "class_counts": dict(sorted(classes.items())),
        "source_class_counts": dict(sorted(source_classes.items())),
        "secret_source_counts": dict(sorted(secret_source_classes.items())),
        "sensitive_image_kind_counts": dict(sorted(sensitive_image_kinds.items())),
        "template_counts": dict(sorted(templates.items())),
        "records_path": str(records_path.relative_to(root)),
        "records_sha256": digest.hexdigest(),
    }


def validate_support(split: str, summary: dict[str, Any]) -> dict[str, Any]:
    records = int(summary["records"])
    if records < MINIMUM_SUPPORT_RECORDS:
        return {
            "enforced": False,
            "reason": f"requires at least {MINIMUM_SUPPORT_RECORDS} records",
        }

    minimum_class_count = max(1, records // 20)
    minimum_subtype_count = max(1, records // 100)
    minimum_hard_negatives = max(1, records // 20)
    minimum_pure_negative_records = max(1, records // 20)
    failures: list[str] = []
    for class_name in DETECTOR_CLASSES:
        actual = int(summary["class_counts"].get(class_name, 0))
        if actual < minimum_class_count:
            failures.append(
                f"{class_name}={actual} below minimum {minimum_class_count}"
            )
    for source_class in SECRET_TEMPLATE_LABELS:
        actual = int(summary["secret_source_counts"].get(source_class, 0))
        if actual < minimum_subtype_count:
            failures.append(
                f"SECRET/{source_class}={actual} below minimum {minimum_subtype_count}"
            )
    for kind in SENSITIVE_IMAGE_KINDS:
        actual = int(summary["sensitive_image_kind_counts"].get(kind, 0))
        if actual < minimum_subtype_count:
            failures.append(
                f"SENSITIVE_IMAGE/{kind}={actual} below minimum {minimum_subtype_count}"
            )
    if int(summary["hard_negative_boxes"]) < minimum_hard_negatives:
        failures.append(
            "hard_negative_boxes="
            f"{summary['hard_negative_boxes']} below minimum {minimum_hard_negatives}"
        )
    if int(summary["pure_negative_records"]) < minimum_pure_negative_records:
        failures.append(
            "pure_negative_records="
            f"{summary['pure_negative_records']} below minimum "
            f"{minimum_pure_negative_records}"
        )
    if failures:
        raise RuntimeError(
            f"{split} synthetic support validation failed: {'; '.join(failures)}"
        )
    return {
        "enforced": True,
        "minimum_detector_class_count": minimum_class_count,
        "minimum_secret_subtype_count": minimum_subtype_count,
        "minimum_sensitive_image_kind_count": minimum_subtype_count,
        "minimum_hard_negative_boxes": minimum_hard_negatives,
        "minimum_pure_negative_records": minimum_pure_negative_records,
        "passed": True,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --force")
        shutil.rmtree(args.output_dir)
    temporary = args.output_dir.with_name(
        f".{args.output_dir.name}.partial-{os.getpid()}"
    )
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        splits = {
            "train": prepare_split(
                temporary,
                "train",
                args.train_records,
                seed=args.seed,
            ),
            "validation": prepare_split(
                temporary,
                "validation",
                args.validation_records,
                seed=args.seed,
            ),
        }
        for split, summary in splits.items():
            summary["support_validation"] = validate_support(split, summary)
        train_templates = set(splits["train"]["template_counts"])
        validation_templates = set(splits["validation"]["template_counts"])
        if train_templates & validation_templates:
            raise RuntimeError("synthetic visual template families overlap")
        manifest = {
            "schema_version": 2,
            "dataset": "plva-screen-native-synthetic",
            "license": "Apache-2.0",
            "synthetic_only": True,
            "all_values_fake": True,
            "classes": list(DETECTOR_CLASSES),
            "visual_class_map": VISUAL_CLASS_MAP,
            "sensitive_image_kinds": list(SENSITIVE_IMAGE_KINDS),
            "coverage_policy": {
                "minimum_records_for_enforcement": MINIMUM_SUPPORT_RECORDS,
                "minimum_detector_class_fraction": 0.05,
                "minimum_secret_subtype_fraction": 0.01,
                "minimum_sensitive_image_kind_fraction": 0.01,
                "minimum_hard_negative_fraction": 0.05,
                "minimum_pure_negative_record_fraction": 0.05,
            },
            "splits": splits,
            "split_policy": "template-family-and-seed-disjoint",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        yaml = [
            f"path: {temporary.resolve()}",
            "train: images/train",
            "val: images/validation",
            f"nc: {len(DETECTOR_CLASSES)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(DETECTOR_CLASSES)],
            "",
        ]
        (temporary / "dataset.yaml").write_text("\n".join(yaml), encoding="utf-8")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        if args.output_dir.exists():
            args.output_dir.rmdir()
        temporary.rename(args.output_dir)
        yaml_path = args.output_dir / "dataset.yaml"
        yaml_path.write_text(
            yaml_path.read_text(encoding="utf-8").replace(
                str(temporary.resolve()), str(args.output_dir.resolve()), 1
            ),
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate screen-native visual detector examples"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-records", type=int, default=20_000)
    parser.add_argument("--validation-records", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=1311)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
