from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def fake_value(value: str) -> str:
    """Guard against accidentally adding production-looking credentials."""

    if value.startswith(("sk_live_", "AKIA")):
        raise ValueError("fixture credentials must be visibly fake")
    return value


def draw_lines(
    output: Path,
    *,
    background: str,
    foreground: str,
    lines: list[tuple[str, int, int] | tuple[str, int, int, list[tuple[str, str]]]],
    family: str,
) -> dict[str, Any]:
    image = Image.new("RGB", (1280, 720), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    truth = []
    for line in lines:
        text, x, y = line[:3]
        declared_annotations = line[3] if len(line) == 4 else []
        draw.text((x, y), text, fill=foreground, font=font)
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        annotations = []
        for label, value in declared_annotations:
            start = text.index(value)
            end = start + len(value)
            annotation_left = x + round(draw.textlength(text[:start], font=font))
            annotation_right = x + round(draw.textlength(text[:end], font=font))
            annotations.append(
                {
                    "label": label,
                    "value": value,
                    "box": [annotation_left, top, annotation_right, bottom],
                }
            )
        truth.append(
            {
                "text": text,
                "box": [left, top, right, bottom],
                "annotations": annotations,
            }
        )
    image.save(output, format="PNG", optimize=False)
    return {
        "id": output.stem,
        "image": output.name,
        "synthetic": True,
        "family": family,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "truth_lines": truth,
    }


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = [
        draw_lines(
            output_dir / "profile-form.png",
            background="#f7f8fa",
            foreground="#101828",
            family="profile-form",
            lines=[
                ("PROFILE SETTINGS", 54, 42),
                ("Full name    Avery Chen", 72, 140, [("NAME", "Avery Chen")]),
                (
                    "Email        avery.chen@example.test",
                    72,
                    215,
                    [("EMAIL", "avery.chen@example.test")],
                ),
                (
                    "Phone        +1 415 555 0136",
                    72,
                    290,
                    [("PHONE", "+1 415 555 0136")],
                ),
                (
                    "Address      275 Sara Summit",
                    72,
                    365,
                    [("ADDRESS", "275 Sara Summit")],
                ),
                ("Save changes", 72, 500),
            ],
        ),
        draw_lines(
            output_dir / "terminal.png",
            background="#101318",
            foreground="#d7f9df",
            family="terminal",
            lines=[
                ("$ plva configure --local", 38, 42),
                (
                    fake_value("API_KEY=sk_test_FAKE0123456789"),
                    38,
                    112,
                    [("API_KEY", "sk_test_FAKE0123456789")],
                ),
                (
                    "AUTH=Bearer fake.header.signature",
                    38,
                    182,
                    [("AUTH_TOKEN", "fake.header.signature")],
                ),
                (
                    "user=avery.chen@example.test",
                    38,
                    252,
                    [("EMAIL", "avery.chen@example.test")],
                ),
                ("status=ready port=4173 version=0.1.0", 38, 322),
            ],
        ),
        draw_lines(
            output_dir / "checkout-table.png",
            background="#ffffff",
            foreground="#111111",
            family="checkout-table",
            lines=[
                ("CHECKOUT REVIEW", 42, 32),
                (
                    "Ship to        Morgan Rivera",
                    42,
                    105,
                    [("NAME", "Morgan Rivera")],
                ),
                (
                    "Street         88 Harbor View",
                    42,
                    165,
                    [("ADDRESS", "88 Harbor View")],
                ),
                (
                    "Card           4111 1111 1111 1111",
                    42,
                    225,
                    [("CARD_NUMBER", "4111 1111 1111 1111")],
                ),
                (
                    "Security code  123",
                    42,
                    285,
                    [("CVC", "123")],
                ),
                ("Order ID       ORD-2026-0711", 42, 345),
                ("Total          $143.20", 42, 405),
            ],
        ),
    ]
    manifest = {
        "schema_version": 1,
        "synthetic_only": True,
        "fixtures": fixtures,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic fake OCR parity fixtures"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
