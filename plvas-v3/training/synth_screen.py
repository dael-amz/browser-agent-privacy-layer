from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import string
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import SPLITS, make_span, validate_record, write_jsonl_atomic


FIRST_NAMES = (
    "Avery",
    "Camila",
    "Darius",
    "Elena",
    "Fatima",
    "Hugo",
    "Imani",
    "Jonah",
    "Keiko",
    "Lina",
    "Mateo",
    "Nadia",
    "Omar",
    "Priya",
    "Quinn",
    "Ravi",
    "Sofia",
    "Tariq",
    "Uma",
    "Victor",
    "Wren",
    "Ximena",
    "Yara",
    "Zane",
)
LAST_NAMES = (
    "Alvarez",
    "Bennett",
    "Chen",
    "Diallo",
    "Eriksen",
    "Flores",
    "Gupta",
    "Haddad",
    "Ito",
    "Johnson",
    "Kowalski",
    "Lopez",
    "Mensah",
    "Novak",
    "Okafor",
    "Patel",
    "Qureshi",
    "Rivera",
    "Singh",
    "Tanaka",
    "Usman",
    "Vega",
    "Williams",
    "Xu",
    "Young",
    "Zoric",
)
STREETS = (
    "Cedar Walk",
    "Glasshouse Lane",
    "Juniper Avenue",
    "Market Street",
    "Orchard Road",
    "Pine Terrace",
    "River Court",
    "Sunset Boulevard",
    "Willow Way",
)
CITIES = ("Austin", "Boston", "Denver", "London", "Oakland", "Portland", "Seattle")
DOMAINS = ("example.com", "example.net", "example.org", "mail.test")
ALPHABET = string.ascii_letters + string.digits

SPLIT_OFFSET = {"train": 100_000, "validation": 700_000, "holdout": 900_000}
SPLIT_PREFIX = {"train": "tr", "validation": "va", "holdout": "ho"}
SPLIT_ROTATION = {"train": 0, "validation": 1, "holdout": 2}

# The screen-native detector generator consumes records in complete cycles.  Two
# slots are reserved for every detector class and two for hard negatives.  The
# text tagger ignores ``visual_elements``; the visual renderer turns those
# descriptors into synthetic, non-person photographs/maps/card thumbnails.
VISUAL_TEMPLATE_COUNT = 20
SECRET_TEMPLATE_LABELS = ("PASSWORD", "API_KEY", "AUTH_TOKEN", "PRIVATE_KEY")
SENSITIVE_FIELD_TEMPLATE_LABELS = ("DOB", "GOV_ID", "BANK_ACCOUNT")
SENSITIVE_IMAGE_KINDS = ("avatar", "map", "id_photo", "payment_card_image")


def _base36(value: int) -> str:
    chars = string.digits + string.ascii_lowercase
    if value == 0:
        return "0"
    output = ""
    while value:
        value, remainder = divmod(value, 36)
        output = chars[remainder] + output
    return output


def _luhn_check_digit(prefix: str) -> str:
    digits = [int(char) for char in prefix + "0"]
    parity = len(digits) % 2
    total = 0
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return str((10 - total % 10) % 10)


def _token(seed: int, length: int) -> str:
    digest = hashlib.sha256(f"plva-synthetic-{seed}".encode()).digest()
    output = []
    for index in range(length):
        output.append(ALPHABET[digest[index % len(digest)] % len(ALPHABET)])
    return "".join(output)


def fake_values(split: str, index: int) -> dict[str, str]:
    uid = SPLIT_OFFSET[split] + index
    prefix = SPLIT_PREFIX[split]
    first = FIRST_NAMES[uid % len(FIRST_NAMES)]
    last = LAST_NAMES[(uid // len(FIRST_NAMES)) % len(LAST_NAMES)]
    middle = string.ascii_uppercase[(uid // 17) % 26]
    suffix = ("", "Jr.", "III", "PhD")[(uid // 29) % 4]
    name = " ".join(part for part in (first, middle, last, suffix) if part)
    house = 100 + uid % 9800
    street = STREETS[(uid // 7) % len(STREETS)]
    city = CITIES[(uid // 13) % len(CITIES)]
    address = f"{house} {street}, Apt {uid % 997 + 1}, {city} {10000 + uid % 89999}"
    email = (
        f"{first.lower()}.{last.lower()}.{_base36(uid)}@{DOMAINS[uid % len(DOMAINS)]}"
    )
    phone = f"+1 (202) 555-{100 + uid % 900:04d} ext {uid}"
    year_ranges = {
        "train": (1940, 1989),
        "validation": (1990, 1999),
        "holdout": (2000, 2005),
    }
    year_low, year_high = year_ranges[split]
    year = year_low + uid % (year_high - year_low + 1)
    month = uid % 12 + 1
    day = uid % 28 + 1
    dob = f"{month:02d}/{day:02d}/{year:04d}"
    gov_id = f"{prefix.upper()}-{uid:09d}"
    bank = f"GB{10 + uid % 89:02d} PLVA {uid:016d}"
    card_prefix = f"4{uid:014d}"[-15:]
    card_digits = card_prefix + _luhn_check_digit(card_prefix)
    card = " ".join(card_digits[pos : pos + 4] for pos in range(0, 16, 4))
    cvc_ranges = {"train": 1000, "validation": 8000, "holdout": 9000}
    cvc = f"{cvc_ranges[split] + index % 1000:04d}"
    password = f"{prefix}!{_token(uid, 12)}"
    api_key = f"sk-test-{prefix}-{_token(uid + 1, 28)}"
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"sub": f"{prefix}-{uid}", "aud": "plva.test"}, separators=(",", ":")
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    auth_token = f"eyJhbGciOiJub25lIn0.{payload}.{_token(uid + 2, 24)}"
    private_key = (
        "-----BEGIN PRIVATE KEY----- "
        f"{base64.b64encode(('FAKE-PLVA-' + prefix + str(uid)).encode()).decode()}"
        " -----END PRIVATE KEY-----"
    )
    negative_order = (
        f"Order {4200 + uid % 700:04d}-{2020 + uid % 8} | "
        f"Build {1 + uid % 30}.{uid % 50}.{uid % 90} | "
        f"Total ${100 + uid % 800}.{uid % 100:02d} | {1 + uid % 40} items"
    )
    negative_dev = (
        f"localhost:{4000 + uid % 2000}  "
        f"commit {hashlib.sha256(f'commit-{uid}'.encode()).hexdigest()[:8]}  "
        f"version {2020 + uid % 8}.{uid % 12 + 1:02d}  "
        f"PID {10000 + uid % 50000}"
    )
    return {
        "NAME": name,
        "ADDRESS": address,
        "DOB": dob,
        "GOV_ID": gov_id,
        "EMAIL": email,
        "PHONE": phone,
        "BANK_ACCOUNT": bank,
        "CARD_NUMBER": card,
        "CVC": cvc,
        "PASSWORD": password,
        "API_KEY": api_key,
        "AUTH_TOKEN": auth_token,
        "PRIVATE_KEY": private_key,
        "NEGATIVE_ORDER": negative_order,
        "NEGATIVE_DEV": negative_dev,
    }


def _render(parts: list[str | tuple[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    text = ""
    spans: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            text += part
            continue
        label, value = part
        start = len(text)
        text += value
        spans.append(make_span(text, start, len(text), label))
    return text, spans


def _template(
    split: str,
    template_index: int,
    cycle_index: int,
    values: dict[str, str],
) -> tuple[str, list[str | tuple[str, str]], list[dict[str, str]]]:
    prefix = {"train": "form", "validation": "panel", "holdout": "drawer"}[split]
    rotation = SPLIT_ROTATION[split]

    if template_index in {6, 15}:
        within_cycle = 0 if template_index == 6 else 1
        label = SECRET_TEMPLATE_LABELS[
            (cycle_index * 2 + within_cycle + rotation) % len(SECRET_TEMPLATE_LABELS)
        ]
        secret_templates: dict[str, tuple[str, list[str | tuple[str, str]]]] = {
            "PASSWORD": (
                "secret-password",
                ["Password: ", ("PASSWORD", values["PASSWORD"]), "  Show"],
            ),
            "API_KEY": (
                "secret-api-key",
                ["PLVA_API_KEY=", ("API_KEY", values["API_KEY"])],
            ),
            "AUTH_TOKEN": (
                "secret-auth-token",
                ["Authorization: Bearer ", ("AUTH_TOKEN", values["AUTH_TOKEN"])],
            ),
            "PRIVATE_KEY": (
                "secret-private-key",
                ["Signing material: ", ("PRIVATE_KEY", values["PRIVATE_KEY"])],
            ),
        }
        name, parts = secret_templates[label]
        return f"{prefix}.{name}", parts, []

    if template_index in {7, 16}:
        within_cycle = 0 if template_index == 7 else 1
        label = SENSITIVE_FIELD_TEMPLATE_LABELS[
            (cycle_index * 2 + within_cycle + rotation)
            % len(SENSITIVE_FIELD_TEMPLATE_LABELS)
        ]
        field_templates: dict[str, tuple[str, list[str | tuple[str, str]]]] = {
            "DOB": (
                "sensitive-date-of-birth",
                ["Date of birth  ", ("DOB", values["DOB"]), "  Verified"],
            ),
            "GOV_ID": (
                "sensitive-government-id",
                ["Identity document number: ", ("GOV_ID", values["GOV_ID"])],
            ),
            "BANK_ACCOUNT": (
                "sensitive-bank-account",
                ["Settlement account  ", ("BANK_ACCOUNT", values["BANK_ACCOUNT"])],
            ),
        }
        name, parts = field_templates[label]
        return f"{prefix}.{name}", parts, []

    if template_index in {8, 17}:
        within_cycle = 0 if template_index == 8 else 1
        kind = SENSITIVE_IMAGE_KINDS[
            (cycle_index * 2 + within_cycle + rotation) % len(SENSITIVE_IMAGE_KINDS)
        ]
        captions = {
            "avatar": "Profile photo preview",
            "map": "Saved location preview",
            "id_photo": "Identity document preview",
            "payment_card_image": "Saved payment card preview",
        }
        caption_prefix = {
            "train": "Account",
            "validation": "Workspace",
            "holdout": "Review",
        }[split]
        return (
            f"{prefix}.sensitive-image-{kind.replace('_', '-')}",
            [f"{caption_prefix} {captions[kind].lower()}"],
            [{"class": "SENSITIVE_IMAGE", "kind": kind}],
        )

    variants: dict[int, tuple[str, list[str | tuple[str, str]]]] = {
        0: ("identity-primary", ["Full legal name: ", ("NAME", values["NAME"])]),
        1: ("address-shipping", ["Shipping address: ", ("ADDRESS", values["ADDRESS"])]),
        2: ("email-primary", ["Primary email address: ", ("EMAIL", values["EMAIL"])]),
        3: ("phone-recovery", ["Recovery phone: ", ("PHONE", values["PHONE"])]),
        4: ("card-entry", ["Card number: ", ("CARD_NUMBER", values["CARD_NUMBER"])]),
        5: ("cvc-entry", ["Security code: ", ("CVC", values["CVC"])]),
        9: ("identity-recipient", ["Recipient: ", ("NAME", values["NAME"])]),
        10: ("address-billing", ["Billing address: ", ("ADDRESS", values["ADDRESS"])]),
        11: ("email-account", ["Signed in as ", ("EMAIL", values["EMAIL"])]),
        12: ("phone-sms", ["SMS destination: ", ("PHONE", values["PHONE"])]),
        13: ("card-saved", ["Saved card: ", ("CARD_NUMBER", values["CARD_NUMBER"])]),
        14: ("cvc-confirm", ["Confirm CVV: ", ("CVC", values["CVC"])]),
        18: ("negative-order", [values["NEGATIVE_ORDER"]]),
        19: ("negative-dev", [values["NEGATIVE_DEV"]]),
    }
    name, parts = variants[template_index]
    return f"{prefix}.{name}", parts, []


OCR_REPLACEMENTS = {
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "B": "8",
}


def apply_ocr_noise(
    text: str,
    spans: list[dict[str, Any]],
    rng: random.Random,
    *,
    replace_rate: float = 0.025,
    delete_rate: float = 0.003,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[str] = []
    boundary_map = [0] * (len(text) + 1)
    operations: list[dict[str, Any]] = []
    protected_singletons = {
        index
        for span in spans
        if span["end"] - span["start"] == 1
        for index in range(span["start"], span["end"])
    }

    for index, char in enumerate(text):
        boundary_map[index] = len(output)
        if (
            index not in protected_singletons
            and not char.isspace()
            and rng.random() < delete_rate
        ):
            operations.append({"op": "delete", "index": index, "from": char})
            boundary_map[index + 1] = len(output)
            continue
        replacement = OCR_REPLACEMENTS.get(char)
        if replacement and rng.random() < replace_rate:
            output.append(replacement)
            operations.append(
                {"op": "replace", "index": index, "from": char, "to": replacement}
            )
        else:
            output.append(char)
        boundary_map[index + 1] = len(output)

    noisy_text = "".join(output)
    noisy_spans: list[dict[str, Any]] = []
    for span in spans:
        start = boundary_map[span["start"]]
        end = boundary_map[span["end"]]
        if end <= start:
            return text, spans, []
        noisy_spans.append(make_span(noisy_text, start, end, span["label"]))
    return noisy_text, noisy_spans, operations


def generate_records(
    split: str,
    count: int,
    *,
    seed: int = 1311,
    noise_fraction: float = 0.35,
) -> Iterator[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}")
    template_count = VISUAL_TEMPLATE_COUNT
    for index in range(count):
        record_seed = seed + SPLIT_OFFSET[split] + index
        rng = random.Random(record_seed)
        values = fake_values(split, index)
        template_index = index % template_count
        cycle_index = index // template_count
        template_id, parts, visual_elements = _template(
            split, template_index, cycle_index, values
        )
        text, spans = _render(parts)
        noisy = rng.random() < noise_fraction
        noise_ops: list[dict[str, Any]] = []
        if noisy:
            text, spans, noise_ops = apply_ocr_noise(text, spans, rng)
        record = {
            "id": f"synthetic:{split}:{record_seed}",
            "text": text,
            "spans": spans,
            "split": split,
            "source": "synthetic_screen",
            "template_id": template_id,
            "seed": record_seed,
            "language": "en",
            "region": "SYNTHETIC",
            "noisy": noisy,
            "visual_elements": visual_elements,
            "provenance": {
                "generator_version": 2,
                "template_cycle": cycle_index,
                "noise_ops": noise_ops,
            },
        }
        validate_record(record)
        yield record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic PLVA screen text"
    )
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1311)
    parser.add_argument("--noise-fraction", type=float, default=0.35)
    args = parser.parse_args()
    summary = write_jsonl_atomic(
        args.output,
        generate_records(
            args.split,
            args.count,
            seed=args.seed,
            noise_fraction=args.noise_fraction,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
