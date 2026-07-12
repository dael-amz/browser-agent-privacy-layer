from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_CONTRACT = ROOT / "reference_contract.json"
DEFAULT_ASSET_ROOT = REPO_ROOT / "models" / "ocr"


def character_list(dictionary_path: Path) -> list[str]:
    characters = dictionary_path.read_text(encoding="utf-8").splitlines()
    # RapidOCR 3.9.1 deliberately appends a space even though en_dict.txt
    # already ends in one, then inserts CTC blank at index zero.
    characters.append(" ")
    characters.insert(0, "blank")
    return characters


def decode_probabilities(
    probabilities: np.ndarray, dictionary_path: Path
) -> list[dict[str, Any]]:
    if probabilities.ndim != 3:
        raise ValueError(
            f"recognizer output must have rank 3, got {probabilities.shape}"
        )
    characters = character_list(dictionary_path)
    if probabilities.shape[-1] != len(characters):
        raise ValueError(
            f"recognizer classes {probabilities.shape[-1]} != decoder characters {len(characters)}"
        )
    token_ids = np.argmax(probabilities, axis=-1)
    token_scores = np.max(probabilities, axis=-1)
    decoded = []
    for row_ids, row_scores in zip(token_ids, token_scores):
        emitted = []
        prior = None
        for timestep, (token_id, score) in enumerate(
            zip(row_ids.tolist(), row_scores.tolist())
        ):
            if token_id == 0 or token_id == prior:
                prior = token_id
                continue
            emitted.append(
                {
                    "character": characters[token_id],
                    "class_id": token_id,
                    "timestep": timestep,
                    "confidence": round(float(score), 7),
                }
            )
            prior = token_id
        decoded.append(
            {
                "text": "".join(item["character"] for item in emitted),
                "characters": emitted,
                "confidence": (
                    round(float(np.mean([item["confidence"] for item in emitted])), 7)
                    if emitted
                    else 0.0
                ),
            }
        )
    return decoded


def create_engine(asset_root: Path, contract: dict[str, Any]):
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import (
        EngineType,
        LangDet,
        LangRec,
        ModelType,
        OCRVersion,
    )

    detector = contract["detector"]
    recognizer = contract["recognizer"]
    return RapidOCR(
        params={
            "Global.use_cls": False,
            "Global.return_word_box": True,
            "Global.return_single_char_box": True,
            "Global.log_level": "error",
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.model_path": str(asset_root / "ch_PP-OCRv4_det_mobile.onnx"),
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.limit_side_len": detector["limit_side_len"],
            "Det.limit_type": detector["limit_type"],
            "Det.thresh": detector["threshold"],
            "Det.box_thresh": detector["box_threshold"],
            "Det.max_candidates": detector["max_candidates"],
            "Det.unclip_ratio": detector["unclip_ratio"],
            "Det.use_dilation": detector["use_dilation"],
            "Det.score_mode": detector["score_mode"],
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.model_path": str(asset_root / "en_PP-OCRv4_rec_mobile.onnx"),
            "Rec.rec_keys_path": str(asset_root / "en_dict.txt"),
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.lang_type": LangRec.EN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.rec_img_shape": recognizer["image_shape"],
            "Rec.rec_batch_num": recognizer["batch_size"],
        }
    )


def serialize_box(box: Any) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in box]


def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dictionary = args.asset_root / "en_dict.txt"
    characters = character_list(dictionary)
    expected_characters = contract["recognizer"]["decoder"]["character_count"]
    if len(characters) != expected_characters or characters[-2:] != [" ", " "]:
        raise ValueError("RapidOCR 97-class dictionary contract changed")
    fixture_manifest = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if fixture_manifest.get("synthetic_only") is not True:
        raise ValueError(
            "reference goldens may include recognized text only for synthetic fixtures"
        )
    engine = create_engine(args.asset_root, contract)
    records = []
    for fixture in fixture_manifest["fixtures"]:
        image_path = args.fixtures.parent / fixture["image"]
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != fixture["sha256"]:
            raise ValueError(f"fixture hash mismatch: {fixture['id']}")
        output = engine(
            image_path,
            return_word_box=True,
            return_single_char_box=True,
        )
        regions = []
        boxes = output.boxes if output.boxes is not None else []
        for index, box in enumerate(boxes):
            words = []
            if output.word_results and index < len(output.word_results):
                words = [
                    {
                        "text": item[0],
                        "confidence": round(float(item[1]), 7),
                        "box": serialize_box(item[2]),
                    }
                    for item in output.word_results[index]
                ]
            regions.append(
                {
                    "quad": serialize_box(box),
                    "text": output.txts[index],
                    "confidence": round(float(output.scores[index]), 7),
                    "words": words,
                }
            )
        records.append(
            {
                "id": fixture["id"],
                "image_sha256": digest,
                "family": fixture["family"],
                "regions": regions,
            }
        )
    result = {
        "schema_version": 1,
        "reference": contract["rapidocr"],
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "detector": contract["detector"],
        "recognizer": contract["recognizer"],
        "fixtures": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pinned RapidOCR reference outputs"
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_reference(parse_args())
    print(
        json.dumps(
            {
                "fixtures": len(result["fixtures"]),
                "regions": sum(len(item["regions"]) for item in result["fixtures"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
