from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.schema import sha256_file
from training.semantic.rampart_reference import (
    DEFAULT_CONTRACT,
    DEFAULT_FIXTURES,
    DEFAULT_MODEL_DIR,
    generate_golden_vectors,
)


REQUIRED_EVALUATION_GATES = (
    "screen_ocr_holdout_present",
    "contextual_recall",
    "secret_safety",
    "offset_round_trip",
    "deterministic_recognizers",
)


def validate_evaluation(path: Path, required_revision: str) -> dict[str, Any]:
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    if evaluation.get("bootstrap_model", {}).get("revision") != required_revision:
        raise ValueError("baseline evaluation uses the wrong Rampart revision")
    gates = evaluation.get("gates", {})
    missing = [name for name in REQUIRED_EVALUATION_GATES if name not in gates]
    if missing:
        raise ValueError(f"baseline evaluation is missing gates: {', '.join(missing)}")
    failed = [
        name
        for name in REQUIRED_EVALUATION_GATES
        if gates[name].get("passed") is not True
    ]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "required_gates": list(REQUIRED_EVALUATION_GATES),
        "failed_gates": failed,
        "all_required_gates_passed": not failed,
        "metrics": evaluation.get("metrics", {}),
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    golden_path = args.output_dir / "rampart_model_goldens.json"
    goldens = generate_golden_vectors(
        args.model_dir,
        args.contract,
        args.fixtures,
        golden_path,
    )
    evaluation = (
        validate_evaluation(args.evaluation, contract["model"]["revision"])
        if args.evaluation
        else {
            "all_required_gates_passed": False,
            "failed_gates": list(REQUIRED_EVALUATION_GATES),
            "reason": "No full Rampart plus deterministic screen/OCR evaluation was supplied.",
        }
    )
    if args.freeze and not evaluation["all_required_gates_passed"]:
        raise RuntimeError(
            "cannot freeze Rampart baseline; failed gates: "
            + ", ".join(evaluation["failed_gates"])
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen" if args.freeze else "contract_verified",
        "baseline_frozen": bool(args.freeze),
        "bootstrap_model": {
            "id": contract["model"]["id"],
            "revision": contract["model"]["revision"],
            "onnx_sha256": contract["model"]["onnx_sha256"],
        },
        "runtime": contract["runtime"],
        "contract_sha256": sha256_file(args.contract),
        "golden_vectors": {
            "path": golden_path.name,
            "sha256": sha256_file(golden_path),
            "count": len(goldens["vectors"]),
            "scope": "model-only",
        },
        "evaluation": evaluation,
        "replacement_training_allowed": bool(args.freeze),
    }
    manifest_path = args.output_dir / "rampart_baseline.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify or freeze the Rampart baseline"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="freeze only when every required full-pipeline evaluation gate passes",
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(freeze(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
