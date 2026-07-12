from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .schema import sha256_file


def validate(artifact_dir: Path, *, release: bool = False) -> dict:
    import onnx
    import onnxruntime as ort

    manifest_path = artifact_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for name, entry in manifest["artifacts"].items():
        path = artifact_dir / entry["path"]
        if not path.exists():
            errors.append(f"missing {entry['path']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            errors.append(f"size mismatch for {entry['path']}")
        if sha256_file(path) != entry["sha256"]:
            errors.append(f"hash mismatch for {entry['path']}")
    runtime_bytes = sum(
        entry["bytes"]
        for _, entry in {
            item["path"]: item
            for item in manifest["artifacts"].values()
            if item["runtime"]
        }.items()
    )
    if runtime_bytes != manifest["runtime_bytes"]:
        errors.append("runtime byte total does not match manifest")
    if runtime_bytes > manifest["runtime_budget_bytes"]:
        errors.append("runtime byte budget exceeded")

    if release:
        if manifest.get("release_eligible") is not True:
            errors.append("manifest is not release eligible")
        if manifest.get("release_checks", {}).get("passed") is not True:
            errors.append("one or more release checks failed")
        if manifest.get("evaluation_gates", {}).get("passed") is not True:
            errors.append("evaluation gates did not pass")
        if manifest.get("cross_runtime", {}).get("passed") is not True:
            errors.append("Node, WASM, and WebGPU parity is incomplete")
        artifact_license = manifest.get("licenses", {}).get("artifact", "")
        if not artifact_license or artifact_license.lower().startswith("pending"):
            errors.append("artifact license is not approved")

    model_path = artifact_dir / "model.int8.onnx"
    onnx.checker.check_model(onnx.load(str(model_path)))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {value.name for value in session.get_inputs()}
    golden = json.loads((artifact_dir / "golden_vectors.json").read_text())
    if golden["model_sha256"] != sha256_file(model_path):
        errors.append("golden vector model hash mismatch")
    checked = 0
    for vector in golden["vectors"]:
        feeds = {
            "input_ids": np.asarray([vector["input_ids"]], dtype=np.int64),
            "attention_mask": np.asarray([vector["attention_mask"]], dtype=np.int64),
        }
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.asarray(
                [vector["token_type_ids"]], dtype=np.int64
            )
        logits = session.run(None, feeds)[0]
        predicted = np.argmax(logits[0], axis=-1).tolist()
        if predicted != vector["predicted_label_ids"]:
            errors.append(f"golden prediction mismatch for {vector['id']}")
            break
        checked += 1
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "valid": True,
        "runtime_bytes": runtime_bytes,
        "golden_vectors_checked": checked,
        "model_sha256": sha256_file(model_path),
        "release_validation": release,
        "release_eligible": manifest.get("release_eligible", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final PLVA model artifacts")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--release",
        action="store_true",
        help="also require every publication gate and cross-runtime parity result",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.artifact_dir, release=args.release),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
