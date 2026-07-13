from __future__ import annotations

import argparse
import json
from pathlib import Path


def sync(artifact_dir: Path, lock_path: Path) -> dict:
    manifest = json.loads(
        (artifact_dir / "model_manifest.json").read_text(encoding="utf-8")
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    model = manifest["artifacts"]["model.int8.onnx"]
    try:
        artifact_path = str(
            (artifact_dir / model["path"])
            .resolve()
            .relative_to(lock_path.parent.resolve())
        )
        manifest_path = str(
            (artifact_dir / "model_manifest.json")
            .resolve()
            .relative_to(lock_path.parent.resolve())
        )
    except ValueError:
        artifact_path = str((artifact_dir / model["path"]).resolve())
        manifest_path = str((artifact_dir / "model_manifest.json").resolve())
    candidate = {
        "status": (
            "release_candidate_local"
            if manifest.get("release_eligible")
            else "development_candidate_local"
        ),
        "name": manifest["model"]["name"],
        "artifact": artifact_path,
        "manifest": manifest_path,
        "bytes": model["bytes"],
        "sha256": model["sha256"],
        "runtime_bytes": manifest["runtime_bytes"],
        "runtime_budget_bytes": manifest["runtime_budget_bytes"],
        "label_count": manifest["model"]["label_count"],
        "max_length": manifest["model"]["max_length"],
        "base_model": manifest["model"]["base_model"],
        "base_revision": manifest["model"]["base_revision"],
        "tensor_contract": manifest["tensor_contract"],
        "license": manifest["licenses"]["artifact"],
        "release_eligible": manifest.get("release_eligible", False),
        "release_checks": manifest.get("release_checks"),
    }
    lock["semantic_replacement"]["status"] = candidate["status"]
    lock["semantic_replacement"]["candidate"] = candidate
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update models.lock.json from a trained artifact"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models.lock.json",
    )
    args = parser.parse_args()
    print(json.dumps(sync(args.artifact_dir, args.lock), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
