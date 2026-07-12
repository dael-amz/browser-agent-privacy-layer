from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from training.schema import sha256_file


ATTEMPT_SCHEMA_VERSION = 1
DEFAULT_ATTEMPT_ID = "default"
ATTEMPT_MANIFEST_NAME = "training-attempt.json"
SEED_CHECKPOINT_NAME = "frozen-seed.pt"
SEED_PROVENANCE_NAME = "frozen-seed-provenance.json"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_attempt_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", value):
        raise ValueError("visual attempt must be 1-32 safe filename characters")
    if value in {".", ".."}:
        raise ValueError("visual attempt cannot be a path component")
    return value


def attempt_output_dir(run_root: Path, attempt_id: str) -> Path:
    attempt_id = validate_attempt_id(attempt_id)
    name = "training" if attempt_id == DEFAULT_ATTEMPT_ID else f"training-{attempt_id}"
    return run_root / "visual" / name


def attempt_artifact_dir(run_root: Path, attempt_id: str) -> Path:
    attempt_id = validate_attempt_id(attempt_id)
    name = "artifacts" if attempt_id == DEFAULT_ATTEMPT_ID else f"artifacts-{attempt_id}"
    return run_root / "visual" / name


def resolve_frozen_seed(
    value: str,
    *,
    volume_root: Path = Path("/vol"),
    snapshot_root: Path | None = None,
) -> Path:
    """Resolve a regular checkpoint confined to the immutable snapshot tree."""
    volume_root = volume_root.resolve()
    allowed_root = (snapshot_root or volume_root / "snapshots").resolve()
    raw = Path(value)
    candidate = raw if raw.is_absolute() else volume_root / raw
    resolved = candidate.resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise RuntimeError("visual seed must be below the Volume snapshots root")
    if candidate.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("visual seed checkpoint is missing or unsafe")
    return resolved


def _read_optional_json(path: Path | None, context: str) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return value


def initialize_seeded_attempt(
    *,
    attempt_root: Path,
    run_id: str,
    attempt_id: str,
    source_checkpoint: Path,
    expected_sha256: str,
    seed_provenance: Mapping[str, Any],
    optimizer_policy: Mapping[str, Any],
    license_path: Path,
    inherited_provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically create a new immutable-seed attempt without overwriting state."""
    attempt_id = validate_attempt_id(attempt_id)
    if attempt_id == DEFAULT_ATTEMPT_ID:
        raise RuntimeError("a frozen-seed retry requires a non-default attempt ID")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("visual seed SHA-256 must be 64 lowercase hex characters")
    if attempt_root.exists():
        raise FileExistsError(
            f"visual attempt already exists; resume it explicitly: {attempt_root}"
        )
    if source_checkpoint.is_symlink() or not source_checkpoint.is_file():
        raise RuntimeError("visual seed checkpoint is missing or unsafe")
    source_sha256 = sha256_file(source_checkpoint)
    if source_sha256 != expected_sha256:
        raise RuntimeError("visual seed checkpoint SHA-256 does not match the CLI pin")
    if license_path.is_symlink() or not license_path.is_file():
        raise RuntimeError("detector seed license is missing or unsafe")
    license_text = license_path.read_text(encoding="utf-8").strip()
    if not license_text or license_text == "NOASSERTION":
        raise RuntimeError("detector seed requires an explicit license")
    inherited = _read_optional_json(
        inherited_provenance_path, "inherited detector provenance"
    )
    optimizer_document = dict(optimizer_policy)
    provenance_document = {
        "schema_version": 1,
        "kind": "frozen-volume-snapshot-seed",
        "source": {
            "path": str(source_checkpoint.resolve()),
            "bytes": source_checkpoint.stat().st_size,
            "sha256": source_sha256,
            "expected_sha256": expected_sha256,
        },
        "declared_provenance": dict(seed_provenance),
        "license": {
            "value": license_text,
            "path": str(license_path.resolve()),
            "sha256": sha256_file(license_path),
        },
        "inherited_detector_provenance": inherited,
        "inherited_detector_provenance_sha256": (
            sha256_file(inherited_provenance_path)
            if inherited is not None and inherited_provenance_path is not None
            else None
        ),
    }
    manifest = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root.resolve()),
        "mode": "new-non-resume-from-frozen-seed",
        "resume_allowed": True,
        "seed": {
            "checkpoint": SEED_CHECKPOINT_NAME,
            "provenance": SEED_PROVENANCE_NAME,
            "source_path": str(source_checkpoint.resolve()),
            "bytes": source_checkpoint.stat().st_size,
            "sha256": source_sha256,
        },
        "optimizer_policy": optimizer_document,
        "optimizer_policy_sha256": _canonical_hash(optimizer_document),
    }

    attempt_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = attempt_root.with_name(
        f".{attempt_root.name}.initializing-{os.getpid()}"
    )
    if temporary.exists():
        raise RuntimeError(f"stale attempt initializer exists: {temporary}")
    temporary.mkdir()
    try:
        staged_seed = temporary / SEED_CHECKPOINT_NAME
        shutil.copy2(source_checkpoint, staged_seed)
        if (
            staged_seed.stat().st_size != source_checkpoint.stat().st_size
            or sha256_file(staged_seed) != source_sha256
        ):
            raise RuntimeError("staged visual seed failed byte/hash verification")
        (temporary / SEED_PROVENANCE_NAME).write_text(
            json.dumps(provenance_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / ATTEMPT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(attempt_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_seeded_attempt(
        attempt_root=attempt_root,
        run_id=run_id,
        attempt_id=attempt_id,
        optimizer_policy=optimizer_policy,
        license_path=license_path,
    )


def load_seeded_attempt(
    *,
    attempt_root: Path,
    run_id: str,
    attempt_id: str,
    optimizer_policy: Mapping[str, Any],
    license_path: Path,
) -> dict[str, Any]:
    attempt_id = validate_attempt_id(attempt_id)
    manifest = _read_optional_json(
        attempt_root / ATTEMPT_MANIFEST_NAME, "visual training attempt manifest"
    )
    if manifest is None:
        raise RuntimeError("visual training attempt manifest is missing")
    expected_policy = dict(optimizer_policy)
    checks = {
        "schema_version": manifest.get("schema_version") == ATTEMPT_SCHEMA_VERSION,
        "run_id": manifest.get("run_id") == run_id,
        "attempt_id": manifest.get("attempt_id") == attempt_id,
        "attempt_root": manifest.get("attempt_root") == str(attempt_root.resolve()),
        "mode": manifest.get("mode") == "new-non-resume-from-frozen-seed",
        "resume_allowed": manifest.get("resume_allowed") is True,
        "optimizer_policy": manifest.get("optimizer_policy") == expected_policy,
        "optimizer_policy_sha256": manifest.get("optimizer_policy_sha256")
        == _canonical_hash(expected_policy),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "visual training attempt contract mismatch: " + ", ".join(failed)
        )
    seed = manifest.get("seed")
    if not isinstance(seed, dict) or seed.get("checkpoint") != SEED_CHECKPOINT_NAME:
        raise RuntimeError("visual training attempt seed contract is invalid")
    checkpoint = attempt_root / SEED_CHECKPOINT_NAME
    provenance = attempt_root / SEED_PROVENANCE_NAME
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise RuntimeError("visual training attempt seed is missing or unsafe")
    if provenance.is_symlink() or not provenance.is_file():
        raise RuntimeError("visual training seed provenance is missing or unsafe")
    if checkpoint.stat().st_size != seed.get("bytes"):
        raise RuntimeError("visual training attempt seed byte count changed")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != seed.get("sha256"):
        raise RuntimeError("visual training attempt seed SHA-256 changed")
    provenance_document = _read_optional_json(
        provenance, "visual training seed provenance"
    )
    if provenance_document is None:
        raise RuntimeError("visual training seed provenance is missing")
    license_entry = provenance_document.get("license")
    if not isinstance(license_entry, dict):
        raise RuntimeError("visual training seed license provenance is missing")
    if license_path.is_symlink() or not license_path.is_file():
        raise RuntimeError("detector seed license is missing or unsafe")
    if sha256_file(license_path) != license_entry.get("sha256"):
        raise RuntimeError("detector seed license changed after attempt creation")
    return {
        "manifest": manifest,
        "manifest_path": attempt_root / ATTEMPT_MANIFEST_NAME,
        "manifest_sha256": sha256_file(attempt_root / ATTEMPT_MANIFEST_NAME),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "provenance": provenance_document,
        "provenance_path": provenance,
        "provenance_sha256": sha256_file(provenance),
        "license": license_entry["value"],
    }
