from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from training.schema import sha256_file
from training.visual.prepare_webpii import DETECTOR_CLASSES
from training.visual.stage_supplemental import validate_prepared_supplemental
from training.visual.train_detector import validate_composed_dataset_contract


STAGE_SCHEMA_VERSION = 1
TREE_HASH_ALGORITHM = "sha256(relative-posix-nul-size-nul-file-sha256-lf-v1)"
COPY_WORKERS = 16
SOURCE_SPLITS = {
    "webpii": ("train", "test"),
    "synthetic": ("train", "validation"),
    "supplemental_ats": ("train", "test"),
}


def _load_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{context} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must contain a JSON object")
    return value


def _resolve_dataset_path(dataset: Mapping[str, Any], value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (Path(str(dataset.get("path", "."))) / path).resolve()


def _source_contract(
    dataset_yaml: Path,
    composed_manifest_path: Path,
    expected_source_roots: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], str]:
    import yaml

    composed = _load_json(composed_manifest_path, "composed detector manifest")
    try:
        dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimeError("composed detector dataset YAML is invalid") from exc
    if not isinstance(dataset, dict):
        raise RuntimeError("composed detector dataset YAML must contain a mapping")
    yaml_hash = sha256_file(dataset_yaml)
    if composed.get("dataset_yaml_sha256") != yaml_hash:
        raise RuntimeError("composed detector dataset YAML SHA-256 mismatch")
    declared_yaml = composed.get("dataset_yaml")
    if (
        not isinstance(declared_yaml, str)
        or Path(declared_yaml).resolve() != dataset_yaml.resolve()
    ):
        raise RuntimeError(
            "composed detector manifest points at a different dataset YAML"
        )

    sources = composed.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("composed detector manifest is missing source contracts")
    expected_names = {"webpii", "synthetic"}
    if "supplemental_ats" in sources:
        expected_names.add("supplemental_ats")
    if set(sources) != expected_names:
        raise RuntimeError(
            "composed detector manifest has unexpected or missing sources"
        )
    if set(expected_source_roots) != expected_names:
        raise RuntimeError(
            "expected detector source roots do not match the composed manifest"
        )

    source_hashes: dict[str, str] = {}
    for name in sorted(expected_names):
        entry = sources[name]
        if not isinstance(entry, dict):
            raise RuntimeError(f"composed {name} source contract must be an object")
        root = expected_source_roots[name].resolve()
        expected_manifest = root / "manifest.json"
        declared_manifest = entry.get("manifest")
        if (
            not isinstance(declared_manifest, str)
            or Path(declared_manifest).resolve() != expected_manifest
        ):
            raise RuntimeError(
                f"composed {name} source path differs from the approved run root"
            )
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"composed {name} source root is missing or unsafe")
        actual_hash = sha256_file(expected_manifest)
        if entry.get("manifest_sha256") != actual_hash:
            raise RuntimeError(f"composed {name} source manifest SHA-256 mismatch")
        source_hashes[name] = actual_hash

    expected_train = [
        expected_source_roots["webpii"].resolve() / "images/train",
        expected_source_roots["synthetic"].resolve() / "images/train",
    ]
    if "supplemental_ats" in expected_names:
        expected_train.append(
            expected_source_roots["supplemental_ats"].resolve() / "images/train"
        )
    raw_train = dataset.get("train")
    train_values = raw_train if isinstance(raw_train, list) else [raw_train]
    actual_train = [_resolve_dataset_path(dataset, value) for value in train_values]
    if actual_train != expected_train or len(set(actual_train)) != len(actual_train):
        raise RuntimeError(
            "composed train paths must be exactly WebPII/train, synthetic/train, "
            "and the optional supplemental/train"
        )
    actual_val = _resolve_dataset_path(dataset, dataset.get("val"))
    actual_test = _resolve_dataset_path(dataset, dataset.get("test"))
    if actual_val != expected_source_roots["synthetic"].resolve() / "images/validation":
        raise RuntimeError("checkpoint selection must use only synthetic validation")
    if actual_test != expected_source_roots["webpii"].resolve() / "images/test":
        raise RuntimeError("published test must use only WebPII/test")
    if any("test" in {part.lower() for part in path.parts} for path in actual_train):
        raise RuntimeError("a published or supplemental test path leaked into training")
    if dataset.get("names") not in (
        list(DETECTOR_CLASSES),
        {index: name for index, name in enumerate(DETECTOR_CLASSES)},
    ):
        raise RuntimeError("composed detector YAML class order is invalid")
    validate_composed_dataset_contract(dataset, composed, composed_manifest_path)

    contract = {
        "dataset_yaml_sha256": yaml_hash,
        "composed_manifest_sha256": sha256_file(composed_manifest_path),
        "source_manifest_sha256": source_hashes,
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return dataset, composed, source_hashes, contract_hash


def _validate_copy_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 256:
        raise ValueError("detector staging copy_workers must be between 1 and 256")
    return value


def _copy_tree_once(
    source: Path, destination: Path, *, copy_workers: int
) -> dict[str, Any]:
    copy_workers = _validate_copy_workers(copy_workers)
    destination.mkdir(parents=True, exist_ok=False)
    files: list[Path] = []
    for path in sorted(
        source.rglob("*"),
        key=lambda item: item.relative_to(source).as_posix().encode("utf-8"),
    ):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise RuntimeError(f"detector source tree contains a symlink: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise RuntimeError(
                f"detector source tree contains a non-regular file: {path}"
            )
        files.append(path)

    def copy_one(path: Path) -> tuple[str, int, str]:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = path.stat()
        digest = hashlib.sha256()
        copied = 0
        with path.open("rb") as source_file, target.open("xb") as destination_file:
            for chunk in iter(lambda: source_file.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
                destination_file.write(chunk)
                copied += len(chunk)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or copied != before.st_size
        ):
            raise RuntimeError(f"detector source file changed while staging: {path}")
        shutil.copystat(path, target)
        return relative.as_posix(), copied, digest.hexdigest()

    with ThreadPoolExecutor(max_workers=copy_workers) as executor:
        fingerprints = list(executor.map(copy_one, files))
    total_bytes = sum(size for _, size, _ in fingerprints)
    return _inventory_summary(fingerprints, total_bytes)


def _inventory_summary(
    fingerprints: list[tuple[str, int, str]], total_bytes: int
) -> dict[str, Any]:
    aggregate = hashlib.sha256()
    for relative, size, digest in sorted(
        fingerprints, key=lambda item: item[0].encode("utf-8")
    ):
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode())
    return {
        "files": len(fingerprints),
        "bytes": total_bytes,
        "tree_sha256": aggregate.hexdigest(),
        "tree_hash_algorithm": TREE_HASH_ALGORITHM,
    }


def _inventory_tree(root: Path) -> dict[str, Any]:
    fingerprints: list[tuple[str, int, str]] = []
    total_bytes = 0
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"local detector source tree is missing or unsafe: {root}")
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    ):
        if path.is_symlink():
            raise RuntimeError(f"local detector source tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        fingerprints.append((relative, size, sha256_file(path)))
        total_bytes += size
    return _inventory_summary(fingerprints, total_bytes)


def _validate_pairs(root: Path, splits: tuple[str, ...], source_name: str) -> None:
    for split in splits:
        images = root / "images" / split
        labels = root / "labels" / split
        if (
            images.is_symlink()
            or labels.is_symlink()
            or not images.is_dir()
            or not labels.is_dir()
        ):
            raise RuntimeError(
                f"local {source_name}/{split} image or label directory is missing"
            )
        image_relatives = {
            path.relative_to(images).with_suffix(".txt").as_posix()
            for path in images.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        label_relatives = {
            path.relative_to(labels).as_posix()
            for path in labels.rglob("*.txt")
            if path.is_file() and not path.is_symlink()
        }
        if not image_relatives or image_relatives != label_relatives:
            raise RuntimeError(
                f"local {source_name}/{split} image-label inventory is incomplete"
            )


def _verify_stage(root: Path, expected_contract_hash: str) -> dict[str, Any]:
    stage_path = root / "staging_manifest.json"
    stage = _load_json(stage_path, "local detector staging manifest")
    if stage.get("schema_version") != STAGE_SCHEMA_VERSION:
        raise RuntimeError("local detector staging manifest schema is unsupported")
    if stage.get("source_contract_sha256") != expected_contract_hash:
        raise RuntimeError("local detector staging source contract is stale")
    sources = stage.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("local detector staging source inventories are missing")
    for name, summary in sources.items():
        if not isinstance(summary, dict):
            raise RuntimeError("local detector staging inventory is invalid")
        if _inventory_tree(root / name) != summary.get("inventory"):
            raise RuntimeError(
                f"local detector staged {name} tree failed hash verification"
            )
        _validate_pairs(root / name, SOURCE_SPLITS[name], name)
    if sha256_file(root / "dataset.yaml") != stage.get("local_dataset_yaml_sha256"):
        raise RuntimeError("local detector dataset YAML failed hash verification")
    if sha256_file(root / "source_manifest.json") != stage.get(
        "local_source_manifest_sha256"
    ):
        raise RuntimeError("local detector source manifest failed hash verification")
    return stage


def stage_composed_dataset(
    dataset_yaml: Path,
    composed_manifest_path: Path,
    expected_source_roots: Mapping[str, Path],
    destination_root: Path,
    *,
    force: bool = False,
    copy_workers: int = COPY_WORKERS,
) -> dict[str, Any]:
    """Copy the frozen detector dataset from a Volume to ephemeral local disk once."""

    copy_workers = _validate_copy_workers(copy_workers)

    _, composed, source_hashes, contract_hash = _source_contract(
        dataset_yaml, composed_manifest_path, expected_source_roots
    )
    had_destination = destination_root.exists()
    if destination_root.exists() and not force:
        stage = _verify_stage(destination_root, contract_hash)
        return {
            "mode": "reused-verified-local-stage",
            "root": str(destination_root),
            "dataset_yaml": str(destination_root / "dataset.yaml"),
            "source_manifest": str(destination_root / "source_manifest.json"),
            "staging_manifest": str(destination_root / "staging_manifest.json"),
            "metadata": stage,
        }

    temporary = destination_root.with_name(
        f".{destination_root.name}.partial-{os.getpid()}"
    )
    backup = destination_root.with_name(
        f".{destination_root.name}.previous-{os.getpid()}"
    )
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    temporary.mkdir(parents=True)
    inventories: dict[str, dict[str, Any]] = {}
    try:
        for name, source_root in expected_source_roots.items():
            inventory = _copy_tree_once(
                source_root, temporary / name, copy_workers=copy_workers
            )
            inventories[name] = {
                "volume_root": str(source_root),
                "volume_manifest_sha256": source_hashes[name],
                "local_root": str(destination_root / name),
                "inventory": inventory,
            }
            _validate_pairs(temporary / name, SOURCE_SPLITS[name], name)
        if "supplemental_ats" in expected_source_roots:
            validate_prepared_supplemental(temporary / "supplemental_ats")

        train_roots = [
            destination_root / "webpii/images/train",
            destination_root / "synthetic/images/train",
        ]
        if "supplemental_ats" in expected_source_roots:
            train_roots.append(destination_root / "supplemental_ats/images/train")
        local_yaml_lines = [
            "path: /",
            "train:",
            *[f"  - {path}" for path in train_roots],
            f"val: {destination_root / 'synthetic/images/validation'}",
            f"test: {destination_root / 'webpii/images/test'}",
            f"nc: {len(DETECTOR_CLASSES)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(DETECTOR_CLASSES)],
            "",
        ]
        local_yaml = temporary / "dataset.yaml"
        local_yaml.write_text("\n".join(local_yaml_lines), encoding="utf-8")

        local_composed = deepcopy(composed)
        for name in expected_source_roots:
            entry = local_composed["sources"][name]
            entry["volume_manifest"] = entry["manifest"]
            entry["manifest"] = str(destination_root / name / "manifest.json")
        local_composed["dataset_yaml"] = str(destination_root / "dataset.yaml")
        local_composed["dataset_yaml_sha256"] = sha256_file(local_yaml)
        local_composed["ephemeral_staging"] = {
            "source_contract_sha256": contract_hash,
            "volume_dataset_yaml": str(dataset_yaml),
            "volume_dataset_yaml_sha256": sha256_file(dataset_yaml),
            "volume_source_manifest": str(composed_manifest_path),
            "volume_source_manifest_sha256": sha256_file(composed_manifest_path),
            "test_used_for_checkpoint_selection": False,
        }
        local_manifest = temporary / "source_manifest.json"
        local_manifest.write_text(
            json.dumps(local_composed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage = {
            "schema_version": STAGE_SCHEMA_VERSION,
            "mode": "volume-to-container-ephemeral-copy",
            "copy_workers": copy_workers,
            "source_contract_sha256": contract_hash,
            "tree_hash_algorithm": TREE_HASH_ALGORITHM,
            "sources": inventories,
            "local_dataset_yaml": str(destination_root / "dataset.yaml"),
            "local_dataset_yaml_sha256": sha256_file(local_yaml),
            "local_source_manifest": str(destination_root / "source_manifest.json"),
            "local_source_manifest_sha256": sha256_file(local_manifest),
            "training_sources": [f"{name}/train" for name in expected_source_roots],
            "selection_source": "synthetic/validation",
            "published_test_source": "webpii/test",
            "supplemental_test_source": (
                "supplemental_ats/test"
                if "supplemental_ats" in expected_source_roots
                else None
            ),
            "test_used_for_checkpoint_selection": False,
        }
        (temporary / "staging_manifest.json").write_text(
            json.dumps(stage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        destination_root.parent.mkdir(parents=True, exist_ok=True)
        if destination_root.exists():
            destination_root.rename(backup)
        temporary.rename(destination_root)
        verified = _verify_stage(destination_root, contract_hash)
        import yaml

        local_dataset = yaml.safe_load(
            (destination_root / "dataset.yaml").read_text(encoding="utf-8")
        )
        local_source = _load_json(
            destination_root / "source_manifest.json",
            "local composed detector manifest",
        )
        validate_composed_dataset_contract(
            local_dataset,
            local_source,
            destination_root / "source_manifest.json",
        )
        shutil.rmtree(backup, ignore_errors=True)
        return {
            "mode": "copied-and-verified-local-stage",
            "root": str(destination_root),
            "dataset_yaml": str(destination_root / "dataset.yaml"),
            "source_manifest": str(destination_root / "source_manifest.json"),
            "staging_manifest": str(destination_root / "staging_manifest.json"),
            "metadata": verified,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if destination_root.exists() and backup.exists():
            shutil.rmtree(destination_root, ignore_errors=True)
        elif destination_root.exists() and not had_destination:
            shutil.rmtree(destination_root, ignore_errors=True)
        if backup.exists() and not destination_root.exists():
            backup.rename(destination_root)
        raise


def persist_stage_metadata(
    stage: Mapping[str, Any], output_dir: Path
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "dataset_yaml": (
            Path(str(stage["dataset_yaml"])),
            output_dir / "staged-dataset.yaml",
        ),
        "source_manifest": (
            Path(str(stage["source_manifest"])),
            output_dir / "staged-source-manifest.json",
        ),
        "staging_manifest": (
            Path(str(stage["staging_manifest"])),
            output_dir / "dataset-staging-manifest.json",
        ),
    }
    result: dict[str, str] = {}
    for name, (source, destination) in copies.items():
        temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        result[f"{name}_path"] = str(destination)
        result[f"{name}_sha256"] = sha256_file(destination)
    return result


def verify_persisted_stage_metadata(
    stage: Mapping[str, Any], output_dir: Path
) -> dict[str, str]:
    """Verify a resumed container staged the exact dataset already persisted.

    Resume must not overwrite the durable provenance it is meant to validate.
    The ephemeral paths are stable per run ID, so byte-for-byte equality is
    required for the rewritten dataset, composed source contract, and staging
    inventory before an existing checkpoint may be loaded.
    """

    copies = {
        "dataset_yaml": (
            Path(str(stage["dataset_yaml"])),
            output_dir / "staged-dataset.yaml",
        ),
        "source_manifest": (
            Path(str(stage["source_manifest"])),
            output_dir / "staged-source-manifest.json",
        ),
        "staging_manifest": (
            Path(str(stage["staging_manifest"])),
            output_dir / "dataset-staging-manifest.json",
        ),
    }
    result: dict[str, str] = {}
    for name, (current, persisted) in copies.items():
        if current.is_symlink() or not current.is_file():
            raise RuntimeError(f"current {name} staging metadata is missing or unsafe")
        if persisted.is_symlink() or not persisted.is_file():
            raise RuntimeError(
                f"persisted {name} staging metadata is missing or unsafe"
            )
        current_hash = sha256_file(current)
        persisted_hash = sha256_file(persisted)
        if current_hash != persisted_hash:
            raise RuntimeError(
                f"resumed {name} staging metadata differs from the original run"
            )
        result[f"{name}_path"] = str(persisted)
        result[f"{name}_sha256"] = persisted_hash
    return result
