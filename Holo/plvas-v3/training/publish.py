from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .schema import sha256_file
from .sync_model_lock import sync
from .validate_artifacts import validate


def publish(args: argparse.Namespace) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    validation = validate(args.artifact_dir, release=True)
    manifest_path = args.artifact_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["licenses"]["artifact"].startswith("pending"):
        raise RuntimeError(
            "artifact license is pending; set it after maintainer review before publication"
        )
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or --token is required")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    try:
        api.create_branch(
            repo_id=args.repo_id,
            branch=args.branch,
            repo_type="model",
        )
    except Exception:
        pass
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.artifact_dir),
        revision=args.branch,
        commit_message=args.commit_message,
        ignore_patterns=[
            "provenance/**",
            "tokenizer_goldens.jsonl",
        ],
    )
    info = api.model_info(args.repo_id, revision=args.branch)
    revision = info.sha
    remote_verification = []
    with tempfile.TemporaryDirectory(prefix="plva-publish-verify-") as temporary:
        published_manifest = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename="model_manifest.json",
                revision=revision,
                repo_type="model",
                token=token,
                cache_dir=temporary,
                force_download=True,
            )
        )
        if sha256_file(published_manifest) != sha256_file(manifest_path):
            raise RuntimeError("published model_manifest.json hash mismatch")
        remote_verification.append("model_manifest.json")
        unique_artifacts = {
            item["path"]: item for item in manifest["artifacts"].values()
        }
        for relative, item in sorted(unique_artifacts.items()):
            downloaded = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=relative,
                    revision=revision,
                    repo_type="model",
                    token=token,
                    cache_dir=temporary,
                    force_download=True,
                )
            )
            if downloaded.stat().st_size != item["bytes"]:
                raise RuntimeError(f"published byte count mismatch for {relative}")
            if sha256_file(downloaded) != item["sha256"]:
                raise RuntimeError(f"published SHA-256 mismatch for {relative}")
            remote_verification.append(relative)
    model = manifest["artifacts"]["model.int8.onnx"]
    model_url = (
        f"https://huggingface.co/{args.repo_id}/resolve/{revision}/{model['path']}"
    )
    if args.tag:
        api.create_tag(
            repo_id=args.repo_id,
            tag=args.tag,
            revision=revision,
            repo_type="model",
        )

    sync(args.artifact_dir, args.lock)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    lock["semantic_replacement"]["status"] = "published"
    candidate = lock["semantic_replacement"]["candidate"]
    candidate.pop("artifact", None)
    candidate.pop("manifest", None)
    candidate |= {
        "status": "published",
        "repository": args.repo_id,
        "revision": revision,
        "url": model_url,
        "manifest_url": (
            f"https://huggingface.co/{args.repo_id}/resolve/{revision}/"
            "model_manifest.json"
        ),
        "license": manifest["licenses"]["artifact"],
    }
    args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return {
        "repo_id": args.repo_id,
        "branch": args.branch,
        "revision": revision,
        "tag": args.tag,
        "model_url": model_url,
        "commit_url": getattr(commit, "commit_url", None),
        "validation": validation,
        "remote_artifacts_verified": remote_verification,
        "lock": lock["semantic_replacement"]["candidate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish verified PLVA artifacts")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--branch", default="plva-tagger-v1")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--commit-message", default="Publish PLVA compact PII tagger")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models.lock.json",
    )
    args = parser.parse_args()
    print(json.dumps(publish(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
