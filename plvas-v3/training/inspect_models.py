from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "models.lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_asset_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    destination = (root / relative).resolve()
    if destination == root or root not in destination.parents:
        raise ValueError(f"unsafe asset path: {relative}")
    return destination


def tensor_description(value: Any) -> dict[str, Any]:
    return {
        "name": value.name,
        "dtype": value.type,
        "shape": [item if isinstance(item, int) else str(item) for item in value.shape],
    }


def inspect_onnx(path: Path) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "ir_version": model.ir_version,
        "opsets": {
            item.domain or "ai.onnx": item.version for item in model.opset_import
        },
        "inputs": [tensor_description(value) for value in session.get_inputs()],
        "outputs": [tensor_description(value) for value in session.get_outputs()],
        "operators": sorted({node.op_type for node in model.graph.node}),
        "providers": session.get_providers(),
    }


def _port(port: ElementTree.Element) -> dict[str, Any]:
    return {
        "name": port.attrib.get("names"),
        "dtype": port.attrib.get("precision"),
        "shape": [int(value.text) for value in port.findall("dim")],
    }


def inspect_openvino_ir(path: Path) -> dict[str, Any]:
    root = ElementTree.parse(path).getroot()
    layers_node = root.find("layers")
    edges_node = root.find("edges")
    if layers_node is None or edges_node is None:
        raise ValueError(f"{path} is missing OpenVINO layers or edges")
    layers = {layer.attrib["id"]: layer for layer in layers_node}
    edges = list(edges_node)
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for layer in layers.values():
        if layer.attrib.get("type") == "Parameter":
            port = layer.find("./output/port")
            if port is None:
                raise ValueError(
                    f"OpenVINO Parameter {layer.attrib['name']} has no output"
                )
            inputs.append(_port(port) | {"name": layer.attrib.get("name")})
        elif layer.attrib.get("type") == "Result":
            port = layer.find("./input/port")
            if port is None:
                raise ValueError(f"OpenVINO Result {layer.attrib['name']} has no input")
            item = _port(port)
            incoming = [
                edge for edge in edges if edge.attrib["to-layer"] == layer.attrib["id"]
            ]
            if len(incoming) != 1:
                raise ValueError(
                    f"OpenVINO Result {layer.attrib['name']} has {len(incoming)} inputs"
                )
            source = layers[incoming[0].attrib["from-layer"]]
            item["source_layer"] = source.attrib.get("name")
            outputs.append(item)
    return {
        "ir_version": int(root.attrib["version"]),
        "inputs": inputs,
        "outputs": outputs,
    }


def validate_tensor_list(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    context: str,
) -> None:
    if len(actual) != len(expected):
        raise ValueError(
            f"{context}: expected {len(expected)} tensors, found {len(actual)}"
        )
    unmatched = list(actual)
    for index, wanted in enumerate(expected):
        if wanted.get("name") is not None:
            candidates = [
                item for item in unmatched if item.get("name") == wanted["name"]
            ]
            if not candidates:
                raise ValueError(f"{context}: missing tensor {wanted['name']}")
            found = candidates[0]
        else:
            found = unmatched[0]
        unmatched.remove(found)
        if wanted.get("dtype") is not None and found.get("dtype") != wanted["dtype"]:
            raise ValueError(
                f"{context}: {found.get('name', index)} dtype {found.get('dtype')} "
                f"!= {wanted['dtype']}"
            )
        shape = found.get("shape", [])
        if wanted.get("rank") is not None and len(shape) != wanted["rank"]:
            raise ValueError(
                f"{context}: {found.get('name', index)} rank {len(shape)} != {wanted['rank']}"
            )
        if wanted.get("shape") is not None and shape != wanted["shape"]:
            raise ValueError(
                f"{context}: {found.get('name', index)} shape {shape} != {wanted['shape']}"
            )
        for raw_dimension, value in wanted.get("fixed_dimensions", {}).items():
            dimension = int(raw_dimension)
            if dimension >= len(shape) or shape[dimension] != value:
                raise ValueError(
                    f"{context}: {found.get('name', index)} dimension {dimension} "
                    f"is {shape[dimension] if dimension < len(shape) else '<missing>'}, expected {value}"
                )


def validate_contract(
    description: dict[str, Any], expected: dict[str, Any], name: str
) -> None:
    if (
        "opset" in expected
        and description.get("opsets", {}).get("ai.onnx") != expected["opset"]
    ):
        raise ValueError(
            f"{name}: ONNX opset {description.get('opsets', {}).get('ai.onnx')} != {expected['opset']}"
        )
    validate_tensor_list(
        description["inputs"], expected.get("inputs", []), context=f"{name} inputs"
    )
    validate_tensor_list(
        description["outputs"], expected.get("outputs", []), context=f"{name} outputs"
    )


def inspect_asset(name: str, asset: dict[str, Any], root: Path) -> dict[str, Any]:
    path = resolve_asset_path(root, asset["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != asset["bytes"]:
        raise ValueError(f"{name}: byte count {size} != {asset['bytes']}")
    digest = sha256_file(path)
    if digest != asset["sha256"]:
        raise ValueError(f"{name}: SHA-256 {digest} != {asset['sha256']}")

    result: dict[str, Any] = {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "format": asset["format"],
    }
    if asset["format"] == "onnx":
        result["contract"] = inspect_onnx(path)
        validate_contract(result["contract"], asset.get("tensor_contract", {}), name)
    elif asset["format"] == "openvino_ir_xml":
        result["contract"] = inspect_openvino_ir(path)
        validate_contract(result["contract"], asset.get("tensor_contract", {}), name)
    elif asset["format"] == "json":
        document = json.loads(path.read_text(encoding="utf-8"))
        result["json_type"] = type(document).__name__
        if asset.get("role") == "bootstrap_config":
            labels = document.get("id2label", {})
            if len(labels) != 35 or document.get("vocab_size") != 19730:
                raise ValueError(
                    f"{name}: Rampart config label or vocabulary count changed"
                )
            result["label_count"] = len(labels)
            result["vocab_size"] = document["vocab_size"]
    elif asset["format"] == "text_dictionary":
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != asset["line_count"]:
            raise ValueError(
                f"{name}: dictionary has {len(lines)} lines, expected {asset['line_count']}"
            )
        result["line_count"] = len(lines)
    return result


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 2:
        raise ValueError("models.lock.json schema_version must be 2")
    root = (args.asset_root or lock_path.parent / lock["asset_root"]).resolve()
    requested = set(args.asset)
    groups = set(args.group)
    unknown = requested - lock["assets"].keys()
    if unknown:
        raise ValueError(f"unknown assets: {', '.join(sorted(unknown))}")
    explicit = bool(requested or groups)
    selected = {
        name: asset
        for name, asset in lock["assets"].items()
        if name in requested
        or asset["group"] in groups
        or (
            not explicit and (args.include_optional or asset.get("default_fetch", True))
        )
    }
    results = {
        name: inspect_asset(name, asset, root) for name, asset in selected.items()
    }
    report = {
        "schema_version": 1,
        "lock": str(lock_path),
        "asset_root": str(root),
        "asset_count": len(results),
        "assets": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect locked PLVA model tensor contracts"
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(inspect(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
