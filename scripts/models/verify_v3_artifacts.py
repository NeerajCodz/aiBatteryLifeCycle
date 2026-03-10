from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_entries(v3_root: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    models = manifest.get("models", {})

    for model_id, meta in models.items():
        rel = meta.get("file")
        sha = meta.get("sha256")
        if rel is None:
            # virtual ensemble model
            continue
        p = v3_root / rel
        if not p.exists():
            errors.append(f"model file missing: {model_id} -> {rel}")
            continue
        if not sha:
            errors.append(f"sha256 missing for model: {model_id}")
            continue
        if sha != sha256_file(p):
            errors.append(f"sha256 mismatch for model: {model_id} -> {rel}")

    return errors


def verify_sections(v3_root: Path, checksums: dict, section: str) -> list[str]:
    errors: list[str] = []
    entries = checksums.get(section, {})
    if not isinstance(entries, dict):
        return [f"checksums.{section} must be an object"]

    for rel, expected in entries.items():
        p = v3_root / rel
        if not p.exists():
            errors.append(f"checksums.{section} missing file: {rel}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            errors.append(f"checksums.{section} mismatch: {rel}")

    return errors


def verify_scalers(v3_root: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    scalers = manifest.get("scalers", {})
    for name, rel in scalers.items():
        p = v3_root / rel
        if not p.exists():
            errors.append(f"scaler missing: {name} -> {rel}")
    scaler_checksums = manifest.get("scaler_checksums", {})
    for name, expected in scaler_checksums.items():
        rel = scalers.get(name)
        if not rel or expected is None:
            continue
        p = v3_root / rel
        if not p.exists():
            continue
        actual = sha256_file(p)
        if actual != expected:
            errors.append(f"scaler checksum mismatch: {name} -> {rel}")
    return errors


def verify_auxiliary(v3_root: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    aux = manifest.get("auxiliary_artifacts", {})
    if not isinstance(aux, dict):
        return ["auxiliary_artifacts must be an object"]
    for aux_id, aux_meta in aux.items():
        if not isinstance(aux_meta, dict):
            errors.append(f"auxiliary_artifacts.{aux_id} must be an object")
            continue
        rel = aux_meta.get("file")
        sha = aux_meta.get("sha256")
        if not rel:
            continue
        p = v3_root / rel
        if not p.exists():
            errors.append(f"auxiliary file missing: {aux_id} -> {rel}")
            continue
        if sha and sha != sha256_file(p):
            errors.append(f"auxiliary sha mismatch: {aux_id} -> {rel}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v3 artifact checksums from models.json")
    parser.add_argument("--v3-root", default="artifacts/v3", help="Path to v3 artifact root")
    args = parser.parse_args()

    v3_root = Path(args.v3_root).resolve()
    manifest_path = v3_root / "models.json"
    if not manifest_path.exists():
        raise SystemExit(f"models.json not found at: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    errors.extend(verify_model_entries(v3_root, manifest))
    errors.extend(verify_scalers(v3_root, manifest))
    errors.extend(verify_auxiliary(v3_root, manifest))

    checksums = manifest.get("checksums", {})
    for section in ("models", "scalers", "results", "features"):
        errors.extend(verify_sections(v3_root, checksums, section))

    if errors:
        print("VERIFICATION FAILED")
        for e in errors:
            print(f" - {e}")
        return 1

    print("VERIFICATION PASSED")
    print(f"manifest: {manifest_path}")
    print(f"models: {len(manifest.get('models', {}))}")
    print(f"auxiliary: {len(manifest.get('auxiliary_artifacts', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
