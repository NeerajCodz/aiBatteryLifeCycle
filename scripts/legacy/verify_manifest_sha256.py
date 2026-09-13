from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_entries(version_root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    models = manifest.get("models", {})
    if isinstance(models, dict):
        for model_id, meta in models.items():
            if not isinstance(meta, dict):
                continue
            rel = meta.get("file")
            expected = meta.get("sha256")
            if rel in (None, ""):
                continue
            p = version_root / rel
            if not p.exists():
                errors.append(f"{version_root.name}: missing model file {model_id} -> {rel}")
                continue
            if not expected:
                errors.append(f"{version_root.name}: missing model hash {model_id}")
                continue
            actual = sha256_file(p)
            if actual != expected:
                errors.append(f"{version_root.name}: model hash mismatch {model_id} -> {rel}")

    aux = manifest.get("auxiliary_artifacts", {})
    if isinstance(aux, dict):
        for aux_id, meta in aux.items():
            if not isinstance(meta, dict):
                continue
            rel = meta.get("file")
            expected = meta.get("sha256")
            if not rel:
                continue
            p = version_root / rel
            if not p.exists():
                errors.append(f"{version_root.name}: missing auxiliary file {aux_id} -> {rel}")
                continue
            if expected and sha256_file(p) != expected:
                errors.append(f"{version_root.name}: auxiliary hash mismatch {aux_id} -> {rel}")

    scaler_checksums = manifest.get("scaler_checksums", {})
    scalers = manifest.get("scalers", {})
    if isinstance(scaler_checksums, dict) and isinstance(scalers, dict):
        for k, expected in scaler_checksums.items():
            rel = scalers.get(k)
            if not rel or expected is None:
                continue
            p = version_root / rel
            if not p.exists():
                errors.append(f"{version_root.name}: missing scaler {k} -> {rel}")
                continue
            if sha256_file(p) != expected:
                errors.append(f"{version_root.name}: scaler hash mismatch {k} -> {rel}")

    checksums = manifest.get("checksums", {})
    if isinstance(checksums, dict):
        for section in ("models", "scalers", "results", "features", "figures"):
            entries = checksums.get(section, {})
            if not isinstance(entries, dict):
                continue
            for rel, expected in entries.items():
                p = version_root / rel
                if not p.exists():
                    errors.append(f"{version_root.name}: checksums.{section} missing {rel}")
                    continue
                if sha256_file(p) != expected:
                    errors.append(f"{version_root.name}: checksums.{section} mismatch {rel}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHA256 entries in models.json")
    parser.add_argument("--artifacts-root", default="artifacts", help="Artifacts root path")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3"],
        help="Versions to verify (default: v1 v2 v3)",
    )
    args = parser.parse_args()

    root = Path(args.artifacts_root).resolve()
    all_errors: list[str] = []
    for v in args.versions:
        version_root = root / v
        manifest_path = version_root / "models.json"
        if not manifest_path.exists():
            all_errors.append(f"{v}: missing models.json")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        all_errors.extend(verify_entries(version_root, manifest))

    if all_errors:
        print("VERIFICATION FAILED")
        for e in all_errors:
            print(f" - {e}")
        return 1

    print("VERIFICATION PASSED")
    for v in args.versions:
        print(f" - {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
