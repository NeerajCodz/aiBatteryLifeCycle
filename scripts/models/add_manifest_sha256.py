from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_checksums(version_root: Path) -> dict[str, Any]:
    checksums: dict[str, Any] = {
        "models": {},
        "scalers": {},
        "results": {},
        "features": {},
        "figures": {},
    }
    for section in ("models", "scalers", "results", "features", "figures"):
        p = version_root / section
        if not p.exists():
            continue
        for f in sorted(p.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(version_root).as_posix()
            checksums[section][rel] = sha256_file(f)
    checksums["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return checksums


def enrich_models_block(version_root: Path, models: dict[str, Any]) -> None:
    for _, meta in models.items():
        if not isinstance(meta, dict):
            continue
        rel = meta.get("file")
        if rel in (None, ""):
            continue
        f = version_root / rel
        if not f.exists():
            continue
        meta["sha256"] = sha256_file(f)
        meta["bytes"] = f.stat().st_size


def enrich_auxiliary(version_root: Path, manifest: dict[str, Any]) -> None:
    aux = manifest.get("auxiliary_artifacts")
    if not isinstance(aux, dict):
        return
    for _, meta in aux.items():
        if not isinstance(meta, dict):
            continue
        rel = meta.get("file")
        if not rel:
            continue
        f = version_root / rel
        if not f.exists():
            continue
        meta["sha256"] = sha256_file(f)
        meta["bytes"] = f.stat().st_size


def enrich_scalers(version_root: Path, manifest: dict[str, Any]) -> None:
    scalers = manifest.get("scalers", {})
    if not isinstance(scalers, dict):
        return
    out: dict[str, str | None] = {}
    for key, rel in scalers.items():
        if not isinstance(rel, str):
            out[key] = None
            continue
        p = version_root / rel
        out[key] = sha256_file(p) if p.exists() else None
    manifest["scaler_checksums"] = out


def process_version(artifacts_root: Path, version: str) -> Path:
    version_root = artifacts_root / version
    manifest_path = version_root / "models.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"models.json not found for {version}: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    models = manifest.get("models", {})
    if isinstance(models, dict):
        enrich_models_block(version_root, models)

    enrich_auxiliary(version_root, manifest)
    enrich_scalers(version_root, manifest)

    manifest["checksums"] = collect_checksums(version_root)
    manifest.setdefault("verification", {})
    if isinstance(manifest["verification"], dict):
        manifest["verification"].update(
            {
                "hash_algorithm": "sha256",
                "required": True,
                "last_verified_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Add SHA256 hashes to models.json for versions")
    parser.add_argument("--artifacts-root", default="artifacts", help="Artifacts root path")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3"],
        help="Version directories to process (default: v1 v2 v3)",
    )
    args = parser.parse_args()

    root = Path(args.artifacts_root).resolve()
    for v in args.versions:
        out = process_version(root, v)
        print(f"updated: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
