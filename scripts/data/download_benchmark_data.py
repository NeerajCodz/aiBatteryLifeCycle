"""Download the public NASA, CALCE, and Oxford benchmark archives."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "datasets" / "raw"

SOURCES = {
    "nasa": {
        "5_Battery_Data_Set.zip": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
    },
    "calce": {
        f"CS2_{cell}.zip": f"https://web.calce.umd.edu/batteries/data/CS2_{cell}.zip"
        for cell in (35, 36, 37, 38)
    },
    "oxford": {
        "Oxford_Battery_Degradation_Dataset_1.mat": (
            "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/"
            "files/m5ac36a1e2073852e4f1f7dee647909a7"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "aiBatteryLifecycle-research/3.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if temporary.stat().st_size < 1024:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded payload is unexpectedly small: {url}")
    with temporary.open("rb") as handle:
        prefix = handle.read(32).lower()
    if b"<html" in prefix or b"<!doctype" in prefix:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Server returned HTML instead of dataset content: {url}")
    temporary.replace(destination)


def fetch_dataset(name: str) -> list[Path]:
    written: list[Path] = []
    for filename, url in SOURCES[name].items():
        destination = RAW_ROOT / name / filename
        if not destination.exists():
            print(f"Downloading {name}/{filename}")
            download(url, destination)
        else:
            print(f"Using cached {name}/{filename}")
        written.append(destination)
        if destination.suffix.lower() == ".zip":
            extract_dir = destination.with_suffix("")
            if not extract_dir.exists():
                print(f"Extracting {destination.name}")
                extract_dir.mkdir(parents=True)
                with zipfile.ZipFile(destination) as archive:
                    archive.extractall(extract_dir)
            # The official NASA bundle contains six ZIP archives inside the
            # outer ZIP. Extract them as well so the original MAT files are
            # available without a manual step.
            if name == "nasa":
                for nested in extract_dir.rglob("*.zip"):
                    nested_dir = nested.with_suffix("")
                    if not nested_dir.exists():
                        nested_dir.mkdir(parents=True)
                        with zipfile.ZipFile(nested) as archive:
                            archive.extractall(nested_dir)
    return written


def write_checksums(paths: list[Path]) -> None:
    checksum_path = RAW_ROOT.parent / "checksums.sha256"
    rows = [f"{sha256(path)}  {path.relative_to(RAW_ROOT).as_posix()}" for path in sorted(paths)]
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Checksums written to {checksum_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=sorted(SOURCES))
    parser.add_argument("--all", action="store_true", help="download every configured dataset")
    args = parser.parse_args()
    selected = sorted(SOURCES) if args.all else args.datasets
    if not selected:
        parser.error("select one or more datasets, or pass --all")
    paths: list[Path] = []
    for name in selected:
        paths.extend(fetch_dataset(name))
    write_checksums(paths)


if __name__ == "__main__":
    main()
