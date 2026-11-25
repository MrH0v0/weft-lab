# SPDX-License-Identifier: MIT
"""Build the installable Blender extension using Python 3.11+ and the standard library."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "extension" / "weft_replay"


def files():
    for path in sorted(SOURCE.rglob("*"), key=lambda p: p.relative_to(SOURCE).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(SOURCE)
        if any(p.startswith(".") or p == "__pycache__" for p in rel.parts):
            continue
        if len(rel.parts) == 1:
            include = path.suffix in {".py", ".cpp", ".inc", ".md", ".txt", ".toml", ".dll"}
        elif rel.parts[0] == "data":
            include = path.suffix in {".png", ".glb", ".sdf", ".i16", ".f32", ".wav", ".json"}
        else:
            include = rel.parts[0] == "licenses" and path.suffix == ".txt"
        if include:
            yield rel.as_posix(), path


def build(folder):
    with (SOURCE / "blender_manifest.toml").open("rb") as file:
        manifest = tomllib.load(file)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{manifest['id']}-{manifest['version']}.zip"
    # Finish the archive on the same volume before replacing an earlier build.
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        draft = Path(tmp) / target.name
        with zipfile.ZipFile(draft, "w") as archive:
            for name, path in files():
                info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=5)
        with zipfile.ZipFile(draft) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Failed CRC check for {bad}")
        draft.replace(target)
    with target.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    print(f"{digest}  {target.name}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    build(parser.parse_args().output_dir)
