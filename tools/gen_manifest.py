"""Write MANIFEST.json for a vector directory. Computes shas; no geometry."""
import hashlib
import json
import sys
from pathlib import Path


def write_manifest(vector_dir, *, set_version: str) -> Path:
    vector_dir = Path(vector_dir)
    files = {}
    for path in sorted(vector_dir.glob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = vector_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"set_version": set_version, "files": files}, indent=1) + "\n"
    )
    return manifest_path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: gen_manifest.py <vector-dir>", file=sys.stderr)
        return 2
    path = write_manifest(argv[1], set_version=Path(argv[1]).name)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
