import argparse
import json
import os
from pathlib import Path
from datetime import datetime

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

def build_manifest(source_dir: Path):
    entries = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_EXTS:
                p = Path(root) / f
                try:
                    stat = p.stat()
                    entries.append({
                        "path": str(p.resolve()),
                        "ext": ext,
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
                    })
                except Exception as e:
                    print(f"[WARN] Could not stat {p}: {e}")
    return entries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Folder with Delhi legal documents")
    ap.add_argument("--out", default="data/manifest.json", help="Output manifest path")
    args = ap.parse_args()

    source_dir = Path(args.source)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = build_manifest(source_dir)
    # Optional: simple heuristic filter if you'd like only likely-Delhi docs by filename
    # entries = [e for e in entries if 'delhi' in Path(e['path']).name.lower()]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"created": datetime.now().isoformat(timespec="seconds"),
                   "source": str(source_dir.resolve()),
                   "count": len(entries),
                   "files": entries}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote manifest with {len(entries)} files to {out_path}")

if __name__ == "__main__":
    main()