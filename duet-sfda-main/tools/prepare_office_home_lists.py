#!/usr/bin/env python
"""Create Office-Home list files expected by the DUET codebase.

The DUET loaders read entries from ./data/office-home/{Domain}_list.txt where
each line is "<image_path> <label>". This helper builds those lists from the
standard Office-Home directory layout:

data/office-home/
  Art/<class>/*.jpg
  Clipart/<class>/*.jpg
  Product/<class>/*.jpg
  RealWorld/<class>/*.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path


DOMAINS = ("Art", "Clipart", "Product", "RealWorld")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/office-home",
        help="Office-Home root containing Art/Clipart/Product/RealWorld.",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Write absolute image paths instead of paths relative to cwd.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Office-Home root not found: {root}")

    missing = [domain for domain in DOMAINS if not (root / domain).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Office-Home domain folders: {missing}")

    class_sets = []
    for domain in DOMAINS:
        classes = {p.name for p in (root / domain).iterdir() if p.is_dir()}
        class_sets.append(classes)

    common_classes = sorted(set.intersection(*class_sets))
    if not common_classes:
        raise RuntimeError("No common class folders found across all domains.")

    class_to_idx = {name: idx for idx, name in enumerate(common_classes)}
    classname_path = root / "classname.txt"
    classname_path.write_text("\n".join(common_classes) + "\n")

    for domain in DOMAINS:
        rows = []
        for class_name in common_classes:
            class_dir = root / domain / class_name
            images = sorted(
                p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            for image_path in images:
                path = image_path.resolve() if args.absolute else image_path
                rows.append(f"{path.as_posix()} {class_to_idx[class_name]}")

        out_path = root / f"{domain}_list.txt"
        out_path.write_text("\n".join(rows) + "\n")
        print(f"Wrote {out_path} with {len(rows)} images")

    print(f"Wrote {classname_path} with {len(common_classes)} classes")


if __name__ == "__main__":
    main()
