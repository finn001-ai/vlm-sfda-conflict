#!/usr/bin/env python
"""Build a deterministic, class-proportional VisDA adaptation proxy list.

The script only writes a list file; it does not copy or modify image files.
Labels are used solely to retain the same fraction of every class. The proxy
is intended for compute-saving experiment screening, not final SFDA reporting.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic fraction of every class from a VisDA list "
            "while preserving the original row order."
        )
    )
    parser.add_argument(
        "--input",
        default="data/VISDA-C/validation_list.txt",
        help="Full VisDA target list in '<image_path> <label>' format.",
    )
    parser.add_argument(
        "--output",
        default="data/VISDA-C/validation_proxy25_seed2020_list.txt",
        help="Proxy adaptation list to create.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.25,
        help="Fraction retained independently in every class (default: 0.25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2020,
        help="Deterministic selection seed (default: 2020).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output list.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            _image_path, label_text = line.rsplit(maxsplit=1)
            label = int(label_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{path}:{line_number}: expected '<image_path> <integer_label>'"
            ) from exc
        rows.append((line, label))
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return rows


def deterministic_rank(seed: int, row_index: int, row: str) -> bytes:
    payload = f"{seed}\0{row_index}\0{row}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def select_indices(
    rows: list[tuple[str, int]], ratio: float, seed: int
) -> tuple[set[int], Counter[int]]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"--ratio must be in (0, 1], got {ratio}")

    by_class: dict[int, list[int]] = defaultdict(list)
    for row_index, (_row, label) in enumerate(rows):
        by_class[label].append(row_index)

    selected: set[int] = set()
    selected_counts: Counter[int] = Counter()
    for label, indices in sorted(by_class.items()):
        keep = max(1, int(round(len(indices) * ratio)))
        keep = min(keep, len(indices))
        ranked = sorted(
            indices,
            key=lambda index: deterministic_rank(seed, index, rows[index][0]),
        )
        chosen = ranked[:keep]
        selected.update(chosen)
        selected_counts[label] = len(chosen)
    return selected, selected_counts


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"VisDA input list not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --input")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --force to replace it"
        )

    rows = read_rows(input_path)
    selected, selected_counts = select_indices(rows, args.ratio, args.seed)
    source_counts = Counter(label for _row, label in rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = [row for index, (row, _label) in enumerate(rows) if index in selected]
    output_path.write_text("\n".join(output_rows) + "\n")

    print(
        f"Wrote {output_path} with {len(output_rows)}/{len(rows)} samples "
        f"({100.0 * len(output_rows) / len(rows):.2f}%), seed={args.seed}"
    )
    print("label,full,proxy,retained_percent")
    for label in sorted(source_counts):
        full = source_counts[label]
        proxy = selected_counts[label]
        print(f"{label},{full},{proxy},{100.0 * proxy / full:.2f}")


if __name__ == "__main__":
    main()
