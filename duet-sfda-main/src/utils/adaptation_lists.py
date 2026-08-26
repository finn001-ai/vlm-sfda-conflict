"""Target-list helpers for compute-saving adaptation proxies."""

from __future__ import annotations

from pathlib import Path


def load_adaptation_and_evaluation_rows(
    default_adaptation_path: str,
    evaluation_path: str,
    adaptation_override: str = "",
) -> tuple[list[str], list[str], Path]:
    """Load adaptation rows separately from full evaluation rows.

    The returned adaptation rows must be used by every loader whose sample
    indices address pseudo-label tensors. Evaluation rows remain independent
    so proxy adaptation can still report accuracy on the complete target set.
    """

    override = str(adaptation_override).strip()
    adaptation_path = Path(override if override else default_adaptation_path)
    evaluation_path_obj = Path(evaluation_path)

    if override and not adaptation_path.is_file():
        raise FileNotFoundError(
            f"Adaptation list override does not exist: {adaptation_path}"
        )

    with adaptation_path.open() as handle:
        adaptation_rows = handle.readlines()
    with evaluation_path_obj.open() as handle:
        evaluation_rows = handle.readlines()

    if not adaptation_rows:
        raise ValueError(f"Adaptation list is empty: {adaptation_path}")
    if not evaluation_rows:
        raise ValueError(f"Evaluation list is empty: {evaluation_path_obj}")

    return adaptation_rows, evaluation_rows, adaptation_path


def resolve_relative_image_rows(
    rows: list[str],
    image_root: str | Path,
) -> list[str]:
    """Resolve relative image-list paths without changing labels or order."""
    root = Path(image_root)
    resolved = []
    for row in rows:
        fields = row.strip().rsplit(maxsplit=1)
        if len(fields) != 2:
            raise ValueError("image-list row must contain path and label")
        image_path, label = fields
        path = Path(image_path)
        if not path.is_absolute():
            path = root / path
        resolved.append(f"{path} {label}\n")
    return resolved
