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
