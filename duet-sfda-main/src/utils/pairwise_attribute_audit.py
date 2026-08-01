"""Label-free helpers for the VisDA pairwise visible-attribute audit.

The audit keeps the released DUET task model and CLIP ViT-B/32 image path
unchanged.  It adds only fixed, human-readable class descriptions generated
from the twelve class names, then asks whether those descriptions can safely
identify task-correct/CLIP-wrong conflicts.  No target image or target label is
used to construct a description, threshold, or routing rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ATTRIBUTE_FAMILIES = ("silhouette", "parts", "proportions", "structure")
PROMPT_TEMPLATES = (
    "a photo of a {class_name}. Its visible appearance has {description}.",
    "this {class_name} is visually identified by {description}.",
)


# Every VisDA class receives the same four description families.  These strings
# are deliberately dataset-image-agnostic: they encode ordinary category
# semantics, not observations from VisDA samples or oracle error analysis.
VISDA_VISIBLE_ATTRIBUTES: dict[str, dict[str, str]] = {
    "aeroplane": {
        "silhouette": "a long streamlined fuselage with fixed wings and a tail",
        "parts": "wings, a vertical tail fin, engines, and landing gear",
        "proportions": "a wide wingspan around a narrow central body",
        "structure": "an enclosed flying vehicle body without road wheels in use",
    },
    "bicycle": {
        "silhouette": "an open lightweight frame between two thin wheels",
        "parts": "pedals, a chain, handlebars, a saddle, and two wheels",
        "proportions": "two similarly sized narrow wheels with a slim frame",
        "structure": "a human-powered vehicle without an engine or enclosed cabin",
    },
    "bus": {
        "silhouette": "a long tall rectangular passenger vehicle body",
        "parts": "many side windows, passenger doors, and large road wheels",
        "proportions": "a high roof and long cabin spanning several window rows",
        "structure": "one large enclosed body built to carry many passengers",
    },
    "car": {
        "silhouette": "a low enclosed passenger cabin with hood, roof, and trunk",
        "parts": "four road wheels, passenger doors, windows, and headlights",
        "proportions": "a relatively low roof over a compact passenger body",
        "structure": "a continuous passenger body without a large cargo bed or box",
    },
    "horse": {
        "silhouette": "a four-legged animal with a long neck, head, and tail",
        "parts": "hooves, mane, ears, muzzle, four legs, and a tail",
        "proportions": "a large horizontal torso supported by four slender legs",
        "structure": "an organic animal body rather than a manufactured object",
    },
    "knife": {
        "silhouette": "a thin elongated handheld object with a blade and handle",
        "parts": "a sharp metal blade, cutting edge, point, and grip",
        "proportions": "a long narrow blade attached to a shorter handle",
        "structure": "a rigid cutting tool without wheels, limbs, or a cabin",
    },
    "motorcycle": {
        "silhouette": "a motorized open frame balanced on two wheels",
        "parts": "an engine, fuel tank, handlebars, seat, and two wheels",
        "proportions": "two road wheels around a compact engine and rider seat",
        "structure": "an engine-powered two-wheeler without an enclosed cabin",
    },
    "person": {
        "silhouette": "an upright human body with head, torso, arms, and legs",
        "parts": "a face, two arms, hands, two legs, and feet",
        "proportions": "a vertical torso above two long supporting legs",
        "structure": "a clothed or unclothed bipedal human figure",
    },
    "plant": {
        "silhouette": "branching stems bearing leaves in an irregular organic shape",
        "parts": "leaves, stems, branches, and sometimes a pot or flowers",
        "proportions": "many thin branches or stems spreading from a rooted base",
        "structure": "rooted vegetation without limbs, wheels, or mechanical parts",
    },
    "skateboard": {
        "silhouette": "a small low narrow board mounted over four tiny wheels",
        "parts": "a flat deck, two trucks, and four small wheels",
        "proportions": "a long thin deck much larger than its small wheels",
        "structure": "a rideable board without handlebars, a seat, or an engine",
    },
    "train": {
        "silhouette": "a long rail vehicle formed by a locomotive or connected cars",
        "parts": "rail wheels, carriage windows, couplers, and a locomotive front",
        "proportions": "a very long body composed of repeated connected sections",
        "structure": "a guided vehicle designed to travel on railway tracks",
    },
    "truck": {
        "silhouette": "a tall driving cab attached to a cargo bed or cargo box",
        "parts": "a cab, heavy chassis, large road wheels, and a cargo compartment",
        "proportions": "a substantial cargo section behind or around the driving cab",
        "structure": "separate passenger and load-carrying regions built for cargo",
    },
}


def _normalized_class_names(class_names: list[str]) -> list[str]:
    return [name.strip().lower().replace("_", " ") for name in class_names]


def build_visda_attribute_prompt_manifest(
    class_names: list[str],
) -> dict[str, Any]:
    """Build the deterministic text-only prompt contract for all VisDA classes."""
    normalized = _normalized_class_names(class_names)
    if len(normalized) != len(set(normalized)):
        raise ValueError("class names must be unique")
    expected = set(VISDA_VISIBLE_ATTRIBUTES)
    observed = set(normalized)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"VisDA attribute classes do not match; missing={missing}, extra={extra}"
        )

    classes = []
    prompts: list[str] = []
    for class_index, class_name in enumerate(normalized):
        attributes = VISDA_VISIBLE_ATTRIBUTES[class_name]
        if set(attributes) != set(ATTRIBUTE_FAMILIES):
            raise ValueError(f"attribute families are incomplete for {class_name}")
        class_prompts = []
        for template_index, template in enumerate(PROMPT_TEMPLATES):
            family_prompts = []
            for family in ATTRIBUTE_FAMILIES:
                prompt = template.format(
                    class_name=class_name,
                    description=attributes[family],
                )
                prompts.append(prompt)
                family_prompts.append(
                    {
                        "family": family,
                        "description": attributes[family],
                        "prompt": prompt,
                    }
                )
            class_prompts.append(
                {"template_index": template_index, "prompts": family_prompts}
            )
        classes.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "templates": class_prompts,
            }
        )
    return {
        "phase": "TEXT_ONLY_PROMPT_CONTRACT",
        "target_images_used": False,
        "target_labels_used": False,
        "external_visual_models_used": False,
        "description_source": "fixed category semantics authored from class names only",
        "attribute_families": list(ATTRIBUTE_FAMILIES),
        "prompt_templates": list(PROMPT_TEMPLATES),
        "shape": [len(normalized), len(PROMPT_TEMPLATES), len(ATTRIBUTE_FAMILIES)],
        "classes": classes,
        "flat_prompts": prompts,
    }


def pairwise_attribute_task_rescue(
    image_features: np.ndarray,
    attribute_text_features: np.ndarray,
    task_prediction: np.ndarray,
    clip_prediction: np.ndarray,
    *,
    min_template_stability: float = 0.90,
    score_chunk_size: int = 4_096,
) -> dict[str, np.ndarray]:
    """Conservatively switch fixed CLIP to task using text attributes only.

    ``attribute_text_features`` must have shape
    ``[class, template, family, embedding]``.  A task rescue requires all four
    template-averaged family margins to favor task, both independent template
    halves to favor task, and cosine agreement between the two four-family
    margin vectors to meet the locked stability threshold.  All other rows keep
    the fixed CLIP top-1 prediction.
    """
    image = np.asarray(image_features, dtype=np.float64)
    text = np.asarray(attribute_text_features, dtype=np.float64)
    task = np.asarray(task_prediction, dtype=np.int64)
    clip = np.asarray(clip_prediction, dtype=np.int64)
    if image.ndim != 2:
        raise ValueError("image_features must be [sample, embedding]")
    if text.ndim != 4 or text.shape[1:3] != (
        len(PROMPT_TEMPLATES),
        len(ATTRIBUTE_FAMILIES),
    ):
        raise ValueError("attribute_text_features have an invalid shape")
    if text.shape[-1] != image.shape[-1]:
        raise ValueError("image and text embedding dimensions do not match")
    if task.shape != (image.shape[0],) or clip.shape != task.shape:
        raise ValueError("predictions must contain one class per image")
    if np.any(task < 0) or np.any(task >= text.shape[0]):
        raise ValueError("task prediction is outside the class range")
    if np.any(clip < 0) or np.any(clip >= text.shape[0]):
        raise ValueError("CLIP prediction is outside the class range")
    if np.any(task == clip):
        raise ValueError("pairwise rescue accepts conflict rows only")
    if not 0.0 <= min_template_stability <= 1.0:
        raise ValueError("min_template_stability must be in [0, 1]")
    if score_chunk_size <= 0:
        raise ValueError("score_chunk_size must be positive")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("image and text features must be finite")

    image_norm = np.linalg.norm(image, axis=1, keepdims=True)
    text_norm = np.linalg.norm(text, axis=3, keepdims=True)
    if np.any(image_norm <= 1e-12) or np.any(text_norm <= 1e-12):
        raise ValueError("image and text features must be nonzero")
    image = image / image_norm
    text = text / text_norm

    score_shape = (
        image.shape[0],
        len(PROMPT_TEMPLATES),
        len(ATTRIBUTE_FAMILIES),
    )
    task_score = np.empty(score_shape, dtype=np.float64)
    clip_score = np.empty(score_shape, dtype=np.float64)
    # Advanced indexing all conflict rows at once would materialize two
    # [sample, template, family, embedding] arrays approaching 2 GB on VisDA.
    # Fixed-size chunks preserve the exact scoring rule with a bounded peak.
    for start in range(0, image.shape[0], score_chunk_size):
        stop = min(start + score_chunk_size, image.shape[0])
        current_image = image[start:stop]
        task_score[start:stop] = np.einsum(
            "nd,ntfd->ntf",
            current_image,
            text[task[start:stop]],
            optimize=True,
        )
        clip_score[start:stop] = np.einsum(
            "nd,ntfd->ntf",
            current_image,
            text[clip[start:stop]],
            optimize=True,
        )
    margin = task_score - clip_score
    family_margin = margin.mean(axis=1)
    template_margin = margin.mean(axis=2)

    first = margin[:, 0]
    second = margin[:, 1]
    numerator = np.einsum("nf,nf->n", first, second)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    stability = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    all_families_support_task = (family_margin > 0.0).all(axis=1)
    both_templates_support_task = (template_margin > 0.0).all(axis=1)
    task_rescue = (
        all_families_support_task
        & both_templates_support_task
        & (stability >= min_template_stability)
    )
    mean_margin = margin.mean(axis=(1, 2))
    descriptor_prediction = np.where(mean_margin > 0.0, task, clip)
    routed_prediction = np.where(task_rescue, task, clip)
    return {
        "task_score": task_score,
        "clip_score": clip_score,
        "margin": margin,
        "family_margin": family_margin,
        "template_margin": template_margin,
        "template_stability": stability,
        "all_families_support_task": all_families_support_task,
        "both_templates_support_task": both_templates_support_task,
        "task_rescue": task_rescue,
        "descriptor_prediction": descriptor_prediction.astype(np.int64),
        "routed_prediction": routed_prediction.astype(np.int64),
    }


def evaluate_pairwise_attribute_gate(
    *,
    reproduction_passed: bool,
    conflict_gain_pp: float,
    conflict_gain_ci: tuple[float, float],
    rescue_coverage: float,
    adjudication_precision: float,
    median_routed_stability: float,
    car_net_corrections: int,
    truck_net_corrections: int,
    min_conflict_gain_pp: float = 1.0,
    min_rescue_coverage: float = 5.0,
    min_adjudication_precision: float = 60.0,
    min_routed_stability: float = 0.90,
) -> dict[str, Any]:
    """Apply the predeclared oracle gate without fitting a label threshold."""
    checks = {
        "baseline_reproduced": bool(reproduction_passed),
        "conflict_gain_at_least_1pp": conflict_gain_pp >= min_conflict_gain_pp,
        "conflict_gain_ci_lower_positive": conflict_gain_ci[0] > 0.0,
        "task_rescue_coverage_at_least_5pct": (
            rescue_coverage >= min_rescue_coverage
        ),
        "task_adjudication_precision_at_least_60pct": (
            adjudication_precision >= min_adjudication_precision
        ),
        "median_routed_template_stability_at_least_0.90": (
            median_routed_stability >= min_routed_stability
        ),
        "car_net_corrections_nonnegative": car_net_corrections >= 0,
        "truck_net_corrections_nonnegative": truck_net_corrections >= 0,
    }
    return {
        "decision": "PASS_OFFLINE_GATE" if all(checks.values()) else "REJECT",
        "thresholds": {
            "min_conflict_gain_pp": min_conflict_gain_pp,
            "conflict_gain_ci_lower_must_be_positive": True,
            "min_task_rescue_coverage_pct": min_rescue_coverage,
            "min_task_adjudication_precision_pct": min_adjudication_precision,
            "min_routed_template_stability": min_routed_stability,
            "car_and_truck_net_corrections_must_be_nonnegative": True,
        },
        "checks": checks,
    }
