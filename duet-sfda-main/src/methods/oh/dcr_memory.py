"""Build DCR's delayed-credit memory from Task/VLM prediction history.

Unlike anchored consensus, this method does not use entropy-weighted PoE, a
fixed initial consensus anchor, or entropy-rank CSM.  Each target sample keeps
its own full-distribution memory and scores each expert by how well its prior
distribution predicts the next Task/CLIP observation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.methods.oh.dcr_common import (
    _build_loaders,
    _build_prompt_model,
    _build_target_model,
    _build_target_optimizer,
    _task_vlm_hard_label_bank,
    _evaluate,
    _load_classnames,
    _poly_schedule,
    _scan_predictions,
)
from src.utils.dcr_consensus import (
    iic_mutual_information_loss,
    prediction_diversity_entropy,
    samplewise_distribution_alignment_loss,
)
from src.utils.dcr_credit_memory import (
    initialize_delayed_credit,
    normalized_js_divergence,
    update_delayed_credit,
)


LOG_PREFIX = "DCR delayed credit memory"


def _validate_config(cfg):
    if not bool(cfg.DCR_MEMORY.ENABLED):
        raise ValueError("DCR_MEMORY.ENABLED must be true for this method")
    if int(cfg.DCR_MEMORY.EPOCHS) < 2:
        raise ValueError("DCR_MEMORY.EPOCHS must be at least 2")
    if not 0.0 <= float(cfg.DCR_MEMORY.CREDIT_DECAY) < 1.0:
        raise ValueError("DCR_MEMORY.CREDIT_DECAY must be in [0, 1)")
    if float(cfg.DCR_MEMORY.CREDIT_ETA) <= 0.0:
        raise ValueError("DCR_MEMORY.CREDIT_ETA must be positive")
    if not 0.0 < float(cfg.DCR_MEMORY.MEMORY_UPDATE_RATE) <= 1.0:
        raise ValueError(
            "DCR_MEMORY.MEMORY_UPDATE_RATE must be in (0, 1]"
        )
    if str(cfg.DCR_MEMORY.CREDIT_MODE) not in {"delayed", "uniform"}:
        raise ValueError(
            "DCR_MEMORY.CREDIT_MODE must be delayed or uniform"
        )
    if str(cfg.DCR_MEMORY.FEEDBACK_MODE) not in {
        "agreement_temporal",
        "agreement_only",
    }:
        raise ValueError(
            "DCR_MEMORY.FEEDBACK_MODE must be agreement_temporal "
            "or agreement_only"
        )
    if str(cfg.DCR_MEMORY.HARD_LABEL_MODE) not in {
        "consensus",
        "task_vlm_agreement",
    }:
        raise ValueError(
            "DCR_MEMORY.HARD_LABEL_MODE must be consensus or task_vlm_agreement"
        )
    if str(cfg.DCR_MEMORY.ALIGNMENT_MODE) not in {
        "batch_iic",
        "samplewise_kl",
    }:
        raise ValueError(
            "DCR_MEMORY.ALIGNMENT_MODE must be batch_iic or samplewise_kl"
        )
    if float(cfg.DCR_MEMORY.AGREEMENT_BETA) <= 0.0:
        raise ValueError("DCR_MEMORY.AGREEMENT_BETA must be positive")
    if float(cfg.DCR_MEMORY.CONFLICT_BETA) <= 0.0:
        raise ValueError("DCR_MEMORY.CONFLICT_BETA must be positive")


def _initial_diagnostics(state, task_probability, clip_probability, epsilon):
    sample_count = int(task_probability.shape[0])
    zeros = torch.zeros(sample_count, dtype=torch.float32)
    agreement = 1.0 - normalized_js_divergence(
        task_probability,
        clip_probability,
        epsilon,
    )
    return {
        "feedback": agreement,
        "agreement": agreement,
        "temporal_stability": torch.ones_like(agreement),
        "task_delayed_loss": zeros,
        "clip_delayed_loss": zeros.clone(),
        "average_task_loss": zeros.clone(),
        "average_clip_loss": zeros.clone(),
        "task_weight": state["task_weight"],
        "clip_weight": state["clip_weight"],
        "update_rate": zeros.clone(),
        "memory_shift_l1": zeros.clone(),
    }


def train_target(cfg):
    """Adapt Task and prompt branches using GT-free delayed expert credit."""
    _validate_config(cfg)
    loaders = _build_loaders(cfg, log_prefix=LOG_PREFIX)
    classnames = _load_classnames(cfg.name_file)
    if len(classnames) != int(cfg.class_num):
        raise ValueError(
            "class-name count {} does not match class_num {}".format(
                len(classnames), int(cfg.class_num)
            )
        )

    net_f, net_b, net_c = _build_target_model(cfg)
    target_optimizer = _build_target_optimizer(cfg, net_f, net_b, net_c)
    prompt_model, prompt_optimizer = _build_prompt_model(cfg, classnames)
    epsilon = float(cfg.DCR_MEMORY.EPSILON)
    epochs = int(cfg.DCR_MEMORY.EPOCHS)
    total_steps = epochs * len(loaders["train"])

    initial_task, initial_clip = _scan_predictions(
        loaders["scan"],
        net_f,
        net_b,
        net_c,
        prompt_model,
        int(cfg.class_num),
    )
    credit_state = initialize_delayed_credit(
        initial_task,
        initial_clip,
        epsilon=epsilon,
    )
    initial_conflict = initial_task.argmax(dim=1) != initial_clip.argmax(dim=1)
    logging.info(
        "{} initialized: samples={}; classes={}; initial_conflicts={}; "
        "initial_conflict_rate={:.2f}%; task_weight_mean=0.5000; "
        "clip_weight_mean=0.5000; soft_coverage=100.00%; "
        "conflict_hard_coverage=100.00%; fixed_initial_anchor=False; "
        "entropy_poe=False; csm=False; comparator=False; "
        "third_visual_model=False; target_gt_affects_training=False".format(
            LOG_PREFIX,
            int(initial_task.shape[0]),
            int(initial_task.shape[1]),
            int(initial_conflict.sum().item()),
            100.0 * float(initial_conflict.float().mean().item()),
        )
    )
    logging.info(
        "{} optimization: epochs={}; steps={}; hard_label_mode={}; "
        "alignment_mode={}; "
        "credit_decay={:.3f}; credit_eta={:.3f}; memory_update_rate={:.3f}; "
        "credit_mode={}; feedback_mode={}; "
        "alpha={:.3f}; agreement_beta={:.3f}; conflict_beta={:.3f}; "
        "diversity_delta={:.3f}; clip_encoders_frozen=True; "
        "prompt_trainable=True; classifier_trainable=True; "
        "sample_self_history_only=True".format(
            LOG_PREFIX,
            epochs,
            total_steps,
            str(cfg.DCR_MEMORY.HARD_LABEL_MODE),
            str(cfg.DCR_MEMORY.ALIGNMENT_MODE),
            float(cfg.DCR_MEMORY.CREDIT_DECAY),
            float(cfg.DCR_MEMORY.CREDIT_ETA),
            float(cfg.DCR_MEMORY.MEMORY_UPDATE_RATE),
            str(cfg.DCR_MEMORY.CREDIT_MODE),
            str(cfg.DCR_MEMORY.FEEDBACK_MODE),
            float(cfg.DCR_MEMORY.ALPHA),
            float(cfg.DCR_MEMORY.AGREEMENT_BETA),
            float(cfg.DCR_MEMORY.CONFLICT_BETA),
            float(cfg.DCR_MEMORY.DIVERSITY_DELTA),
        )
    )

    global_step = 0
    for epoch in range(epochs):
        scan_task, scan_clip = _scan_predictions(
            loaders["scan"],
            net_f,
            net_b,
            net_c,
            prompt_model,
            int(cfg.class_num),
        )
        if epoch == 0:
            # The first epoch has no future observation yet, so neither expert
            # receives retrospective credit.  It starts from an equal mixture.
            credit_diagnostics = _initial_diagnostics(
                credit_state,
                scan_task,
                scan_clip,
                epsilon,
            )
        else:
            credit_state, credit_diagnostics = update_delayed_credit(
                credit_state,
                scan_task,
                scan_clip,
                decay=float(cfg.DCR_MEMORY.CREDIT_DECAY),
                credit_eta=float(cfg.DCR_MEMORY.CREDIT_ETA),
                memory_update_rate=float(
                    cfg.DCR_MEMORY.MEMORY_UPDATE_RATE
                ),
                credit_mode=str(cfg.DCR_MEMORY.CREDIT_MODE),
                feedback_mode=str(cfg.DCR_MEMORY.FEEDBACK_MODE),
                epsilon=epsilon,
            )
        teacher_bank = credit_state["memory"].detach().cpu()
        task_vlm_agreement, joint_label, prior_active = _task_vlm_hard_label_bank(
            cfg,
            scan_task,
            scan_clip,
            epoch,
        )

        net_f.train()
        net_b.train()
        net_c.train()
        # Frozen CLIP encoders remain deterministic while gradients flow only
        # to the shared prompt context tensor.
        prompt_model.eval()
        task_loss_sum = 0.0
        prompt_loss_sum = 0.0
        hard_loss_sum = 0.0
        batches = 0

        for views, _, indices in loaders["train"]:
            if int(indices.numel()) < 2:
                continue
            global_step += 1
            batches += 1
            _poly_schedule(target_optimizer, global_step, total_steps)
            _poly_schedule(prompt_optimizer, global_step, total_steps)
            task_image = views[0].cuda(non_blocking=True)
            clip_image = views[1].cuda(non_blocking=True)

            task_logits = net_c(net_b(net_f(task_image)))
            clip_logits, _ = prompt_model(clip_image)
            task_probability = task_logits.float().softmax(dim=1)
            clip_probability = clip_logits.float().softmax(dim=1)
            cpu_indices = indices.long().cpu()
            shared_teacher = teacher_bank[cpu_indices].to(
                task_logits.device
            ).detach()

            consensus_hard = shared_teacher.argmax(dim=1)
            if str(cfg.DCR_MEMORY.HARD_LABEL_MODE) == "task_vlm_agreement":
                agreement_batch = task_vlm_agreement[cpu_indices].to(
                    task_logits.device
                )
                joint_label_batch = joint_label[cpu_indices].to(
                    task_logits.device
                )
                hard_label = torch.where(
                    agreement_batch,
                    joint_label_batch,
                    consensus_hard,
                )
                hard_weight = torch.where(
                    agreement_batch,
                    task_logits.new_full(
                        (task_logits.shape[0],),
                        float(cfg.DCR_MEMORY.AGREEMENT_BETA),
                    ),
                    task_logits.new_full(
                        (task_logits.shape[0],),
                        float(cfg.DCR_MEMORY.CONFLICT_BETA),
                    ),
                )
            else:
                hard_label = consensus_hard
                hard_weight = task_logits.new_full(
                    (task_logits.shape[0],),
                    float(cfg.DCR_MEMORY.CONFLICT_BETA),
                )

            if str(cfg.DCR_MEMORY.ALIGNMENT_MODE) == "samplewise_kl":
                task_alignment = samplewise_distribution_alignment_loss(
                    task_probability,
                    shared_teacher,
                    epsilon=epsilon,
                )
                prompt_alignment = samplewise_distribution_alignment_loss(
                    clip_probability,
                    shared_teacher,
                    epsilon=epsilon,
                )
            else:
                task_alignment = iic_mutual_information_loss(
                    task_probability,
                    shared_teacher,
                    epsilon=epsilon,
                )
                prompt_alignment = iic_mutual_information_loss(
                    clip_probability,
                    shared_teacher,
                    epsilon=epsilon,
                )
            hard_ce = F.cross_entropy(
                task_logits,
                hard_label,
                reduction="none",
            )
            hard_loss = (hard_weight * hard_ce).mean()
            diversity = prediction_diversity_entropy(
                task_probability,
                epsilon=epsilon,
            )
            task_loss = (
                float(cfg.DCR_MEMORY.ALPHA) * task_alignment
                + hard_loss
                - float(cfg.DCR_MEMORY.DIVERSITY_DELTA) * diversity
            )
            prompt_loss = prompt_alignment
            total_loss = task_loss + prompt_loss

            target_optimizer.zero_grad()
            prompt_optimizer.zero_grad()
            total_loss.backward()
            target_optimizer.step()
            prompt_optimizer.step()

            task_loss_sum += float(task_loss.detach().item())
            prompt_loss_sum += float(prompt_loss.detach().item())
            hard_loss_sum += float(hard_loss.detach().item())

        evaluate_now = (
            (epoch + 1) % int(cfg.DCR_MEMORY.EVAL_INTERVAL) == 0
            or epoch + 1 == epochs
        )
        if evaluate_now:
            accuracy, detail = _evaluate(
                loaders["test"],
                net_f,
                net_b,
                net_c,
                str(cfg.SETTING.DATASET) == "VISDA-C",
            )
        else:
            accuracy, detail = float("nan"), ""
        current_conflict = scan_task.argmax(dim=1) != scan_clip.argmax(dim=1)
        log_message = (
            "{} epoch: {}/{}; accuracy={:.2f}%; task_loss={:.6f}; "
            "prompt_loss={:.6f}; hard_loss={:.6f}; current_conflicts={}; "
            "task_vlm_agreements={}; prior_active={}; feedback_mean={:.4f}; "
            "task_delayed_loss={:.4f}; clip_delayed_loss={:.4f}; "
            "task_weight_mean={:.4f}; clip_weight_mean={:.4f}; "
            "task_preferred_rate={:.2f}%; memory_shift_l1={:.4f}; "
            "soft_coverage=100.00%; conflict_hard_coverage=100.00%; "
            "target_gt_affects_training=False"
        ).format(
            LOG_PREFIX,
            epoch + 1,
            epochs,
            accuracy,
            task_loss_sum / max(batches, 1),
            prompt_loss_sum / max(batches, 1),
            hard_loss_sum / max(batches, 1),
            int(current_conflict.sum().item()),
            int(task_vlm_agreement.sum().item()),
            prior_active,
            float(credit_diagnostics["feedback"].mean().item()),
            float(credit_diagnostics["task_delayed_loss"].mean().item()),
            float(credit_diagnostics["clip_delayed_loss"].mean().item()),
            float(credit_diagnostics["task_weight"].mean().item()),
            float(credit_diagnostics["clip_weight"].mean().item()),
            100.0
            * float(
                (
                    credit_diagnostics["task_weight"]
                    > credit_diagnostics["clip_weight"]
                )
                .float()
                .mean()
                .item()
            ),
            float(credit_diagnostics["memory_shift_l1"].mean().item()),
        )
        if detail:
            log_message += "\n" + detail
        logging.info(log_message)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net_f.state_dict(), output_dir / "target_F.pt")
    torch.save(net_b.state_dict(), output_dir / "target_B.pt")
    torch.save(net_c.state_dict(), output_dir / "target_C.pt")
    torch.save(
        prompt_model.prompt_learner.state_dict(),
        output_dir / "target_prompt.pt",
    )
    torch.save(
        {key: value.cpu() for key, value in credit_state.items()},
        output_dir / "dcr_memory_state.pt",
    )
    logging.info(
        "{} completed: saved_dir={}; inference_uses_target_only=True".format(
            LOG_PREFIX,
            output_dir,
        )
    )
    return net_f, net_b, net_c
