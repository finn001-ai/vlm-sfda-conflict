"""Full-coverage anchored Task/CLIP consensus adaptation.

This path intentionally does not use the pairwise Comparator, synthetic
conflicts, sample-anchor K, a coverage gate, or target labels.  A frozen
pre-adaptation Task/CLIP prediction pair is cached once as a per-sample anchor.
During adaptation, the current Task model and a prompt-adapted, encoder-frozen
CLIP model are updated synchronously from the same detached dynamic consensus.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from clip.custom_clip import ClipTestTimeTuning
from src.data.data_list import ImageList_idx, NCropsTransform
from src.models import network
from src.utils.adaptation_lists import (
    load_adaptation_and_evaluation_rows,
    resolve_relative_image_rows,
)
from src.utils.domainnet126_source import (
    load_domainnet126_source_into_split,
)
from src.utils.duet_anchored_consensus import (
    consensus_shift_factors,
    entropy_weighted_poe,
    iic_mutual_information_loss,
    modulate_anchored_consensus,
    prediction_diversity_entropy,
)
from src.utils.first_cycle_prior import prior_calibrate


logger = logging.getLogger(__name__)


def _task_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _task_test_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _clip_transform():
    interpolation = getattr(transforms, "InterpolationMode", None)
    bicubic = interpolation.BICUBIC if interpolation is not None else Image.BICUBIC
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=bicubic),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26126152, 0.27577711),
            ),
        ]
    )


def _load_classnames(path: str) -> list[str]:
    classnames = []
    with Path(path).open() as stream:
        for line in stream:
            classnames.extend(token.strip() for token in line.split() if token.strip())
    if not classnames:
        raise ValueError("class-name file is empty: {}".format(path))
    return [name.replace("_", " ") for name in classnames]


def _build_loaders(cfg, log_prefix="DUET anchored consensus"):
    adaptation_override = str(cfg.ACTIVE.ADAPTATION_LIST).strip()
    adaptation_rows, evaluation_rows, adaptation_path = (
        load_adaptation_and_evaluation_rows(
            cfg.t_dset_path,
            cfg.test_dset_path,
            adaptation_override,
        )
    )
    if str(cfg.SETTING.DATASET) == "domainnet126":
        image_root = Path(cfg.DATA_DIR) / "domainnet126"
        adaptation_rows = resolve_relative_image_rows(
            adaptation_rows,
            image_root,
        )
        evaluation_rows = resolve_relative_image_rows(
            evaluation_rows,
            image_root,
        )
    if adaptation_override:
        logging.info(
            "{} adaptation list: {}; adaptation_samples={}; "
            "full_evaluation_samples={}".format(
                log_prefix,
                adaptation_path,
                len(adaptation_rows),
                len(evaluation_rows),
            )
        )

    train_transform = NCropsTransform([_task_train_transform(), _clip_transform()])
    scan_transform = NCropsTransform([_task_test_transform(), _clip_transform()])
    train_set = ImageList_idx(adaptation_rows, transform=train_transform)
    scan_set = ImageList_idx(adaptation_rows, transform=scan_transform)
    test_set = ImageList_idx(evaluation_rows, transform=_task_test_transform())
    batch_size = int(cfg.TEST.BATCH_SIZE)
    return {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(cfg.NUM_WORKERS),
            drop_last=False,
        ),
        "scan": DataLoader(
            scan_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg.NUM_WORKERS),
            drop_last=False,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size * 3,
            shuffle=False,
            num_workers=int(cfg.NUM_WORKERS),
            drop_last=False,
        ),
    }


def _set_lr0(optimizer):
    for group in optimizer.param_groups:
        group["lr0"] = group["lr"]
    return optimizer


def _poly_schedule(optimizer, step: int, total_steps: int):
    decay = (1.0 + 10.0 * float(step) / float(total_steps)) ** (-0.75)
    for group in optimizer.param_groups:
        group["lr"] = group["lr0"] * decay


def _build_target_model(cfg):
    if str(cfg.MODEL.ARCH).startswith("res"):
        net_f = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif str(cfg.MODEL.ARCH).startswith("vgg"):
        net_f = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()
    else:
        raise ValueError("unsupported target architecture: {}".format(cfg.MODEL.ARCH))
    net_b = network.feat_bottleneck(
        type="bn",
        feature_dim=net_f.in_features,
        bottleneck_dim=int(cfg.bottleneck),
    ).cuda()
    net_c = network.feat_classifier(
        type="wn",
        class_num=int(cfg.class_num),
        bottleneck_dim=int(cfg.bottleneck),
    ).cuda()
    if str(cfg.SETTING.DATASET) == "domainnet126":
        load_domainnet126_source_into_split(cfg, net_f, net_b, net_c)
    else:
        net_f.load_state_dict(torch.load(cfg.output_dir_src + "/source_F.pt"))
        net_b.load_state_dict(torch.load(cfg.output_dir_src + "/source_B.pt"))
        net_c.load_state_dict(torch.load(cfg.output_dir_src + "/source_C.pt"))
    return net_f, net_b, net_c


def _build_target_optimizer(cfg, net_f, net_b, net_c):
    base_lr = float(cfg.DUET_CONSENSUS.TARGET_LR)
    groups = [
        {
            "params": [parameter for parameter in net_f.parameters()],
            "lr": base_lr * float(cfg.DUET_CONSENSUS.FEATURE_LR_SCALE),
        },
        {
            "params": [parameter for parameter in net_b.parameters()],
            "lr": base_lr * float(cfg.DUET_CONSENSUS.BOTTLENECK_LR_SCALE),
        },
        {
            "params": [parameter for parameter in net_c.parameters()],
            "lr": base_lr * float(cfg.DUET_CONSENSUS.CLASSIFIER_LR_SCALE),
        },
    ]
    return _set_lr0(
        optim.SGD(
            groups,
            momentum=0.9,
            weight_decay=1e-3,
            nesterov=True,
        )
    )


def _build_prompt_model(cfg, classnames):
    prompt_model = ClipTestTimeTuning(
        int(cfg.GPU_ID),
        classnames,
        None,
        arch=str(cfg.ACTIVE.ARCH),
        n_ctx=int(cfg.ACTIVE.N_CTX),
        ctx_init=str(cfg.ACTIVE.CTX_INIT),
    ).cuda()
    for parameter in prompt_model.parameters():
        parameter.requires_grad_(False)
    # PromptLearner keeps a reference to the complete CLIP module for prompt
    # resets, so prompt_learner.parameters() would accidentally include and
    # unfreeze the encoders.  The shared context tensor is the only trainable
    # VLM parameter in this method.
    prompt_parameters = [prompt_model.prompt_learner.ctx]
    for parameter in prompt_parameters:
        parameter.requires_grad_(True)
    prompt_optimizer = _set_lr0(
        optim.SGD(
            prompt_parameters,
            lr=float(cfg.DUET_CONSENSUS.PROMPT_LR),
            momentum=0.9,
            weight_decay=1e-3,
            nesterov=True,
        )
    )
    return prompt_model, prompt_optimizer


@torch.no_grad()
def _scan_predictions(loader, net_f, net_b, net_c, prompt_model, num_classes):
    sample_count = len(loader.dataset)
    task_bank = torch.empty(sample_count, num_classes, dtype=torch.float32)
    clip_bank = torch.empty_like(task_bank)
    seen = torch.zeros(sample_count, dtype=torch.bool)
    modes = (net_f.training, net_b.training, net_c.training, prompt_model.training)
    net_f.eval()
    net_b.eval()
    net_c.eval()
    prompt_model.eval()
    for views, _, indices in loader:
        task_image = views[0].cuda(non_blocking=True)
        clip_image = views[1].cuda(non_blocking=True)
        task_logits = net_c(net_b(net_f(task_image)))
        clip_logits, _ = prompt_model(clip_image)
        cpu_indices = indices.long().cpu()
        task_bank[cpu_indices] = task_logits.float().softmax(dim=1).cpu()
        clip_bank[cpu_indices] = clip_logits.float().softmax(dim=1).cpu()
        seen[cpu_indices] = True
    if not bool(seen.all()):
        raise RuntimeError("full-set prediction scan missed adaptation samples")
    net_f.train(modes[0])
    net_b.train(modes[1])
    net_c.train(modes[2])
    prompt_model.train(modes[3])
    return task_bank, clip_bank


@torch.no_grad()
def _evaluate(loader, net_f, net_b, net_c, visda):
    modes = (net_f.training, net_b.training, net_c.training)
    net_f.eval()
    net_b.eval()
    net_c.eval()
    predictions = []
    labels = []
    for image, label, _ in loader:
        logits = net_c(net_b(net_f(image.cuda(non_blocking=True))))
        predictions.append(logits.argmax(dim=1).cpu())
        labels.append(label.long().cpu())
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(labels).numpy()
    if visda:
        matrix = confusion_matrix(target, prediction)
        per_class = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
        score = float(per_class.mean() * 100.0)
        detail = " ".join(str(np.round(value * 100.0, 2)) for value in per_class)
    else:
        score = float((prediction == target).mean() * 100.0)
        detail = ""
    net_f.train(modes[0])
    net_b.train(modes[1])
    net_c.train(modes[2])
    return score, detail


def _duet_hard_label_bank(cfg, task_probability, clip_probability, epoch):
    prior_active = epoch < int(cfg.DUET_CONSENSUS.FIRST_PRIOR_EPOCHS)
    if prior_active:
        task_for_label = prior_calibrate(
            task_probability,
            power=float(cfg.DUET_FCP.POWER),
            epsilon=float(cfg.DUET_CONSENSUS.EPSILON),
        )
        clip_for_label = prior_calibrate(
            clip_probability,
            power=float(cfg.DUET_FCP.POWER),
            epsilon=float(cfg.DUET_CONSENSUS.EPSILON),
        )
    else:
        task_for_label = task_probability
        clip_for_label = clip_probability
    agreement = task_for_label.argmax(dim=1) == clip_for_label.argmax(dim=1)
    label = (task_for_label + clip_for_label).argmax(dim=1)
    return agreement, label, prior_active


def _validate_config(cfg):
    if not bool(cfg.DUET_CONSENSUS.ENABLED):
        raise ValueError("DUET_CONSENSUS.ENABLED must be true for this method")
    if int(cfg.DUET_CONSENSUS.EPOCHS) < 2:
        raise ValueError("DUET_CONSENSUS.EPOCHS must be at least 2")
    if not 0.0 <= float(cfg.DUET_CONSENSUS.CSM_STRENGTH) < 1.0:
        raise ValueError("DUET_CONSENSUS.CSM_STRENGTH must be in [0, 1)")
    if str(cfg.DUET_CONSENSUS.HARD_LABEL_MODE) not in {
        "consensus",
        "duet_agreement",
    }:
        raise ValueError(
            "DUET_CONSENSUS.HARD_LABEL_MODE must be consensus or duet_agreement"
        )
    if float(cfg.DUET_CONSENSUS.AGREEMENT_BETA) <= 0.0:
        raise ValueError("DUET_CONSENSUS.AGREEMENT_BETA must be positive")
    if float(cfg.DUET_CONSENSUS.CONFLICT_BETA) <= 0.0:
        raise ValueError("DUET_CONSENSUS.CONFLICT_BETA must be positive")


def train_target(cfg):
    """Train Task and prompt branches through full-coverage anchored consensus."""
    _validate_config(cfg)
    loaders = _build_loaders(cfg)
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
    epsilon = float(cfg.DUET_CONSENSUS.EPSILON)
    epochs = int(cfg.DUET_CONSENSUS.EPOCHS)
    total_steps = epochs * len(loaders["train"])

    initial_task, initial_clip = _scan_predictions(
        loaders["scan"],
        net_f,
        net_b,
        net_c,
        prompt_model,
        int(cfg.class_num),
    )
    initial_consensus = entropy_weighted_poe(
        initial_task,
        initial_clip,
        epsilon=epsilon,
    )
    anchor_bank = initial_consensus["centered"].detach().cpu()
    initial_conflict = initial_task.argmax(dim=1) != initial_clip.argmax(dim=1)
    logging.info(
        "DUET anchored consensus initialized: samples={}; classes={}; "
        "initial_conflicts={}; initial_conflict_rate={:.2f}%; "
        "task_weight_mean={:.4f}; clip_weight_mean={:.4f}; "
        "coverage=100.00%; comparator=False; synthetic_supervision=False; "
        "third_visual_model=False; target_gt_affects_training=False".format(
            int(initial_task.shape[0]),
            int(initial_task.shape[1]),
            int(initial_conflict.sum().item()),
            100.0 * float(initial_conflict.float().mean().item()),
            float(initial_consensus["left_weight"].mean().item()),
            float(initial_consensus["right_weight"].mean().item()),
        )
    )
    logging.info(
        "DUET anchored consensus optimization: epochs={}; steps={}; "
        "hard_label_mode={}; alpha={:.3f}; agreement_beta={:.3f}; "
        "conflict_beta={:.3f}; diversity_delta={:.3f}; csm_strength={:.3f}; "
        "clip_encoders_frozen=True; prompt_trainable=True; classifier_trainable=True; "
        "synchronous_shared_snapshot=True".format(
            epochs,
            total_steps,
            str(cfg.DUET_CONSENSUS.HARD_LABEL_MODE),
            float(cfg.DUET_CONSENSUS.ALPHA),
            float(cfg.DUET_CONSENSUS.AGREEMENT_BETA),
            float(cfg.DUET_CONSENSUS.CONFLICT_BETA),
            float(cfg.DUET_CONSENSUS.DIVERSITY_DELTA),
            float(cfg.DUET_CONSENSUS.CSM_STRENGTH),
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
        scan_dynamic = entropy_weighted_poe(
            scan_task,
            scan_clip,
            epsilon=epsilon,
        )
        shift_state = consensus_shift_factors(
            scan_dynamic["probability"],
            epoch=epoch,
            total_epochs=epochs,
            strength=float(cfg.DUET_CONSENSUS.CSM_STRENGTH),
            epsilon=epsilon,
        )
        gamma_bank = shift_state["gamma"].detach().cpu()
        duet_agreement, duet_label, prior_active = _duet_hard_label_bank(
            cfg,
            scan_task,
            scan_clip,
            epoch,
        )

        net_f.train()
        net_b.train()
        net_c.train()
        # Evaluation mode keeps every frozen CLIP encoder deterministic;
        # gradients still flow from the text encoder into the prompt context.
        prompt_model.eval()
        task_loss_sum = 0.0
        prompt_loss_sum = 0.0
        hard_loss_sum = 0.0
        batches = 0

        for views, _, indices in loaders["train"]:
            # The target bottleneck contains training-mode BatchNorm1d, which
            # cannot estimate variance from a singleton final batch.
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
            dynamic = entropy_weighted_poe(
                task_probability,
                clip_probability,
                epsilon=epsilon,
            )
            modulated = modulate_anchored_consensus(
                anchor_bank[indices.long()].to(task_logits.device),
                dynamic["centered"],
                gamma_bank[indices.long()].to(task_logits.device),
            )
            shared_teacher = modulated["probability"].detach()

            consensus_hard = shared_teacher.argmax(dim=1)
            if str(cfg.DUET_CONSENSUS.HARD_LABEL_MODE) == "duet_agreement":
                agreement_batch = duet_agreement[indices.long()].to(task_logits.device)
                duet_label_batch = duet_label[indices.long()].to(task_logits.device)
                hard_label = torch.where(
                    agreement_batch,
                    duet_label_batch,
                    consensus_hard,
                )
                hard_weight = torch.where(
                    agreement_batch,
                    task_logits.new_full(
                        (task_logits.shape[0],),
                        float(cfg.DUET_CONSENSUS.AGREEMENT_BETA),
                    ),
                    task_logits.new_full(
                        (task_logits.shape[0],),
                        float(cfg.DUET_CONSENSUS.CONFLICT_BETA),
                    ),
                )
            else:
                hard_label = consensus_hard
                hard_weight = task_logits.new_full(
                    (task_logits.shape[0],),
                    float(cfg.DUET_CONSENSUS.CONFLICT_BETA),
                )

            task_iic = iic_mutual_information_loss(
                task_probability,
                shared_teacher,
                epsilon=epsilon,
            )
            prompt_iic = iic_mutual_information_loss(
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
                float(cfg.DUET_CONSENSUS.ALPHA) * task_iic
                + hard_loss
                - float(cfg.DUET_CONSENSUS.DIVERSITY_DELTA) * diversity
            )
            prompt_loss = prompt_iic
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
            (epoch + 1) % int(cfg.DUET_CONSENSUS.EVAL_INTERVAL) == 0
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
            "DUET anchored consensus epoch: {}/{}; accuracy={:.2f}%; "
            "task_loss={:.6f}; prompt_loss={:.6f}; hard_loss={:.6f}; "
            "current_conflicts={}; duet_agreements={}; prior_active={}; "
            "gamma_mean={:.4f}; gamma_min={:.4f}; gamma_max={:.4f}; "
            "soft_coverage=100.00%; conflict_hard_coverage=100.00%; "
            "target_gt_affects_training=False"
        ).format(
            epoch + 1,
            epochs,
            accuracy,
            task_loss_sum / max(batches, 1),
            prompt_loss_sum / max(batches, 1),
            hard_loss_sum / max(batches, 1),
            int(current_conflict.sum().item()),
            int(duet_agreement.sum().item()),
            prior_active,
            float(gamma_bank.mean().item()),
            float(gamma_bank.min().item()),
            float(gamma_bank.max().item()),
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
    logging.info(
        "DUET anchored consensus completed: saved_dir={}; "
        "inference_uses_target_only=True".format(output_dir)
    )
    return net_f, net_b, net_c
