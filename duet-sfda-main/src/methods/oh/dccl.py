"""Stage14 DUET 主循环，以及可选的 Boundary-Flip 扩展。

当前文件只保留论文主线真正运行的组件：

1. source/CLIP 双视角伪标签；
2. both-prior 类别校准与稳定伪标签记忆；
3. Stage14 blend target head；
4. graph-temporal residual (GTR)；
5. Boundary-Flip 候选和方向性监督。

Stage15--23 与更早 DCCL 实验已由 Git 标签
``archive/dccl-full-pre-prune-20260728`` 固定，不再混入主循环。
"""

import logging
import os
import os.path as osp

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from data.datautils_domain import build_dataset
from data.domain_datasets import domain_datasets
from src.data.data_list import GaussianBlur, ImageList_idx, NCropsTransform
from src.models import network
from src.utils import IID_losses, loss
from src.utils.boundary_flip import (
    boundary_flip_loss,
    update_boundary_flip_state,
)
from src.utils.conflict_diffusion import (
    adaptive_graph_teacher_fusion,
    dual_space_diffusion,
    graph_temporal_residual_weights,
    update_temporal_resolution,
)
from src.utils.consistency import prediction_consistency_kl


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def cosine_scheduler(cfg, optimizer, iter_num, max_iter, lr_min=1e-6):
    for param_group in optimizer.param_groups:
        lr_max = param_group['lr0']  # Initial learning rate
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * iter_num / max_iter))
        param_group['lr'] = lr
        param_group['weight_decay'] = (
            cfg.OPTIM.WD * param_group.get('weight_decay_scale', 1.0)
        )
        param_group['momentum'] = cfg.OPTIM.MOMENTUM
        param_group['nesterov'] = cfg.OPTIM.NESTEROV
    return optimizer


def get_augmentation(aug_type, normalize=True):
    if normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    if aug_type == "moco-v2":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "moco-v1":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "plain":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "clip_inference":
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "test":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return None


def get_augmentation_versions(cfg):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.

    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in 'tws':
        if version == "s":
            transform_list.append(get_augmentation("moco-v2"))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == 't':
            transform_list.append(get_augmentation("test"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    transform = NCropsTransform(transform_list)

    return transform


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    #   else:
    #     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(cfg):
    """建立 adaptation、伪标签刷新和完整评估三个 loader。"""
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    adaptation_path = str(cfg.DCCL.ADAPTATION_LIST).strip()
    target_list_path = adaptation_path if adaptation_path else cfg.t_dset_path
    if adaptation_path and not osp.isfile(target_list_path):
        raise FileNotFoundError(
            "DCCL.ADAPTATION_LIST does not exist: {}".format(target_list_path)
        )
    txt_tar = open(target_list_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    if adaptation_path:
        logging.info(
            "DCCL adaptation proxy list: {}; adaptation_samples={}; "
            "full_evaluation_samples={}".format(
                target_list_path, len(txt_tar), len(txt_test)
            )
        )
    # txt_test = open(cfg.t_dset_path).readlines()

    # if not cfg.da == 'uda':
    #     label_map_s = {}
    #     for i in range(len(cfg.src_classes)):
    #         label_map_s[cfg.src_classes[i]] = i

    #     new_tar = []
    #     for i in range(len(txt_tar)):
    #         rec = txt_tar[i]
    #         reci = rec.strip().split(' ')
    #         if int(reci[1]) in cfg.tar_classes:
    #             if int(reci[1]) in cfg.src_classes:
    #                 line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
    #                 new_tar.append(line)
    #             else:
    #                 line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
    #                 new_tar.append(line)
    #     txt_tar = new_tar.copy()
    #     txt_test = txt_tar.copy()

    train_transform = get_augmentation_versions(cfg)

    dsets["target"] = ImageList_idx(txt_tar, transform=train_transform)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=cfg.NUM_WORKERS, drop_last=False)
    # Pseudo-label indices must match the adaptation loader. Final evaluation
    # remains on txt_test, which is always the complete target list.
    dsets["test_aug"] = ImageList_idx(txt_tar, transform=train_transform)
    dset_loaders["test_aug"] = DataLoader(dsets["test_aug"], batch_size=train_bs, shuffle=False,
                                          num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders


def apply_target_head_logits(cfg, features, source_logits, target_head, curr_cycle):
    """Stage14 固定使用 source/target 两个分类头的线性 blend。"""
    if target_head is None or curr_cycle < cfg.DCCL.TARGET_HEAD_START_CYCLE:
        return source_logits
    if not 0.0 <= cfg.DCCL.TARGET_HEAD_MIX <= 1.0:
        raise ValueError("DCCL.TARGET_HEAD_MIX must be in [0, 1]")
    target_logits = target_head(features)
    mix = float(cfg.DCCL.TARGET_HEAD_MIX)
    return (1.0 - mix) * source_logits + mix * target_logits
def build_target_classifier_head(cfg, source_head):
    """建立 Stage14 的可训练 target head，并从 source head 初始化。"""
    target_head = network.feat_classifier(
        type=source_head.type,
        class_num=cfg.class_num,
        bottleneck_dim=cfg.bottleneck,
    ).cuda()
    target_head.load_state_dict(source_head.state_dict())
    return target_head


def cal_acc(
    loader,
    netF,
    netB,
    netC,
    cfg,
    target_head=None,
    curr_cycle=0,
    flag=False,
):
    """使用 Stage14 blend head 评估完整目标域。"""
    start_test = True
    with torch.no_grad():
        for data in loader:
            inputs = data[0].cuda()
            labels = data[1]
            features = netB(netF(inputs))
            outputs = apply_target_head_logits(
                cfg, features, netC(features), target_head, curr_cycle
            )
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def consistency_loss(weak_output, strong_output):
    return prediction_consistency_kl(weak_output, strong_output)


def init_loss_diagnostics():
    return {
        "steps": 0,
        "terms": {
            name: {"raw_sum": 0.0, "weighted_sum": 0.0, "active_batches": 0}
            for name in (
                "consistency",
                "stable_ce",
                "clip_kl",
                "gtr",
                "boundary_flip",
            )
        },
    }


def record_loss_diagnostic(diagnostics, name, loss_value, weight):
    if diagnostics is None:
        return
    term = diagnostics["terms"][name]
    raw_value = float(loss_value.detach().item())
    term["raw_sum"] += raw_value
    term["weighted_sum"] += raw_value * float(weight)
    term["active_batches"] += 1


def log_loss_diagnostics(diagnostics, cycle):
    if diagnostics is None:
        return
    steps = max(int(diagnostics["steps"]), 1)
    weighted_means = {
        name: values["weighted_sum"] / steps
        for name, values in diagnostics["terms"].items()
    }
    tracked_total = sum(weighted_means.values())
    fields = [
        "DCCL loss diagnostics: cycle={}; steps={}; tracked_weighted_total={:.6f}".format(
            int(cycle), int(diagnostics["steps"]), tracked_total
        )
    ]
    for name, values in diagnostics["terms"].items():
        active_batches = int(values["active_batches"])
        raw_active_mean = (
            values["raw_sum"] / active_batches if active_batches > 0 else 0.0
        )
        weighted_step_mean = weighted_means[name]
        share = (
            weighted_step_mean / tracked_total if tracked_total > 0.0 else 0.0
        )
        fields.append(
            "{}_raw={:.6f}; {}_weighted={:.6f}; {}_share={:.4f}; "
            "{}_active_batches={}".format(
                name,
                raw_active_mean,
                name,
                weighted_step_mean,
                name,
                share,
                name,
                active_batches,
            )
        )
    logging.info("; ".join(fields))


def train_clip(cfg, model, confi_imag, confi_dis, text_features, clip_optimizer, q_value):
    if cfg.SETTING.DATASET in domain_datasets:
        cfg.domain_name = cfg.domain[cfg.SETTING.T]

    set_id = 'sfuda'
    val_dataset = build_dataset(set_id, image_test(), confi_imag, confi_dis, cfg.DATA_DIR, cfg.domain_name,
                                mode='test')
    batchsize = cfg.TEST.BATCH_SIZE
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize, shuffle=True,
        num_workers=cfg.NUM_WORKERS, drop_last=False)

    max_iter = len(val_loader)
    iter_num = 0
    iter_test = iter(val_loader)
    total_corrects = 0
    total_samples = 0
    beta = cfg.ACTIVE.BETA

    while iter_num < max_iter:
        try:
            images, target, pseudo_label, _ = next(iter_test)
        except StopIteration:
            iter_test = iter(val_loader)
            images, target, pseudo_label, _ = next(iter_test)

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)

        images = images.cuda(int(cfg.GPU_ID), non_blocking=True)
        image = images
        target = target.cuda(int(cfg.GPU_ID), non_blocking=True)
        pseudo_label = pseudo_label.cuda()

        iter_num = iter_num + 1

        logits, _ = model(image, text_features)

        clip_preds = nn.Softmax(dim=1)(logits)
        loss, q_value = IID_losses.tsallis_mutual_info(clip_preds, pseudo_label, q_value, beta)
        # print(f"q_value: {q_value}")

        predicted_labels = clip_preds.argmax(dim=1)
        correct = (predicted_labels == target).sum().item()
        total_corrects += correct
        total_samples += target.size(0)

        clip_optimizer.zero_grad()
        loss.backward()
        clip_optimizer.step()

    avg_acc = total_corrects / total_samples if total_samples > 0 else 0.0
    log_str = ('CLIP visual Accuracy = {:.2f}%;').format(avg_acc * 100)
    logging.info(log_str)

    return clip_optimizer, q_value


def init_gtr_state(num_samples):
    """GTR 只需要三项时序状态，不再复用旧 candidate/ACCD 状态表。"""
    return {
        "pending_label": torch.full((num_samples,), -1, dtype=torch.long),
        "pending_count": torch.zeros(num_samples, dtype=torch.long),
        "stable_label": torch.full((num_samples,), -1, dtype=torch.long),
    }


def save_temporal_diagnostics(
    cfg,
    curr_cycle,
    mem_label,
    label_mask,
    clip_soft,
    source_label,
    clip_label,
    model_soft,
    teacher_soft,
    target_label,
    memory_weight=None,
    pl_state=None,
    boundary_flip_result=None,
):
    out_dir = osp.join(cfg.output_dir, "temporal_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"{cfg.name}_cycle{curr_cycle + 1:02d}.npz")
    payload = dict(
        cycle=np.array(curr_cycle + 1, dtype=np.int64),
        task=np.array(cfg.name),
        mix_label=mem_label.cpu().numpy().astype(np.int64),
        label_mask=label_mask.cpu().numpy().astype(bool),
        source_label=source_label.cpu().numpy().astype(np.int64),
        clip_label=clip_label.cpu().numpy().astype(np.int64),
        task_prob=model_soft.cpu().numpy().astype(np.float32),
        clip_prob=clip_soft.cpu().numpy().astype(np.float32),
        teacher_label=teacher_soft.argmax(dim=1).cpu().numpy().astype(np.int64),
        teacher_prob=teacher_soft.cpu().numpy().astype(np.float32),
        target_label=target_label.cpu().numpy().astype(np.int64),
    )
    if memory_weight is not None:
        payload["memory_weight"] = (
            memory_weight.cpu().numpy().astype(np.float32)
        )
    if pl_state is not None:
        for state_name in (
            "last_current_mask",
            "last_stable_mask",
            "last_conflict_mask",
        ):
            if state_name in pl_state:
                payload[state_name.removeprefix("last_")] = (
                    pl_state[state_name].cpu().numpy().astype(bool)
                )
    if boundary_flip_result is not None:
        boundary_fields = {
            "initial_label": np.int64,
            "base_label": np.int64,
            "adjusted_label": np.int64,
            "candidate_mask": bool,
            "stable_mask": bool,
            "active_mask": bool,
            "weight": np.float32,
            "semantic_similarity": np.float32,
            "flip_margin": np.float32,
            "switch_count": np.int64,
            "class_prior": np.float32,
            "class_mean_confidence": np.float32,
        }
        for name, dtype in boundary_fields.items():
            payload[f"boundary_flip_{name}"] = (
                boundary_flip_result[name].detach().cpu().numpy().astype(dtype)
            )
    np.savez_compressed(out_path, **payload)
    logging.info("DCCL temporal diagnostics wrote: {}".format(out_path))


def build_graph_fused_teacher(cfg, task_features, clip_features, model_soft, clip_soft, source_label, clip_label):
    """构造 Stage14 GTR 使用的双空间图后验与融合 teacher。"""
    _, _, graph_post, anchors = dual_space_diffusion(
        task_features,
        clip_features,
        model_soft,
        clip_soft,
        source_label,
        clip_label,
        anchor_ratio=cfg.DCCL.GTF_ANCHOR_RATIO,
        anchor_min_per_class=cfg.DCCL.GTF_ANCHOR_MIN_PER_CLASS,
        k=cfg.DCCL.GTF_GRAPH_K,
        temperature=cfg.DCCL.GTF_TEMPERATURE,
        alpha=cfg.DCCL.GTF_ALPHA,
        steps=cfg.DCCL.GTF_STEPS,
        chunk_size=cfg.DCCL.GTF_CHUNK_SIZE,
    )
    base_teacher = (model_soft + clip_soft) / 2
    teacher_soft, graph_weight = adaptive_graph_teacher_fusion(
        base_teacher,
        graph_post,
        strength=cfg.DCCL.GTF_STRENGTH,
        eps=cfg.DCCL.EPSILON,
    )
    logging.info(
        "DCCL graph-teacher fusion: anchors={}; strength={:.3f}; "
        "mean_graph_weight={:.4f}; max_graph_weight={:.4f}; "
        "changed_top1={}".format(
            int(anchors.sum().item()),
            float(cfg.DCCL.GTF_STRENGTH),
            float(graph_weight.mean().item()),
            float(graph_weight.max().item()),
            int((base_teacher.argmax(dim=1) != teacher_soft.argmax(dim=1)).sum().item()),
        )
    )
    return teacher_soft, graph_post, graph_weight, anchors


def train_target(cfg):
    """训练 DUET/Stage14 及其 Boundary-Flip 扩展。

    中文主流程导航
    --------------
    1. 加载冻结的源分类器、可适配的特征提取器和 DUET 的 CLIP 分支；
    2. 每个 cycle 全量刷新 task/CLIP 概率与稳定 agreement 伪标签；
    3. Stage14 使用稳定伪标签训练 target head，并保留图时序残差（GTR）；
    4. 若 ``BOUNDARY_FLIP.ENABLED``，仅从类别校准产生的稳定翻转中选出
       方向性监督；该分支不会运行旧 DCCL promotion/candidate 路径；
    5. minibatch 中把 flip loss 以较小权重并入原 DUET/Stage14 损失。

    目标真实标签只用于日志和最终评估，不进入候选门控或训练损失。
    """
    clip_model, _, _ = clip.load(cfg.ACTIVE.ARCH)
    clip_model.float()
    text_inputs = clip_pre_text(cfg)

    dset_loaders = data_load(cfg)
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    netF.eval()
    netB.eval()
    netC.eval()

    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if cfg.OPTIM.LR_DECAY2 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2}]
        else:
            v.requires_grad = False
    if cfg.DCCL.TARGET_HEAD_LR_MULT <= 0:
        raise ValueError("DCCL.TARGET_HEAD_LR_MULT must be positive")
    target_head = build_target_classifier_head(cfg, netC)
    target_head.train()
    for parameter in target_head.parameters():
        parameter.requires_grad = True
        param_group.append(
            {
                "params": parameter,
                "lr": cfg.OPTIM.LR * cfg.DCCL.TARGET_HEAD_LR_MULT,
            }
        )
    logging.info(
        "Stage14 blend target head: mix={:.3f}; start_cycle={}; "
        "lr_mult={:.3f}".format(
            float(cfg.DCCL.TARGET_HEAD_MIX),
            int(cfg.DCCL.TARGET_HEAD_START_CYCLE),
            float(cfg.DCCL.TARGET_HEAD_LR_MULT),
        )
    )

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    for param in clip_model.transformer.parameters():
        param.requires_grad = False
    for param in clip_model.token_embedding.parameters():
        param.requires_grad = False
    clip_model.positional_embedding.requires_grad = False
    for param in clip_model.ln_final.parameters():
        param.requires_grad = False
    clip_model.text_projection.requires_grad = False

    vision_params = [p for p in clip_model.visual.parameters() if p.requires_grad]

    clip_optimizer = optim.Adam(vision_params, lr=cfg.ACTIVE.FINE_LR, betas=(0.9, 0.999), eps=1e-8)
    clip_optimizer = op_copy(clip_optimizer)

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL

    if cfg.BOUNDARY_FLIP.ENABLED:
        if cfg.BOUNDARY_FLIP.START_CYCLE < 1:
            raise ValueError("BOUNDARY_FLIP.START_CYCLE must be at least 1")
        if cfg.BOUNDARY_FLIP.STABLE_CYCLES <= 0:
            raise ValueError("BOUNDARY_FLIP.STABLE_CYCLES must be positive")
        if cfg.BOUNDARY_FLIP.MAX_PER_PAIR <= 0:
            raise ValueError("BOUNDARY_FLIP.MAX_PER_PAIR must be positive")
        if cfg.BOUNDARY_FLIP.LOSS_PAR <= 0:
            raise ValueError("BOUNDARY_FLIP.LOSS_PAR must be positive")
        logging.info(
            "Boundary-Flip DUET enabled: start_cycle={}; alpha={:.4f}; "
            "min_confidence={:.4f}; min_margin={:.4f}; semantic_threshold={:.4f}; "
            "stable_cycles={}; max_switches={}; max_per_pair={}; "
            "loss_par={:.4f}; negative_weight={:.4f}".format(
                int(cfg.BOUNDARY_FLIP.START_CYCLE),
                float(cfg.BOUNDARY_FLIP.LOGIT_ALPHA),
                float(cfg.BOUNDARY_FLIP.MIN_ADJUSTED_CONFIDENCE),
                float(cfg.BOUNDARY_FLIP.MIN_MARGIN),
                float(cfg.BOUNDARY_FLIP.SEMANTIC_THRESHOLD),
                int(cfg.BOUNDARY_FLIP.STABLE_CYCLES),
                int(cfg.BOUNDARY_FLIP.MAX_SWITCHES),
                int(cfg.BOUNDARY_FLIP.MAX_PER_PAIR),
                float(cfg.BOUNDARY_FLIP.LOSS_PAR),
                float(cfg.BOUNDARY_FLIP.NEGATIVE_WEIGHT),
            )
        )

    pl_state = None
    gtr_state = None
    boundary_flip_state = None
    with torch.no_grad():
        text_features = F.normalize(
            clip_model.encode_text(text_inputs), dim=1
        ).detach()
    boundary_flip_text_features = text_features.float().cpu()
    curr_cycle = 0
    q_value = cfg.ACTIVE.Q_VALUE
    while curr_cycle < cfg.ACTIVE.CYCLE:
        iter_num = 0

        netF.eval()
        netB.eval()
        target_head.eval()
        (
            mem_label,
            label_mask,
            memory_weight,
            confi_imag,
            confi_dis,
            clip_soft,
            source_label,
            clip_label,
            model_soft,
            task_features,
            clip_features,
            target_label,
            pl_state,
        ) = obtain_label(
            cfg,
            dset_loaders['test_aug'],
            netF,
            netB,
            netC,
            target_head,
            text_features,
            clip_model,
            pl_state,
            curr_cycle,
        )
        if gtr_state is None:
            gtr_state = init_gtr_state(source_label.size(0))
        teacher_soft, graph_teacher, _, _ = build_graph_fused_teacher(
            cfg,
            task_features,
            clip_features,
            model_soft,
            clip_soft,
            source_label,
            clip_label,
        )
        boundary_flip_result = None
        if cfg.BOUNDARY_FLIP.ENABLED:
            # 中文阅读顺序：
            # agreement anchors -> 类别频率/置信度统计 -> 动态校准翻转
            # -> 视角/语义/时序门控 -> 类别对预算。
            boundary_flip_state, boundary_flip_result = (
                update_boundary_flip_state(
                    model_soft,
                    clip_soft,
                    source_label,
                    clip_label,
                    label_mask,
                    boundary_flip_text_features,
                    boundary_flip_state,
                    curr_cycle=curr_cycle,
                    start_cycle=int(cfg.BOUNDARY_FLIP.START_CYCLE),
                    alpha=float(cfg.BOUNDARY_FLIP.LOGIT_ALPHA),
                    min_adjusted_confidence=float(
                        cfg.BOUNDARY_FLIP.MIN_ADJUSTED_CONFIDENCE
                    ),
                    min_margin=float(cfg.BOUNDARY_FLIP.MIN_MARGIN),
                    semantic_threshold=float(
                        cfg.BOUNDARY_FLIP.SEMANTIC_THRESHOLD
                    ),
                    stable_cycles=int(cfg.BOUNDARY_FLIP.STABLE_CYCLES),
                    max_switches=int(cfg.BOUNDARY_FLIP.MAX_SWITCHES),
                    max_per_pair=int(cfg.BOUNDARY_FLIP.MAX_PER_PAIR),
                    min_weight=float(cfg.BOUNDARY_FLIP.MIN_WEIGHT),
                    epsilon=float(cfg.DCCL.EPSILON),
                )
            )
            flip_active = boundary_flip_result["active_mask"]
            flip_oracle_accuracy = (
                float(
                    (
                        boundary_flip_result["adjusted_label"][flip_active]
                        == target_label[flip_active]
                    )
                    .float()
                    .mean()
                    .item()
                )
                if flip_active.any()
                else 0.0
            )
            active_pairs = torch.unique(
                boundary_flip_result["initial_label"][flip_active] * cfg.class_num
                + boundary_flip_result["adjusted_label"][flip_active]
            )
            logging.info(
                "Boundary-Flip DUET proposals: cycle={}; changed_top1={}; "
                "candidates={}; temporally_stable={}; active_after_pair_budget={}; "
                "active_pairs={}; switched_or_interrupted={}; mean_weight={:.6f}; "
                "oracle_active_accuracy={:.2f}%".format(
                    curr_cycle + 1,
                    int(
                        (
                            boundary_flip_result["base_label"]
                            != boundary_flip_result["adjusted_label"]
                        ).sum().item()
                    ),
                    int(boundary_flip_result["candidate_mask"].sum().item()),
                    int(boundary_flip_result["stable_mask"].sum().item()),
                    int(flip_active.sum().item()),
                    int(active_pairs.numel()),
                    int(
                        (
                            boundary_flip_result["switch_count"] > 0
                        ).sum().item()
                    ),
                    float(
                        boundary_flip_result["weight"][flip_active].mean().item()
                    )
                    if flip_active.any()
                    else 0.0,
                    flip_oracle_accuracy * 100.0,
                )
            )
        save_temporal_diagnostics(
            cfg,
            curr_cycle,
            mem_label,
            label_mask,
            clip_soft,
            source_label,
            clip_label,
            model_soft,
            teacher_soft,
            target_label,
            memory_weight=memory_weight,
            pl_state=pl_state,
            boundary_flip_result=boundary_flip_result,
        )
        # DUET 主 KL 始终以当前 CLIP posterior 为 teacher。
        kl_target = clip_soft
        gtr_target = teacher_soft
        gtr_weight = torch.zeros(source_label.size(0), dtype=torch.float)
        if cfg.DCCL.GTR_PAR > 0:
            if graph_teacher is None:
                raise RuntimeError("Stage14 graph teacher construction failed")
            teacher_label = teacher_soft.argmax(dim=1)
            graph_label = graph_teacher.argmax(dim=1)
            gtr_eligible = (
                (source_label != clip_label)
                & (teacher_label == graph_label)
            )
            (
                gtr_state["pending_label"],
                gtr_state["pending_count"],
                gtr_state["stable_label"],
                gtr_newly_stable,
                gtr_stable_mask,
                gtr_demoted,
            ) = update_temporal_resolution(
                gtr_state["pending_label"],
                gtr_state["pending_count"],
                gtr_state["stable_label"],
                gtr_eligible,
                teacher_label,
                cfg.DCCL.GTR_STABLE_CYCLES,
                cfg.DCCL.GTR_MEMORY,
            )
            gtr_weight, gtr_graph_conf, gtr_disagreement = graph_temporal_residual_weights(
                clip_soft,
                graph_teacher,
                teacher_label,
                source_label,
                clip_label,
                gtr_state["stable_label"],
                cfg.DCCL.GTR_MIN_GRAPH_CONF,
                cfg.DCCL.GTR_MIN_DISAGREEMENT,
                eps=cfg.DCCL.EPSILON,
            )
            active_gtr = gtr_weight > 0
            logging.info(
                "DCCL graph-temporal residual: eligible={}; newly_stable={}; "
                "stable_active={}; demoted={}; loss_active={}; mean_weight={:.4f}; "
                "mean_graph_conf={:.4f}; mean_disagreement={:.4f}".format(
                    int(gtr_eligible.sum().item()),
                    int(gtr_newly_stable.sum().item()),
                    int(gtr_stable_mask.sum().item()),
                    int(gtr_demoted.sum().item()),
                    int(active_gtr.sum().item()),
                    float(gtr_weight[active_gtr].mean().item()) if active_gtr.any() else 0.0,
                    float(gtr_graph_conf[active_gtr].mean().item()) if active_gtr.any() else 0.0,
                    float(gtr_disagreement[active_gtr].mean().item()) if active_gtr.any() else 0.0,
                )
            )

        # Stable agreement 是唯一 hard CE 来源；历史 promotion、candidate、
        # ACCD 与 dual-tier 分支均已从当前主线删除。
        hard_mask = label_mask

        boundary_flip_active_mask = None
        boundary_flip_early_label = None
        boundary_flip_late_label = None
        boundary_flip_weight = None
        if boundary_flip_result is not None:
            boundary_flip_active_mask = boundary_flip_result["active_mask"]
            boundary_flip_early_label = boundary_flip_result[
                "initial_label"
            ].cuda()
            boundary_flip_late_label = boundary_flip_result[
                "adjusted_label"
            ].cuda()
            boundary_flip_weight = boundary_flip_result["weight"].cuda()

        kl_target = kl_target.cuda()
        gtr_target = gtr_target.cuda()
        gtr_weight = gtr_weight.cuda()
        mem_label = mem_label.cuda()
        clip_optimizer, q_value = train_clip(
            cfg,
            clip_model,
            confi_imag,
            confi_dis,
            text_inputs,
            clip_optimizer,
            q_value,
        )

        cfg.load = 'prompt_model.pt'
        netF.train()
        netB.train()
        target_head.train()
        boundary_flip_loss_sum = 0.0
        boundary_flip_loss_batches = 0
        loss_diagnostics = (
            init_loss_diagnostics() if cfg.DCCL.LOSS_DIAG else None
        )
        iter_test = iter(dset_loaders["target"])
        while iter_num < max_iter:
            try:
                inputs_test, _, tar_idx = next(iter_test)
            except StopIteration:
                iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = next(iter_test)

            if inputs_test[0].size(0) == 1:
                continue

            weak_x = inputs_test[1].cuda()
            strong_x = inputs_test[2].cuda()

            iter_num += 1
            if loss_diagnostics is not None:
                loss_diagnostics["steps"] += 1
            optimizer = cosine_scheduler(cfg, optimizer, iter_num=iter_num, max_iter=max_iter)

            weak_feas = netB(netF(weak_x))
            strong_feas = netB(netF(strong_x))
            weak_logits = apply_target_head_logits(
                cfg, weak_feas, netC(weak_feas), target_head, curr_cycle
            )
            strong_logits = apply_target_head_logits(
                cfg, strong_feas, netC(strong_feas), target_head, curr_cycle
            )

            weak_preds = F.softmax(weak_logits, dim=1)

            # 网络仍会对整个 batch 计算 weak/strong logits；hard_mask 只决定
            # 哪些样本进入下面的伪标签分类 CE。mask=False 不代表跳过前向传播。
            filtered_idx = tar_idx[hard_mask[tar_idx]]

            con_loss = consistency_loss(weak_logits, strong_logits)
            classifier_loss = con_loss * cfg.ACTIVE.CON_PAR
            record_loss_diagnostic(
                loss_diagnostics,
                "consistency",
                con_loss,
                cfg.ACTIVE.CON_PAR,
            )
            # classifier_loss = metric_loss * cfg.ACTIVE.CLS_PAR
            if cfg.ACTIVE.CLS_PAR > 0:
                pred = mem_label[filtered_idx]
                supervised_logits = weak_logits[hard_mask[tar_idx]]
                if pred.size(0) != 0:
                    stable_ce_loss = nn.CrossEntropyLoss()(
                        supervised_logits, pred
                    )
                    classifier_loss += stable_ce_loss * cfg.ACTIVE.CLS_PAR
                    record_loss_diagnostic(
                        loss_diagnostics,
                        "stable_ce",
                        stable_ce_loss,
                        cfg.ACTIVE.CLS_PAR,
                    )
            if boundary_flip_active_mask is not None:
                # 对每个稳定翻转同时执行：
                #   (1) 提升 late label；(2) 抑制 early label。
                # 它只作用于 active flip，不覆盖原有 teacher 或 hard CE。
                batch_flip_mask = boundary_flip_active_mask[tar_idx]
                if batch_flip_mask.any():
                    flip_positions = torch.nonzero(
                        batch_flip_mask, as_tuple=False
                    ).squeeze(1).cuda()
                    flip_indices = tar_idx[batch_flip_mask].cuda()
                    flip_loss = boundary_flip_loss(
                        weak_logits[flip_positions],
                        boundary_flip_early_label[flip_indices],
                        boundary_flip_late_label[flip_indices],
                        boundary_flip_weight[flip_indices],
                        negative_weight=float(
                            cfg.BOUNDARY_FLIP.NEGATIVE_WEIGHT
                        ),
                        epsilon=float(cfg.DCCL.EPSILON),
                    )
                    classifier_loss += (
                        flip_loss * cfg.BOUNDARY_FLIP.LOSS_PAR
                    )
                    record_loss_diagnostic(
                        loss_diagnostics,
                        "boundary_flip",
                        flip_loss,
                        cfg.BOUNDARY_FLIP.LOSS_PAR,
                    )
                    boundary_flip_loss_sum += float(flip_loss.detach().item())
                    boundary_flip_loss_batches += 1
            kl_target_batch = kl_target[tar_idx]
            per_sample_kl = F.kl_div(weak_preds.log(), kl_target_batch, reduction="none").sum(dim=1)
            mi_loss = per_sample_kl.mean()
            classifier_loss += mi_loss * cfg.ACTIVE.KL_PAR
            record_loss_diagnostic(
                loss_diagnostics,
                "clip_kl",
                mi_loss,
                cfg.ACTIVE.KL_PAR,
            )
            if cfg.DCCL.GTR_PAR > 0:
                gtr_weight_batch = gtr_weight[tar_idx]
                if gtr_weight_batch.sum() > 0:
                    gtr_target_batch = gtr_target[tar_idx]
                    per_sample_gtr = F.kl_div(
                        weak_preds.log(), gtr_target_batch, reduction="none"
                    ).sum(dim=1)
                    gtr_loss = (
                        per_sample_gtr * gtr_weight_batch
                    ).sum() / gtr_weight_batch.sum()
                    classifier_loss += gtr_loss * cfg.DCCL.GTR_PAR
                    record_loss_diagnostic(
                        loss_diagnostics,
                        "gtr",
                        gtr_loss,
                        cfg.DCCL.GTR_PAR,
                    )

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netF.eval()
                netB.eval()
                target_head.eval()
                if cfg.SETTING.DATASET == 'VISDA-C':
                    acc_s_te, acc_list = cal_acc(
                        dset_loaders['test'],
                        netF,
                        netB,
                        netC,
                        cfg=cfg,
                        target_head=target_head,
                        curr_cycle=curr_cycle,
                        flag=True,
                    )
                    log_str = ('Task: {}, Iter:{}/{}; Cycle: {}/{}; '
                               'Accuracy = {:.2f}%; classifier_loss = {}').format(cfg.name, iter_num, max_iter,
                                                                                  curr_cycle + 1, cfg.ACTIVE.CYCLE,
                                                                                  acc_s_te,
                                                                                  classifier_loss) + '\n' + acc_list
                else:
                    acc_s_te, _ = cal_acc(
                        dset_loaders['test'],
                        netF,
                        netB,
                        netC,
                        cfg=cfg,
                        target_head=target_head,
                        curr_cycle=curr_cycle,
                        flag=False,
                    )
                    log_str = ('Task: {}, Iter:{}/{}; Cycle: {}/{}; '
                               'Accuracy = {:.2f}%; classifier_loss = {}').format(cfg.name, iter_num, max_iter,
                                                                                  curr_cycle + 1, cfg.ACTIVE.CYCLE,
                                                                                  acc_s_te, classifier_loss)

                if boundary_flip_result is not None:
                    log_str += (
                        "; boundary_flip_candidates={}; "
                        "boundary_flip_stable={}; boundary_flip_active={}; "
                        "boundary_flip_loss={:.6f}; boundary_flip_batches={}"
                    ).format(
                        int(
                            boundary_flip_result["candidate_mask"].sum().item()
                        ),
                        int(
                            boundary_flip_result["stable_mask"].sum().item()
                        ),
                        int(
                            boundary_flip_result["active_mask"].sum().item()
                        ),
                        (
                            boundary_flip_loss_sum
                            / boundary_flip_loss_batches
                            if boundary_flip_loss_batches > 0
                            else 0.0
                        ),
                        boundary_flip_loss_batches,
                    )
                logging.info(log_str)
                boundary_flip_loss_sum = 0.0
                boundary_flip_loss_batches = 0
                netF.train()
                netB.train()
                target_head.train()
        log_loss_diagnostics(loss_diagnostics, curr_cycle + 1)
        curr_cycle += 1

    return netF, netB, netC



def init_pseudo_label_state(num_samples):
    """初始化逐样本的连续一致性状态。"""
    return {
        "pending_label": torch.full((num_samples,), -1, dtype=torch.long),
        "pending_count": torch.zeros(num_samples, dtype=torch.long),
        "stable_label": torch.full((num_samples,), -1, dtype=torch.long),
    }


def apply_pseudo_label_memory(
    cfg,
    matching_indices,
    mixed_label,
    mixed_confidence,
    pl_state,
    curr_cycle,
):
    """把单轮双视角一致，提升为跨 cycle 连续一致的 hard CE 标签。

    warmup 期间直接使用当前 agreement；之后只保留连续达到阈值的标签。
    若本轮冲突或标签改变，stable 身份会撤销，避免旧伪标签永久污染训练。
    """
    if cfg.DCCL.PL_STABLE_CYCLES <= 0:
        raise ValueError("DCCL.PL_STABLE_CYCLES must be positive")
    current_mask = matching_indices & (
        mixed_confidence >= cfg.DCCL.PL_MEMORY_MIN_CONF
    )
    if pl_state is None:
        pl_state = init_pseudo_label_state(mixed_label.numel())

    same_label = pl_state["pending_label"] == mixed_label
    pl_state["pending_count"] = torch.where(
        current_mask & same_label,
        pl_state["pending_count"] + 1,
        torch.where(
            current_mask,
            torch.ones_like(pl_state["pending_count"]),
            torch.zeros_like(pl_state["pending_count"]),
        ),
    )
    pl_state["pending_label"] = torch.where(
        current_mask,
        mixed_label,
        torch.full_like(pl_state["pending_label"], -1),
    )
    newly_stable = current_mask & (
        pl_state["pending_count"] >= cfg.DCCL.PL_STABLE_CYCLES
    )
    pl_state["stable_label"] = torch.where(
        newly_stable,
        mixed_label,
        torch.full_like(pl_state["stable_label"], -1),
    )

    warmup = curr_cycle < cfg.DCCL.PL_MEMORY_WARMUP_CYCLES
    stable_mask = pl_state["stable_label"] >= 0
    label_mask = current_mask if warmup else stable_mask
    memory_label = torch.where(
        stable_mask, pl_state["stable_label"], mixed_label
    )
    memory_weight = label_mask.float()
    pl_state["last_current_mask"] = current_mask.clone()
    pl_state["last_stable_mask"] = stable_mask.clone()
    pl_state["last_conflict_mask"] = (~matching_indices).clone()

    logging.info(
        "DCCL pseudo-label memory: warmup={}; current={}; stable={}; selected={}".format(
            int(warmup),
            int(current_mask.sum().item()),
            int(stable_mask.sum().item()),
            int(label_mask.sum().item()),
        )
    )
    return label_mask, memory_label, memory_weight, pl_state


def prior_calibrate(probability, power, epsilon):
    """用预测类别先验做温和的长尾校准。"""
    prior = probability.mean(dim=0).clamp_min(epsilon)
    calibrated = probability / prior.pow(power)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(epsilon)


def apply_both_prior_calibration(cfg, source_prob, clip_prob):
    """Stage14 固定校准：source 与 CLIP 分别校准后等权融合。"""
    source_calibrated = prior_calibrate(
        source_prob, cfg.DCCL.CALIB_POWER, cfg.DCCL.EPSILON
    )
    clip_calibrated = prior_calibrate(
        clip_prob, cfg.DCCL.CALIB_POWER, cfg.DCCL.EPSILON
    )
    mixed_prob = (source_calibrated + clip_calibrated) / 2
    logging.info(
        "DCCL both-prior calibration: power={:.3f}; "
        "source_prior_range=({:.4f},{:.4f}); clip_prior_range=({:.4f},{:.4f})".format(
            float(cfg.DCCL.CALIB_POWER),
            float(source_prob.mean(dim=0).min().item()),
            float(source_prob.mean(dim=0).max().item()),
            float(clip_prob.mean(dim=0).min().item()),
            float(clip_prob.mean(dim=0).max().item()),
        )
    )
    return source_calibrated, clip_calibrated, mixed_prob


def obtain_label(
    cfg,
    loader,
    netF,
    netB,
    netC,
    target_head,
    text_features,
    clip_model,
    pl_state,
    curr_cycle,
):
    """生成 Stage14 双视角伪标签及 Boundary-Flip 所需的诊断量。"""
    first_batch = True
    with torch.no_grad():
        normalized_text = F.normalize(text_features, dim=1)
        clip_logit_scale = clip_model.logit_scale.exp()
        for inputs_test, labels, _ in loader:
            weak_x = inputs_test[1].cuda()
            task_features = netB(netF(weak_x))
            source_logits = apply_target_head_logits(
                cfg,
                task_features,
                netC(task_features),
                target_head,
                curr_cycle,
            )
            clip_features = F.normalize(
                clip_model.encode_image(weak_x), dim=1
            )
            clip_logits = (
                clip_logit_scale * clip_features @ normalized_text.t()
            )

            if first_batch:
                all_source_logits = source_logits.float().cpu()
                all_clip_logits = clip_logits.float().cpu()
                all_task_features = task_features.float().cpu()
                all_clip_features = clip_features.float().cpu()
                all_labels = labels.long().cpu()
                first_batch = False
            else:
                all_source_logits = torch.cat(
                    (all_source_logits, source_logits.float().cpu()), dim=0
                )
                all_clip_logits = torch.cat(
                    (all_clip_logits, clip_logits.float().cpu()), dim=0
                )
                all_task_features = torch.cat(
                    (all_task_features, task_features.float().cpu()), dim=0
                )
                all_clip_features = torch.cat(
                    (all_clip_features, clip_features.float().cpu()), dim=0
                )
                all_labels = torch.cat((all_labels, labels.long().cpu()), dim=0)

    source_prob = F.softmax(all_source_logits, dim=1)
    clip_prob = F.softmax(all_clip_logits, dim=1)
    source_prob, clip_prob, mixed_prob = apply_both_prior_calibration(
        cfg, source_prob, clip_prob
    )
    source_label = source_prob.argmax(dim=1)
    clip_label = clip_prob.argmax(dim=1)
    mixed_confidence, mixed_label = mixed_prob.max(dim=1)
    matching_indices = source_label == clip_label

    label_mask, memory_label, memory_weight, pl_state = (
        apply_pseudo_label_memory(
            cfg,
            matching_indices,
            mixed_label,
            mixed_confidence,
            pl_state,
            curr_cycle,
        )
    )

    selected_accuracy = (
        float((memory_label[label_mask] == all_labels[label_mask]).float().mean().item())
        if label_mask.any()
        else 0.0
    )
    logging.info(
        "DCCL pseudo labels: selected={}/{}; oracle_accuracy={:.2f}%; "
        "source_accuracy={:.2f}%; clip_accuracy={:.2f}%; mixed_accuracy={:.2f}%".format(
            int(label_mask.sum().item()),
            int(label_mask.numel()),
            selected_accuracy * 100.0,
            float((source_label == all_labels).float().mean().item()) * 100.0,
            float((clip_label == all_labels).float().mean().item()) * 100.0,
            float((mixed_label == all_labels).float().mean().item()) * 100.0,
        )
    )

    return (
        memory_label,
        label_mask,
        memory_weight,
        loader.dataset.imgs,
        mixed_prob.detach(),
        clip_prob,
        source_label,
        clip_label,
        source_prob,
        all_task_features,
        all_clip_features,
        all_labels,
        pl_state,
    )

def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.ACTIVE.CTX_INIT.replace("_", " ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts
