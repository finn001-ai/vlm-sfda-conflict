"""
Builds upon: https://github.com/tim-learn/SHOT
Corresponding paper: http://proceedings.mlr.press/v119/liang20a/liang20a.pdf
"""

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
import seaborn as sns

from torchvision import transforms
from src.utils import loss
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList, ImageList_idx
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from src.utils.utils import *
from src.data.data_list import *
from src.utils import loss, prompt_tuning, IID_losses
from src.utils.conflict_diffusion import (
    conflict_diffusion_evidence,
    dual_space_diffusion,
    update_temporal_resolution,
)
# from src.utils import loss, active_prompt, IID_losses
# from proposed_method import *
from torch.nn.functional import normalize
from data.datautils_domain import build_dataset
from data.cls_to_names import *
from data.domain_datasets import domain_datasets
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(cfg, optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = cfg.OPTIM.WD
        param_group['momentum'] = cfg.OPTIM.MOMENTUM
        param_group['nesterov'] = cfg.OPTIM.NESTEROV
    return optimizer


def cosine_scheduler(cfg, optimizer, iter_num, max_iter, lr_min=1e-6):
    for param_group in optimizer.param_groups:
        lr_max = param_group['lr0']  # Initial learning rate
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * iter_num / max_iter))
        param_group['lr'] = lr
        param_group['weight_decay'] = cfg.OPTIM.WD
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


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    #   else:
    #     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


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


def create_white_image(resize_size=256, crop_size=224):
    white_image = Image.new("RGB", (resize_size, resize_size), (255, 255, 255))
    # white_image = Image.new("RGB", (resize_size, resize_size), (0, 0, 0))
    transform_pipeline = image_test(resize_size, crop_size)
    return transform_pipeline(white_image)


def data_load(cfg):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
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
    dsets["test_aug"] = ImageList_idx(txt_test, transform=train_transform)
    dset_loaders["test_aug"] = DataLoader(dsets["test_aug"], batch_size=train_bs, shuffle=False,
                                          num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
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
    # Apply softmax to both outputs to get probabilities
    # weak_probs = F.softmax(weak_output, dim=1)
    # strong_probs = F.softmax(strong_output, dim=1)
    weak_probs = nn.Softmax(dim=1)(weak_output)
    strong_probs = nn.Softmax(dim=1)(strong_output)

    # Compute KL divergence between the weak and strong probabilities
    loss = F.kl_div(strong_probs.log(), weak_probs, reduction="batchmean")
    return loss


def train_clip(cfg, model, confi_imag, confi_dis, text_features, clip_optimizer, q_value):
    if cfg.SETTING.DATASET in domain_datasets:
        cfg.domain_name = cfg.domain[cfg.SETTING.T]
        classnames = cfg.classname

    if 'RN' in cfg.DIFO.ARCH:
        data_transform = image_test_50()
    else:
        data_transform = image_test()
        # data_transform = get_augmentation("plain")

    set_id = 'sfuda'
    val_dataset = build_dataset(set_id, data_transform, confi_imag, confi_dis, cfg.DATA_DIR, cfg.domain_name,
                                mode='test')
    batchsize = cfg.TEST.BATCH_SIZE
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize, shuffle=True,
        num_workers=cfg.NUM_WORKERS, drop_last=False)

    max_iter = len(val_loader)
    iter_num = 0
    total_corrects = 0
    total_samples = 0
    beta = cfg.ACTIVE.BETA

    while iter_num < max_iter:
        try:
            images, target, pseudo_label, _ = next(iter_test)
        except:
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


def spectral_entropy(text_features, EPS=1e-9):
    corr_matrix = torch.corrcoef(text_features)
    eigenvalues = torch.linalg.eigvalsh(corr_matrix)
    eigenvalues = eigenvalues / eigenvalues.sum()
    spectral_ent = - (eigenvalues * torch.log(eigenvalues + EPS)).sum().item()
    return spectral_ent


def init_conflict_state(num_samples):
    return {
        "promoted_label": torch.full((num_samples,), -1, dtype=torch.long),
        "candidate_side": torch.full((num_samples,), -1, dtype=torch.long),
        "candidate_count": torch.zeros(num_samples, dtype=torch.long),
        "rejected": torch.zeros(num_samples, dtype=torch.bool),
        "accd_pending_label": torch.full((num_samples,), -1, dtype=torch.long),
        "accd_pending_count": torch.zeros(num_samples, dtype=torch.long),
        "accd_resolved_label": torch.full((num_samples,), -1, dtype=torch.long),
        "accd_anchor_label": torch.full((num_samples,), -1, dtype=torch.long),
    }


def update_conflict_state(cfg, state, source_label, clip_label, model_soft):
    conflict_mask = source_label != clip_label
    promoted_mask = state["promoted_label"] >= 0
    sample_idx = torch.arange(source_label.size(0))

    source_prob = model_soft[sample_idx, source_label]
    clip_prob = model_soft[sample_idx, clip_label]
    candidate_mass = source_prob + clip_prob
    candidate_gap = torch.abs(source_prob - clip_prob)
    preferred_side = torch.where(source_prob >= clip_prob, torch.zeros_like(source_label), torch.ones_like(source_label))

    dominates = (
        conflict_mask
        & (~promoted_mask)
        & (candidate_mass >= cfg.DCCL.TAU_HIGH)
        & (candidate_gap >= cfg.DCCL.GAP_PROMOTE)
    )
    same_side = state["candidate_side"] == preferred_side
    state["candidate_count"] = torch.where(
        dominates & same_side,
        state["candidate_count"] + 1,
        torch.where(dominates, torch.ones_like(state["candidate_count"]), torch.zeros_like(state["candidate_count"])),
    )
    state["candidate_side"] = torch.where(dominates, preferred_side, state["candidate_side"])

    promote_mask = dominates & (state["candidate_count"] >= cfg.DCCL.PROMOTE_K)
    promoted = torch.where(preferred_side == 0, source_label, clip_label)
    state["promoted_label"] = torch.where(promote_mask, promoted, state["promoted_label"])

    # Rejection is re-evaluated every cycle so a sample can recover later.
    state["rejected"] = conflict_mask & (state["promoted_label"] < 0) & (candidate_mass < cfg.DCCL.TAU_LOW)

    logging.info(
        "DCCL states: conflicts={}; promoted={}; rejected={}; candidates={}".format(
            int(conflict_mask.sum().item()),
            int((state["promoted_label"] >= 0).sum().item()),
            int(state["rejected"].sum().item()),
            int((conflict_mask & (state["promoted_label"] < 0) & (~state["rejected"])).sum().item()),
        )
    )


def get_candidate_weight(cfg, candidate_mass):
    if cfg.DCCL.CAND_WEIGHT == "none":
        return torch.ones_like(candidate_mass)
    if cfg.DCCL.CAND_WEIGHT == "mass":
        return candidate_mass
    if cfg.DCCL.CAND_WEIGHT == "ramp":
        denom = max(float(1.0 - cfg.DCCL.CAND_TAU), float(cfg.DCCL.EPSILON))
        return ((candidate_mass - cfg.DCCL.CAND_TAU) / denom).clamp(0.0, 1.0)
    raise ValueError(f"Unknown DCCL.CAND_WEIGHT: {cfg.DCCL.CAND_WEIGHT}")


def build_conflict_kl_target(cfg, clip_soft, source_label, clip_label, model_soft):
    if cfg.DCCL.KL_MODE == "clip":
        return clip_soft, torch.ones(source_label.size(0), dtype=torch.float)

    conflict_mask = source_label != clip_label
    if cfg.DCCL.KL_MODE == "non_conflict":
        return clip_soft, (~conflict_mask).float()

    if cfg.DCCL.KL_MODE != "candidate":
        raise ValueError(f"Unknown DCCL.KL_MODE: {cfg.DCCL.KL_MODE}")

    sample_idx = torch.arange(source_label.size(0))
    candidate_target = torch.zeros_like(clip_soft)
    if cfg.DCCL.KL_CANDIDATE == "balanced":
        source_weight = torch.full_like(model_soft[sample_idx, source_label], 0.5)
        clip_weight = torch.full_like(source_weight, 0.5)
    elif cfg.DCCL.KL_CANDIDATE == "confidence":
        source_weight = model_soft[sample_idx, source_label]
        clip_weight = clip_soft[sample_idx, clip_label]
        norm = (source_weight + clip_weight).clamp_min(cfg.DCCL.EPSILON)
        source_weight = source_weight / norm
        clip_weight = clip_weight / norm
    else:
        raise ValueError(f"Unknown DCCL.KL_CANDIDATE: {cfg.DCCL.KL_CANDIDATE}")

    candidate_target[sample_idx, source_label] = source_weight
    candidate_target[sample_idx, clip_label] += clip_weight
    kl_target = torch.where(conflict_mask.unsqueeze(1), candidate_target, clip_soft)
    return kl_target, torch.ones(source_label.size(0), dtype=torch.float)


def update_accd_state(cfg, state, evidence, curr_cycle):
    """Promote only graph-supported conflict labels stable across cycles."""
    eligible = evidence["eligible"] & (curr_cycle >= cfg.ACCD.START_CYCLE)
    (
        state["accd_pending_label"],
        state["accd_pending_count"],
        state["accd_resolved_label"],
        newly_resolved,
        resolved_mask,
        demoted,
    ) = update_temporal_resolution(
        state["accd_pending_label"],
        state["accd_pending_count"],
        state["accd_resolved_label"],
        eligible,
        evidence["graph_label"],
        cfg.ACCD.STABLE_CYCLES,
        cfg.ACCD.RESOLUTION_MEMORY,
    )
    return newly_resolved, resolved_mask, demoted


def train_target(cfg):
    clip_model, preprocess, _ = clip.load(cfg.ACTIVE.ARCH)
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

    iter_sample = iter(dset_loaders["target"])
    inputs_sample, _, _ = next(iter_sample)
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
    # max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"]) * cfg.ACTIVE.CYCLE
    interval_iter = max_iter // cfg.TEST.INTERVAL

    prev_label_mask = None
    conflict_state = None
    text_features = None
    curr_cycle = 0
    # office-home : 1.0 / VisDA-C : 1.05
    q_value = cfg.ACTIVE.Q_VALUE
    print(f"train_clip")
    while curr_cycle < cfg.ACTIVE.CYCLE:
        iter_num = 0

        netF.eval()
        netB.eval()
        # netC.eval()
        (
            mem_label,
            label_mask,
            confi_imag,
            confi_dis,
            clip_soft,
            source_label,
            clip_label,
            model_soft,
            task_features,
            clip_features,
            target_label,
        ) = obtain_label(
            cfg,
            dset_loaders['test_aug'], netF, netB, netC, text_inputs, text_features, clip_model, prev_label_mask,
            curr_cycle,
        )
        if conflict_state is None:
            conflict_state = init_conflict_state(source_label.size(0))
        if not cfg.ACCD.ENABLED:
            update_conflict_state(cfg, conflict_state, source_label, clip_label, model_soft)
        sample_idx = torch.arange(source_label.size(0))
        candidate_mass = model_soft[sample_idx, source_label] + model_soft[sample_idx, clip_label]
        candidate_weight = get_candidate_weight(cfg, candidate_mass)
        kl_target, kl_weight = build_conflict_kl_target(cfg, clip_soft, source_label, clip_label, model_soft)

        accd_resolved_mask = torch.zeros_like(label_mask)
        if cfg.ACCD.ENABLED:
            if cfg.ACCD.ANCHOR_MEMORY == "dynamic":
                fixed_anchor_mask = None
                fixed_anchor_label = None
            elif cfg.ACCD.ANCHOR_MEMORY == "frozen_initial":
                fixed_anchor_mask = conflict_state["accd_anchor_label"] >= 0
                if fixed_anchor_mask.any():
                    fixed_anchor_label = conflict_state["accd_anchor_label"]
                else:
                    fixed_anchor_mask = None
                    fixed_anchor_label = None
            else:
                raise ValueError(f"Unknown ACCD.ANCHOR_MEMORY: {cfg.ACCD.ANCHOR_MEMORY}")

            task_graph, clip_graph, graph_posterior, anchor_mask = dual_space_diffusion(
                task_features,
                clip_features,
                model_soft,
                clip_soft,
                source_label,
                clip_label,
                anchor_ratio=cfg.ACCD.ANCHOR_RATIO,
                anchor_min_per_class=cfg.ACCD.ANCHOR_MIN_PER_CLASS,
                k=cfg.ACCD.GRAPH_K,
                temperature=cfg.ACCD.TEMPERATURE,
                alpha=cfg.ACCD.ALPHA,
                steps=cfg.ACCD.STEPS,
                chunk_size=cfg.ACCD.CHUNK_SIZE,
                anchor_mask=fixed_anchor_mask,
                anchor_label=fixed_anchor_label,
            )
            if cfg.ACCD.ANCHOR_MEMORY == "frozen_initial" and fixed_anchor_mask is None:
                conflict_state["accd_anchor_label"][anchor_mask] = source_label[anchor_mask]
            evidence = conflict_diffusion_evidence(
                task_graph,
                clip_graph,
                graph_posterior,
                source_label,
                clip_label,
                candidate_mass_threshold=cfg.ACCD.CANDIDATE_MASS,
                candidate_margin_threshold=cfg.ACCD.CANDIDATE_MARGIN,
            )
            resolution_evidence = dict(evidence)
            if cfg.ACCD.RESOLUTION_TARGET == "source_only":
                resolution_evidence["eligible"] = (
                    evidence["eligible"] & (evidence["graph_label"] == source_label)
                )
            elif cfg.ACCD.RESOLUTION_TARGET != "both":
                raise ValueError(f"Unknown ACCD.RESOLUTION_TARGET: {cfg.ACCD.RESOLUTION_TARGET}")
            newly_resolved, accd_resolved_mask, demoted = update_accd_state(
                cfg, conflict_state, resolution_evidence, curr_cycle
            )
            kl_target[accd_resolved_mask] = graph_posterior[accd_resolved_mask]

            eligible = evidence["eligible"]
            resolved_correct = (
                conflict_state["accd_resolved_label"][accd_resolved_mask]
                == target_label[accd_resolved_mask]
            )
            eligible_correct = evidence["graph_label"][eligible] == target_label[eligible]
            eligible_clip_correct = clip_label[eligible] == target_label[eligible]
            resolution_eligible = resolution_evidence["eligible"]
            resolution_correct = (
                evidence["graph_label"][resolution_eligible]
                == target_label[resolution_eligible]
            )
            resolution_clip_correct = (
                clip_label[resolution_eligible] == target_label[resolution_eligible]
            )
            resolved_to_source = eligible & (evidence["graph_label"] == source_label)
            resolved_to_clip = eligible & (evidence["graph_label"] == clip_label)
            eligible_net_gain = int(eligible_correct.sum().item() - eligible_clip_correct.sum().item())
            resolution_net_gain = int(
                resolution_correct.sum().item() - resolution_clip_correct.sum().item()
            )
            logging.info(
                "ACCD cycle: anchors={}; conflicts={}; cross_space={}; eligible={}; "
                "outside_candidate={}; newly_resolved={}; demoted={}; resolved_active={}; "
                "resolution_eligible={}; anchor_memory={}; resolution_memory={}; "
                "resolution_target={}".format(
                    int(anchor_mask.sum().item()),
                    int(evidence["conflict"].sum().item()),
                    int((evidence["conflict"] & evidence["cross_space_agreement"]).sum().item()),
                    int(eligible.sum().item()),
                    int(evidence["outside_candidate"].sum().item()),
                    int(newly_resolved.sum().item()),
                    int(demoted.sum().item()),
                    int(accd_resolved_mask.sum().item()),
                    int(resolution_evidence["eligible"].sum().item()),
                    cfg.ACCD.ANCHOR_MEMORY,
                    cfg.ACCD.RESOLUTION_MEMORY,
                    cfg.ACCD.RESOLUTION_TARGET,
                )
            )
            logging.info(
                "ACCD oracle diagnostics only: eligible_accuracy={:.2f}%; clip_accuracy={:.2f}%; "
                "net_gain={}; to_source={}; to_clip={}; resolution_accuracy={:.2f}%; "
                "resolution_clip_accuracy={:.2f}%; resolution_net_gain={}; "
                "resolved_accuracy={:.2f}%".format(
                    float(eligible_correct.float().mean().item() * 100.0) if eligible.any() else 0.0,
                    float(eligible_clip_correct.float().mean().item() * 100.0) if eligible.any() else 0.0,
                    eligible_net_gain,
                    int(resolved_to_source.sum().item()),
                    int(resolved_to_clip.sum().item()),
                    float(resolution_correct.float().mean().item() * 100.0)
                    if resolution_eligible.any() else 0.0,
                    float(resolution_clip_correct.float().mean().item() * 100.0)
                    if resolution_eligible.any() else 0.0,
                    resolution_net_gain,
                    float(resolved_correct.float().mean().item() * 100.0) if accd_resolved_mask.any() else 0.0,
                )
            )

        promoted_mask = conflict_state["promoted_label"] >= 0
        hard_label = mem_label.clone()
        hard_label[promoted_mask] = conflict_state["promoted_label"][promoted_mask]
        hard_label[accd_resolved_mask] = conflict_state["accd_resolved_label"][accd_resolved_mask]
        hard_mask = label_mask | promoted_mask | accd_resolved_mask
        candidate_mask = (
            (source_label != clip_label)
            & (~promoted_mask)
            & (~accd_resolved_mask)
            & (~conflict_state["rejected"])
            & (candidate_mass >= cfg.DCCL.CAND_TAU)
            & (curr_cycle >= cfg.DCCL.CAND_START_CYCLE)
        )
        logging.info(
            "DCCL candidate gate: start_cycle={}; tau={:.3f}; weight={}; selected={}/{}".format(
                int(cfg.DCCL.CAND_START_CYCLE),
                float(cfg.DCCL.CAND_TAU),
                cfg.DCCL.CAND_WEIGHT,
                int(candidate_mask.sum().item()),
                int((source_label != clip_label).sum().item()),
            )
        )

        clip_soft = clip_soft.cuda()
        kl_target = kl_target.cuda()
        kl_weight = kl_weight.cuda()
        mem_label = hard_label.cuda()
        source_label = source_label.cuda()
        clip_label = clip_label.cuda()
        candidate_weight = candidate_weight.cuda()
        prev_label_mask = label_mask

        # clip_optimizer = train_clip_lr(cfg, clip_model, confi_imag, confi_dis, text_inputs, clip_optimizer, curr_cycle)
        clip_optimizer, q_value = train_clip(cfg, clip_model, confi_imag, confi_dis, text_inputs, clip_optimizer,
                                             q_value)

        cfg.load = 'prompt_model.pt'
        # mem_label = torch.from_numpy(mem_label).cuda()
        netF.train()
        netB.train()
        # netC.train()
        while iter_num < max_iter:
            try:
                inputs_test, _, tar_idx = next(iter_test)
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = next(iter_test)

            if inputs_test[0].size(0) == 1:
                continue

            weak_x = inputs_test[1].cuda()
            strong_x = inputs_test[2].cuda()

            iter_num += 1
            optimizer = cosine_scheduler(cfg, optimizer, iter_num=iter_num, max_iter=max_iter)

            weak_feas = netB(netF(weak_x))
            strong_feas = netB(netF(strong_x))

            weak_logits = netC(weak_feas)
            strong_logits = netC(strong_feas)

            # batch_cos = cal_cosine(weak_feas, strong_feas)
            # weak_logits = weak_logits * batch_cos

            weak_preds = nn.Softmax(dim=1)(weak_logits)

            filtered_idx = tar_idx[hard_mask[tar_idx]]

            con_loss = consistency_loss(weak_logits, strong_logits)
            classifier_loss = con_loss * cfg.ACTIVE.CON_PAR
            # classifier_loss = metric_loss * cfg.ACTIVE.CLS_PAR
            if cfg.ACTIVE.CLS_PAR > 0:
                pred = mem_label[filtered_idx]
                supervised_logits = weak_logits[hard_mask[tar_idx]]
                if pred.size(0) != 0:
                    classifier_loss += nn.CrossEntropyLoss()(supervised_logits, pred) * cfg.ACTIVE.CLS_PAR
            batch_candidate_mask = candidate_mask[tar_idx]
            if cfg.DCCL.CAND_PAR > 0 and batch_candidate_mask.any():
                candidate_positions = torch.nonzero(batch_candidate_mask, as_tuple=False).squeeze(1).cuda()
                candidate_indices = tar_idx[batch_candidate_mask].cuda()
                source_candidates = source_label[candidate_indices]
                clip_candidates = clip_label[candidate_indices]
                candidate_prob = (
                    weak_preds[candidate_positions, source_candidates]
                    + weak_preds[candidate_positions, clip_candidates]
                ).clamp_min(cfg.DCCL.EPSILON)
                candidate_losses = -torch.log(candidate_prob)
                weights = candidate_weight[candidate_indices].clamp_min(cfg.DCCL.EPSILON)
                candidate_loss = (candidate_losses * weights).sum() / weights.sum()
                classifier_loss += candidate_loss * cfg.DCCL.CAND_PAR
            # pseudo_output = weak_preds[filtered_idx]
            kl_target_batch = kl_target[tar_idx]
            kl_weight_batch = kl_weight[tar_idx]
            # mixed_soft_batch = confi_dis[tar_idx].cuda()
            # mi_loss = F.kl_div(weak_preds.log(), mixed_soft_batch, reduction="batchmean")
            per_sample_kl = F.kl_div(weak_preds.log(), kl_target_batch, reduction="none").sum(dim=1)
            if kl_weight_batch.sum() > 0:
                mi_loss = (per_sample_kl * kl_weight_batch).sum() / kl_weight_batch.sum()
                classifier_loss += mi_loss * cfg.ACTIVE.KL_PAR

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netF.eval()
                netB.eval()
                # netC.eval()
                if cfg.SETTING.DATASET == 'VISDA-C':
                    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                    log_str = ('Task: {}, Iter:{}/{}; Cycle: {}/{}; '
                               'Accuracy = {:.2f}%; classifier_loss = {}').format(cfg.name, iter_num, max_iter,
                                                                                  curr_cycle + 1, cfg.ACTIVE.CYCLE,
                                                                                  acc_s_te,
                                                                                  classifier_loss) + '\n' + acc_list
                else:
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                    log_str = ('Task: {}, Iter:{}/{}; Cycle: {}/{}; '
                               'Accuracy = {:.2f}%; classifier_loss = {}').format(cfg.name, iter_num, max_iter,
                                                                                  curr_cycle + 1, cfg.ACTIVE.CYCLE,
                                                                                  acc_s_te, classifier_loss)

                # cfg.out_file.write(log_str + '\n')
                # cfg.out_file.flush()
                # print(log_str+'\n')
                logging.info(log_str)
                netF.train()
                netB.train()
                # netC.train()
        curr_cycle += 1

    # torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.MODEL.METHOD + ".pt"))
    # torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.MODEL.METHOD + ".pt"))
    # torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.MODEL.METHOD + ".pt"))

    # if cfg.ISSAVE:
    #     torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.SHOT.CLS_PAR + ".pt"))
    #     torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.SHOT.CLS_PAR + ".pt"))
    #     torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.SHOT.CLS_PAR + ".pt"))

    return netF, netB, netC


def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def cal_cosine(weak_feas, strong_feas):
    normalized_weak = F.normalize(weak_feas, p=2, dim=1)
    normalized_strong = F.normalize(strong_feas, p=2, dim=1)

    cos_sim = torch.sum(normalized_weak * normalized_strong, dim=1)
    mean_cos = cos_sim.mean()
    return mean_cos


def expand_pseudo_label_mask(cfg, label_mask, all_mix_output, all_mix_output_pred):
    if cfg.DCCL.PL_EXPAND == "none":
        return label_mask
    if cfg.DCCL.PL_EXPAND != "balanced_topk":
        raise ValueError(f"Unknown DCCL.PL_EXPAND: {cfg.DCCL.PL_EXPAND}")
    if cfg.DCCL.PL_TOPK_PER_CLASS <= 0:
        return label_mask

    expanded_mask = label_mask.clone()
    mix_conf, _ = torch.max(all_mix_output, dim=1)
    num_classes = all_mix_output.size(1)
    for class_idx in range(num_classes):
        class_mask = all_mix_output_pred == class_idx
        class_candidates = torch.nonzero(class_mask & (mix_conf >= cfg.DCCL.PL_MIN_CONF), as_tuple=False).squeeze(1)
        if class_candidates.numel() == 0:
            continue
        current_count = int((expanded_mask & class_mask).sum().item())
        need = int(cfg.DCCL.PL_TOPK_PER_CLASS) - current_count
        if need <= 0:
            continue
        topk = min(need, class_candidates.numel())
        class_conf = mix_conf[class_candidates]
        selected = class_candidates[torch.topk(class_conf, k=topk).indices]
        expanded_mask[selected] = True

    logging.info(
        "DCCL pseudo-label expansion: mode={}; topk_per_class={}; min_conf={:.3f}; selected={}->{}".format(
            cfg.DCCL.PL_EXPAND,
            int(cfg.DCCL.PL_TOPK_PER_CLASS),
            float(cfg.DCCL.PL_MIN_CONF),
            int(label_mask.sum().item()),
            int(expanded_mask.sum().item()),
        )
    )
    return expanded_mask


def prior_calibrate(prob, power, eps):
    prior = prob.mean(dim=0).clamp_min(eps)
    calibrated = prob / prior.pow(power)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(eps)


def apply_classwise_calibration_mode(cfg, source_prob, clip_prob, mode):
    if mode == "none":
        mix_prob = (source_prob + clip_prob) / 2
        return source_prob, clip_prob, mix_prob

    source_cal = source_prob
    clip_cal = clip_prob
    if mode in {"source_prior", "both_prior"}:
        source_cal = prior_calibrate(source_cal, cfg.DCCL.CALIB_POWER, cfg.DCCL.EPSILON)
    if mode in {"clip_prior", "both_prior"}:
        clip_cal = prior_calibrate(clip_cal, cfg.DCCL.CALIB_POWER, cfg.DCCL.EPSILON)

    mix_prob = (source_cal + clip_cal) / 2
    if mode == "mix_prior":
        mix_prob = prior_calibrate(mix_prob, cfg.DCCL.CALIB_POWER, cfg.DCCL.EPSILON)
    elif mode not in {"source_prior", "clip_prior", "both_prior"}:
        raise ValueError(f"Unknown DCCL.CALIB_MODE: {mode}")

    return source_cal, clip_cal, mix_prob


def score_calibration_candidate(cfg, source_prob, clip_prob, mix_prob):
    _, source_pred = torch.max(source_prob, dim=1)
    _, clip_pred = torch.max(clip_prob, dim=1)
    agreement = source_pred == clip_pred
    coverage = agreement.float().mean()
    mix_conf, _ = torch.max(mix_prob, dim=1)
    if agreement.any():
        agreement_conf = mix_conf[agreement].mean()
    else:
        agreement_conf = torch.tensor(0.0)
    score = coverage + float(cfg.DCCL.CALIB_AUTO_LAMBDA) * agreement_conf
    return float(score.item()), float(coverage.item()), float(agreement_conf.item())


def apply_classwise_calibration(cfg, source_prob, clip_prob):
    mode = cfg.DCCL.CALIB_MODE
    selected_mode = mode
    if mode == "auto_agree":
        candidate_modes = ["none", "source_prior", "clip_prior", "both_prior", "mix_prior"]
        scored = []
        for candidate_mode in candidate_modes:
            candidate_source, candidate_clip, candidate_mix = apply_classwise_calibration_mode(
                cfg, source_prob, clip_prob, candidate_mode
            )
            score, coverage, agreement_conf = score_calibration_candidate(
                cfg, candidate_source, candidate_clip, candidate_mix
            )
            scored.append((score, coverage, agreement_conf, candidate_mode))
        scored.sort(reverse=True)
        selected_mode = scored[0][3]
        logging.info(
            "DCCL auto calibration scores: {}".format(
                "; ".join(
                    "{} score={:.4f} cov={:.4f} conf={:.4f}".format(name, score, coverage, conf)
                    for score, coverage, conf, name in scored
                )
            )
        )

    source_cal, clip_cal, mix_prob = apply_classwise_calibration_mode(cfg, source_prob, clip_prob, selected_mode)

    logging.info(
        "DCCL classwise calibration: requested_mode={}; selected_mode={}; power={:.3f}; source_prior_range=({:.4f},{:.4f}); clip_prior_range=({:.4f},{:.4f})".format(
            mode,
            selected_mode,
            float(cfg.DCCL.CALIB_POWER),
            float(source_prob.mean(dim=0).min().item()),
            float(source_prob.mean(dim=0).max().item()),
            float(clip_prob.mean(dim=0).min().item()),
            float(clip_prob.mean(dim=0).max().item()),
        )
    )
    return source_cal, clip_cal, mix_prob


def obtain_label(cfg, loader, netF, netB, netC, text_inputs, text_features, clip_model, prev_label_mask, curr_cycle):
    # class_logit_bias = get_class_bias(netF, netB, netC)
    start_test = True
    with torch.no_grad():
        if text_features is None:
            current_text_features = clip_model.encode_text(text_inputs)
        else:
            current_text_features = text_features
        current_text_features = F.normalize(current_text_features, dim=1)
        clip_logit_scale = clip_model.logit_scale.exp()
        iter_test = iter(loader)
        for _ in range(len(loader)):
            inputs_test, labels, _ = next(iter_test)
            weak_x = inputs_test[1].cuda()
            # strong_x = inputs_test[2].cuda()

            weak_feas = netB(netF(weak_x))
            # strong_feas = netB(netF(strong_x))

            weak_outputs = netC(weak_feas)
            # strong_outputs = netC(strong_feas)

            clip_image_features = F.normalize(clip_model.encode_image(weak_x), dim=1)
            clip_score = clip_logit_scale * clip_image_features @ current_text_features.t()

            clip_score = clip_score.cpu()
            if start_test:
                all_output = weak_outputs.float().cpu()
                all_clip_score = clip_score.float().cpu()
                all_task_features = weak_feas.float().cpu()
                all_clip_features = clip_image_features.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, weak_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                all_task_features = torch.cat((all_task_features, weak_feas.float().cpu()), 0)
                all_clip_features = torch.cat((all_clip_features, clip_image_features.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
    all_output, clip_all_output, all_mix_output = apply_classwise_calibration(cfg, all_output, clip_all_output)

    # Compute predictions for all_output and clip_all_output
    _, all_output_pred = torch.max(all_output, dim=1)
    _, clip_all_output_pred = torch.max(clip_all_output, dim=1)

    _, all_mix_output_pred = torch.max(all_mix_output, dim=1)

    # Find indices where predictions match
    matching_indices = all_output_pred == clip_all_output_pred

    # Update label mask based on previous label mask
    if prev_label_mask is not None:
        label_mask = prev_label_mask | (~prev_label_mask & matching_indices)
    else:
        label_mask = matching_indices
    label_mask = expand_pseudo_label_mask(cfg, label_mask, all_mix_output, all_mix_output_pred)

    # Filter predictions and labels based on the updated label mask
    valid_preds = all_output_pred[label_mask]
    valid_labels = all_label[label_mask]

    # Calculate pseudo label accuracy
    if len(valid_preds) > 0:
        pseudo_label_accuracy = torch.sum(valid_preds == valid_labels).item() / float(len(valid_preds))
        # plot_confusion_matrix(valid_labels, valid_preds, curr_cycle)
        # breakpoint()
    else:
        pseudo_label_accuracy = 0.0

    # Print accuracy and number of valid samples
    log_str = "Number of valid pseudo-labeled samples: {}/{}; Accuracy = {:.2f}%".format(
        len(valid_preds), len(all_output_pred), pseudo_label_accuracy * 100
    )
    logging.info(log_str)
    valid_mixed = all_mix_output_pred[label_mask]
    mixed_output_accuracy = torch.sum(valid_mixed == valid_labels).item() / float(len(valid_preds))
    log_str_valid = "Mixed output with valid mask: {:.2f}%".format(mixed_output_accuracy * 100)
    logging.info(log_str_valid)

    # _, all_mix_output_pred = torch.max(all_mix_output, dim=1)
    mix_output_accuracy = torch.sum(all_mix_output_pred == all_label).item() / float(len(all_label))
    clip_output_accuracy = torch.sum(clip_all_output_pred == all_label).item() / float(len(all_label))
    pure_output_accuracy = torch.sum(all_output_pred == all_label).item() / float(len(all_label))

    log_str_mix = ("all_mix_output Accuracy = {:.2f}%; clip_output_accuracy = {:.2f}%; "
                   "pure_output_accuracy = {:.2f}%;").format(mix_output_accuracy * 100,
                                                             clip_output_accuracy * 100, pure_output_accuracy * 100)
    logging.info(log_str_mix)

    confi_imag = loader.dataset.imgs
    confi_dis = all_mix_output.detach()

    return (
        all_mix_output_pred,
        label_mask,
        confi_imag,
        confi_dis,
        clip_all_output,
        all_output_pred,
        clip_all_output_pred,
        all_output,
        all_task_features,
        all_clip_features,
        all_label.long(),
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


def clip_text(model, text_features, inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits
