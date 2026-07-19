"""
Builds upon: https://github.com/tim-learn/SHOT
Corresponding paper: http://proceedings.mlr.press/v119/liang20a/liang20a.pdf
"""

import os
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
    adaptive_graph_teacher_fusion,
    class_balanced_mask_by_prior,
    conflict_diffusion_evidence,
    dual_space_diffusion,
    graph_temporal_residual_weights,
    topology_prior_calibrate,
    topology_target_prior_calibrate,
    transport_candidate_mass,
    update_temporal_resolution,
)
from src.utils.model_ema import update_model_ema
from src.utils.target_head import bounded_residual_logits
from src.utils.trajectory_ensemble import (
    capture_trajectory_snapshot,
    load_trajectory_snapshot,
)
from src.utils.class_pair_flow import (
    ClassPairFlowAdapter,
    update_class_pair_flow,
    update_soft_class_pair_flow,
)
from src.utils.class_pair_feature_adapter import (
    ClassPairFeatureAdapter,
    weighted_graph_temporal_kl,
)
from src.utils.agreement_covariance_transport import AgreementCovarianceTransport
from src.utils.agreement_whitened_transport import AgreementWhitenedTransport
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
        param_group['weight_decay'] = (
            cfg.OPTIM.WD * param_group.get('weight_decay_scale', 1.0)
        )
        param_group['momentum'] = cfg.OPTIM.MOMENTUM
        param_group['nesterov'] = cfg.OPTIM.NESTEROV
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


def apply_target_prototype_logits(cfg, features, logits, proto_state):
    if not cfg.DCCL.PROTO_ADAPT or proto_state is None:
        return logits
    prototypes = proto_state.get("prototypes")
    proto_mask = proto_state.get("mask")
    if prototypes is None or proto_mask is None or not proto_mask.any():
        return logits
    if cfg.DCCL.PROTO_MIX <= 0:
        return logits
    if cfg.DCCL.PROTO_TEMPERATURE <= 0:
        raise ValueError("DCCL.PROTO_TEMPERATURE must be positive")

    prototypes = prototypes.to(device=features.device, dtype=features.dtype)
    proto_mask = proto_mask.to(device=features.device)
    proto_logits = F.normalize(features.float(), dim=1) @ prototypes.t()
    proto_logits = proto_logits / float(cfg.DCCL.PROTO_TEMPERATURE)
    active_proto = proto_logits[:, proto_mask]
    if active_proto.numel() == 0:
        return logits

    proto_center = active_proto.mean(dim=1, keepdim=True)
    proto_scale = active_proto.std(dim=1, keepdim=True, unbiased=False).clamp_min(cfg.DCCL.EPSILON)
    source_scale = logits.detach().float().std(dim=1, keepdim=True, unbiased=False).clamp_min(cfg.DCCL.EPSILON)
    proto_delta = torch.zeros_like(logits.float())
    proto_delta[:, proto_mask] = (
        (active_proto - proto_center) / proto_scale * source_scale
    ).to(proto_delta.dtype)
    return logits + float(cfg.DCCL.PROTO_MIX) * proto_delta.to(logits.dtype)


class SourceAnchoredResidualClassifier(nn.Module):
    def __init__(self, cfg, source_head):
        super().__init__()
        self.type = source_head.type
        self.max_gate = float(cfg.DCCL.TARGET_RESIDUAL_MAX_GATE)
        self.epsilon = float(cfg.DCCL.EPSILON)
        self.residual = network.feat_classifier(
            type=source_head.type,
            class_num=cfg.class_num,
            bottleneck_dim=cfg.bottleneck,
        ).cuda()
        residual_device = next(self.residual.parameters()).device
        self.gate_logit = nn.Parameter(torch.tensor(
            float(cfg.DCCL.TARGET_RESIDUAL_GATE_INIT),
            device=residual_device,
        ))
        with torch.no_grad():
            if hasattr(self.residual.fc, "weight_g"):
                self.residual.fc.weight_g.zero_()
            else:
                self.residual.fc.weight.zero_()
            if self.residual.fc.bias is not None:
                self.residual.fc.bias.zero_()

    def effective_gate(self):
        return self.max_gate * torch.sigmoid(self.gate_logit.detach())

    def forward(self, features, source_logits):
        residual_logits = self.residual(features)
        return bounded_residual_logits(
            source_logits,
            residual_logits,
            self.gate_logit,
            self.max_gate,
            self.epsilon,
        )


def apply_target_head_logits(cfg, features, source_logits, target_head, curr_cycle):
    if (
        not cfg.DCCL.TARGET_HEAD_ADAPT
        or target_head is None
        or curr_cycle < cfg.DCCL.TARGET_HEAD_START_CYCLE
    ):
        return source_logits
    if cfg.DCCL.TARGET_HEAD_VARIANT in {"residual", "pair_flow"}:
        return target_head(features, source_logits)
    if cfg.DCCL.TARGET_HEAD_VARIANT != "blend":
        raise ValueError(
            "DCCL.TARGET_HEAD_VARIANT must be blend, residual, or pair_flow"
        )
    if not 0.0 <= cfg.DCCL.TARGET_HEAD_MIX <= 1.0:
        raise ValueError("DCCL.TARGET_HEAD_MIX must be in [0, 1]")
    target_logits = target_head(features)
    mix = float(cfg.DCCL.TARGET_HEAD_MIX)
    return (1.0 - mix) * source_logits + mix * target_logits


def apply_pair_feature_adapter(cfg, features, adapter, curr_cycle):
    if (
        not cfg.DCCL.PAIR_FEATURE_ADAPT
        or adapter is None
        or curr_cycle < cfg.DCCL.PAIR_FEATURE_START_CYCLE
    ):
        return features
    gradient_mode = cfg.DCCL.PAIR_FEATURE_GRADIENT_MODE
    if gradient_mode not in {"joint", "gtr_only"}:
        raise ValueError(
            "DCCL.PAIR_FEATURE_GRADIENT_MODE must be joint or gtr_only"
        )
    return adapter(
        features,
        detach_delta=gradient_mode == "gtr_only" and torch.is_grad_enabled(),
    )


def apply_agreement_covariance_transport(
    cfg, features, transport, curr_cycle, sample_indices
):
    if (
        not cfg.DCCL.COV_TRANSPORT_ADAPT
        or transport is None
        or curr_cycle < cfg.DCCL.COV_TRANSPORT_START_CYCLE
    ):
        return features
    if sample_indices is None:
        raise ValueError("Agreement covariance transport requires sample indices")
    return transport(features, sample_indices)


def build_target_classifier_head(cfg, source_head):
    if cfg.DCCL.TARGET_HEAD_VARIANT == "residual":
        return SourceAnchoredResidualClassifier(cfg, source_head)
    if cfg.DCCL.TARGET_HEAD_VARIANT == "pair_flow":
        adapter = ClassPairFlowAdapter(
            feature_dim=cfg.bottleneck,
            num_classes=cfg.class_num,
            rank=int(cfg.DCCL.PAIR_FLOW_RANK),
            max_gate=float(cfg.DCCL.PAIR_FLOW_MAX_GATE),
            gate_init=float(cfg.DCCL.PAIR_FLOW_GATE_INIT),
            epsilon=float(cfg.DCCL.EPSILON),
        ).cuda()
        adapter.type = source_head.type
        return adapter
    if cfg.DCCL.TARGET_HEAD_VARIANT != "blend":
        raise ValueError(
            "DCCL.TARGET_HEAD_VARIANT must be blend, residual, or pair_flow"
        )
    target_head = network.feat_classifier(
        type=source_head.type,
        class_num=cfg.class_num,
        bottleneck_dim=cfg.bottleneck,
    ).cuda()
    target_head.load_state_dict(source_head.state_dict())
    return target_head


def build_target_head_ema(cfg, target_head):
    if cfg.DCCL.TARGET_HEAD_VARIANT != "blend":
        raise ValueError("Target-head EMA currently supports only the blend variant")
    ema_head = build_target_classifier_head(cfg, target_head)
    ema_head.eval()
    for parameter in ema_head.parameters():
        parameter.requires_grad = False
    return ema_head


@torch.no_grad()
def update_target_prototype_state(cfg, features, labels, mask, proto_state):
    if not cfg.DCCL.PROTO_ADAPT:
        return proto_state
    if cfg.DCCL.PROTO_MIN_PER_CLASS <= 0:
        raise ValueError("DCCL.PROTO_MIN_PER_CLASS must be positive")
    if not 0.0 <= cfg.DCCL.PROTO_MOMENTUM < 1.0:
        raise ValueError("DCCL.PROTO_MOMENTUM must be in [0, 1)")

    features = F.normalize(features.float(), dim=1).cpu()
    labels = labels.long().cpu()
    mask = mask.bool().cpu()
    num_classes = cfg.class_num
    prototypes = torch.zeros(num_classes, features.size(1), dtype=torch.float)
    proto_mask = torch.zeros(num_classes, dtype=torch.bool)
    for class_idx in range(num_classes):
        class_rows = mask & (labels == class_idx)
        if int(class_rows.sum().item()) < int(cfg.DCCL.PROTO_MIN_PER_CLASS):
            continue
        proto = features[class_rows].mean(dim=0)
        prototypes[class_idx] = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
        proto_mask[class_idx] = True

    if proto_state is not None and proto_state.get("prototypes") is not None:
        prev_proto = proto_state["prototypes"].float().cpu()
        prev_mask = proto_state["mask"].bool().cpu()
        keep_prev = prev_mask & (~proto_mask)
        prototypes[keep_prev] = prev_proto[keep_prev]
        proto_mask = proto_mask | keep_prev
        update_mask = proto_mask & prev_mask & (~keep_prev)
        if update_mask.any() and cfg.DCCL.PROTO_MOMENTUM > 0:
            momentum = float(cfg.DCCL.PROTO_MOMENTUM)
            blended = momentum * prev_proto[update_mask] + (1.0 - momentum) * prototypes[update_mask]
            prototypes[update_mask] = F.normalize(blended, dim=1)

    logging.info(
        "DCCL target prototypes: active_classes={}/{}; min_per_class={}; mix={:.3f}; temp={:.3f}".format(
            int(proto_mask.sum().item()),
            int(num_classes),
            int(cfg.DCCL.PROTO_MIN_PER_CLASS),
            float(cfg.DCCL.PROTO_MIX),
            float(cfg.DCCL.PROTO_TEMPERATURE),
        )
    )
    return {"prototypes": prototypes, "mask": proto_mask}


def cal_acc(
    loader,
    netF,
    netB,
    netC,
    cfg=None,
    proto_state=None,
    target_head=None,
    curr_cycle=0,
    pair_feature_adapter=None,
    covariance_transport=None,
    flag=False,
):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            adapted_feas = apply_pair_feature_adapter(
                cfg, feas, pair_feature_adapter, curr_cycle
            ) if cfg is not None else feas
            if cfg is not None:
                adapted_feas = apply_agreement_covariance_transport(
                    cfg,
                    adapted_feas,
                    covariance_transport,
                    curr_cycle,
                    data[2],
                )
            outputs = netC(adapted_feas)
            if cfg is not None:
                outputs = apply_target_head_logits(
                    cfg, adapted_feas, outputs, target_head, curr_cycle
                )
                outputs = apply_target_prototype_logits(
                    cfg, adapted_feas, outputs, proto_state
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


def cal_acc_trajectory_ensemble(
    loader,
    netF,
    netB,
    netC,
    cfg,
    proto_state,
    target_head,
    curr_cycle,
    snapshots,
):
    if len(snapshots) < 2:
        raise ValueError("Trajectory ensemble requires at least two snapshots")

    ensemble_output = None
    all_label = None
    for snapshot in snapshots:
        load_trajectory_snapshot(snapshot, netF, netB, target_head)
        member_outputs = []
        member_labels = []
        with torch.no_grad():
            for data in loader:
                inputs = data[0].cuda()
                labels = data[1]
                features = netB(netF(inputs))
                outputs = netC(features)
                outputs = apply_target_head_logits(
                    cfg, features, outputs, target_head, curr_cycle
                )
                outputs = apply_target_prototype_logits(
                    cfg, features, outputs, proto_state
                )
                member_outputs.append(outputs.float().cpu())
                member_labels.append(labels.float())
        member_output = torch.cat(member_outputs, dim=0)
        member_label = torch.cat(member_labels, dim=0)
        if ensemble_output is None:
            ensemble_output = member_output
            all_label = member_label
        else:
            if not torch.equal(all_label, member_label):
                raise ValueError("Trajectory ensemble loader order changed")
            ensemble_output += member_output

    ensemble_output /= float(len(snapshots))
    load_trajectory_snapshot(snapshots[-1], netF, netB, target_head)
    predict = ensemble_output.argmax(dim=1)
    accuracy = (predict.float() == all_label).float().mean().item() * 100.0
    mean_ent = torch.mean(
        loss.Entropy(nn.Softmax(dim=1)(ensemble_output))
    ).cpu().item()
    return accuracy, mean_ent


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
        "gtr_pending_label": torch.full((num_samples,), -1, dtype=torch.long),
        "gtr_pending_count": torch.zeros(num_samples, dtype=torch.long),
        "gtr_stable_label": torch.full((num_samples,), -1, dtype=torch.long),
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
):
    if not cfg.DCCL.TEMPORAL_DIAG:
        return

    out_dir = osp.join(cfg.output_dir, cfg.DCCL.TEMPORAL_DIAG_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"{cfg.name}_cycle{curr_cycle + 1:02d}.npz")
    np.savez_compressed(
        out_path,
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
    logging.info("DCCL temporal diagnostics wrote: {}".format(out_path))


def build_graph_fused_teacher(cfg, task_features, clip_features, model_soft, clip_soft, source_label, clip_label):
    if not cfg.DCCL.GRAPH_TEACHER_FUSION:
        teacher_soft = (model_soft + clip_soft) / 2
        return teacher_soft, None, None, None
    if cfg.DCCL.GTF_APPLY_TO not in {"both", "clip", "kl", "none"}:
        raise ValueError(f"Unknown DCCL.GTF_APPLY_TO: {cfg.DCCL.GTF_APPLY_TO}")

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
        "apply_to={}; mean_graph_weight={:.4f}; max_graph_weight={:.4f}; "
        "changed_top1={}".format(
            int(anchors.sum().item()),
            float(cfg.DCCL.GTF_STRENGTH),
            cfg.DCCL.GTF_APPLY_TO,
            float(graph_weight.mean().item()),
            float(graph_weight.max().item()),
            int((base_teacher.argmax(dim=1) != teacher_soft.argmax(dim=1)).sum().item()),
        )
    )
    return teacher_soft, graph_post, graph_weight, anchors


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
    target_head = None
    target_head_ema = None
    pair_feature_adapter = None
    covariance_transport = None

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
    if cfg.DCCL.TARGET_HEAD_ADAPT:
        if cfg.DCCL.TARGET_HEAD_LR_MULT <= 0:
            raise ValueError("DCCL.TARGET_HEAD_LR_MULT must be positive")
        target_head = build_target_classifier_head(cfg, netC)
        target_head.train()
        for k, v in target_head.named_parameters():
            v.requires_grad = True
            group = {
                'params': v,
                'lr': cfg.OPTIM.LR * cfg.DCCL.TARGET_HEAD_LR_MULT,
            }
            if k == "gate_logit":
                group['weight_decay_scale'] = 0.0
            param_group.append(group)
        logging.info(
            "DCCL target head enabled: variant={}; mix={:.3f}; "
            "start_cycle={}; lr_mult={:.3f}".format(
                cfg.DCCL.TARGET_HEAD_VARIANT,
                float(cfg.DCCL.TARGET_HEAD_MIX),
                int(cfg.DCCL.TARGET_HEAD_START_CYCLE),
                float(cfg.DCCL.TARGET_HEAD_LR_MULT),
            )
        )
        if cfg.DCCL.TARGET_HEAD_EMA:
            if not 0.0 <= cfg.DCCL.TARGET_HEAD_EMA_MOMENTUM < 1.0:
                raise ValueError("DCCL.TARGET_HEAD_EMA_MOMENTUM must be in [0, 1)")
            target_head_ema = build_target_head_ema(cfg, target_head)
            logging.info(
                "DCCL target-head EMA enabled: momentum={:.4f}; "
                "teacher used for pseudo labels and evaluation".format(
                    float(cfg.DCCL.TARGET_HEAD_EMA_MOMENTUM)
                )
            )

    if cfg.DCCL.PAIR_FEATURE_ADAPT:
        if not cfg.DCCL.TARGET_HEAD_ADAPT or cfg.DCCL.TARGET_HEAD_VARIANT != "blend":
            raise ValueError(
                "Pair-feature adaptation requires the Stage14 blend target head"
            )
        if cfg.DCCL.PAIR_FEATURE_LR_MULT <= 0:
            raise ValueError("DCCL.PAIR_FEATURE_LR_MULT must be positive")
        if cfg.DCCL.PAIR_FEATURE_GRADIENT_MODE not in {"joint", "gtr_only"}:
            raise ValueError(
                "DCCL.PAIR_FEATURE_GRADIENT_MODE must be joint or gtr_only"
            )
        if (
            cfg.DCCL.PAIR_FEATURE_GRADIENT_MODE == "gtr_only"
            and cfg.DCCL.GTR_PAR <= 0
        ):
            raise ValueError(
                "gtr_only pair-feature training requires DCCL.GTR_PAR > 0"
            )
        pair_feature_adapter = ClassPairFeatureAdapter(
            feature_dim=cfg.bottleneck,
            rank=int(cfg.DCCL.PAIR_FLOW_RANK),
            min_active_rank=int(cfg.DCCL.PAIR_FEATURE_MIN_ACTIVE_RANK),
            max_gate=float(cfg.DCCL.PAIR_FEATURE_MAX_GATE),
            gate_init=float(cfg.DCCL.PAIR_FEATURE_GATE_INIT),
            epsilon=float(cfg.DCCL.EPSILON),
        ).cuda()
        pair_feature_adapter.train()
        for name, parameter in pair_feature_adapter.named_parameters():
            parameter.requires_grad = True
            group = {
                "params": parameter,
                "lr": cfg.OPTIM.LR * cfg.DCCL.PAIR_FEATURE_LR_MULT,
            }
            if (
                name == "gate_logit"
                or cfg.DCCL.PAIR_FEATURE_GRADIENT_MODE == "gtr_only"
            ):
                group["weight_decay_scale"] = 0.0
            param_group.append(group)
        logging.info(
            "DCCL pair-feature adapter enabled: rank={}; min_active_rank={}; max_gate={:.4f}; "
            "start_cycle={}; lr_mult={:.3f}; gradient_mode={}".format(
                int(cfg.DCCL.PAIR_FLOW_RANK),
                int(cfg.DCCL.PAIR_FEATURE_MIN_ACTIVE_RANK),
                float(cfg.DCCL.PAIR_FEATURE_MAX_GATE),
                int(cfg.DCCL.PAIR_FEATURE_START_CYCLE),
                float(cfg.DCCL.PAIR_FEATURE_LR_MULT),
                cfg.DCCL.PAIR_FEATURE_GRADIENT_MODE,
            )
        )

    if cfg.DCCL.COV_TRANSPORT_ADAPT:
        if not cfg.DCCL.TARGET_HEAD_ADAPT or cfg.DCCL.TARGET_HEAD_VARIANT != "blend":
            raise ValueError(
                "Agreement covariance transport requires the Stage14 blend head"
            )
        if pair_feature_adapter is not None:
            raise ValueError(
                "Agreement covariance transport cannot use the learned pair adapter"
            )
        if cfg.DCCL.COV_TRANSPORT_MODE == "conditional":
            covariance_transport = AgreementCovarianceTransport(
                num_classes=cfg.class_num,
                feature_dim=cfg.bottleneck,
                rank=int(cfg.DCCL.COV_TRANSPORT_RANK),
                min_anchors=int(cfg.DCCL.COV_TRANSPORT_MIN_ANCHORS),
                max_gate=float(cfg.DCCL.COV_TRANSPORT_MAX_GATE),
                epsilon=float(cfg.DCCL.EPSILON),
            ).cuda()
            logging.info(
                "DCCL agreement covariance transport enabled: mode=conditional; "
                "rank={}; min_anchors={}; max_gate={:.4f}; start_cycle={}".format(
                    int(cfg.DCCL.COV_TRANSPORT_RANK),
                    int(cfg.DCCL.COV_TRANSPORT_MIN_ANCHORS),
                    float(cfg.DCCL.COV_TRANSPORT_MAX_GATE),
                    int(cfg.DCCL.COV_TRANSPORT_START_CYCLE),
                )
            )
        elif cfg.DCCL.COV_TRANSPORT_MODE == "global_whitened":
            covariance_transport = AgreementWhitenedTransport(
                num_classes=cfg.class_num,
                feature_dim=cfg.bottleneck,
                min_anchors=int(cfg.DCCL.COV_GLOBAL_MIN_ANCHORS),
                shrinkage=float(cfg.DCCL.COV_GLOBAL_SHRINKAGE),
                holdout_ratio=float(cfg.DCCL.COV_GLOBAL_HOLDOUT_RATIO),
                max_gate=float(cfg.DCCL.COV_TRANSPORT_MAX_GATE),
                min_improvement=float(cfg.DCCL.COV_GLOBAL_MIN_IMPROVEMENT),
                epsilon=float(cfg.DCCL.EPSILON),
            ).cuda()
            logging.info(
                "DCCL agreement-whitened transport enabled: min_anchors={}; "
                "shrinkage={:.4f}; holdout_ratio={:.4f}; max_gate={:.4f}; "
                "min_improvement={:.6f}; start_cycle={}".format(
                    int(cfg.DCCL.COV_GLOBAL_MIN_ANCHORS),
                    float(cfg.DCCL.COV_GLOBAL_SHRINKAGE),
                    float(cfg.DCCL.COV_GLOBAL_HOLDOUT_RATIO),
                    float(cfg.DCCL.COV_TRANSPORT_MAX_GATE),
                    float(cfg.DCCL.COV_GLOBAL_MIN_IMPROVEMENT),
                    int(cfg.DCCL.COV_TRANSPORT_START_CYCLE),
                )
            )
        else:
            raise ValueError(
                "DCCL.COV_TRANSPORT_MODE must be conditional or global_whitened"
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
    # max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"]) * cfg.ACTIVE.CYCLE
    interval_iter = max_iter // cfg.TEST.INTERVAL

    trajectory_snapshots = []
    trajectory_intervals = [
        int(value) for value in cfg.DCCL.TRAJECTORY_SNAPSHOT_INTERVALS
    ]
    if cfg.DCCL.TRAJECTORY_ENSEMBLE:
        if target_head is None:
            raise ValueError("Trajectory ensemble requires target-head adaptation")
        if target_head_ema is not None:
            raise ValueError("Trajectory ensemble cannot be combined with target-head EMA")
        if pair_feature_adapter is not None:
            raise ValueError("Trajectory ensemble cannot be combined with pair-feature adaptation")
        if covariance_transport is not None:
            raise ValueError("Trajectory ensemble cannot use covariance transport")
        if len(trajectory_intervals) < 2:
            raise ValueError("Trajectory ensemble requires at least two intervals")
        if trajectory_intervals != sorted(set(trajectory_intervals)):
            raise ValueError("Trajectory snapshot intervals must be sorted and unique")
        if trajectory_intervals[0] < 1 or trajectory_intervals[-1] > cfg.TEST.INTERVAL:
            raise ValueError("Trajectory snapshot interval is outside TEST.INTERVAL")
        if trajectory_intervals[-1] != cfg.TEST.INTERVAL:
            raise ValueError("Trajectory snapshots must include the final interval")
        logging.info(
            "DCCL trajectory ensemble enabled: final_cycle={}; intervals={}".format(
                int(cfg.ACTIVE.CYCLE), trajectory_intervals
            )
        )

    prev_label_mask = None
    pl_state = None
    proto_state = None
    conflict_state = None
    pair_flow_state = None
    text_features = None
    curr_cycle = 0
    # office-home : 1.0 / VisDA-C : 1.05
    q_value = cfg.ACTIVE.Q_VALUE
    print(f"train_clip")
    while curr_cycle < cfg.ACTIVE.CYCLE:
        iter_num = 0

        netF.eval()
        netB.eval()
        if target_head is not None:
            target_head.eval()
        if pair_feature_adapter is not None:
            pair_feature_adapter.eval()
        inference_head = target_head_ema if target_head_ema is not None else target_head
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
            pl_state,
        ) = obtain_label(
            cfg,
            dset_loaders['test_aug'], netF, netB, netC, inference_head,
            pair_feature_adapter,
            covariance_transport,
            text_inputs, text_features, clip_model, prev_label_mask,
            proto_state, pl_state, curr_cycle,
        )
        if covariance_transport is not None and not bool(
            covariance_transport.fitted.item()
        ):
            if isinstance(covariance_transport, AgreementWhitenedTransport):
                transport_diagnostics = covariance_transport.fit(
                    task_features,
                    source_label,
                    clip_label,
                    netC.fc.weight,
                    netC.fc.bias,
                )
                logging.info(
                    "DCCL agreement-whitened geometry frozen: anchors={}; "
                    "train_anchors={}; heldout_anchors={}; active_classes={}; "
                    "selected_strength={:.6f}; heldout_loss_improvement={:.6f}; "
                    "heldout_accuracy_delta={:.6f}; mean_relative_shift={:.6f}".format(
                        transport_diagnostics["anchors"],
                        transport_diagnostics["train_anchors"],
                        transport_diagnostics["heldout_anchors"],
                        transport_diagnostics["active_classes"],
                        transport_diagnostics["selected_strength"],
                        transport_diagnostics["heldout_loss_improvement"],
                        transport_diagnostics["heldout_accuracy_delta"],
                        transport_diagnostics["mean_relative_shift"],
                    )
                )
            else:
                transport_diagnostics = covariance_transport.fit(
                    task_features,
                    source_label,
                    clip_label,
                    confi_dis,
                )
                logging.info(
                    "DCCL agreement covariance geometry frozen: anchors={}; "
                    "active_classes={}; fixed_conflicts={}; eligible_conflicts={}; "
                    "eligible_coverage={:.6f}; mean_relative_shift={:.6f}".format(
                        transport_diagnostics["anchors"],
                        transport_diagnostics["active_classes"],
                        transport_diagnostics["fixed_conflicts"],
                        transport_diagnostics["eligible_conflicts"],
                        transport_diagnostics["eligible_coverage"],
                        transport_diagnostics["mean_relative_shift"],
                    )
                )
        if isinstance(target_head, ClassPairFlowAdapter):
            was_frozen = bool(pair_flow_state and pair_flow_state["frozen"])
            pair_flow_state = update_class_pair_flow(
                source_label,
                clip_label,
                mem_label,
                label_mask,
                pair_flow_state,
                num_classes=cfg.class_num,
                rank=int(cfg.DCCL.PAIR_FLOW_RANK),
                min_count=int(cfg.DCCL.PAIR_FLOW_MIN_COUNT),
                min_cycles=int(cfg.DCCL.PAIR_FLOW_MIN_CYCLES),
            )
            target_head.set_basis(pair_flow_state["basis"])
            logging.info(
                "DCCL class-pair flow: cycle={}; active_rank={}; frozen={}; "
                "resolved_flow_count={:.0f}".format(
                    curr_cycle + 1,
                    int(pair_flow_state["active_rank"]),
                    bool(pair_flow_state["frozen"]),
                    float(pair_flow_state["counts"].sum().item()),
                )
            )
            if pair_flow_state["frozen"] and not was_frozen:
                logging.info(
                    "DCCL class-pair basis frozen: pairs={}".format(
                        pair_flow_state["pairs"]
                    )
                )
        if pair_feature_adapter is not None:
            was_frozen = bool(pair_flow_state and pair_flow_state["frozen"])
            pair_flow_state = update_soft_class_pair_flow(
                source_label,
                clip_label,
                confi_dis,
                pair_flow_state,
                num_classes=cfg.class_num,
                rank=int(cfg.DCCL.PAIR_FLOW_RANK),
                min_count=int(cfg.DCCL.PAIR_FLOW_MIN_COUNT),
                min_cycles=int(cfg.DCCL.PAIR_FLOW_MIN_CYCLES),
                fixed_candidates=True,
            )
            if pair_flow_state["frozen"]:
                pair_feature_adapter.set_pairs(
                    pair_flow_state["pairs"], netC.fc.weight
                )
            logging.info(
                "DCCL fixed-candidate pair flow: cycle={}; valid={}; "
                "candidate_mass={:.2f}; active_rank={}; frozen={}; "
                "resolved_flow_mass={:.2f}".format(
                    curr_cycle + 1,
                    int(pair_flow_state["last_valid_count"]),
                    float(pair_flow_state["last_candidate_mass"]),
                    int(pair_flow_state["active_rank"]),
                    bool(pair_flow_state["frozen"]),
                    float(pair_flow_state["counts"].sum().item()),
                )
            )
            if pair_flow_state["frozen"] and not was_frozen:
                logging.info(
                    "DCCL pair-feature directions frozen: pairs={}".format(
                        pair_flow_state["pairs"]
                    )
                )
        proto_state = update_target_prototype_state(
            cfg, task_features, mem_label, label_mask, proto_state
        )
        if conflict_state is None:
            conflict_state = init_conflict_state(source_label.size(0))
        if not cfg.ACCD.ENABLED:
            update_conflict_state(cfg, conflict_state, source_label, clip_label, model_soft)
        teacher_soft, graph_teacher, graph_weight, graph_anchors = build_graph_fused_teacher(
            cfg,
            task_features,
            clip_features,
            model_soft,
            clip_soft,
            source_label,
            clip_label,
        )
        if cfg.DCCL.GRAPH_TEACHER_FUSION and cfg.DCCL.GTF_APPLY_TO in {"both", "clip"}:
            confi_dis = teacher_soft.detach()
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
        )
        sample_idx = torch.arange(source_label.size(0))
        candidate_mass = model_soft[sample_idx, source_label] + model_soft[sample_idx, clip_label]
        candidate_weight = get_candidate_weight(cfg, candidate_mass)
        kl_base_soft = (
            teacher_soft
            if cfg.DCCL.GRAPH_TEACHER_FUSION and cfg.DCCL.GTF_APPLY_TO in {"both", "kl"}
            else clip_soft
        )
        kl_target, kl_weight = build_conflict_kl_target(cfg, kl_base_soft, source_label, clip_label, model_soft)
        gtr_target = teacher_soft
        gtr_weight = torch.zeros(source_label.size(0), dtype=torch.float)
        if cfg.DCCL.GTR_PAR > 0:
            if graph_teacher is None:
                raise ValueError("DCCL.GTR_PAR requires DCCL.GRAPH_TEACHER_FUSION=True")
            teacher_label = teacher_soft.argmax(dim=1)
            graph_label = graph_teacher.argmax(dim=1)
            gtr_eligible = (
                (source_label != clip_label)
                & (teacher_label == graph_label)
            )
            (
                conflict_state["gtr_pending_label"],
                conflict_state["gtr_pending_count"],
                conflict_state["gtr_stable_label"],
                gtr_newly_stable,
                gtr_stable_mask,
                gtr_demoted,
            ) = update_temporal_resolution(
                conflict_state["gtr_pending_label"],
                conflict_state["gtr_pending_count"],
                conflict_state["gtr_stable_label"],
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
                conflict_state["gtr_stable_label"],
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

        accd_resolved_mask = torch.zeros_like(label_mask)
        accd_hard_mask = torch.zeros_like(label_mask)
        accd_transport_mask = torch.zeros_like(label_mask)
        accd_shifted_mass = torch.zeros_like(kl_weight)
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
            if cfg.ACCD.RESOLUTION_ACTION == "hard_label":
                kl_target[accd_resolved_mask] = graph_posterior[accd_resolved_mask]
                accd_hard_mask = accd_resolved_mask
            elif cfg.ACCD.RESOLUTION_ACTION == "teacher_abstain":
                kl_weight[accd_resolved_mask] = 0.0
                accd_hard_mask = torch.zeros_like(accd_resolved_mask)
            elif cfg.ACCD.RESOLUTION_ACTION == "candidate_transport":
                accd_transport_mask = (
                    accd_resolved_mask
                    & resolution_evidence["eligible"]
                    & (
                        conflict_state["accd_resolved_label"]
                        == evidence["graph_label"]
                    )
                )
                kl_target, accd_shifted_mass = transport_candidate_mass(
                    kl_target,
                    graph_posterior,
                    source_label,
                    clip_label,
                    accd_transport_mask,
                )
                accd_hard_mask = torch.zeros_like(accd_resolved_mask)
            else:
                raise ValueError(
                    f"Unknown ACCD.RESOLUTION_ACTION: {cfg.ACCD.RESOLUTION_ACTION}"
                )

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
                "resolution_target={}; resolution_action={}; teacher_abstained={}; "
                "candidate_transported={}; mean_shifted_mass={:.4f}".format(
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
                    cfg.ACCD.RESOLUTION_ACTION,
                    int((kl_weight == 0).sum().item()),
                    int(accd_transport_mask.sum().item()),
                    float(accd_shifted_mass[accd_transport_mask].mean().item())
                    if accd_transport_mask.any() else 0.0,
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
        hard_label[accd_hard_mask] = conflict_state["accd_resolved_label"][accd_hard_mask]
        hard_mask = label_mask | promoted_mask | accd_hard_mask
        candidate_mask = (
            (source_label != clip_label)
            & (~promoted_mask)
            & (~accd_hard_mask)
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
        gtr_target = gtr_target.cuda()
        gtr_weight = gtr_weight.cuda()
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
        if target_head is not None:
            target_head.train()
        if pair_feature_adapter is not None:
            pair_feature_adapter.train()
        pair_feature_gtr_loss_sum = 0.0
        pair_feature_gtr_loss_batches = 0
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

            weak_base_feas = netB(netF(weak_x))
            strong_base_feas = netB(netF(strong_x))
            weak_feas = apply_pair_feature_adapter(
                cfg, weak_base_feas, pair_feature_adapter, curr_cycle
            )
            strong_feas = apply_pair_feature_adapter(
                cfg, strong_base_feas, pair_feature_adapter, curr_cycle
            )
            weak_feas = apply_agreement_covariance_transport(
                cfg, weak_feas, covariance_transport, curr_cycle, tar_idx
            )
            strong_feas = apply_agreement_covariance_transport(
                cfg, strong_feas, covariance_transport, curr_cycle, tar_idx
            )

            weak_logits = netC(weak_feas)
            strong_logits = netC(strong_feas)
            weak_logits = apply_target_head_logits(
                cfg, weak_feas, weak_logits, target_head, curr_cycle
            )
            strong_logits = apply_target_head_logits(
                cfg, strong_feas, strong_logits, target_head, curr_cycle
            )
            weak_logits = apply_target_prototype_logits(cfg, weak_feas, weak_logits, proto_state)
            strong_logits = apply_target_prototype_logits(cfg, strong_feas, strong_logits, proto_state)

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
                    if (
                        pair_feature_adapter is not None
                        and cfg.DCCL.PAIR_FEATURE_GRADIENT_MODE == "gtr_only"
                        and curr_cycle >= cfg.DCCL.PAIR_FEATURE_START_CYCLE
                        and pair_feature_adapter.is_effective()
                    ):
                        pair_gtr_feas = pair_feature_adapter(
                            weak_base_feas.detach(), detach_delta=False
                        )
                        pair_gtr_logits = netC(pair_gtr_feas)
                        pair_gtr_loss = weighted_graph_temporal_kl(
                            pair_gtr_logits,
                            gtr_target_batch,
                            gtr_weight_batch,
                            cfg.DCCL.EPSILON,
                        )
                        classifier_loss += pair_gtr_loss * cfg.DCCL.GTR_PAR
                        pair_feature_gtr_loss_sum += float(
                            pair_gtr_loss.detach().item()
                        )
                        pair_feature_gtr_loss_batches += 1

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
            if target_head_ema is not None:
                update_model_ema(
                    target_head_ema,
                    target_head,
                    float(cfg.DCCL.TARGET_HEAD_EMA_MOMENTUM),
                )

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netF.eval()
                netB.eval()
                if target_head is not None:
                    target_head.eval()
                if pair_feature_adapter is not None:
                    pair_feature_adapter.eval()
                captured_trajectory_snapshot = False
                if cfg.DCCL.TRAJECTORY_ENSEMBLE and (
                    curr_cycle == cfg.ACTIVE.CYCLE - 1
                ):
                    checkpoint_index = (
                        cfg.TEST.INTERVAL
                        if iter_num == max_iter
                        else iter_num // interval_iter
                    )
                    if checkpoint_index in trajectory_intervals:
                        trajectory_snapshots.append(
                            capture_trajectory_snapshot(
                                netF, netB, target_head, checkpoint_index
                            )
                        )
                        captured_trajectory_snapshot = True
                        logging.info(
                            "DCCL trajectory snapshot captured: interval={}; members={}".format(
                                int(checkpoint_index), len(trajectory_snapshots)
                            )
                        )
                # netC.eval()
                if cfg.SETTING.DATASET == 'VISDA-C':
                    acc_s_te, acc_list = cal_acc(
                        dset_loaders['test'], netF, netB, netC, cfg,
                        proto_state, inference_head, curr_cycle,
                        pair_feature_adapter, covariance_transport, True
                    )
                    log_str = ('Task: {}, Iter:{}/{}; Cycle: {}/{}; '
                               'Accuracy = {:.2f}%; classifier_loss = {}').format(cfg.name, iter_num, max_iter,
                                                                                  curr_cycle + 1, cfg.ACTIVE.CYCLE,
                                                                                  acc_s_te,
                                                                                  classifier_loss) + '\n' + acc_list
                else:
                    acc_s_te, _ = cal_acc(
                        dset_loaders['test'], netF, netB, netC, cfg,
                        proto_state, inference_head, curr_cycle,
                        pair_feature_adapter, covariance_transport, False
                    )
                    log_str = ('Task: {}, Iter:{}/{}; Cycle: {}/{}; '
                               'Accuracy = {:.2f}%; classifier_loss = {}').format(cfg.name, iter_num, max_iter,
                                                                                  curr_cycle + 1, cfg.ACTIVE.CYCLE,
                                                                                  acc_s_te, classifier_loss)

                if isinstance(target_head, SourceAnchoredResidualClassifier):
                    log_str += "; residual_gate={:.6f}".format(
                        float(target_head.effective_gate().item())
                    )
                elif isinstance(target_head, ClassPairFlowAdapter):
                    active_rank = (
                        int(pair_flow_state["active_rank"])
                        if pair_flow_state is not None
                        else 0
                    )
                    log_str += (
                        "; pair_flow_gate={:.6f}; pair_flow_active_rank={}"
                    ).format(
                        float(target_head.effective_gate().item()), active_rank
                    )
                if pair_feature_adapter is not None:
                    active_rank = (
                        int(pair_flow_state["active_rank"])
                        if pair_flow_state is not None
                        else 0
                    )
                    log_str += (
                        "; pair_feature_gate={:.6f}; pair_feature_router_norm={:.6f}; "
                        "pair_flow_active_rank={}; pair_feature_effective={}; "
                        "pair_feature_gradient_mode={}; pair_feature_gtr_active={}; "
                        "pair_feature_gtr_loss={:.6f}; pair_feature_gtr_batches={}"
                    ).format(
                        float(pair_feature_adapter.effective_gate().item()),
                        float(pair_feature_adapter.router.weight.detach().norm().item()),
                        active_rank,
                        bool(pair_feature_adapter.is_effective()),
                        cfg.DCCL.PAIR_FEATURE_GRADIENT_MODE,
                        int(
                            (gtr_weight > 0).sum().item()
                            if (
                                cfg.DCCL.PAIR_FEATURE_GRADIENT_MODE == "gtr_only"
                                and curr_cycle >= cfg.DCCL.PAIR_FEATURE_START_CYCLE
                                and pair_feature_adapter.is_effective()
                            )
                            else 0
                        ),
                        (
                            pair_feature_gtr_loss_sum
                            / pair_feature_gtr_loss_batches
                            if pair_feature_gtr_loss_batches > 0
                            else 0.0
                        ),
                        pair_feature_gtr_loss_batches,
                    )
                if covariance_transport is not None:
                    transport_diagnostics = covariance_transport.diagnostics()
                    if isinstance(
                        covariance_transport, AgreementWhitenedTransport
                    ):
                        log_str += (
                            "; cov_global_active_classes={}; "
                            "cov_global_selected_strength={:.6f}; "
                            "cov_global_heldout_improvement={:.6f}; "
                            "cov_global_accuracy_delta={:.6f}; "
                            "cov_global_mean_shift={:.6f}"
                        ).format(
                            transport_diagnostics["active_classes"],
                            transport_diagnostics["selected_strength"],
                            transport_diagnostics["heldout_loss_improvement"],
                            transport_diagnostics["heldout_accuracy_delta"],
                            transport_diagnostics["mean_relative_shift"],
                        )
                    else:
                        log_str += (
                            "; cov_transport_active_classes={}; "
                            "cov_transport_coverage={:.6f}; "
                            "cov_transport_mean_shift={:.6f}"
                        ).format(
                            transport_diagnostics["active_classes"],
                            transport_diagnostics["eligible_coverage"],
                            transport_diagnostics["mean_relative_shift"],
                        )

                # cfg.out_file.write(log_str + '\n')
                # cfg.out_file.flush()
                # print(log_str+'\n')
                logging.info(log_str)
                pair_feature_gtr_loss_sum = 0.0
                pair_feature_gtr_loss_batches = 0
                if captured_trajectory_snapshot and len(trajectory_snapshots) >= 2:
                    trajectory_acc, _ = cal_acc_trajectory_ensemble(
                        dset_loaders['test'],
                        netF,
                        netB,
                        netC,
                        cfg,
                        proto_state,
                        target_head,
                        curr_cycle,
                        trajectory_snapshots,
                    )
                    logging.info(
                        "Trajectory Ensemble Task: {}, Iter:{}/{}; Cycle: {}/{}; "
                        "Accuracy = {:.2f}%; Members={}".format(
                            cfg.name,
                            iter_num,
                            max_iter,
                            curr_cycle + 1,
                            cfg.ACTIVE.CYCLE,
                            trajectory_acc,
                            len(trajectory_snapshots),
                        )
                    )
                netF.train()
                netB.train()
                if target_head is not None:
                    target_head.train()
                if pair_feature_adapter is not None:
                    pair_feature_adapter.train()
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


def init_pseudo_label_state(num_samples):
    return {
        "pending_label": torch.full((num_samples,), -1, dtype=torch.long),
        "pending_count": torch.zeros(num_samples, dtype=torch.long),
        "stable_label": torch.full((num_samples,), -1, dtype=torch.long),
    }


def apply_pseudo_label_memory(
    cfg,
    prev_label_mask,
    matching_indices,
    all_mix_output_pred,
    all_mix_output,
    mix_conf,
    pl_state,
    curr_cycle,
):
    mode = cfg.DCCL.PL_MEMORY
    confidence_mask = mix_conf >= cfg.DCCL.PL_MEMORY_MIN_CONF
    current_mask = matching_indices & confidence_mask

    if mode == "monotonic":
        if prev_label_mask is not None:
            label_mask = prev_label_mask | (~prev_label_mask & current_mask)
        else:
            label_mask = current_mask
        return label_mask, all_mix_output_pred, pl_state

    if mode == "current":
        return current_mask, all_mix_output_pred, pl_state

    if mode != "stable":
        raise ValueError(f"Unknown DCCL.PL_MEMORY: {mode}")
    if cfg.DCCL.PL_STABLE_CYCLES <= 0:
        raise ValueError("DCCL.PL_STABLE_CYCLES must be positive")
    if cfg.DCCL.PL_STABLE_MEMORY not in {"persistent", "reversible"}:
        raise ValueError(f"Unknown DCCL.PL_STABLE_MEMORY: {cfg.DCCL.PL_STABLE_MEMORY}")

    if pl_state is None:
        pl_state = init_pseudo_label_state(all_mix_output_pred.numel())

    same_label = pl_state["pending_label"] == all_mix_output_pred
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
        all_mix_output_pred,
        torch.full_like(pl_state["pending_label"], -1),
    )
    stable = current_mask & (pl_state["pending_count"] >= cfg.DCCL.PL_STABLE_CYCLES)
    if cfg.DCCL.PL_STABLE_MEMORY == "persistent":
        pl_state["stable_label"] = torch.where(
            stable, all_mix_output_pred, pl_state["stable_label"]
        )
    else:
        pl_state["stable_label"] = torch.where(
            stable,
            all_mix_output_pred,
            torch.full_like(pl_state["stable_label"], -1),
        )

    stable_mask = pl_state["stable_label"] >= 0
    warmup = curr_cycle < cfg.DCCL.PL_MEMORY_WARMUP_CYCLES
    if warmup:
        label_mask = current_mask
        memory_label = all_mix_output_pred
    else:
        label_mask = stable_mask
        memory_label = torch.where(
            stable_mask,
            pl_state["stable_label"],
            all_mix_output_pred,
        )

    if cfg.DCCL.PL_CLASS_BALANCE and not warmup:
        budget = int(round(float(cfg.DCCL.PL_BALANCE_COVERAGE) * memory_label.numel()))
        if budget <= 0:
            budget = int(label_mask.sum().item())
        target_prior = all_mix_output.mean(dim=0)
        balanced_mask, quotas = class_balanced_mask_by_prior(
            memory_label,
            label_mask,
            mix_conf,
            target_prior,
            budget,
            min_per_class=cfg.DCCL.PL_BALANCE_MIN_PER_CLASS,
            eps=cfg.DCCL.EPSILON,
        )
        logging.info(
            "DCCL pseudo-label class balance: selected={}->{}; budget={}; "
            "active_classes={}; quota_range=({},{})".format(
                int(label_mask.sum().item()),
                int(balanced_mask.sum().item()),
                int(budget),
                int((quotas > 0).sum().item()),
                int(quotas[quotas > 0].min().item()) if (quotas > 0).any() else 0,
                int(quotas.max().item()) if quotas.numel() else 0,
            )
        )
        label_mask = balanced_mask

    logging.info(
        "DCCL pseudo-label memory: mode={}; stable_memory={}; warmup={}; "
        "current={}; stable={}; selected={}".format(
            cfg.DCCL.PL_MEMORY,
            cfg.DCCL.PL_STABLE_MEMORY,
            int(warmup),
            int(current_mask.sum().item()),
            int(stable_mask.sum().item()),
            int(label_mask.sum().item()),
        )
    )
    return label_mask, memory_label, pl_state


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


def apply_classwise_calibration(cfg, source_prob, clip_prob, task_features=None, clip_features=None):
    mode = cfg.DCCL.CALIB_MODE
    selected_mode = mode
    if mode == "topo_prior":
        if task_features is None or clip_features is None:
            raise ValueError("topo_prior calibration requires task and CLIP features")
        raw_source_label = source_prob.argmax(dim=1)
        raw_clip_label = clip_prob.argmax(dim=1)
        source_cal, clip_cal, mix_prob, graph_prior, anchors = topology_prior_calibrate(
            task_features,
            clip_features,
            source_prob,
            clip_prob,
            raw_source_label,
            raw_clip_label,
            power=cfg.DCCL.CALIB_POWER,
            anchor_ratio=cfg.DCCL.TOPO_ANCHOR_RATIO,
            anchor_min_per_class=cfg.DCCL.TOPO_ANCHOR_MIN_PER_CLASS,
            k=cfg.DCCL.TOPO_GRAPH_K,
            temperature=cfg.DCCL.TOPO_TEMPERATURE,
            alpha=cfg.DCCL.TOPO_ALPHA,
            steps=cfg.DCCL.TOPO_STEPS,
            chunk_size=cfg.DCCL.TOPO_CHUNK_SIZE,
            eps=cfg.DCCL.EPSILON,
        )
        logging.info(
            "DCCL topology-prior calibration: anchors={}; graph_prior_range=({:.4f},{:.4f}); "
            "graph_prior_entropy={:.4f}; k={}; alpha={:.3f}; steps={}".format(
                int(anchors.sum().item()),
                float(graph_prior.min().item()),
                float(graph_prior.max().item()),
                float(-(graph_prior * graph_prior.clamp_min(cfg.DCCL.EPSILON).log()).sum().item()),
                int(cfg.DCCL.TOPO_GRAPH_K),
                float(cfg.DCCL.TOPO_ALPHA),
                int(cfg.DCCL.TOPO_STEPS),
            )
        )
        return source_cal, clip_cal, mix_prob

    if mode == "topo_target_prior":
        if task_features is None or clip_features is None:
            raise ValueError("topo_target_prior calibration requires task and CLIP features")
        raw_source_label = source_prob.argmax(dim=1)
        raw_clip_label = clip_prob.argmax(dim=1)
        (
            source_cal,
            clip_cal,
            mix_prob,
            graph_prior,
            target_prior,
            anchors,
            target_mix,
        ) = topology_target_prior_calibrate(
            task_features,
            clip_features,
            source_prob,
            clip_prob,
            raw_source_label,
            raw_clip_label,
            power=cfg.DCCL.CALIB_POWER,
            target_mix=cfg.DCCL.TOPO_TARGET_MIX,
            anchor_ratio=cfg.DCCL.TOPO_ANCHOR_RATIO,
            anchor_min_per_class=cfg.DCCL.TOPO_ANCHOR_MIN_PER_CLASS,
            k=cfg.DCCL.TOPO_GRAPH_K,
            temperature=cfg.DCCL.TOPO_TEMPERATURE,
            alpha=cfg.DCCL.TOPO_ALPHA,
            steps=cfg.DCCL.TOPO_STEPS,
            chunk_size=cfg.DCCL.TOPO_CHUNK_SIZE,
            eps=cfg.DCCL.EPSILON,
        )
        logging.info(
            "DCCL topology-target prior alignment: anchors={}; target_mix={:.4f}; "
            "graph_prior_range=({:.4f},{:.4f}); target_prior_range=({:.4f},{:.4f}); "
            "target_prior_entropy={:.4f}; k={}; alpha={:.3f}; steps={}".format(
                int(anchors.sum().item()),
                float(target_mix),
                float(graph_prior.min().item()),
                float(graph_prior.max().item()),
                float(target_prior.min().item()),
                float(target_prior.max().item()),
                float(-(target_prior * target_prior.clamp_min(cfg.DCCL.EPSILON).log()).sum().item()),
                int(cfg.DCCL.TOPO_GRAPH_K),
                float(cfg.DCCL.TOPO_ALPHA),
                int(cfg.DCCL.TOPO_STEPS),
            )
        )
        return source_cal, clip_cal, mix_prob

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


def obtain_label(
    cfg,
    loader,
    netF,
    netB,
    netC,
    target_head,
    pair_feature_adapter,
    covariance_transport,
    text_inputs,
    text_features,
    clip_model,
    prev_label_mask,
    proto_state,
    pl_state,
    curr_cycle,
):
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
            inputs_test, labels, sample_indices = next(iter_test)
            weak_x = inputs_test[1].cuda()
            # strong_x = inputs_test[2].cuda()

            weak_feas = netB(netF(weak_x))
            # strong_feas = netB(netF(strong_x))

            adapted_weak_feas = apply_pair_feature_adapter(
                cfg, weak_feas, pair_feature_adapter, curr_cycle
            )
            adapted_weak_feas = apply_agreement_covariance_transport(
                cfg,
                adapted_weak_feas,
                covariance_transport,
                curr_cycle,
                sample_indices,
            )
            weak_outputs = netC(adapted_weak_feas)
            weak_outputs = apply_target_head_logits(
                cfg, adapted_weak_feas, weak_outputs, target_head, curr_cycle
            )
            weak_outputs = apply_target_prototype_logits(
                cfg, adapted_weak_feas, weak_outputs, proto_state
            )
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
    all_output, clip_all_output, all_mix_output = apply_classwise_calibration(
        cfg, all_output, clip_all_output, all_task_features, all_clip_features
    )

    # Compute predictions for all_output and clip_all_output
    _, all_output_pred = torch.max(all_output, dim=1)
    _, clip_all_output_pred = torch.max(clip_all_output, dim=1)

    _, all_mix_output_pred = torch.max(all_mix_output, dim=1)

    # Find indices where predictions match
    matching_indices = all_output_pred == clip_all_output_pred

    mix_conf, _ = torch.max(all_mix_output, dim=1)
    label_mask, all_mix_output_pred, pl_state = apply_pseudo_label_memory(
        cfg,
        prev_label_mask,
        matching_indices,
        all_mix_output_pred,
        all_mix_output,
        mix_conf,
        pl_state,
        curr_cycle,
    )
    label_mask = expand_pseudo_label_mask(cfg, label_mask, all_mix_output, all_mix_output_pred)

    # Filter predictions and labels based on the updated label mask
    valid_preds = all_mix_output_pred[label_mask]
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
    if len(valid_preds) > 0:
        mixed_output_accuracy = torch.sum(valid_mixed == valid_labels).item() / float(len(valid_preds))
    else:
        mixed_output_accuracy = 0.0
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


def clip_text(model, text_features, inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits
