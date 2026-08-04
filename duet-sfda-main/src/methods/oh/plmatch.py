"""
Builds upon: https://github.com/tim-learn/SHOT
Corresponding paper: http://proceedings.mlr.press/v119/liang20a/liang20a.pdf
"""

import os.path as osp
import math
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
# from src.utils import loss, active_prompt, IID_losses
# from proposed_method import *
from torch.nn.functional import normalize
from data.datautils_domain import build_dataset
from data.cls_to_names import *
from data.domain_datasets import domain_datasets
from sklearn.metrics import confusion_matrix
from src.utils.adaptation_lists import load_adaptation_and_evaluation_rows
from src.utils.conflict_boundary import (
    pairwise_first_order_boundary,
    route_conflict_probabilities,
)
from src.utils.failure_audit import save_failure_audit_snapshot
from src.utils.first_cycle_prior import apply_first_cycle_prior
from src.utils.topk_conflict_probe import write_topk_conflict_probe
from src.utils.swap_conflict_selection import (
    select_swap_labels,
    summarize_swap_decisions,
)
from src.utils.attribute_reliability import (
    entropy_anchored_attribute_target,
    pairwise_attribute_margin,
)
from src.utils.pairwise_attribute_audit import build_visda_attribute_prompt_manifest
from src.utils.support_conditioned_clip import (
    condition_clip_on_task_clip_top2_union,
)
from src.utils.clip_confidence_delay import (
    LOCKED_DELAY_FRACTION,
    class_balanced_clip_confidence_delay,
)
from src.utils.pcgrad_parameter_runtime import run_exact_pcgrad_parameter_audit
from src.utils.pcgrad_compatibility import (
    build_pcgrad_parameter_correction,
    merge_compatible_parameter_correction_,
)

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
    adaptation_override = str(cfg.ACTIVE.ADAPTATION_LIST).strip()
    txt_tar, txt_test, adaptation_path = load_adaptation_and_evaluation_rows(
        cfg.t_dset_path,
        cfg.test_dset_path,
        adaptation_override,
    )
    if adaptation_override:
        logging.info(
            "PLMatch adaptation proxy list: {}; adaptation_samples={}; "
            "full_evaluation_samples={}".format(
                adaptation_path, len(txt_tar), len(txt_test)
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
    # tar_idx indexes pseudo-label tensors produced by test_aug, so both
    # loaders must use the same ordered adaptation rows. Only test stays full.
    dsets["test_aug"] = ImageList_idx(txt_tar, transform=train_transform)
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


def train_target(
    cfg,
    *,
    first_cycle_prior=False,
    boundary_router=False,
    attribute_reliability_kl=False,
    support_conditioned_clip=False,
    support_conditioned_clip_memory=False,
    clip_confidence_delay=False,
    pcgrad_compatibility=False,
    topk_conflict_probe=False,
    swap_conflict_selection=False,
):
    """训练原始 DUET，或启用一个明确隔离的候选改动。

    ``first_cycle_prior=False`` 是发布版 DUET 的原始路径。DUET-FCP 入口只把
    该参数设为 ``True``；stable memory、target head、graph teacher 和 GTR
    均不在本文件中，因此不会混入候选。

    ``boundary_router=True`` 仅在首轮未准入冲突中，将固定 top fraction 的
    算术融合软分布替换为局部边界距离更大的完整 task 或 CLIP 分布。硬 CE
    mask、consistency、KL 和所有优化器均保持发布版 DUET 不变。

    ``attribute_reliability_kl=True`` 只在首轮未准入冲突中，将原本的 CLIP
    KL 软目标替换为离线审计锁定的属性可靠性后验。agreement、硬 CE mask、
    consistency、损失权重和优化器均保持发布版 DUET 不变。

    ``support_conditioned_clip=True`` 只在首轮未准入冲突中，将 CLIP KL
    软目标限制到 task/CLIP top-2 并集后重新归一化。task 只提供候选支持集；
    CLIP 的集内相对概率及 top-1 均保持不变。

    ``support_conditioned_clip_memory=True`` 使用同一软目标公式，但将它持续用于
    首轮冲突中尚未被累积 agreement mask 解决的样本。这个候选只改时间作用域；
    目标公式、KL 权重、硬 CE mask、consistency 和优化器均不变。

    ``clip_confidence_delay=True`` 仅在首轮 top-1 一致样本中，按共同伪类别
    暂缓 CLIP 置信度最低的固定 10%。后续仍由原始单调 agreement 规则自然
    准入；CLIP KL、融合分布、损失权重和优化器全部保持不变。

    ``pcgrad_compatibility=True`` 仅在 cycle 2 对尚未准入的冲突样本组合
    consistency 与 CLIP-KL 的输出梯度。PCGrad 修正量再按其与完整 DUET
    参数梯度的非负投影比例截断；其余 cycle、目标、mask、权重和优化器不变。

    ``topk_conflict_probe=True`` 只导出每个 cycle 的 Task/CLIP top-2
    冲突覆盖 oracle diagnostic。它使用 detach 后的 prior 校准前概率，
    不返回任何训练信号，也不改变 mask、loss 或优化器。

    ``swap_conflict_selection=True`` 只对 bidirectional_cross_support
    （纯 swap）冲突产生硬伪标签：cycle 0 直接取 CLIP top1；cycle >= 1 按
    eA=pA*qA、eB=pB*qB 的 log 差与 ``DUET_SWAP.GATE_D`` 门槛决定 A/B 或
    abstain。abstain 样本不进入 label_mask，不进训练损失。检测与决策使用
    prior 校准前的概率（与 Top-k probe 导出口径一致）；非 swap 冲突和非
    冲突样本不进入该规则，原 DUET 策略不变。
    """
    candidate_count = sum(
        bool(value)
        for value in (
            first_cycle_prior,
            boundary_router,
            attribute_reliability_kl,
            support_conditioned_clip,
            support_conditioned_clip_memory,
            clip_confidence_delay,
            pcgrad_compatibility,
        )
    )
    if candidate_count > 1:
        raise ValueError("DUET candidate interventions must be run separately")
    parameter_audit = bool(cfg.PCGRAD_PARAMETER_AUDIT.ENABLED)
    if parameter_audit and candidate_count:
        raise ValueError("PCGrad parameter audit requires pure arithmetic DUET")
    if parameter_audit and (
        cfg.SETTING.DATASET != "VISDA-C"
        or int(cfg.ACTIVE.CYCLE) != 2
        or int(cfg.TEST.BATCH_SIZE) != 64
        or cfg.FAILURE_AUDIT.ENABLED
    ):
        raise ValueError(
            "PCGrad parameter audit is locked to VisDA-C, two cycles, "
            "batch size 64, and FAILURE_AUDIT disabled"
        )
    if pcgrad_compatibility and (
        cfg.SETTING.DATASET != "VISDA-C"
        or int(cfg.ACTIVE.CYCLE) != 4
        or int(cfg.TEST.BATCH_SIZE) != 64
        or cfg.FAILURE_AUDIT.ENABLED
    ):
        raise ValueError(
            "PCGrad compatibility is locked to VisDA-C, four cycles, "
            "batch size 64, and FAILURE_AUDIT disabled"
        )
    if swap_conflict_selection and not cfg.DUET_SWAP.ENABLED:
        raise ValueError(
            "swap-conflict selection requires DUET_SWAP.ENABLED=True"
        )
    if swap_conflict_selection and not first_cycle_prior:
        raise ValueError(
            "swap-conflict selection requires first_cycle_prior=True"
        )
    if swap_conflict_selection and candidate_count:
        raise ValueError(
            "swap-conflict selection cannot be combined with other DUET "
            "candidates"
        )
    if swap_conflict_selection and (
        cfg.SETTING.DATASET != "VISDA-C"
        or str(cfg.ACTIVE.ARCH) != "ViT-B/32"
    ):
        raise ValueError(
            "swap-conflict selection is locked to VisDA-C with CLIP ViT-B/32"
        )
    swap_gate_D = float(cfg.DUET_SWAP.GATE_D)
    if swap_conflict_selection and swap_gate_D < 0.0:
        raise ValueError("DUET_SWAP.GATE_D must be non-negative")
    if first_cycle_prior and cfg.DUET_FCP.POWER < 0:
        raise ValueError("DUET_FCP.POWER must be non-negative")
    boundary_fraction = float(cfg.DUET_BOUNDARY.TOP_FRACTION)
    if boundary_router and not 0.0 < boundary_fraction <= 1.0:
        raise ValueError("DUET_BOUNDARY.TOP_FRACTION must be in (0, 1]")
    if attribute_reliability_kl and (
        cfg.SETTING.DATASET != "VISDA-C" or str(cfg.ACTIVE.ARCH) != "ViT-B/32"
    ):
        raise ValueError(
            "attribute-reliability KL is locked to VisDA-C with CLIP ViT-B/32"
        )
    if (support_conditioned_clip or support_conditioned_clip_memory) and (
        cfg.SETTING.DATASET != "VISDA-C" or str(cfg.ACTIVE.ARCH) != "ViT-B/32"
    ):
        raise ValueError(
            "support-conditioned CLIP is locked to VisDA-C with CLIP ViT-B/32"
        )
    clip_delay_fraction = float(cfg.DUET_CLIP_DELAY.FRACTION)
    if clip_confidence_delay and (
        cfg.SETTING.DATASET != "VISDA-C" or str(cfg.ACTIVE.ARCH) != "ViT-B/32"
    ):
        raise ValueError(
            "CLIP-confidence delay is locked to VisDA-C with CLIP ViT-B/32"
        )
    if clip_confidence_delay and not math.isclose(
        clip_delay_fraction,
        LOCKED_DELAY_FRACTION,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("DUET_CLIP_DELAY.FRACTION is locked to 0.10")
    stop_after_pre_cycle = int(cfg.FAILURE_AUDIT.STOP_AFTER_PRE_CYCLE)
    if stop_after_pre_cycle < 0 or stop_after_pre_cycle > int(cfg.ACTIVE.CYCLE):
        raise ValueError(
            "FAILURE_AUDIT.STOP_AFTER_PRE_CYCLE must be between 0 and ACTIVE.CYCLE"
        )
    if stop_after_pre_cycle and not cfg.FAILURE_AUDIT.ENABLED:
        raise ValueError(
            "FAILURE_AUDIT.STOP_AFTER_PRE_CYCLE requires FAILURE_AUDIT.ENABLED"
        )
    logging.info(
        "DUET first-cycle prior: enabled={}; power={:.3f}".format(
            bool(first_cycle_prior),
            float(cfg.DUET_FCP.POWER) if first_cycle_prior else 0.0,
        )
    )
    logging.info(
        "DUET boundary router: enabled={}; first_cycle_only=True; top_fraction={:.3f}".format(
            bool(boundary_router),
            boundary_fraction,
        )
    )
    logging.info(
        "DUET attribute reliability KL: enabled={}; first_cycle_only=True; "
        "target=unresolved_conflicts".format(bool(attribute_reliability_kl))
    )
    logging.info(
        "DUET support-conditioned CLIP: enabled={}; first_cycle_only=True; "
        "support=task_clip_top2_union; target=unresolved_conflicts".format(
            bool(support_conditioned_clip)
        )
    )
    logging.info(
        "DUET support-conditioned CLIP memory: enabled={}; "
        "memory=cycle1_task_clip_conflicts; target=currently_unresolved; "
        "support=task_clip_top2_union".format(
            bool(support_conditioned_clip_memory)
        )
    )
    logging.info(
        "DUET CLIP-confidence delay: enabled={}; first_cycle_only=True; "
        "per_pseudo_class_fraction={:.3f}".format(
            bool(clip_confidence_delay),
            clip_delay_fraction,
        )
    )
    logging.info(
        "DUET exact PCGrad parameter audit: enabled={}; cycle=2; "
        "pure_arithmetic_duet=True; optimizer_updates_in_audit=0".format(
            parameter_audit
        )
    )
    logging.info(
        "DUET PCGrad compatibility: enabled={}; active_cycle=2; "
        "fraction=clip(dot(full_duet_grad,pcgrad_correction)/"
        "norm2(pcgrad_correction),0,1); target_labels=False; "
        "fitted_thresholds=False".format(bool(pcgrad_compatibility))
    )
    logging.info(
        "DUET Top-k conflict probe: enabled={}; probability_stage=pre_first_cycle_prior; "
        "target_labels=oracle_diagnostic_only".format(bool(topk_conflict_probe))
    )
    logging.info(
        "DUET swap-conflict selection: enabled={}; scope=bidirectional_cross_support; "
        "gate_D={:.2f}; cycle0=always_clip_top1; abstain=not_in_loss; "
        "probability_stage=pre_first_cycle_prior".format(
            bool(swap_conflict_selection),
            swap_gate_D,
        )
    )
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
    trainable_parameters = tuple(
        parameter
        for model in (netF, netB)
        for parameter in model.parameters()
        if parameter.requires_grad
    )

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
    text_features = None
    attribute_text_features = None
    if attribute_reliability_kl:
        with open(cfg.name_file) as handle:
            class_names = [
                line.strip().replace("_", " ") for line in handle if line.strip()
            ]
        prompt_manifest = build_visda_attribute_prompt_manifest(class_names)
        device = next(clip_model.parameters()).device
        attribute_tokens = torch.cat(
            [clip.tokenize(prompt) for prompt in prompt_manifest["flat_prompts"]]
        ).to(device)
        with torch.no_grad():
            text_features = F.normalize(
                clip_model.encode_text(text_inputs), dim=1
            ).detach()
            attribute_text_features = F.normalize(
                clip_model.encode_text(attribute_tokens), dim=1
            ).detach()
        class_count, template_count, family_count = prompt_manifest["shape"]
        attribute_text_features = attribute_text_features.reshape(
            class_count,
            template_count,
            family_count,
            -1,
        )
        logging.info(
            "DUET attribute text contract: classes={}; templates={}; families={}; "
            "target_labels=False; fitted_thresholds=False".format(
                class_count,
                template_count,
                family_count,
            )
        )
    curr_cycle = 0
    # office-home : 1.0 / VisDA-C : 1.05
    q_value = cfg.ACTIVE.Q_VALUE
    print(f"train_clip")
    while curr_cycle < cfg.ACTIVE.CYCLE:
        iter_num = 0

        netF.eval()
        netB.eval()
        # netC.eval()
        parameter_audit_this_cycle = parameter_audit and curr_cycle + 1 == 2
        diagnostic_payload_requested = bool(
            cfg.FAILURE_AUDIT.ENABLED or parameter_audit_this_cycle
        )
        label_result = obtain_label(
            dset_loaders['test_aug'], netF, netB, netC, text_inputs, text_features, clip_model, prev_label_mask,
            curr_cycle,
            return_diagnostics=diagnostic_payload_requested,
            first_cycle_prior=first_cycle_prior,
            prior_power=(
                float(cfg.DUET_FCP.POWER) if first_cycle_prior else 0.0
            ),
            prior_epsilon=float(cfg.ACTIVE.EPSILON),
            boundary_router=boundary_router,
            boundary_top_fraction=boundary_fraction,
            attribute_reliability_kl=attribute_reliability_kl,
            attribute_text_features=attribute_text_features,
            support_conditioned_clip=support_conditioned_clip,
            support_conditioned_clip_memory=support_conditioned_clip_memory,
            clip_confidence_delay=clip_confidence_delay,
            clip_delay_fraction=clip_delay_fraction,
            topk_conflict_probe=topk_conflict_probe,
            probe_cfg=cfg,
            swap_conflict_selection=swap_conflict_selection,
            swap_gate_D=swap_gate_D,
        )
        if diagnostic_payload_requested:
            (
                mem_label,
                label_mask,
                confi_imag,
                confi_dis,
                kl_soft,
                audit_payload,
            ) = label_result
        else:
            mem_label, label_mask, confi_imag, confi_dis, kl_soft = label_result
        if cfg.FAILURE_AUDIT.ENABLED:
            save_failure_audit_snapshot(
                cfg,
                f"pre_cycle{curr_cycle + 1:02d}.npz",
                cycle=np.array(curr_cycle + 1, dtype=np.int64),
                task=np.array(cfg.name),
                phase=np.array("pre_cycle"),
                **audit_payload,
            )
            if stop_after_pre_cycle == curr_cycle + 1:
                logging.info(
                    "Failure audit stop: after_pre_cycle={}; optimizer_steps_in_cycle=0".format(
                        curr_cycle + 1
                    )
                )
                return netF, netB, netC
        if parameter_audit_this_cycle:
            run_exact_pcgrad_parameter_audit(
                cfg,
                netF=netF,
                netB=netB,
                netC=netC,
                target_dataset=dset_loaders["target"].dataset,
                mem_label=mem_label,
                label_mask=label_mask,
                kl_soft=kl_soft,
                audit_payload={
                    "source_label": audit_payload["source_label"],
                    "clip_label": audit_payload["clip_label"],
                },
            )
            logging.info(
                "PCGrad exact parameter audit stop: after_pre_cycle=2; "
                "cycle2_optimizer_steps=0; parameters_updated_by_audit=False"
            )
            return netF, netB, netC
        kl_soft = kl_soft.cuda()
        mem_label = mem_label.cuda()
        prev_label_mask = label_mask

        # clip_optimizer = train_clip_lr(cfg, clip_model, confi_imag, confi_dis, text_inputs, clip_optimizer, curr_cycle)
        clip_optimizer, q_value = train_clip(cfg, clip_model, confi_imag, confi_dis, text_inputs, clip_optimizer,
                                             q_value)

        cfg.load = 'prompt_model.pt'
        # mem_label = torch.from_numpy(mem_label).cuda()
        netF.train()
        netB.train()
        # netC.train()
        compatibility_batches = 0
        compatibility_applied_batches = 0
        compatibility_unresolved = 0
        compatibility_output_active = 0
        compatibility_fraction_sum = 0.0
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

            filtered_idx = tar_idx[label_mask[tar_idx]]

            con_loss = consistency_loss(weak_logits, strong_logits)
            classifier_loss = con_loss * cfg.ACTIVE.CON_PAR
            # classifier_loss = metric_loss * cfg.ACTIVE.CLS_PAR
            if cfg.ACTIVE.CLS_PAR > 0:
                pred = mem_label[filtered_idx]
                supervised_logits = weak_logits[label_mask[tar_idx]]
                if pred.size(0) != 0:
                    classifier_loss += nn.CrossEntropyLoss()(supervised_logits, pred) * cfg.ACTIVE.CLS_PAR
            # pseudo_output = weak_preds[filtered_idx]
            clip_soft_batch = kl_soft[tar_idx]
            # mixed_soft_batch = confi_dis[tar_idx].cuda()
            # mi_loss = F.kl_div(weak_preds.log(), mixed_soft_batch, reduction="batchmean")
            mi_loss = F.kl_div(weak_preds.log(), clip_soft_batch, reduction="batchmean")
            classifier_loss += mi_loss * cfg.ACTIVE.KL_PAR

            correction_payload = None
            if pcgrad_compatibility and curr_cycle == 1:
                unresolved_batch = (~label_mask[tar_idx]).to(
                    weak_logits.device, non_blocking=True
                )
                correction_payload = build_pcgrad_parameter_correction(
                    weak_logits=weak_logits,
                    strong_logits=strong_logits,
                    weak_probability=weak_preds,
                    strong_probability=nn.Softmax(dim=1)(strong_logits),
                    clip_target=clip_soft_batch,
                    unresolved_mask=unresolved_batch,
                    parameters=trainable_parameters,
                    consistency_weight=float(cfg.ACTIVE.CON_PAR),
                    clip_weight=float(cfg.ACTIVE.KL_PAR),
                )

            optimizer.zero_grad()
            classifier_loss.backward()
            if correction_payload is not None:
                compatibility = merge_compatible_parameter_correction_(
                    trainable_parameters,
                    correction_payload["parameter_correction"],
                )
                compatibility_batches += 1
                compatibility_applied_batches += int(
                    compatibility["fraction"] > 0.0
                )
                compatibility_unresolved += correction_payload["unresolved"]
                compatibility_output_active += correction_payload[
                    "output_pcgrad_active"
                ]
                compatibility_fraction_sum += compatibility["fraction"]
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
        if pcgrad_compatibility:
            mean_fraction = (
                compatibility_fraction_sum / compatibility_batches
                if compatibility_batches
                else 0.0
            )
            logging.info(
                "DUET PCGrad compatibility cycle summary: cycle={}; active={}; "
                "audited_batches={}; applied_batches={}; unresolved_rows={}; "
                "output_pcgrad_active_rows={}; mean_fraction={:.6f}; "
                "target_labels=False; fitted_thresholds=False".format(
                    curr_cycle + 1,
                    curr_cycle == 1,
                    compatibility_batches,
                    compatibility_applied_batches,
                    compatibility_unresolved,
                    compatibility_output_active,
                    mean_fraction,
                )
            )
        curr_cycle += 1

    if cfg.FAILURE_AUDIT.ENABLED:
        final_payload = collect_final_failure_audit(dset_loaders['test'], netF, netB, netC)
        save_failure_audit_snapshot(
            cfg,
            "final_full.npz",
            cycle=np.array(cfg.ACTIVE.CYCLE, dtype=np.int64),
            task=np.array(cfg.name),
            phase=np.array("final_full"),
            **final_payload,
        )

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


def obtain_label(
    loader,
    netF,
    netB,
    netC,
    text_inputs,
    text_features,
    clip_model,
    prev_label_mask,
    curr_cycle,
    return_diagnostics=False,
    first_cycle_prior=False,
    prior_power=0.5,
    prior_epsilon=1e-6,
    boundary_router=False,
    boundary_top_fraction=0.2,
    attribute_reliability_kl=False,
    attribute_text_features=None,
    support_conditioned_clip=False,
    support_conditioned_clip_memory=False,
    clip_confidence_delay=False,
    clip_delay_fraction=LOCKED_DELAY_FRACTION,
    topk_conflict_probe=False,
    probe_cfg=None,
    swap_conflict_selection=False,
    swap_gate_D=4.0,
):
    # class_logit_bias = get_class_bias(netF, netB, netC)
    start_test = True
    collect_boundary = bool(boundary_router and curr_cycle == 0)
    collect_attribute = bool(attribute_reliability_kl and curr_cycle == 0)
    collect_support_conditioned = bool(
        (support_conditioned_clip and curr_cycle == 0)
        or support_conditioned_clip_memory
    )
    collect_sample_indices = bool(return_diagnostics or topk_conflict_probe)
    if collect_attribute and (text_features is None or attribute_text_features is None):
        raise ValueError("attribute-reliability KL requires fixed text features")
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            inputs_test, labels, sample_index = next(iter_test)
            weak_x = inputs_test[1].cuda()

            weak_feas = netB(netF(weak_x))

            weak_outputs = netC(weak_feas)
            if return_diagnostics:
                # 审计模式额外记录 strong-view 预测，用于判断后期模型对
                # 首轮伪标签的反对是否稳定；默认训练路径不会执行这次前向。
                strong_x = inputs_test[2].cuda()
                strong_outputs = netC(netB(netF(strong_x)))

            if collect_attribute:
                clip_image_feature = F.normalize(
                    clip_model.encode_image(weak_x), dim=1
                )
                clip_score = (
                    clip_model.logit_scale.exp()
                    * clip_image_feature
                    @ text_features.t()
                )
                batch_attribute_margin = pairwise_attribute_margin(
                    clip_image_feature,
                    attribute_text_features,
                    weak_outputs.argmax(dim=1),
                    clip_score.argmax(dim=1),
                )
            elif text_features is not None:
                clip_score = clip_text(clip_model, text_features, weak_x)
            else:
                clip_score, _ = clip_model(weak_x, text_inputs)

            if collect_boundary:
                batch_task_pred = weak_outputs.argmax(dim=1)
                batch_clip_pred = clip_score.argmax(dim=1)
                batch_conflict = batch_task_pred != batch_clip_pred
                batch_task_radius = torch.zeros(
                    weak_x.size(0), dtype=torch.float32, device=weak_x.device
                )
                batch_clip_radius = torch.zeros_like(batch_task_radius)
                conflict_position = torch.nonzero(
                    batch_conflict, as_tuple=False
                ).flatten()
                if conflict_position.numel() > 0:
                    conflict_task_pred = batch_task_pred[conflict_position]
                    conflict_clip_pred = batch_clip_pred[conflict_position]
                    with torch.enable_grad():
                        task_x = weak_x[conflict_position].detach().requires_grad_(True)
                        task_boundary_logits = netC(netB(netF(task_x)))
                        task_radius, _, _ = pairwise_first_order_boundary(
                            task_boundary_logits,
                            task_x,
                            conflict_task_pred,
                            conflict_clip_pred,
                        )
                        clip_x = weak_x[conflict_position].detach().requires_grad_(True)
                        if text_features is None:
                            clip_boundary_logits, _ = clip_model(clip_x, text_inputs)
                        else:
                            clip_boundary_features = F.normalize(
                                clip_model.encode_image(clip_x), dim=1
                            )
                            clip_boundary_logits = (
                                clip_model.logit_scale.exp()
                                * clip_boundary_features
                                @ text_features.t()
                            )
                        clip_radius, _, _ = pairwise_first_order_boundary(
                            clip_boundary_logits,
                            clip_x,
                            conflict_clip_pred,
                            conflict_task_pred,
                        )
                    batch_task_radius[conflict_position] = task_radius.float()
                    batch_clip_radius[conflict_position] = clip_radius.float()

            clip_score = clip_score.cpu()
            if start_test:
                all_output = weak_outputs.float().cpu()
                all_clip_score = clip_score.float().cpu()
                all_label = labels.float()
                if collect_sample_indices:
                    all_sample_index = sample_index.long().cpu()
                if collect_boundary:
                    all_task_radius = batch_task_radius.cpu()
                    all_clip_radius = batch_clip_radius.cpu()
                if collect_attribute:
                    all_attribute_margin = batch_attribute_margin.float().cpu()
                if return_diagnostics:
                    all_task_features = weak_feas.float().cpu()
                    all_strong_output = strong_outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, weak_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                if collect_sample_indices:
                    all_sample_index = torch.cat(
                        (all_sample_index, sample_index.long().cpu()), 0
                    )
                if collect_boundary:
                    all_task_radius = torch.cat(
                        (all_task_radius, batch_task_radius.cpu()), 0
                    )
                    all_clip_radius = torch.cat(
                        (all_clip_radius, batch_clip_radius.cpu()), 0
                    )
                if collect_attribute:
                    all_attribute_margin = torch.cat(
                        (
                            all_attribute_margin,
                            batch_attribute_margin.float().cpu(),
                        ),
                        0,
                    )
                if return_diagnostics:
                    all_task_features = torch.cat(
                        (all_task_features, weak_feas.float().cpu()), 0
                    )
                    all_strong_output = torch.cat(
                        (all_strong_output, strong_outputs.float().cpu()), 0
                    )

    all_output = nn.Softmax(dim=1)(all_output)
    clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
    if topk_conflict_probe:
        if probe_cfg is None:
            raise ValueError("Top-k conflict probe requires the active config")
        with open(probe_cfg.name_file) as handle:
            probe_class_names = [line.strip() for line in handle if line.strip()]
        with torch.no_grad():
            probe_summary = write_topk_conflict_probe(
                output_root=probe_cfg.output_dir,
                task_probability=all_output.detach(),
                clip_probability=clip_all_output.detach(),
                labels=all_label.detach(),
                sample_indices=all_sample_index.detach(),
                dataset_items=loader.dataset.imgs,
                class_names=probe_class_names,
                task_name=str(probe_cfg.name),
                source_domain=str(probe_cfg.domain[probe_cfg.SETTING.S]),
                target_domain=str(probe_cfg.domain[probe_cfg.SETTING.T]),
                seed=int(probe_cfg.SETTING.SEED),
                cycle=int(curr_cycle),
                probability_stage="pre_first_cycle_prior",
            )
        probe_overall = probe_summary["overall"]
        logging.info(
            "DUET Top-k conflict probe: cycle={}; conflicts={}; "
            "top1_union={:.4f}%; top2_union={:.4f}%; recovered={}; "
            "ground_truth_affects_training=False".format(
                curr_cycle + 1,
                probe_overall["conflict_samples"],
                probe_overall["top1_union_coverage"],
                probe_overall["top2_union_coverage"],
                probe_overall["top2_recovered_count"],
            )
        )
    # Swap-conflict hard-label selection uses the pre-prior probabilities
    # (same probability_stage as the Top-k probe export), so decisions match
    # the archived offline analysis exactly.  Ground truth is used only for
    # the evaluation log below, never to build the labels.
    swap_selection_payload = None
    if swap_conflict_selection:
        with torch.no_grad():
            swap_labels, swap_selected = select_swap_labels(
                all_output.detach(),
                clip_all_output.detach(),
                cycle=int(curr_cycle),
                gate_D=float(swap_gate_D),
            )
        swap_selection_payload = {
            "labels": torch.from_numpy(swap_labels).long(),
            "selected": torch.from_numpy(swap_selected).bool(),
        }
        swap_stats = summarize_swap_decisions(
            all_output.detach(),
            clip_all_output.detach(),
            all_label,
            cycle=int(curr_cycle),
            gate_D=float(swap_gate_D),
        )
        logging.info(
            "DUET swap-conflict selection: cycle={}; swap_conflicts={}; "
            "decisions={}; abstain={}; correct={}; precision={:.2f}%; "
            "gate_D={:.2f}; ground_truth_eval_only=True".format(
                curr_cycle + 1,
                swap_stats["swap_conflicts"],
                swap_stats["decisions"],
                swap_stats["abstain"],
                swap_stats["correct"],
                swap_stats["precision_pct"],
                float(swap_gate_D),
            )
        )
    if return_diagnostics:
        strong_task_prob = nn.Softmax(dim=1)(all_strong_output)
    if first_cycle_prior:
        all_output, clip_all_output, prior_active = apply_first_cycle_prior(
            all_output,
            clip_all_output,
            curr_cycle=curr_cycle,
            power=prior_power,
            epsilon=prior_epsilon,
        )
        logging.info(
            "DUET first-cycle prior schedule: cycle={}; active={}".format(
                curr_cycle + 1, prior_active
            )
        )

    # Compute predictions for all_output and clip_all_output
    _, all_output_pred = torch.max(all_output, dim=1)
    _, clip_all_output_pred = torch.max(clip_all_output, dim=1)

    # Find indices where predictions match
    matching_indices = all_output_pred == clip_all_output_pred

    admission_matching = matching_indices
    if clip_confidence_delay and curr_cycle == 0:
        delay = class_balanced_clip_confidence_delay(
            matching_indices,
            all_output_pred,
            clip_all_output,
            fraction=clip_delay_fraction,
        )
        admission_matching = delay["retained_matching"]
        delayed = delay["delayed"]
        delayed_confidence = clip_all_output[
            delayed, all_output_pred[delayed]
        ]
        logging.info(
            "DUET CLIP-confidence delay applied: cycle=1; "
            "original_agreements={}; delayed={}; retained={}; "
            "mean_delayed_clip_confidence={:.6f}; target_labels=False; "
            "fitted_thresholds=False".format(
                int(matching_indices.sum().item()),
                int(delayed.sum().item()),
                int(admission_matching.sum().item()),
                float(delayed_confidence.mean().item()),
            )
        )
    elif clip_confidence_delay:
        logging.info(
            "DUET CLIP-confidence delay applied: cycle={}; delayed=0; "
            "first_cycle_only=True".format(curr_cycle + 1)
        )

    # Update label mask based on previous label mask
    if prev_label_mask is not None:
        label_mask = prev_label_mask | (~prev_label_mask & admission_matching)
    else:
        label_mask = admission_matching
    if swap_selection_payload is not None:
        # Admit gate-passing swap samples (cycle 0 admits all swaps).
        # Abstained swap samples stay out of label_mask and therefore out of
        # the hard-label CE loss; every other sample is untouched.
        label_mask = label_mask | swap_selection_payload["selected"]

    kl_soft_output = clip_all_output
    if collect_attribute:
        active_conflict = (~label_mask) & (~matching_indices)
        conflict_count = int(active_conflict.sum().item())
        if conflict_count <= 0:
            raise RuntimeError("attribute-reliability KL found no active conflicts")
        reliability = entropy_anchored_attribute_target(
            all_output[active_conflict],
            clip_all_output[active_conflict],
            all_output_pred[active_conflict].long(),
            clip_all_output_pred[active_conflict].long(),
            all_attribute_margin[active_conflict],
            clip_logit_scale=float(
                clip_model.logit_scale.exp().detach().cpu().item()
            ),
        )
        kl_soft_output = clip_all_output.clone()
        kl_soft_output[active_conflict] = reliability["probability"]
        changed_top1 = int(
            (
                reliability["probability"].argmax(dim=1)
                != clip_all_output_pred[active_conflict]
            )
            .sum()
            .item()
        )
        logging.info(
            "DUET attribute reliability KL applied: cycle=1; "
            "active_conflicts={}; changed_top1={}; mean_weight={:.6f}; "
            "target_labels=False; fitted_thresholds=False".format(
                conflict_count,
                changed_top1,
                float(reliability["attribute_weight"].mean().item()),
            )
        )
    elif attribute_reliability_kl:
        logging.info(
            "DUET attribute reliability KL applied: cycle={}; "
            "active_conflicts=0; changed_top1=0; mean_weight=0.000000; "
            "first_cycle_only=True".format(curr_cycle + 1)
        )
    elif collect_support_conditioned:
        active_conflict = (~label_mask) & (~matching_indices)
        conflict_count = int(active_conflict.sum().item())
        if conflict_count <= 0:
            raise RuntimeError("support-conditioned CLIP found no active conflicts")
        conditioned = condition_clip_on_task_clip_top2_union(
            all_output[active_conflict],
            clip_all_output[active_conflict],
        )
        kl_soft_output = clip_all_output.clone()
        kl_soft_output[active_conflict] = conditioned["probability"]
        changed_top1 = int(
            (
                conditioned["probability"].argmax(dim=1)
                != clip_all_output_pred[active_conflict]
            )
            .sum()
            .item()
        )
        logging.info(
            "DUET support-conditioned CLIP applied: cycle={}; "
            "active_conflicts={}; changed_top1={}; mean_support_size={:.6f}; "
            "mean_retained_clip_mass={:.6f}; target_labels=False; "
            "fitted_thresholds=False".format(
                curr_cycle + 1,
                conflict_count,
                changed_top1,
                float(conditioned["support_size"].float().mean().item()),
                float(conditioned["retained_clip_mass"].mean().item()),
            )
        )
    elif support_conditioned_clip or support_conditioned_clip_memory:
        logging.info(
            "DUET support-conditioned CLIP applied: cycle={}; "
            "active_conflicts=0; changed_top1=0; mean_support_size=0.000000; "
            "mean_retained_clip_mass=0.000000; first_cycle_only={}".format(
                curr_cycle + 1,
                bool(support_conditioned_clip),
            )
        )

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
    # Combine outputs for confidence distribution and other uses

    all_mix_output = (all_output + clip_all_output) / 2.0
    boundary_payload = None
    if collect_boundary:
        active_conflict = (~label_mask) & (~matching_indices)
        (
            all_mix_output,
            boundary_selected,
            boundary_choose_task,
            boundary_separation,
        ) = route_conflict_probabilities(
            all_output,
            clip_all_output,
            active_conflict,
            all_task_radius,
            all_clip_radius,
            fraction=boundary_top_fraction,
        )
        boundary_payload = {
            "boundary_selected": boundary_selected,
            "boundary_choose_task": boundary_choose_task,
            "boundary_separation": boundary_separation,
            "task_boundary_radius": all_task_radius,
            "clip_boundary_radius": all_clip_radius,
        }
        logging.info(
            "DUET boundary routing: cycle={}; active_conflicts={}; selected={}; "
            "choose_task={}; choose_clip={}".format(
                curr_cycle + 1,
                int(active_conflict.sum().item()),
                int(boundary_selected.sum().item()),
                int((boundary_selected & boundary_choose_task).sum().item()),
                int((boundary_selected & ~boundary_choose_task).sum().item()),
            )
        )
    elif boundary_router:
        logging.info(
            "DUET boundary routing: cycle={}; active_conflicts=0; selected=0; "
            "choose_task=0; choose_clip=0".format(curr_cycle + 1)
        )

    _, all_mix_output_pred = torch.max(all_mix_output, dim=1)
    if swap_selection_payload is not None:
        # Override the mixed argmax with the swap rule's chosen side so the
        # hard pseudo labels (mem_label) equal A or B for admitted swaps.
        selected = swap_selection_payload["selected"]
        all_mix_output_pred[selected] = swap_selection_payload["labels"][selected]
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

    result = (
        all_mix_output_pred,
        label_mask,
        confi_imag,
        confi_dis,
        kl_soft_output,
    )
    if not return_diagnostics:
        return result
    audit_payload = {
        "mix_label": all_mix_output_pred.long(),
        "label_mask": label_mask.bool(),
        "source_label": all_output_pred.long(),
        "clip_label": clip_all_output_pred.long(),
        "task_prob": all_output.float(),
        "clip_prob": clip_all_output.float(),
        "strong_task_prob": strong_task_prob.float(),
        "target_label": all_label.long(),
        "task_feature": all_task_features.float(),
        "sample_index": all_sample_index.long(),
    }
    if boundary_payload is not None:
        audit_payload.update(boundary_payload)
    return result + (audit_payload,)


def collect_final_failure_audit(loader, netF, netB, netC):
    """Collect deterministic full-target features and source-head predictions."""
    features = []
    probabilities = []
    labels = []
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        for data in loader:
            inputs = data[0].cuda()
            task_feature = netB(netF(inputs))
            task_prob = F.softmax(netC(task_feature), dim=1)
            features.append(task_feature.float().cpu())
            probabilities.append(task_prob.float().cpu())
            labels.append(data[1].long().cpu())
    task_feature = torch.cat(features, dim=0)
    task_prob = torch.cat(probabilities, dim=0)
    target_label = torch.cat(labels, dim=0)
    return {
        "target_label": target_label,
        "task_feature": task_feature,
        "base_task_prob": task_prob,
        "task_prob": task_prob,
        "source_label": task_prob.argmax(dim=1),
    }


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
    logit_scale = model.logit_scale.detach().exp()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits
