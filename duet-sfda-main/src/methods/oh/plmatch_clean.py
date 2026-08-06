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
import clip

from torchvision import transforms
from src.utils import loss
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList, ImageList_idx
from sklearn.metrics import confusion_matrix
from src.utils.utils import *
from src.data.data_list import *
from src.utils import loss, IID_losses
# from src.utils import loss, active_prompt, IID_losses
# from proposed_method import *
from torch.nn.functional import normalize
from data.datautils_domain import build_dataset
from data.cls_to_names import *
from data.domain_datasets import domain_datasets
from src.utils.adaptation_lists import load_adaptation_and_evaluation_rows
from src.utils.first_cycle_prior import apply_first_cycle_prior
from src.utils.swap_conflict_selection import (
    select_swap_labels,
    summarize_swap_decisions,
)
from src.utils.swap_intervention_audit import (
    SwapInterventionAuditor,
    build_swap_audit_payload,
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
    swap_conflict_selection=False,
):
    """训练原始 DUET，或启用一个明确隔离的候选改动。

    ``first_cycle_prior=False`` 是发布版 DUET 的原始路径。DUET-FCP 入口只把
    该参数设为 ``True``；stable memory、target head、graph teacher 和 GTR
    均不在本文件中，因此不会混入候选。

    ``swap_conflict_selection=True`` 只对 bidirectional_cross_support
    （纯 swap）冲突产生硬伪标签：cycle 0 直接取 CLIP top1；cycle >= 1 按
    eA=pA*qA、eB=pB*qB 的 log 差与 ``DUET_SWAP.GATE_D`` 门槛决定 A/B 或
    abstain。abstain 样本不进入 label_mask，不进训练损失。检测与决策使用
    prior 校准前的概率（与 Top-k probe 导出口径一致）；非 swap 冲突和非
    冲突样本不进入该规则，原 DUET 策略不变。
    """
    if swap_conflict_selection and not cfg.DUET_SWAP.ENABLED:
        raise ValueError(
            "swap-conflict selection requires DUET_SWAP.ENABLED=True"
        )
    if swap_conflict_selection and not first_cycle_prior:
        raise ValueError(
            "swap-conflict selection requires first_cycle_prior=True"
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
    swap_min_direction_accuracy = float(
        cfg.DUET_SWAP.MIN_DIRECTION_ACCURACY
    )
    if swap_conflict_selection and not (
        0.0 <= swap_min_direction_accuracy <= 1.0
    ):
        raise ValueError(
            "DUET_SWAP.MIN_DIRECTION_ACCURACY must be in [0, 1]"
        )
    swap_last_active_cycle = int(cfg.DUET_SWAP.LAST_ACTIVE_CYCLE)
    if swap_conflict_selection and not (
        1 <= swap_last_active_cycle <= int(cfg.ACTIVE.CYCLE)
    ):
        raise ValueError(
            "DUET_SWAP.LAST_ACTIVE_CYCLE must be between 1 and "
            "ACTIVE.CYCLE"
        )
    swap_audit_enabled = bool(cfg.DUET_SWAP_AUDIT.ENABLED)
    if swap_audit_enabled and not swap_conflict_selection:
        raise ValueError(
            "DUET_SWAP_AUDIT.ENABLED requires swap-conflict selection"
        )
    if first_cycle_prior and cfg.DUET_FCP.POWER < 0:
        raise ValueError("DUET_FCP.POWER must be non-negative")
    logging.info(
        "DUET first-cycle prior: enabled={}; power={:.3f}".format(
            bool(first_cycle_prior),
            float(cfg.DUET_FCP.POWER) if first_cycle_prior else 0.0,
        )
    )
    logging.info(
        "DUET swap-conflict selection: enabled={}; scope=bidirectional_cross_support; "
        "gate_D={:.2f}; min_direction_accuracy={:.2f}; last_active_cycle={}; "
        "cycle0=always_clip_top1; "
        "abstain=not_in_loss; probability_stage=pre_first_cycle_prior".format(
            bool(swap_conflict_selection),
            swap_gate_D,
            swap_min_direction_accuracy,
            swap_last_active_cycle,
        )
    )
    logging.info(
        "DUET swap-intervention audit: enabled={}; cycles=2,3; "
        "read_only=True; ground_truth=diagnostic_only".format(
            bool(swap_audit_enabled)
        )
    )
    clip_model, preprocess, _ = clip.load(cfg.ACTIVE.ARCH)
    clip_model.float()
    text_inputs = clip_pre_text(cfg)

    dset_loaders = data_load(cfg)
    swap_auditor = None
    if swap_audit_enabled:
        with open(cfg.name_file) as handle:
            audit_class_names = [
                line.strip() for line in handle if line.strip()
            ]
        swap_auditor = SwapInterventionAuditor(
            output_root=cfg.output_dir,
            class_names=audit_class_names,
        )
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
    curr_cycle = 0
    # office-home : 1.0 / VisDA-C : 1.05
    q_value = cfg.ACTIVE.Q_VALUE
    print(f"train_clip")
    while curr_cycle < cfg.ACTIVE.CYCLE:
        iter_num = 0

        netF.eval()
        netB.eval()
        # netC.eval()
        label_result = obtain_label(
            dset_loaders['test_aug'], netF, netB, netC, text_inputs, text_features, clip_model, prev_label_mask,
            curr_cycle,
            first_cycle_prior=first_cycle_prior,
            prior_power=(
                float(cfg.DUET_FCP.POWER) if first_cycle_prior else 0.0
            ),
            prior_epsilon=float(cfg.ACTIVE.EPSILON),
            swap_conflict_selection=swap_conflict_selection,
            swap_gate_D=swap_gate_D,
            swap_min_direction_accuracy=swap_min_direction_accuracy,
            swap_last_active_cycle=swap_last_active_cycle,
            swap_audit_enabled=swap_audit_enabled,
            swap_auditor=swap_auditor,
            swap_audit_probe_cfg=cfg,
        )
        mem_label, label_mask, confi_imag, confi_dis, kl_soft = label_result
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
    first_cycle_prior=False,
    prior_power=0.5,
    prior_epsilon=1e-6,
    swap_conflict_selection=False,
    swap_gate_D=4.0,
    swap_min_direction_accuracy=0.0,
    swap_last_active_cycle=8,
    swap_audit_enabled=False,
    swap_auditor=None,
    swap_audit_probe_cfg=None,
):
    # class_logit_bias = get_class_bias(netF, netB, netC)
    start_test = True
    collect_sample_indices = bool(
        swap_audit_enabled
    )
    collect_strong = bool(swap_audit_enabled)
    collect_features = bool(swap_audit_enabled)
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            inputs_test, labels, sample_index = next(iter_test)
            weak_x = inputs_test[1].cuda()

            weak_feas = netB(netF(weak_x))

            weak_outputs = netC(weak_feas)
            if collect_strong:
                # 审计/failure 模式额外记录 strong-view 特征与预测；
                # 默认训练路径不会执行这次前向，避免影响训练状态。
                strong_x = inputs_test[2].cuda()
                strong_feas = netB(netF(strong_x))
                strong_outputs = netC(strong_feas)

            if text_features is not None:
                clip_score = clip_text(clip_model, text_features, weak_x)
            else:
                clip_score, _ = clip_model(weak_x, text_inputs)

            clip_score = clip_score.cpu()
            if start_test:
                all_output = weak_outputs.float().cpu()
                all_clip_score = clip_score.float().cpu()
                all_label = labels.float()
                if collect_sample_indices:
                    all_sample_index = sample_index.long().cpu()
                if collect_features:
                    all_task_features = weak_feas.float().cpu()
                    all_strong_features = strong_feas.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, weak_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                if collect_sample_indices:
                    all_sample_index = torch.cat(
                        (all_sample_index, sample_index.long().cpu()), 0
                    )
                if collect_features:
                    all_task_features = torch.cat(
                        (all_task_features, weak_feas.float().cpu()), 0
                    )
                    all_strong_features = torch.cat(
                        (all_strong_features, strong_feas.float().cpu()), 0
                    )

    all_output = nn.Softmax(dim=1)(all_output)
    clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
    # Swap-conflict hard-label selection uses the pre-prior probabilities
    # (same probability_stage as the Top-k probe export), so decisions match
    # the archived offline analysis exactly.  Ground truth is used only for
    # the evaluation log below, never to build the labels.
    swap_selection_payload = None
    swap_diagnostics = None
    if swap_conflict_selection:
        with torch.no_grad():
            swap_result = select_swap_labels(
                all_output.detach(),
                clip_all_output.detach(),
                cycle=int(curr_cycle),
                gate_D=float(swap_gate_D),
                min_direction_accuracy=float(swap_min_direction_accuracy),
                last_active_cycle=int(swap_last_active_cycle),
                return_diagnostics=bool(swap_audit_enabled),
            )
            if swap_audit_enabled:
                swap_labels, swap_selected, swap_diagnostics = swap_result
            else:
                swap_labels, swap_selected = swap_result
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
            min_direction_accuracy=float(swap_min_direction_accuracy),
            last_active_cycle=int(swap_last_active_cycle),
        )
        logging.info(
            "DUET swap-conflict selection: cycle={}; swap_conflicts={}; "
            "decisions={}; abstain={}; correct={}; precision={:.2f}%; "
            "gate_D={:.2f}; min_direction_accuracy={:.2f}; "
            "ground_truth_eval_only=True".format(
                curr_cycle + 1,
                swap_stats["swap_conflicts"],
                swap_stats["decisions"],
                swap_stats["abstain"],
                swap_stats["correct"],
                swap_stats["precision_pct"],
                float(swap_gate_D),
                float(swap_min_direction_accuracy),
            )
        )
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

    # Update label mask based on previous label mask
    if prev_label_mask is not None:
        label_mask = prev_label_mask | (~prev_label_mask & admission_matching)
    else:
        label_mask = admission_matching
    if swap_selection_payload is not None:
        # Audit snapshot of the mask before swap admission (Cycle 2/3 only).
        if swap_audit_enabled:
            base_label_mask = label_mask.clone()
        # Admit gate-passing swap samples (cycle 0 admits all swaps).
        # Abstained swap samples stay out of label_mask and therefore out of
        # the hard-label CE loss; every other sample is untouched.
        label_mask = label_mask | swap_selection_payload["selected"]
    elif swap_audit_enabled:
        base_label_mask = label_mask

    kl_soft_output = clip_all_output

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

    _, all_mix_output_pred = torch.max(all_mix_output, dim=1)
    base_mix_label = all_mix_output_pred.clone()
    if swap_selection_payload is not None:
        # Override the mixed argmax with the swap rule's chosen side so the
        # hard pseudo labels (mem_label) equal A or B for admitted swaps.
        selected = swap_selection_payload["selected"]
        all_mix_output_pred[selected] = swap_selection_payload["labels"][selected]
    if swap_audit_enabled and curr_cycle in (1, 2):
        if swap_auditor is None or swap_audit_probe_cfg is None:
            raise ValueError(
                "swap-intervention audit requires the auditor and config"
            )
        with open(swap_audit_probe_cfg.name_file) as handle:
            audit_class_names = [
                line.strip() for line in handle if line.strip()
            ]
        with torch.no_grad():
            audit_payload = build_swap_audit_payload(
                cycle=int(curr_cycle + 1),
                task_prob=all_output.detach(),
                clip_prob=clip_all_output.detach(),
                task_feat=all_task_features.detach(),
                strong_feat=all_strong_features.detach(),
                base_mix_label=base_mix_label.detach(),
                final_mem_label=all_mix_output_pred.detach(),
                base_label_mask=base_label_mask,
                final_label_mask=label_mask,
                prev_label_mask=prev_label_mask,
                current_agreement=admission_matching,
                swap_selected=swap_selection_payload["selected"],
                swap_diagnostics=swap_diagnostics,
                real_label=all_label.detach(),
                sample_index=all_sample_index.detach(),
                image_paths=[item[0] for item in loader.dataset.imgs],
                class_names=audit_class_names,
                gate_D=float(swap_gate_D),
                min_direction_accuracy=float(swap_min_direction_accuracy),
            )
        swap_auditor.record_cycle(int(curr_cycle), audit_payload)
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
    return result


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
