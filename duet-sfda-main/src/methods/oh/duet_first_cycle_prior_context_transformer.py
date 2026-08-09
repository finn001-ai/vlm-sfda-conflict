"""DUET-FCP + class-balanced anchor Context Transformer (candidate method).

这是一个自包含的 DUET 训练文件，只保留两条路径：

1. ``context_conflict_transformer=False``（或 ``DUET_CONTEXT.ENABLED=False``）
   -> 与原始 ``duet_first_cycle_prior`` 完全一致（仅首轮 both-prior）；
2. ``context_conflict_transformer=True`` + ``DUET_CONTEXT.ENABLED=True``
   -> 第一轮（cycle index 0）保持纯 DUET-FCP，不运行 Transformer；从
   ``DUET_CONTEXT.ACTIVE_CYCLES``（默认第 2 个 cycle，index 1）开始，
   用 Task/CLIP 高置信一致样本构建 class-balanced anchor bank，用轻量
   Cross-Attention Transformer（或 cosine-kNN / prototype 对照）对 strict
   conflict / weak-agreement 查询做确认、纠正或拒绝，然后按原始 DUET 规则
   进入硬 CE / consistency / CLIP soft KL。这样 anchor 与 agreement 建立在
   至少一轮 Task/CLIP target 适配之后，避免第一轮基线未形成时引入噪声。

本文件不包含也不允许混入其他 DUET 候选方法（boundary router、attribute
reliability、support-conditioned CLIP、CLIP confidence delay、PCGrad、
Top-k probe、swap-conflict selection 及其 Gate D 规则等）。

Builds upon: https://github.com/tim-learn/SHOT
Corresponding paper: http://proceedings.mlr.press/v119/liang20a/liang20a.pdf
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import clip

from torchvision import transforms
from src.utils import loss, prompt_tuning, IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList, ImageList_idx
from sklearn.metrics import confusion_matrix
from src.utils.utils import *
from src.data.data_list import *
from data.datautils_domain import build_dataset
from data.domain_datasets import domain_datasets
from src.utils.adaptation_lists import load_adaptation_and_evaluation_rows
from src.utils.first_cycle_prior import apply_first_cycle_prior
from src.utils.duet_context import (
    ComparatorReplayMemory,
    DuetContextConflictTransformer,
    PairwiseConflictComparator,
    run_context_refinement,
)

logger = logging.getLogger(__name__)


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
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
                    p=0.8,
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
    """返回 test/weak/strong 三种增广的 NCrops 变换。"""
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
    return NCropsTransform(transform_list)


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(cfg):
    """准备 target / test / test_aug 三个 loader。"""
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

    train_transform = get_augmentation_versions(cfg)

    dsets["target"] = ImageList_idx(txt_tar, transform=train_transform)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False,
                                      num_workers=cfg.NUM_WORKERS, drop_last=False)
    # tar_idx 指向 test_aug 生成的伪标签；两个 loader 必须使用同一顺序的
    # adaptation 行，只有 test 保持全量。
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
    """weak/strong 一致性：KL(strong || weak)。"""
    weak_probs = nn.Softmax(dim=1)(weak_output)
    strong_probs = nn.Softmax(dim=1)(strong_output)
    return F.kl_div(strong_probs.log(), weak_probs, reduction="batchmean")


def train_clip(cfg, model, confi_imag, confi_dis, text_features, clip_optimizer, q_value):
    """原 DUET 的 CLIP visual 自训练（IID/tsallis 互信息）。"""
    if cfg.SETTING.DATASET in domain_datasets:
        cfg.domain_name = cfg.domain[cfg.SETTING.T]
        classnames = cfg.classname

    if 'RN' in cfg.DIFO.ARCH:
        data_transform = prompt_tuning.image_test_50()
    else:
        data_transform = image_test()

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

        predicted_labels = clip_preds.argmax(dim=1)
        correct = (predicted_labels == target).sum().item()
        total_corrects += correct
        total_samples += target.size(0)

        clip_optimizer.zero_grad()
        loss.backward()
        clip_optimizer.step()

    avg_acc = total_corrects / total_samples if total_samples > 0 else 0.0
    logging.info('CLIP visual Accuracy = {:.2f}%;'.format(avg_acc * 100))

    return clip_optimizer, q_value


def train_target(
    cfg,
    *,
    first_cycle_prior=True,
    context_conflict_transformer=True,
):
    """训练 DUET-FCP（默认），或在其上启用 Context Transformer 候选。

    - ``first_cycle_prior=True`` 是基线（原始 duet_first_cycle_prior）；
    - ``context_conflict_transformer=True`` 且 ``cfg.DUET_CONTEXT.ENABLED=True``
      时，从 ``cfg.DUET_CONTEXT.ACTIVE_CYCLES``（默认 index 1 = 第 2 个
      cycle）开始运行 anchor bank + Context Transformer；第一轮（index 0）
      保持纯 DUET-FCP，不运行 Transformer；
    - 关闭 Context Transformer 后完全退化为基线，无任何其他候选混入。
    """
    if not first_cycle_prior:
        raise ValueError(
            "duet_first_cycle_prior_context_transformer requires "
            "first_cycle_prior=True"
        )
    if first_cycle_prior and float(cfg.DUET_FCP.POWER) < 0:
        raise ValueError("DUET_FCP.POWER must be non-negative")

    context_requested = bool(context_conflict_transformer)
    context_enabled = bool(cfg.DUET_CONTEXT.ENABLED) if context_requested else False
    context_active_cycles = tuple(int(v) for v in cfg.DUET_CONTEXT.ACTIVE_CYCLES)
    context_refiner = str(cfg.DUET_CONTEXT.REFINER_TYPE)
    if context_enabled and context_refiner not in (
        "transformer",
        "cosine_knn",
        "prototype",
        "comparator",
    ):
        raise ValueError(
            "DUET_CONTEXT.REFINER_TYPE must be one of "
            "transformer | cosine_knn | prototype | comparator"
        )
    if context_enabled and int(cfg.DUET_CONTEXT.ANCHORS_PER_CLASS) < 1:
        raise ValueError("DUET_CONTEXT.ANCHORS_PER_CLASS must be >= 1")
    if context_enabled and int(cfg.DUET_CONTEXT.MODEL_DIM) % int(
        cfg.DUET_CONTEXT.NUM_HEADS
    ) != 0:
        raise ValueError("DUET_CONTEXT.MODEL_DIM must be divisible by NUM_HEADS")

    logging.info(
        "DUET first-cycle prior: enabled=True; power={:.3f}".format(
            float(cfg.DUET_FCP.POWER)
        )
    )
    logging.info(
        "DUET context transformer: requested={}; enabled={}; refiner={}; "
        "active_cycles={}; anchors_per_class={}; anchor_task_conf={:.2f}; "
        "anchor_clip_conf={:.2f}; strict_conflict={}; weak_agreement={}; "
        "ground_truth_affects_training=False".format(
            context_requested,
            context_enabled,
            context_refiner,
            list(context_active_cycles),
            int(cfg.DUET_CONTEXT.ANCHORS_PER_CLASS),
            float(cfg.DUET_CONTEXT.ANCHOR_TASK_CONF),
            float(cfg.DUET_CONTEXT.ANCHOR_CLIP_CONF),
            bool(cfg.DUET_CONTEXT.USE_STRICT_CONFLICT),
            bool(cfg.DUET_CONTEXT.USE_WEAK_AGREEMENT),
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

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features,
                                   bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num,
                                   bottleneck_dim=cfg.bottleneck).cuda()

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

    # 冻结 CLIP 文本侧，只训练 visual 侧
    for param in clip_model.transformer.parameters():
        param.requires_grad = False
    for param in clip_model.token_embedding.parameters():
        param.requires_grad = False
    clip_model.positional_embedding.requires_grad = False
    for param in clip_model.ln_final.parameters():
        param.requires_grad = False
    clip_model.text_projection.requires_grad = False

    vision_params = [p for p in clip_model.visual.parameters() if p.requires_grad]
    clip_optimizer = optim.Adam(vision_params, lr=cfg.ACTIVE.FINE_LR,
                                betas=(0.9, 0.999), eps=1e-8)
    clip_optimizer = op_copy(clip_optimizer)

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL

    prev_label_mask = None
    text_features = None
    context_transformer = None
    context_optimizer = None
    context_comparator = None
    context_comparator_optimizer = None
    if context_enabled and context_refiner != "comparator":
        context_transformer = DuetContextConflictTransformer(
            feature_dim=int(cfg.bottleneck),
            num_classes=int(cfg.class_num),
            model_dim=int(cfg.DUET_CONTEXT.MODEL_DIM),
            num_heads=int(cfg.DUET_CONTEXT.NUM_HEADS),
            ffn_dim=int(cfg.DUET_CONTEXT.FFN_DIM),
            dropout=float(cfg.DUET_CONTEXT.DROPOUT),
        ).cuda()
        context_optimizer = optim.Adam(
            context_transformer.parameters(),
            lr=float(cfg.DUET_CONTEXT.LR),
            weight_decay=float(cfg.DUET_CONTEXT.WEIGHT_DECAY),
        )
        logging.info(
            "DUET context transformer initialized: feature_dim={}; "
            "num_classes={}; model_dim={}; num_heads={}; ffn_dim={}; "
            "dropout={:.2f}; trainable_parameters={}".format(
                int(cfg.bottleneck),
                int(cfg.class_num),
                int(cfg.DUET_CONTEXT.MODEL_DIM),
                int(cfg.DUET_CONTEXT.NUM_HEADS),
                int(cfg.DUET_CONTEXT.FFN_DIM),
                float(cfg.DUET_CONTEXT.DROPOUT),
                sum(p.numel() for p in context_transformer.parameters()),
            )
        )
    if context_enabled and context_refiner == "comparator":
        # 第一个 comparator 在训练开始前创建（与旧版 persistent 相同的
        # 初始化时机）：Cycle 1 的全局 RNG 轨迹必须和旧版一致，
        # 否则连 baseline（65.6 vs 64.9）都会漂。
        context_comparator = PairwiseConflictComparator(
            input_dim=16,
            hidden=int(cfg.DUET_CONTEXT.COMPARATOR_HIDDEN),
            layers=int(cfg.DUET_CONTEXT.COMPARATOR_LAYERS),
            dropout=float(cfg.DUET_CONTEXT.DROPOUT),
        ).cuda()
        context_comparator_optimizer = optim.Adam(
            context_comparator.parameters(),
            lr=float(cfg.DUET_CONTEXT.LR),
            weight_decay=float(cfg.DUET_CONTEXT.WEIGHT_DECAY),
        )
        logging.info(
            "DUET pairwise comparator initialized: input_dim=16; "
            "hidden={}; layers={}; dropout={:.2f}; "
            "trainable_parameters={}".format(
                int(cfg.DUET_CONTEXT.COMPARATOR_HIDDEN),
                int(cfg.DUET_CONTEXT.COMPARATOR_LAYERS),
                float(cfg.DUET_CONTEXT.DROPOUT),
                sum(p.numel() for p in context_comparator.parameters()),
            )
        )
        context_replay_memory = ComparatorReplayMemory(
            per_direction_capacity=int(cfg.DUET_CONTEXT.REPLAY_PER_DIRECTION),
            # replay 存的是 comparator 的 16 维 pair evidence，不是 Task feature。
            feature_dim=16,
            device=next(context_comparator.parameters()).device,
        )

    curr_cycle = 0
    # office-home : 1.0 / VisDA-C : 1.05
    q_value = cfg.ACTIVE.Q_VALUE
    print("train_clip")
    while curr_cycle < cfg.ACTIVE.CYCLE:
        iter_num = 0

        netF.eval()
        netB.eval()
        # persistent comparator：Cycle 2/3/4 沿用同一个 comparator + optimizer，
        # 不 reset（Run 9/10 行为）；Cycle 1 不激活，obtain_label 自动忽略。
        label_result = obtain_label(
            dset_loaders['test_aug'], netF, netB, netC, text_inputs, text_features,
            clip_model, prev_label_mask, curr_cycle,
            first_cycle_prior=first_cycle_prior,
            prior_power=float(cfg.DUET_FCP.POWER),
            prior_epsilon=float(cfg.ACTIVE.EPSILON),
            context_conflict_transformer=context_enabled,
            context_transformer=context_transformer,
            context_optimizer=context_optimizer,
            comparator=context_comparator,
            comparator_optimizer=context_comparator_optimizer,
            replay_memory=(
                context_replay_memory
                if context_enabled and context_refiner == "comparator"
                else None
            ),
            context_cfg=cfg.DUET_CONTEXT,
            context_active_cycles=context_active_cycles,
            context_num_classes=int(cfg.class_num),
        )
        mem_label, label_mask, confi_imag, confi_dis, kl_soft = label_result
        kl_soft = kl_soft.cuda()
        mem_label = mem_label.cuda()
        prev_label_mask = label_mask

        clip_optimizer, q_value = train_clip(
            cfg, clip_model, confi_imag, confi_dis, text_inputs, clip_optimizer, q_value
        )

        cfg.load = 'prompt_model.pt'
        netF.train()
        netB.train()
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
            weak_preds = nn.Softmax(dim=1)(weak_logits)

            filtered_idx = tar_idx[label_mask[tar_idx]]

            con_loss = consistency_loss(weak_logits, strong_logits)
            classifier_loss = con_loss * cfg.ACTIVE.CON_PAR
            if cfg.ACTIVE.CLS_PAR > 0:
                pred = mem_label[filtered_idx]
                supervised_logits = weak_logits[label_mask[tar_idx]]
                if pred.size(0) != 0:
                    classifier_loss += nn.CrossEntropyLoss()(supervised_logits, pred) * cfg.ACTIVE.CLS_PAR
            clip_soft_batch = kl_soft[tar_idx]
            mi_loss = F.kl_div(weak_preds.log(), clip_soft_batch, reduction="batchmean")
            classifier_loss += mi_loss * cfg.ACTIVE.KL_PAR

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netF.eval()
                netB.eval()
                if cfg.SETTING.DATASET == 'VISDA-C':
                    acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                    log_str = ('Task: {}, Iter:{}/{}; Cycle: {}/{}; '
                               'Accuracy = {:.2f}%; classifier_loss = {}').format(
                                   cfg.name, iter_num, max_iter,
                                   curr_cycle + 1, cfg.ACTIVE.CYCLE,
                                   acc_s_te, classifier_loss) + '\n' + acc_list
                else:
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                    log_str = ('Task: {}, Iter:{}/{}; Cycle: {}/{}; '
                               'Accuracy = {:.2f}%; classifier_loss = {}').format(
                                   cfg.name, iter_num, max_iter,
                                   curr_cycle + 1, cfg.ACTIVE.CYCLE,
                                   acc_s_te, classifier_loss)
                logging.info(log_str)
                netF.train()
                netB.train()
        curr_cycle += 1

    return netF, netB, netC


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
    *,
    first_cycle_prior=True,
    prior_power=0.5,
    prior_epsilon=1e-6,
    context_conflict_transformer=False,
    context_transformer=None,
    context_optimizer=None,
    comparator=None,
    comparator_optimizer=None,
    replay_memory=None,
    context_cfg=None,
    context_active_cycles=(0,),
    context_num_classes=None,
):
    """收集全 target 的 Task/CLIP 概率与 Task feature，产出伪标签。

    返回 ``(mem_label, label_mask, confi_imag, confi_dis, kl_soft)``：

    - 基线路径与原始 DUET-FCP 完全一致；
    - Context 激活时，在 prior 校准后、admission 之前运行
      ``run_context_refinement``，再按 resolved / weak-rejected 规则覆盖
      ``label_mask`` / ``kl_soft`` / ``mem_label``。
    """
    start_test = True
    context_active = bool(
        context_conflict_transformer
        and curr_cycle in tuple(context_active_cycles)
    )
    comparator_mode = bool(
        context_active
        and context_cfg is not None
        and str(context_cfg.REFINER_TYPE) == "comparator"
    )
    if context_conflict_transformer and not context_active:
        # 每个 cycle 都输出一行：未激活时修正统计为 0，便于逐轮对比。
        logging.info(
            "DUET context correction: cycle={}; active=False; resolved=0; "
            "resolved_rate=0.00%; weak_deferred=0; weak_defer_rate=0.00%; "
            "corrections=0; ground_truth_affects_training=False".format(
                curr_cycle + 1
            )
        )
    if context_active and not comparator_mode:
        if (
            context_cfg is None
            or context_transformer is None
            or context_optimizer is None
        ):
            raise ValueError(
                "context-conflict transformer requires cfg, module and optimizer"
            )
    if comparator_mode:
        if comparator is None or comparator_optimizer is None:
            raise ValueError(
                "comparator mode requires the comparator module and optimizer"
            )
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            inputs_test, labels, sample_index = next(iter_test)
            weak_x = inputs_test[1].cuda()

            weak_feas = netB(netF(weak_x))
            weak_outputs = netC(weak_feas)
            if text_features is not None:
                clip_score = clip_text(clip_model, text_features, weak_x)
            else:
                clip_score, _ = clip_model(weak_x, text_inputs)
            clip_score = clip_score.cpu()
            if comparator_mode:
                clip_image_feature = clip_model.encode_image(weak_x).float().cpu()
                strong_x = inputs_test[2].cuda()
                strong_feas = netB(netF(strong_x))
                strong_task_outputs = netC(strong_feas)
                strong_task_feature = strong_feas.float().cpu()
                strong_clip_logits, _ = clip_model(strong_x, text_inputs)
                strong_clip_feature = clip_model.encode_image(strong_x).float().cpu()
                strong_task_outputs = strong_task_outputs.float().cpu()
                strong_clip_logits = strong_clip_logits.float().cpu()

            if start_test:
                all_output = weak_outputs.float().cpu()
                all_clip_score = clip_score.float().cpu()
                all_label = labels.float()
                if context_active:
                    all_sample_index = sample_index.long().cpu()
                    all_task_features = weak_feas.float().cpu()
                if comparator_mode:
                    all_clip_features = clip_image_feature
                    all_strong_task_outputs = strong_task_outputs
                    all_strong_clip_outputs = strong_clip_logits
                    all_strong_task_features = strong_task_feature
                    all_strong_clip_features = strong_clip_feature
                start_test = False
            else:
                all_output = torch.cat((all_output, weak_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                if context_active:
                    all_sample_index = torch.cat(
                        (all_sample_index, sample_index.long().cpu()), 0
                    )
                    all_task_features = torch.cat(
                        (all_task_features, weak_feas.float().cpu()), 0
                    )
                if comparator_mode:
                    all_clip_features = torch.cat(
                        (all_clip_features, clip_image_feature), 0
                    )
                    all_strong_task_outputs = torch.cat(
                        (all_strong_task_outputs, strong_task_outputs), 0
                    )
                    all_strong_clip_outputs = torch.cat(
                        (all_strong_clip_outputs, strong_clip_logits), 0
                    )
                    all_strong_task_features = torch.cat(
                        (all_strong_task_features, strong_task_feature), 0
                    )
                    all_strong_clip_features = torch.cat(
                        (all_strong_clip_features, strong_clip_feature), 0
                    )

    all_output = nn.Softmax(dim=1)(all_output)
    clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()

    pre_prior_task_probs = None
    pre_prior_clip_probs = None
    if context_active:
        # REQUIRE_PRE_POST_PRIOR_AGREEMENT 需要 prior 校准前的概率。
        pre_prior_task_probs = all_output.detach().clone()
        pre_prior_clip_probs = clip_all_output.detach().clone()
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

    # 计算 prior 校准后的 Task / CLIP Top-1
    _, all_output_pred = torch.max(all_output, dim=1)
    _, clip_all_output_pred = torch.max(clip_all_output, dim=1)
    matching_indices = all_output_pred == clip_all_output_pred

    # Context Transformer 插入点：prior 之后、admission_matching 之前。
    admission_matching = matching_indices
    context_payload = None
    if context_active:
        context_payload = run_context_refinement(
            task_probs=all_output,
            clip_probs=clip_all_output,
            task_features=all_task_features,
            num_classes=int(context_num_classes),
            context_cfg=context_cfg,
            pre_prior_task_probs=pre_prior_task_probs,
            pre_prior_clip_probs=pre_prior_clip_probs,
            labels=all_label,
            sample_indices=all_sample_index,
            transformer=context_transformer,
            optimizer=context_optimizer,
            clip_features=(
                all_clip_features if comparator_mode else None
            ),
            strong_task_probs=(
                nn.Softmax(dim=1)(all_strong_task_outputs)
                if comparator_mode
                else None
            ),
            strong_clip_probs=(
                nn.Softmax(dim=1)(all_strong_clip_outputs)
                if comparator_mode
                else None
            ),
            strong_task_features=(
                all_strong_task_features if comparator_mode else None
            ),
            strong_clip_features=(
                all_strong_clip_features if comparator_mode else None
            ),
            comparator=(comparator if comparator_mode else None),
            comparator_optimizer=(
                comparator_optimizer if comparator_mode else None
            ),
            replay_memory=(replay_memory if comparator_mode else None),
            cycle=int(curr_cycle + 1),
        )
        # weak-agreement 未通过验证：暂缓进入硬 CE（admission_matching=False），
        # 但仍参加 consistency 与 CLIP soft KL（kl_soft 保持 clip 概率）。
        admission_matching = matching_indices.clone()
        if bool(context_cfg.USE_WEAK_AGREEMENT):
            admission_matching[context_payload["weak_rejected_mask"]] = False

    # label_mask 单调累积（原始 DUET 规则不变）
    if prev_label_mask is not None:
        label_mask = prev_label_mask | (~prev_label_mask & admission_matching)
    else:
        label_mask = admission_matching

    kl_soft_output = clip_all_output
    if context_payload is not None:
        soft_only_admission = bool(
            getattr(context_cfg, "SOFT_ONLY_ADMISSION", False)
        )
        resolved_mask = context_payload["resolved_mask"]
        if int(resolved_mask.sum().item()) > 0:
            kl_soft_output = clip_all_output.clone()
            kl_soft_output[resolved_mask] = context_payload["refined_targets"][
                resolved_mask
            ]
            if soft_only_admission:
                # 消融：resolved 决策只做 KL soft target，不产生 hard label。
                logging.info(
                    "DUET context soft-only: cycle={}; resolved_soft_targets={}; "
                    "hard_admission=0; ground_truth_affects_training=False".format(
                        curr_cycle + 1,
                        int(resolved_mask.sum().item()),
                    )
                )
            else:
                # resolved 的 strict conflict 必须同时修改：
                #   label_mask -> True（进入硬 CE）
                #   mem_label  -> context_top1（见下方 all_mix_output 覆盖）
                #   kl_soft    -> refined context target
                label_mask = label_mask | resolved_mask
                logging.info(
                    "DUET context transformer admitted: cycle={}; resolved={}; "
                    "unresolved_conflicts_stay_in_consistency_and_clip_kl=True; "
                    "ground_truth_affects_training=False".format(
                        curr_cycle + 1,
                        int(resolved_mask.sum().item()),
                    )
                )

    # 伪标签精度日志（与原 DUET 一致）
    valid_preds = all_output_pred[label_mask]
    valid_labels = all_label[label_mask]
    if len(valid_preds) > 0:
        pseudo_label_accuracy = torch.sum(valid_preds == valid_labels).item() / float(len(valid_preds))
    else:
        pseudo_label_accuracy = 0.0
    log_str = "Number of valid pseudo-labeled samples: {}/{}; Accuracy = {:.2f}%".format(
        len(valid_preds), len(all_output_pred), pseudo_label_accuracy * 100
    )
    logging.info(log_str)

    # 混合分布：先按原始 DUET 生成，再覆盖 resolved 行
    all_mix_output = (all_output + clip_all_output) / 2.0
    if context_payload is not None and not bool(
        getattr(context_cfg, "SOFT_ONLY_ADMISSION", False)
    ):
        all_mix_output[context_payload["resolved_mask"]] = context_payload[
            "refined_targets"
        ][context_payload["resolved_mask"]]
    _, all_mix_output_pred = torch.max(all_mix_output, dim=1)
    if context_payload is not None and not bool(
        getattr(context_cfg, "SOFT_ONLY_ADMISSION", False)
    ):
        resolved_mask = context_payload["resolved_mask"]
        if int(resolved_mask.sum().item()) > 0:
            mem_kl_consistent = bool(
                (
                    all_mix_output_pred[resolved_mask]
                    == kl_soft_output[resolved_mask].argmax(dim=1)
                )
                .all()
                .item()
            )
            label_mask_consistent = bool(
                (label_mask[resolved_mask]).all().item()
            )
            logging.info(
                "DUET context consistency: cycle={}; resolved={}; "
                "mem_label==kl_soft_argmax={}; label_mask_holds_all_resolved={}; "
                "ground_truth_affects_training=False".format(
                    curr_cycle + 1,
                    int(resolved_mask.sum().item()),
                    mem_kl_consistent,
                    label_mask_consistent,
                )
            )

    valid_mixed = all_mix_output_pred[label_mask]
    mixed_output_accuracy = torch.sum(valid_mixed == valid_labels).item() / float(len(valid_preds))
    logging.info("Mixed output with valid mask: {:.2f}%".format(mixed_output_accuracy * 100))

    mix_output_accuracy = torch.sum(all_mix_output_pred == all_label).item() / float(len(all_label))
    clip_output_accuracy = torch.sum(clip_all_output_pred == all_label).item() / float(len(all_label))
    pure_output_accuracy = torch.sum(all_output_pred == all_label).item() / float(len(all_label))
    log_str_mix = ("all_mix_output Accuracy = {:.2f}%; clip_output_accuracy = {:.2f}%; "
                   "pure_output_accuracy = {:.2f}%;").format(
                       mix_output_accuracy * 100,
                       clip_output_accuracy * 100,
                       pure_output_accuracy * 100)
    logging.info(log_str_mix)

    confi_imag = loader.dataset.imgs
    confi_dis = all_mix_output.detach()
    return all_mix_output_pred, label_mask, confi_imag, confi_dis, kl_soft_output


def clip_pre_text(cfg):
    """构造 CLIP 文本 prompt（class name + 模板）。"""
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
    """用固定 text features 计算 CLIP logits。"""
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.detach().exp()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits
