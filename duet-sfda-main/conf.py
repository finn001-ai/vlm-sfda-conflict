# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import logging
from pickle import TRUE
import torch
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode
import os.path as osp

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - see README.md for more information
# Data directory
_C.DATA_DIR = "/home/sfda/data/"

# Weight directory
_C.CKPT_DIR = "/home/sfda/"

# GPU id
_C.GPU_ID = '0'
# Output directory
_C.SAVE_DIR = "./output"

_C.ISSAVE = False
# Path to a specific checkpoint
_C.CKPT_PATH = ""

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Optional description of a config
_C.DESC = ""

_C.DA = "uda"

_C.FOLDER = './data/'

_C.NUM_WORKERS = 4

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Some of the available models can be found here:
# Torchvision: https://pytorch.org/vision/0.14/models.html
# timm: https://github.com/huggingface/pytorch-image-models/tree/v0.6.13
# RobustBench: https://github.com/RobustBench/robustbench
_C.MODEL.ARCH = 'resnet50'

_C.MODEL.METHOD = "lcfd"

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

_C.MODEL.EPISODIC = False

_C.MODEL.WEIGHTS = 'IMAGENET1K_V1'
# ----------------------------- SETTING options -------------------------- #
_C.SETTING = CfgNode()

# Dataset for evaluation
_C.SETTING.DATASET = 'office-home'

# The index of source domain
_C.SETTING.S = 0 
# The index of Target domain
_C.SETTING.T = 1

#Seed
_C.SETTING.SEED = 2021

#Sorce model directory
_C.SETTING.OUTPUT_SRC = 'weight_512/seed2021'

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Choices: Adam, SGD
_C.OPTIM.METHOD = "SGD"

# Learning rate
_C.OPTIM.LR = 1e-3

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 5e-4

_C.OPTIM.LR_DECAY1 = 0.1

_C.OPTIM.LR_DECAY2 = 1

_C.OPTIM.LR_DECAY3 = 0.01

# ------------------------------- Test options ------------------------- #
_C.TEST = CfgNode()


# Batch size
_C.TEST.BATCH_SIZE = 64

# Max epoch 
_C.TEST.MAX_EPOCH = 15

# Interval
_C.TEST.INTERVAL = 15

# --------------------------------- SOURCE options ---------------------------- #
_C.SOURCE = CfgNode()

_C.SOURCE.EPSILON = 1e-5

_C.SOURCE.TRTE = 'val'
# --------------------------------- NRC options --------------------------- #
_C.NRC = CfgNode()

_C.NRC.K = 5

_C.NRC.KK = 4

_C.NRC.EPSILON = 1e-5

# --------------------------------- SHOT options ---------------------------- #
_C.SHOT = CfgNode()

_C.SHOT.CLS_PAR = 0.3
_C.SHOT.ENT = True
_C.SHOT.GENT = True
_C.SHOT.EPSILON = 1e-5
_C.SHOT.ENT_PAR = 1.0
_C.SHOT.THRESHOLD = 0.0
_C.SHOT.DISTANCE = 'cosine'# ["cosine", "euclidean"]
# --------------------------------- SCLM options ---------------------------- #
_C.SCLM = CfgNode()

_C.SCLM.CLS_PAR = 0.3
_C.SCLM.ENT = True
_C.SCLM.GENT = True
_C.SCLM.EPSILON = 1e-5
_C.SCLM.CLS_SNT = 0.1
_C.SCLM.ENT_PAR = 1.0
_C.SCLM.NEW_ENT_PAR = 0.3
_C.SCLM.DISTANCE = 'cosine'# ["cosine", "euclidean"]
_C.SCLM.THRESHOLD = 0.0
_C.SCLM.INITC_PAR = 0.3
_C.SCLM.CONFI_PAR = 0.3
# --------------------------------- GKD options ---------------------------- #
_C.GKD = CfgNode()

_C.GKD.CLS_PAR = 0.3
_C.GKD.ENT = True
_C.GKD.GENT = True
_C.GKD.EPSILON = 1e-5
_C.GKD.ENT_PAR = 1.0
_C.GKD.THRESHOLD = 0.0
_C.GKD.DISTANCE = 'cosine'# ["cosine", "euclidean"]
# --------------------------------- TPDS options ---------------------------- #
_C.TPDS = CfgNode()

_C.TPDS.EPSILON = 1e-5
_C.TPDS.THRESHOLD = 0.0
_C.TPDS.DISTANCE = 'cosine'# ["cosine", "euclidean"]

# --------------------------------- COWA options ----------------------------- #
_C.COWA = CfgNode()

_C.COWA.ALPHA = 0.2
_C.COWA.WARM = 0.0
_C.COWA.COEFF = 'JMDS' #['LPG', 'JMDS', 'PPL','NO']
_C.COWA.EPSILON = 1e-5
_C.COWA.EPSILON2 = 1e-6
_C.COWA.DISTANCE = 'cosine'# ["cosine", "euclidean"]
_C.COWA.PICKLE = False
# --------------------------------- PLUE options --------------------- #
_C.PLUE = CfgNode()

_C.PLUE.TEMPORAL_LENGTH = 5
_C.PLUE.LABEL_REFINEMENT = True
_C.PLUE.CTR = True
_C.PLUE.EPSILON = 1e-5
_C.PLUE.NEG_L = True
_C.PLUE.REWEIGHTING = True
# _C.PLUE.QUEUE_SIZE = 16384
_C.PLUE.NUM_NEIGHBORS = 10
# ---------------------------------ADACONTRAST  options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard" # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0  # lambda for classification loss
_C.ADACONTRAST.BETA = 1.0   # lambda for instance loss
_C.ADACONTRAST.ETA = 1.0    # lambda for diversity loss
_C.ADACONTRAST.OPTIM_COS = True
_C.ADACONTRAST.OPTIM_EXP = False
_C.ADACONTRAST.FULL_PROGRESS = 0
_C.ADACONTRAST.SCHEDULE = [10,20]
_C.ADACONTRAST.GAMMA = 0.2
_C.ADACONTRAST.DIST_TYPE = "cosine" # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong" # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# --------------------------------- LCFD options ----------------------------- #
_C.LCFD = CfgNode()

_C.LCFD.CLS_PAR = 0.4
_C.LCFD.LOSS_FUNC = 'sce' #['l1',''l2','kl','sce']
_C.LCFD.ENT = True
_C.LCFD.GENT = True
_C.LCFD.EPSILON = 1e-5
_C.LCFD.GENT_PAR = 1.0
_C.LCFD.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.LCFD.N_CTX = 4 
_C.LCFD.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.LCFD.TTA_STEPS = 1
# --------------------------------- DIFO options ----------------------------- #
_C.DIFO = CfgNode()

_C.DIFO.CLS_PAR = 0.4
_C.DIFO.ENT = True
_C.DIFO.GENT = True
_C.DIFO.EPSILON = 1e-5
_C.DIFO.GENT_PAR = 1.0
_C.DIFO.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.DIFO.N_CTX = 4 
_C.DIFO.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.DIFO.TTA_STEPS = 1
_C.DIFO.IIC_PAR = 1.0
_C.DIFO.LOAD = None
# --------------------------------- ACTIVE options ----------------------------- #
_C.ACTIVE = CfgNode()

_C.ACTIVE.CLS_PAR = 0.4
_C.ACTIVE.ENT = True
_C.ACTIVE.GENT = True
_C.ACTIVE.EPSILON = 1e-5
_C.ACTIVE.GENT_PAR = 1.0
_C.ACTIVE.CTX_INIT = 'a_photo_of_a' #initialize context
_C.ACTIVE.N_CTX = 4
_C.ACTIVE.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.ACTIVE.TTA_STEPS = 1
_C.ACTIVE.IIC_PAR = 1.0
_C.ACTIVE.LOAD = None
_C.ACTIVE.FINE_LR = 1e-7
_C.ACTIVE.Q_VALUE = 1.05
_C.ACTIVE.BETA = 0.99
_C.ACTIVE.CYCLE = 4
_C.ACTIVE.CON_PAR = 0.2
_C.ACTIVE.KL_PAR = 0.4
# Optional target-domain adaptation list. Methods that support this field use
# it for training and pseudo-label inference while retaining full evaluation.
_C.ACTIVE.ADAPTATION_LIST = ""

# ------------------------ DUET boundary-router options ------------------- #
_C.DUET_BOUNDARY = CfgNode()
_C.DUET_BOUNDARY.TOP_FRACTION = 0.2

# Locked first-cycle class-balanced agreement delay.  This fraction comes from
# a predeclared offline audit and must not be tuned with target labels.
_C.DUET_CLIP_DELAY = CfgNode()
_C.DUET_CLIP_DELAY.FRACTION = 0.10

# --------------------------- Failure-audit options ------------------------ #
# These options only write read-only diagnostic snapshots. They do not alter
# pseudo labels, losses, optimizers, or inference logits.
_C.FAILURE_AUDIT = CfgNode()
_C.FAILURE_AUDIT.ENABLED = False
_C.FAILURE_AUDIT.DIR = "failure_audit"
_C.FAILURE_AUDIT.FEATURE_DTYPE = "float16"
# Diagnostic-only early stop.  Zero preserves every training path.  A positive
# value stops immediately after writing that pre-cycle snapshot, before the
# corresponding cycle performs CLIP or task optimization.
_C.FAILURE_AUDIT.STOP_AFTER_PRE_CYCLE = 0

# -------- Exact cycle-2 PCGrad parameter preflight (diagnostic only) ----- #
# This mode replays one pure arithmetic-DUET cycle, measures locked parameter
# gradients at the next cycle boundary, restores every model buffer, and
# stops before any cycle-2 optimizer step.  It never changes a training run
# when disabled.
_C.PCGRAD_PARAMETER_AUDIT = CfgNode()
_C.PCGRAD_PARAMETER_AUDIT.ENABLED = False
_C.PCGRAD_PARAMETER_AUDIT.DIR = "conflict_pcgrad_parameter_audit"

# --------------------------- DUET-FCP options ----------------------------- #
_C.DUET_FCP = CfgNode()
# source/CLIP 两路只在第 1 个 cycle 使用的 prior 强度。
_C.DUET_FCP.POWER = 0.5

# ---------- DUET-FCP + Class-Balanced Anchor Context Transformer -------- #
# 候选方法 duet_first_cycle_prior_context_transformer 的配置。
# 仅在 DUET_FCP（first_cycle_prior）之上运行；关闭 ENABLED 后完全退化为
# 原始 duet_first_cycle_prior。
_C.DUET_CONTEXT = CfgNode()
# 总开关：False 时新方法完全退化为原始 DUET-FCP，不创建 Transformer。
_C.DUET_CONTEXT.ENABLED = False
# 激活的 cycle（0-based）。第一轮（index 0）只跑纯 DUET-FCP：基线伪标签和
# CLIP 视觉分支尚未经过任何 target 适配，anchor/agreement 质量不够稳。
# 默认从第 2 个 cycle（index 1）开始运行 Context Transformer；想后续每轮
# 都跑可改为 [1, 2, 3]。
_C.DUET_CONTEXT.ACTIVE_CYCLES = [1]
# 是否处理 strict conflict（Task/CLIP Top-1 不一致）查询。
_C.DUET_CONTEXT.USE_STRICT_CONFLICT = True
# 是否处理 weak agreement（Top-1 一致但置信度低/熵高）查询。
_C.DUET_CONTEXT.USE_WEAK_AGREEMENT = True
# 每个类别最多保留的 anchor 数量（class-balanced）。
_C.DUET_CONTEXT.ANCHORS_PER_CLASS = 8
# anchor 准入条件：Task/CLIP 一致，且两路置信度均 >= 阈值、熵均 <= 阈值。
_C.DUET_CONTEXT.ANCHOR_TASK_CONF = 0.90
_C.DUET_CONTEXT.ANCHOR_CLIP_CONF = 0.90
_C.DUET_CONTEXT.ANCHOR_TASK_ENTROPY = 0.40
_C.DUET_CONTEXT.ANCHOR_CLIP_ENTROPY = 0.40
# anchor 可靠性分数中的熵权重：
# reliability = task_conf + clip_conf - entropy_weight * (task_entropy + clip_entropy)
_C.DUET_CONTEXT.ENTROPY_WEIGHT = 1.0
# 严格模式：anchor 还必须满足 prior 校准前 Task/CLIP 一致、且 prior 校准前后
# 共同 Top-1 不变，避免仅因 prior 校准才一致的低质样本进入 bank。
_C.DUET_CONTEXT.REQUIRE_PRE_POST_PRIOR_AGREEMENT = True
# weak-agreement 判定阈值：任一分支置信度低于该值或熵高于该值即为 weak。
_C.DUET_CONTEXT.WEAK_CONF_THRESHOLD = 0.70
_C.DUET_CONTEXT.WEAK_ENTROPY_THRESHOLD = 1.00
# Context Transformer 结构。
_C.DUET_CONTEXT.MODEL_DIM = 256
_C.DUET_CONTEXT.NUM_HEADS = 4
_C.DUET_CONTEXT.FFN_DIM = 512
_C.DUET_CONTEXT.DROPOUT = 0.10
# leave-one-out anchor 自训练。
_C.DUET_CONTEXT.TRAIN_STEPS_PER_CYCLE = 100
_C.DUET_CONTEXT.TRAIN_BATCH_SIZE = 64
_C.DUET_CONTEXT.LR = 1e-4
_C.DUET_CONTEXT.WEIGHT_DECAY = 1e-4
# strict conflict 接受阈值（context 置信度 / 边际）。
_C.DUET_CONTEXT.ACCEPT_CONF = 0.75
_C.DUET_CONTEXT.ACCEPT_MARGIN = 0.20
# weak-agreement 验证通过阈值（必须保持共同 Top-1）。
_C.DUET_CONTEXT.WEAK_ACCEPT_CONF = 0.75
_C.DUET_CONTEXT.WEAK_ACCEPT_MARGIN = 0.20
# 第三类（既非 Task 也非 CLIP Top-1）的更严格阈值。
_C.DUET_CONTEXT.THIRD_CLASS_CONF = 0.85
_C.DUET_CONTEXT.THIRD_CLASS_MARGIN = 0.30
_C.DUET_CONTEXT.ALLOW_THIRD_CLASS = True
# False 时跳过置信度/边际阈值（强制 resolve，但第三类限制仍然生效）。
_C.DUET_CONTEXT.ABSTAIN_WHEN_UNCERTAIN = True
# 对照实现：transformer | cosine_knn | prototype | comparator。
# comparator = pairwise conflict-resolution：只学 trust Task vs trust CLIP，
# 输出空间不再是 65 类，而是二选一 + 边际 abstain。
_C.DUET_CONTEXT.REFINER_TYPE = "transformer"
# cosine kNN 的邻居数（仅 REFINER_TYPE=cosine_knn 使用）。
_C.DUET_CONTEXT.KNN_K = 5
# pairwise comparator（REFINER_TYPE=comparator）参数。
# 两两比较器 MLP 的隐藏层宽度 / 层数。
_C.DUET_CONTEXT.COMPARATOR_HIDDEN = 64
_C.DUET_CONTEXT.COMPARATOR_LAYERS = 2
# anchor 相似度取 top-k 余弦的平均（不用 max，防异常 anchor 骗过高分）。
_C.DUET_CONTEXT.SIM_TOPK = 3
# synthetic conflict 门槛：runner-up（或翻转类）概率必须 >= 该值，
# 且 Top1/Top2 margin <= MAX_TOP1_MARGIN，否则该 anchor 不造这一侧样本。
_C.DUET_CONTEXT.MIN_RUNNER_PROB = 0.10
_C.DUET_CONTEXT.MAX_TOP1_MARGIN = 0.60
# abstain 门槛：|trust_task - trust_clip| < COMPARATOR_GATE 时 abstain。
_C.DUET_CONTEXT.COMPARATOR_GATE = 0.20
# >0 时不再使用绝对 margin gate，改为按 margin 排名只 resolve
# 当前 cycle 的固定比例。该选择不使用 target GT。
_C.DUET_CONTEXT.COMPARATOR_COVERAGE_FRACTION = 0.0
# comparator synthetic replay memory（persistent + replay 实验）：
# 每个信任方向最多保留 REPLAY_PER_DIRECTION 个历史 matched synthetic；
# 训练时每个 step 用 当前 matched : memory = (1-REPLAY_MIX_FRACTION) : REPLAY_MIX_FRACTION。
_C.DUET_CONTEXT.REPLAY_PER_DIRECTION = 64
_C.DUET_CONTEXT.REPLAY_MIX_FRACTION = 0.25
# GT-free adaptive training budget for the pairwise comparator.  When enabled,
# TRAIN_STEPS_PER_CYCLE is the maximum number of optimizer updates.
_C.DUET_CONTEXT.EARLY_STOP_ENABLED = False
_C.DUET_CONTEXT.EARLY_STOP_VAL_FRACTION = 0.20
_C.DUET_CONTEXT.EARLY_STOP_MIN_VAL_PER_DIRECTION = 6
_C.DUET_CONTEXT.EARLY_STOP_CHECK_INTERVAL = 10
_C.DUET_CONTEXT.EARLY_STOP_PATIENCE = 3
# 仅用于论文诊断：固定训练预算内，每隔若干步保存同一批真实 conflict
# 的 comparator 输出。GT 只在训练全部结束后计算日志，不用于停止、选步或恢复。
_C.DUET_CONTEXT.EVAL_TRAJECTORY_ENABLED = False
_C.DUET_CONTEXT.EVAL_TRAJECTORY_INTERVAL = 10
_C.DUET_CONTEXT.EVAL_TRAJECTORY_COVERAGES = [10, 20, 40, 60, 80]
# epoch-based 训练（替代固定 TRAIN_STEPS_PER_CYCLE）：
# 每个 epoch 把当前 matched synthetic 基本看一遍（配合 25% replay），
# 避免几十个样本被 200 步反复背诵（loss 下降 / confidence 膨胀 /
# real correctness 不升的根源）。=0 时退回固定 steps。
_C.DUET_CONTEXT.COMPARATOR_EPOCHS = 20
# soft-only 消融：comparator 的 resolved 决策只用于 KL soft target
# （refined_targets / kl_soft），不再执行 label_mask |= resolved_mask，
# 即不产生新的 hard pseudo-label。用于验证 arbitration 信号本身有没有价值。
_C.DUET_CONTEXT.SOFT_ONLY_ADMISSION = False
# distribution matching：synthetic conflict 只保留“长得像真实 conflict”的子集。
# 真实 conflict 的 confidence/entropy/margin/similarity 分布无需标签即可统计；
# 用 z-score（|z| <= DIST_MATCH_Z_MAX）在指定维度上过滤 synthetic 池，
# 避免用“一边明显坏、一边明显好”的简单样本训练 comparator。
_C.DUET_CONTEXT.DIST_MATCH_SYNTHETIC = True
_C.DUET_CONTEXT.DIST_MATCH_Z_MAX = 1.5
# 参与匹配的特征维度（v1 只匹配“困难程度”）：
# 4 = task_entropy, 5 = clip_entropy, 6 = task_margin, 7 = clip_margin
# （0-3 是双分支候选概率，8-11 是 anchor similarity；第一版不参与，
#  避免多维度同时卡 z-score 把 synthetic 池筛没）
_C.DUET_CONTEXT.DIST_MATCH_DIMS = [4, 5, 6, 7]
# 命中数下限：z-score 过滤后不足该数量时，退化为保留 mean|z| 最小的
# MIN_DIST_MATCH_KEPT 个样本，避免 synthetic 池被全部滤掉导致无法训练。
_C.DUET_CONTEXT.MIN_DIST_MATCH_KEPT = 16
# 固定随机种子（leave-one-out 采样等）。
_C.DUET_CONTEXT.SEED = 2020
# 是否打印 evaluation-only 指标（target label 只进日志，不进训练）。
_C.DUET_CONTEXT.EVAL_ONLY_LOGGING = True

# --------------------- DUET swap-conflict hard-label selection ---------- #
_C.DUET_SWAP = CfgNode()
# 独立配置开关：默认关闭，必须显式置 True 才启用 swap 选边规则。
_C.DUET_SWAP.ENABLED = False
# 决策强度门槛 D：|log(eA)-log(eB)| >= D 才产生标签，否则 abstain。
_C.DUET_SWAP.GATE_D = 4.0
# 方向门槛：低于离线锁定 cycle-0 方向精度的方向直接 abstain。
# 0.0 = 关闭；0.8 为推荐值（保护 car/truck 等 CLIP 不可靠方向）。
_C.DUET_SWAP.MIN_DIRECTION_ACCURACY = 0.0
# 最后一个激活的 cycle（1-based）：从该 cycle 起不再产生新 swap 标签。
# 后期（cycle 7-8）标签精度仅 ~60-65%，且大部分是重复样本；推荐 6。
_C.DUET_SWAP.LAST_ACTIVE_CYCLE = 8

# --------------------- Cycle 2/3 swap-intervention audit ---------------- #
_C.DUET_SWAP_AUDIT = CfgNode()
# 纯诊断开关：只记录 Cycle 2/3 的逐样本干预明细，不改变任何训练信号。
_C.DUET_SWAP_AUDIT.ENABLED = False

# --------------------- Top-k conflict probe full-softmax dump ------------ #
_C.CONFLICT_PROBE = CfgNode()
# 可选：每个 cycle 完整 12 类 task/CLIP softmax 的导出目录（一个 npz / cycle）。
# 空串 = 不导出（默认，保持现有 probe 输出不变）。DCPL 噪声转移矩阵的离线
# 诊断需要全量样本的完整 softmax，不只是冲突子集的 top-2。
_C.CONFLICT_PROBE.DUMP_DIR = ""

# --------------------------------- TSD options ----------------------------- #
_C.TSD = CfgNode()

_C.TSD.CLS_PAR = 0.4
_C.TSD.ENT = True
_C.TSD.GENT = True
_C.TSD.EPSILON = 1e-5
_C.TSD.GENT_PAR = 1.0
_C.TSD.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.TSD.N_CTX = 4 
_C.TSD.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.TSD.TTA_STEPS = 1
_C.TSD.IIC_PAR = 1.0
_C.TSD.LOAD = None
_C.TSD.LENT_PAR = 0.05
# --------------------------------- ProDe options ----------------------------- #
_C.ProDe = CfgNode()

_C.ProDe.ENT = True
_C.ProDe.GENT = True
_C.ProDe.EPSILON = 1e-5
_C.ProDe.GENT_PAR = 0.1
_C.ProDe.CTX_INIT = 'a_photo_of_a' #initialize context 
_C.ProDe.N_CTX = 4 
_C.ProDe.ARCH = 'ViT-B/32' #['RN50', 'ViT-B/32','RN101','ViT-B/16']
_C.ProDe.TTA_STEPS = 1
_C.ProDe.IIC_PAR = 1.3
_C.ProDe.LOAD = None
# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args():
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--cfg", dest="cfg_file",default="cfgs/imagenet_a/sclm.yaml", type=str,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    args = parser.parse_args()
    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.bottleneck = 512
    if cfg.SETTING.DATASET == 'office-home':
        cfg.domain = ['Art', 'Clipart', 'Product', 'RealWorld']
        cfg.class_num = 65 
        cfg.name_file = './data/office-home/classname.txt'
    if cfg.SETTING.DATASET == 'VISDA-C':
        cfg.domain = ['train', 'validation']
        cfg.class_num = 12
        cfg.name_file = './data/VISDA-C/classname.txt'
    if cfg.SETTING.DATASET == 'office':
        cfg.domain = ['amazon', 'dslr', 'webcam']
        cfg.name_file = './data/office/classname.txt'
        cfg.class_num = 31
    if cfg.SETTING.DATASET == 'imagenet_a':
        cfg.domain = ['target']
        cfg.class_num = 200
        cfg.bottleneck = 2048
    if cfg.SETTING.DATASET == 'imagenet_r':
        cfg.domain = ['target']
        cfg.class_num = 200
        cfg.bottleneck = 2048
    if cfg.SETTING.DATASET == 'imagenet_k':
        cfg.domain = ['target']
        cfg.class_num = 1000
        cfg.bottleneck = 2048
    if cfg.SETTING.DATASET == 'imagenet_v':
        cfg.domain = ['target']
        cfg.class_num = 1000
        cfg.bottleneck = 2048
    if cfg.SETTING.DATASET == 'domainnet126':
        cfg.domain = ["clipart", "painting", "real", "sketch"]
        cfg.name_file = './data/domainnet126/classname.txt'
        cfg.class_num = 126
        cfg.bottleneck = 256

    cfg.output_dir_src = os.path.join(cfg.CKPT_DIR,cfg.SETTING.OUTPUT_SRC,cfg.DA,cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.S][0].upper())
    cfg.output_dir = os.path.join(cfg.SAVE_DIR,cfg.DA,cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper(),cfg.MODEL.METHOD)
    cfg.name = cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper()
    cfg.name_src = cfg.domain[cfg.SETTING.S][0].upper()
    g_pathmgr.mkdirs(cfg.output_dir)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.output_dir, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])


    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda
               ]
    logger.info("PyTorch Version: torch={}, cuda={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(root, dataset_name):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_a": "imagenet-a",
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "domainnet126", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "office31": "office-31",
               "visda": "visda-2017",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "",
               "cifar100": "",
               "cifar100_c": "",
               "imagenet_v": "imagenetv2-matched-frequency-format-val"
               }
    return os.path.join(root, mapping[dataset_name])


def get_domain_sequence(ckpt_path):
    assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt')
    domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    mapping = {"real": ["clipart", "painting", "sketch"],
               "clipart": ["sketch", "real", "painting"],
               "painting": ["real", "sketch", "clipart"],
               "sketch": ["painting", "clipart", "real"],
               }
    return mapping[domain]


def adaptation_method_lookup(adaptation):
    lookup_table = {"source": "Norm",
                    "norm_test": "Norm",
                    "norm_alpha": "Norm",
                    "norm_ema": "Norm",
                    "ttaug": "TTAug",
                    "memo": "MEMO",
                    "lame": "LAME",
                    "tent": "Tent",
                    "eata": "EATA",
                    "sar": "SAR",
                    "adacontrast": "AdaContrast",
                    "cotta": "CoTTA",
                    "rotta": "RoTTA",
                    "gtta": "GTTA",
                    "rmt": "RMT",
                    "roid": "ROID",
                    "proib": "Proib"
                    }
    assert adaptation in lookup_table.keys(), \
        f"Adaptation method '{adaptation}' is not supported! Choose from: {list(lookup_table.keys())}"
    return lookup_table[adaptation]
