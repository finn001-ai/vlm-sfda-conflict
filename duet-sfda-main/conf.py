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
# --------------------------------- DCCL options ----------------------------- #
_C.DCCL = CfgNode()

_C.DCCL.CAND_PAR = 0.05
_C.DCCL.CAND_START_CYCLE = 0
_C.DCCL.CAND_TAU = 0.0
_C.DCCL.CAND_WEIGHT = "none"
_C.DCCL.KL_MODE = "clip"
_C.DCCL.KL_CANDIDATE = "confidence"
_C.DCCL.CALIB_MODE = "none"
_C.DCCL.CALIB_POWER = 0.5
_C.DCCL.CALIB_AUTO_LAMBDA = 0.2
_C.DCCL.TOPO_GRAPH_K = 15
_C.DCCL.TOPO_TEMPERATURE = 0.07
_C.DCCL.TOPO_ALPHA = 0.9
_C.DCCL.TOPO_STEPS = 20
_C.DCCL.TOPO_CHUNK_SIZE = 512
_C.DCCL.TOPO_ANCHOR_RATIO = 0.5
_C.DCCL.TOPO_ANCHOR_MIN_PER_CLASS = 5
_C.DCCL.TOPO_TARGET_MIX = -1.0
_C.DCCL.PL_EXPAND = "none"
_C.DCCL.PL_TOPK_PER_CLASS = 0
_C.DCCL.PL_MIN_CONF = 0.0
_C.DCCL.PL_MEMORY = "monotonic"
_C.DCCL.PL_STABLE_CYCLES = 2
_C.DCCL.PL_STABLE_MEMORY = "reversible"
_C.DCCL.PL_MEMORY_WARMUP_CYCLES = 1
_C.DCCL.PL_MEMORY_MIN_CONF = 0.0
_C.DCCL.PL_CLASS_BALANCE = False
_C.DCCL.PL_BALANCE_COVERAGE = 0.75
_C.DCCL.PL_BALANCE_MIN_PER_CLASS = 1
_C.DCCL.PROTO_ADAPT = False
_C.DCCL.PROTO_MIX = 0.0
_C.DCCL.PROTO_TEMPERATURE = 0.2
_C.DCCL.PROTO_MIN_PER_CLASS = 3
_C.DCCL.PROTO_MOMENTUM = 0.0
_C.DCCL.TARGET_HEAD_ADAPT = False
_C.DCCL.TARGET_HEAD_VARIANT = "blend"
_C.DCCL.TARGET_HEAD_MIX = 0.3
_C.DCCL.TARGET_HEAD_START_CYCLE = 1
_C.DCCL.TARGET_HEAD_LR_MULT = 1.0
_C.DCCL.TARGET_HEAD_EMA = False
_C.DCCL.TARGET_HEAD_EMA_MOMENTUM = 0.99
_C.DCCL.TARGET_RESIDUAL_MAX_GATE = 0.3
_C.DCCL.TARGET_RESIDUAL_GATE_INIT = -2.0
_C.DCCL.TRAJECTORY_ENSEMBLE = False
_C.DCCL.TRAJECTORY_SNAPSHOT_INTERVALS = [2, 3, 4]
_C.DCCL.PAIR_FLOW_RANK = 16
_C.DCCL.PAIR_FLOW_MIN_COUNT = 5
_C.DCCL.PAIR_FLOW_MIN_CYCLES = 2
_C.DCCL.PAIR_FLOW_MAX_GATE = 0.3
_C.DCCL.PAIR_FLOW_GATE_INIT = -2.0
_C.DCCL.PAIR_FEATURE_ADAPT = False
_C.DCCL.PAIR_FEATURE_START_CYCLE = 1
_C.DCCL.PAIR_FEATURE_LR_MULT = 1.0
_C.DCCL.PAIR_FEATURE_MIN_ACTIVE_RANK = 1
_C.DCCL.PAIR_FEATURE_GRADIENT_MODE = "joint"
_C.DCCL.PAIR_FEATURE_MAX_GATE = 0.05
_C.DCCL.PAIR_FEATURE_GATE_INIT = -2.0
_C.DCCL.COV_TRANSPORT_ADAPT = False
_C.DCCL.COV_TRANSPORT_MODE = "conditional"
_C.DCCL.COV_TRANSPORT_START_CYCLE = 1
_C.DCCL.COV_TRANSPORT_MIN_ANCHORS = 8
_C.DCCL.COV_TRANSPORT_RANK = 4
_C.DCCL.COV_TRANSPORT_MAX_GATE = 0.05
_C.DCCL.COV_GLOBAL_MIN_ANCHORS = 512
_C.DCCL.COV_GLOBAL_SHRINKAGE = 0.1
_C.DCCL.COV_GLOBAL_HOLDOUT_RATIO = 0.2
_C.DCCL.COV_GLOBAL_MIN_IMPROVEMENT = 0.001
_C.DCCL.EPSILON = 1e-6
_C.DCCL.TAU_LOW = 0.4
_C.DCCL.TAU_HIGH = 0.7
_C.DCCL.GAP_PROMOTE = 0.3
_C.DCCL.PROMOTE_K = 2
_C.DCCL.TEMPORAL_DIAG = False
_C.DCCL.TEMPORAL_DIAG_DIR = "temporal_diagnostics"
_C.DCCL.GRAPH_TEACHER_FUSION = False
_C.DCCL.GTF_APPLY_TO = "both"
_C.DCCL.GTF_STRENGTH = 0.5
_C.DCCL.GTF_GRAPH_K = 15
_C.DCCL.GTF_TEMPERATURE = 0.07
_C.DCCL.GTF_ALPHA = 0.9
_C.DCCL.GTF_STEPS = 20
_C.DCCL.GTF_CHUNK_SIZE = 512
_C.DCCL.GTF_ANCHOR_RATIO = 0.5
_C.DCCL.GTF_ANCHOR_MIN_PER_CLASS = 5
_C.DCCL.GTR_PAR = 0.0
_C.DCCL.GTR_STABLE_CYCLES = 2
_C.DCCL.GTR_MEMORY = "reversible"
_C.DCCL.GTR_MIN_GRAPH_CONF = 0.05
_C.DCCL.GTR_MIN_DISAGREEMENT = 0.25
_C.DCCL.THREE_VIEW_EM = False
_C.DCCL.THREE_VIEW_EM_START_CYCLE = 1
_C.DCCL.THREE_VIEW_EM_STEPS = 5
_C.DCCL.THREE_VIEW_EM_DIRICHLET = 5.0
_C.DCCL.THREE_VIEW_EM_MIN_CLASS_ANCHORS = 3
_C.DCCL.THREE_VIEW_EM_PAR = 0.05

# ------------------------------- ACCD options ----------------------------- #
_C.ACCD = CfgNode()

_C.ACCD.ENABLED = False
_C.ACCD.GRAPH_K = 15
_C.ACCD.TEMPERATURE = 0.07
_C.ACCD.ALPHA = 0.9
_C.ACCD.STEPS = 20
_C.ACCD.CHUNK_SIZE = 512
_C.ACCD.ANCHOR_RATIO = 0.5
_C.ACCD.ANCHOR_MIN_PER_CLASS = 5
_C.ACCD.ANCHOR_MEMORY = "dynamic"
_C.ACCD.CANDIDATE_MASS = 0.6
_C.ACCD.CANDIDATE_MARGIN = 0.1
_C.ACCD.START_CYCLE = 0
_C.ACCD.STABLE_CYCLES = 2
_C.ACCD.RESOLUTION_MEMORY = "persistent"
_C.ACCD.RESOLUTION_TARGET = "both"
_C.ACCD.RESOLUTION_ACTION = "hard_label"  # hard_label, teacher_abstain, candidate_transport
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
