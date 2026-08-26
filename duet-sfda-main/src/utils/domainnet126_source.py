"""Load DomainNet-126's monolithic source checkpoint into F/B/C modules."""

from __future__ import annotations

import logging
from pathlib import Path

from src.models.model import ResNetDomainNet126


_BACKBONE_PARTS = (
    "conv1",
    "bn1",
    "relu",
    "maxpool",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "avgpool",
)


def copy_domainnet126_source_into_split(source_model, net_f, net_b, net_c) -> None:
    """Copy equivalent ResNet, bottleneck, BN, and classifier parameters."""
    source_stack = source_model.encoder[-1]
    source_resnet = source_stack[0]
    source_bn = source_stack[1]
    for name in _BACKBONE_PARTS:
        getattr(net_f, name).load_state_dict(
            getattr(source_resnet, name).state_dict()
        )
    net_b.bottleneck.load_state_dict(source_resnet.fc.state_dict())
    net_b.bn.load_state_dict(source_bn.state_dict())
    net_c.fc.load_state_dict(source_model.fc.state_dict())


def load_domainnet126_source_into_split(cfg, net_f, net_b, net_c) -> Path:
    """Load the official DomainNet-126 source checkpoint and split it."""
    source_domain = str(cfg.domain[int(cfg.SETTING.S)])
    checkpoint = Path(cfg.output_dir_src) / f"best_{source_domain}_2020.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"DomainNet-126 source checkpoint does not exist: {checkpoint}"
        )
    source_model = ResNetDomainNet126(
        arch=str(cfg.MODEL.ARCH),
        checkpoint_path=str(checkpoint),
        num_classes=int(cfg.class_num),
        bottleneck_dim=int(cfg.bottleneck),
    )
    copy_domainnet126_source_into_split(
        source_model,
        net_f,
        net_b,
        net_c,
    )
    logging.info(
        "DomainNet-126 source split: checkpoint=%s; source=%s; "
        "F/B/C_equivalent=True",
        checkpoint,
        source_domain,
    )
    return checkpoint
