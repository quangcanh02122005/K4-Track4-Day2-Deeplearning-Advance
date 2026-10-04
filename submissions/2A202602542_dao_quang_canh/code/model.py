"""model.py - tạo backbone (timm), đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import copy

import timm
import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Model 9 lớp. init: scratch (không tiền huấn luyện) | frozen (chỉ train head) | finetune (train hết)."""
    name = SUGGESTED_BACKBONES.get(name, name)
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    model = timm.create_model(name, pretrained=(init != "scratch") and pretrained,
                              num_classes=num_classes, drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model) -> str:
    """Tag trọng số timm thực sự được tải (ghi vào results.xlsx)."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return str(cfg.get("hf_hub_id") or cfg.get("tag") or cfg.get("url") or "none")


def data_config(model) -> dict:
    """mean/std/input_size đúng với trọng số đang dùng (timm.data.resolve_data_config)."""
    return timm.data.resolve_data_config({}, model=model)


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số ngoài head. BN backbone phải eval: dùng set_train_mode()."""
    head_params = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_params


def set_train_mode(model, frozen: bool = False) -> None:
    """model.train(); nếu backbone đóng băng thì đưa mọi module ngoài head về eval (giữ BN không cập nhật
    thống kê chạy, nếu không BN lệch dần so với trọng số đóng băng)."""
    model.train()
    if frozen:
        for m in model.modules():
            m.eval()  # eval() của module cha lan xuống cả head, nên bật lại train cho head ngay sau
        model.get_classifier().train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm (slide trang 52): backbone ndim>1 | norm+bias backbone (wd=0) | head."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    bb_decay, bb_no_decay, head = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim <= 1:
            bb_no_decay.append(p)
        else:
            bb_decay.append(p)
    groups = [
        {"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
        {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_norm_bias"},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), gồm cả phần đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC/ảnh bằng torch.utils.flop_counter (FLOPs / 2). Chỉ tính conv/matmul/attention; bỏ qua phép
    phần tử (BN, activation), nên lệch vài % so với fvcore/thop."""
    from torch.utils.flop_counter import FlopCounterMode

    m = copy.deepcopy(model).cpu().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    with FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops() / 2 / 1e9
