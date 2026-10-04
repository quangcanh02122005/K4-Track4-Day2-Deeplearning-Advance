"""experiments.py - bảng thí nghiệm B (backbone) và T (công thức huấn luyện), chạy hàng loạt, gom kết quả.

Mỗi thí nghiệm = dict ghi đè lên Config. `run_group` bỏ qua thí nghiệm đã có summary.json (an toàn khi Colab ngắt
phiên: chạy lại ô là tiếp tục). Chỉ khác công thức nền T00 đúng MỘT yếu tố (nguyên tắc N1).
"""
from __future__ import annotations

import pandas as pd

from train import Config, run

# ---- Bước 1: backbone, cùng công thức nền, cùng seed ----
BACKBONES = {
    "B01": ("resnet50", "resnet50"),
    "B02": ("resnext50_32x4d", "resnext50"),
    "B03": ("convnext_tiny", "convnext_tiny"),
    "B04": ("deit_small_patch16_224", "deit_small"),
    "B05": ("swin_tiny_patch4_window7_224", "swin_tiny"),
    "B06": ("efficientnet_b0", "efficientnet_b0"),
    "B07": ("mobilenetv3_large_100", "mobilenetv3"),
}

# ---- Bước 2: mỗi dòng: (trục, mô tả, ghi đè). Nền T00 = {} (công thức GUIDE mục 1.4) ----
TRAINING = {
    "T00": ("-", "công thức nền", {}),
    "T01": ("A", "đóng băng backbone, chỉ train head", {"init": "frozen"}),
    "T02": ("A", "từ đầu (không tiền huấn luyện)", {"init": "scratch"}),
    "T03": ("B", "aug: + ColorJitter", {"aug": "color"}),
    "T04": ("B", "aug: TrivialAugmentWide", {"aug": "trivial"}),
    "T05": ("B", "aug: + lật dọc, xoay 90", {"aug": "geom"}),
    "T06": ("B", "CutMix (alpha=1)", {"mix": "cutmix"}),
    "T07": ("B", "Mixup (alpha=1)", {"mix": "mixup"}),
    "T08": ("C", "label smoothing 0.1", {"loss": "ls", "label_smoothing": 0.1}),
    "T09": ("C", "focal loss gamma=2", {"loss": "focal", "focal_gamma": 2.0}),
    "T10": ("C", "CE trọng số 1/n_c", {"loss": "ce_weighted"}),
    "T11": ("D", "sampler cân bằng lớp", {"sampler": "balanced"}),
    "T12": ("E", "LR head = LR backbone (1e-4)", {"lr_head": 1e-4}),
    "T13": ("F", "EMA 0.998", {"ema_decay": 0.998}),
    "T14": ("G", "độ phân giải 256", {"img_size": 256}),
    "T15": ("G", "20 epoch", {"epochs": 20}),
}


def make_config(exp_id: str, base: dict, overrides: dict, **extra) -> Config:
    return Config(**{**base, **overrides, "exp_id": exp_id, **extra})


def run_group(experiments: dict[str, dict], base: dict, seed: int = 0, **extra) -> pd.DataFrame:
    """experiments: {exp_id: overrides}. base: cấu hình chung (đường dẫn, backbone, nền mới nếu tham lam)."""
    rows = []
    for exp_id, ov in experiments.items():
        cfg = make_config(exp_id, base, ov, seed=seed, **extra)
        res = run(cfg)
        rows.append({**res, "overrides": ov})
    return pd.DataFrame(rows)


def backbone_experiments() -> dict[str, dict]:
    return {k: {"backbone": v[0], "tag": v[1]} for k, v in BACKBONES.items()}


def training_experiments(only: list[str] | None = None) -> dict[str, dict]:
    keys = only or list(TRAINING)
    def slug(ov):  # tên ảnh curves chỉ dùng ASCII, ví dụ "init-frozen"
        return "_".join(f"{k}-{v}" for k, v in ov.items()) or "baseline"
    return {k: {**TRAINING[k][2], "tag": slug(TRAINING[k][2])} for k in keys}
