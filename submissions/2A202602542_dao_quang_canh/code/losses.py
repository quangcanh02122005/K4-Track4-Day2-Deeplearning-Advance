"""losses.py - label smoothing, focal loss, trọng số lớp, Mixup/CutMix."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """kind: "ce" | "ls" (smoothing=) | "focal" (gamma=, alpha=) | "ce_weighted" (weight=)."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """q'(k) = (1-eps)*1[k==y] + eps/K  =>  loss = (1-eps)*NLL + eps*mean_k(-log p_k). eps=0 cho đúng CE."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


class FocalLoss(nn.Module):
    """FL = -alpha_t * (1-p_t)^gamma * log p_t, trung bình theo batch. gamma=0, alpha=None cho đúng CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp.exp()).clamp(min=0) ** self.gamma) * logp
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp của TRAIN. beta=0: 1/n_c (trung bình 1); beta>0: class-balanced
    (1-beta)/(1-beta^n_c), tổng chuẩn hoá về số lớp."""
    n = np.asarray(counts, dtype=np.float64)
    if beta == 0:
        w = 1.0 / n
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
    return torch.as_tensor(w, dtype=torch.float32)


def _rand_box(h: int, w: int, lam: float):
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn batch. Trả về (x_mix, (y_a, y_b, lam)); lam của CutMix được tính lại theo diện tích hộp thực."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        x_mix = x.clone()
        y1, y2, x1, x2 = _rand_box(x.size(2), x.size(3), lam)
        x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3))
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
