"""inference.py - các phương pháp suy luận (Bước 3). Chọn phương pháp CHỈ dựa trên val; T khớp trên VAL."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@torch.inference_mode()
def predict_logits(model, loader, device, view=None, amp: bool = False):
    """(filenames, y_true[N], logits[N,9]) theo thứ tự loader. `view`: hàm batch -> batch (vd view_hflip)."""
    model.eval()
    names, ys, outs = [], [], []
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        if view is not None:
            x = view(x)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        names += list(f)
        ys.append(y)
        outs.append(logits.float().cpu())
    return names, torch.cat(ys).numpy(), torch.cat(outs).numpy()


@torch.inference_mode()
def predict_logits_views(model, loader, device, views_fn, amp: bool = False):
    """Như predict_logits nhưng `views_fn(x)` trả về LIST batch (multi-crop/scale). Trả về
    (filenames, y_true, [logits_view_0, ..., logits_view_{K-1}])."""
    model.eval()
    names, ys, outs = [], [], None
    for x, y, f in loader:
        views = views_fn(x.to(device, non_blocking=True))
        if outs is None:
            outs = [[] for _ in views]
        for i, v in enumerate(views):
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                outs[i].append(model(v).float().cpu())
        names += list(f)
        ys.append(y)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) cỡ `crop`; flip=True thêm bản lật ngang của từng crop (10 view)."""
    h, w = x.shape[-2:]
    boxes = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), ((h - crop) // 2, (w - crop) // 2)]
    views = [x[..., t:t + crop, l:l + crop] for t, l in boxes]
    if flip:
        views += [torch.flip(v, dims=[3]) for v in views]
    return views


def views_multiscale(x, sizes):
    """Resize batch về từng cỡ trong `sizes`. CNN có global pooling chạy được; ViT/Swin cần model hỗ trợ
    kích thước động (timm: dynamic_img_size / strict_img_size=False), nếu không sẽ báo lỗi kích thước."""
    return [x if s == x.shape[-1] else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False,
                                                     antialias=s < x.shape[-1]) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: space="prob" (trung bình softmax) hoặc "logit" (trung bình logit rồi softmax)."""
    if space == "prob":
        return np.mean([_softmax(l) for l in logits_per_view], axis=0)
    if space == "logit":
        return _softmax(np.mean([np.asarray(l, dtype=np.float64) for l in logits_per_view], axis=0))
    raise ValueError(f"space không hợp lệ: {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    shapes = {np.asarray(p).shape for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"Các dự đoán khác kích thước: {shapes}")
    return np.mean(list_of_probs, axis=0)


def _nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL (tìm golden-section theo log T trong [0.05, 20])."""
    z = np.asarray(val_logits, dtype=np.float64)
    y = np.asarray(val_labels)
    lo, hi = math.log(0.05), math.log(20.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = _nll(z, y, math.exp(a)), _nll(z, y, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _nll(z, y, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _nll(z, y, math.exp(b))
    return math.exp((lo + hi) / 2)


def apply_temperature(logits, T: float):
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    fused = fused.to(conv.weight.device, conv.weight.dtype)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(bn.bias + (bias - bn.running_mean) * scale)
    return fused


@torch.no_grad()
def fuse_conv_bn(model, check_input=None, tol: float = 1e-3):
    """Gộp BN vào Conv liền trước (các cặp con liền kề trong cùng module cha) trên BẢN SAO của model.
    BatchNormAct2d của timm (BN+activation) được thay bằng phần activation của nó.
    Nếu truyền `check_input`, in sai số lớn nhất của đầu ra trước/sau và báo lỗi nếu > tol (chặn trường hợp
    hai module liền kề trong khai báo nhưng không nối tiếp nhau trong forward).
    Kiến trúc dùng LayerNorm (ViT, Swin, ConvNeXt) không có BN: trả về bản sao không đổi, n_fused = 0."""
    fused_model = copy.deepcopy(model).eval()
    n = [0]

    def walk(parent):
        names = list(parent._modules.keys())
        i = 0
        while i < len(names) - 1:
            a, b = parent._modules[names[i]], parent._modules[names[i + 1]]
            if (isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and b.track_running_stats
                    and a.out_channels == b.num_features):
                parent._modules[names[i]] = _fuse_pair(a, b)
                act = getattr(b, "act", None)
                drop = getattr(b, "drop", None)
                tail = [m for m in (drop, act) if m is not None and not isinstance(m, nn.Identity)]
                parent._modules[names[i + 1]] = nn.Sequential(*tail) if tail else nn.Identity()
                n[0] += 1
                i += 2
            else:
                i += 1
        for child in parent._modules.values():
            if child is not None:
                walk(child)

    walk(fused_model)
    fused_model.n_fused = n[0]
    if check_input is not None:
        model.eval()
        diff = (model(check_input) - fused_model(check_input)).abs().max().item()
        print(f"fuse_conv_bn: gộp {n[0]} cặp, sai số lớn nhất = {diff:.2e}")
        if diff > tol:
            raise RuntimeError(f"Gộp BN sai: sai số {diff:.2e} > {tol}")
    return fused_model
