"""benchmark.py - đo độ trễ suy luận đúng cách: warmup, synchronize, >= 50 lần, p50/p95/p99.

Mặc định đo THUẦN forward của model (không tính đọc ảnh/tiền xử lý), đầu vào ngẫu nhiên đã nằm sẵn trên thiết bị.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Thời gian (ms) của fn(): bỏ `warmup` lần đầu, đồng bộ trước và sau mỗi lần đo."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"p50": float(np.percentile(ts, 50)), "p95": float(np.percentile(ts, 95)),
            "p99": float(np.percentile(ts, 99)), "mean": float(ts.mean()), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, k_views: int = 1, fused_bn: bool = False) -> dict:
    """Độ trễ forward với đầu vào ngẫu nhiên (B,3,S,S). dtype: fp32 | amp | fp16. k_views: số lần forward
    liên tiếp (mô phỏng TTA). `fused_bn` chỉ để ghi nhãn: truyền sẵn model đã gộp BN."""
    assert iters >= 50 and warmup >= 10, "Quy tắc đo: >= 50 lần đo, >= 10 lần warmup"
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    m = copy.deepcopy(model).to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        m, x = m.half(), x.half()
    amp = dtype == "amp" and dev.type == "cuda"

    def fn():
        with torch.inference_mode(), torch.autocast(dev.type, enabled=amp):
            for _ in range(k_views):
                m(x)

    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    r = bench(fn, warmup, iters, sync)
    return {"gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU", "dtype": dtype,
            "batch": batch_size, "img_size": img_size, "k_views": k_views, "fused_bn": fused_bn,
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"],
            "images_per_s": batch_size / (r["p50"] / 1000), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view (đo thật) kèm tỉ lệ so với K lần một lượt chạy."""
    one = latency_report(model, k_views=1, **kw)
    many = latency_report(model, k_views=k_views, **kw)
    many["ratio_vs_k_times_single"] = many["p50"] / (k_views * one["p50"])
    return many
