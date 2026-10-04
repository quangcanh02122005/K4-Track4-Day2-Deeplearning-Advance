"""train.py - vòng huấn luyện dùng chung cho mọi thí nghiệm (B, T, F).

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc (cùng định nghĩa lúc chấm).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _import_eval():
    """Tìm eval.py của repo gốc: cùng thư mục, hoặc thư mục cha gần nhất có eval.py."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        if (d / "eval.py").exists():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            break
    import eval as ev  # noqa: E402
    return ev


ev = _import_eval()
import dataset as D  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    tag: str = ""                     # mô tả ngắn cho tên ảnh curves/<exp_id>_<tag>.png (rỗng -> tên backbone)
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | geom | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    cache_images: bool = False
    resume: bool = True
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    tag = cfg.tag or cfg.backbone
    suffix = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{tag}{suffix}.png"


def set_seed(seed: int) -> None:
    """Cố định random/numpy/torch. cudnn.benchmark=False, deterministic=False (tốc độ > tái lập từng bit;
    AMP và một số kernel CUDA vốn không tất định). Seed worker: dataset._worker_init + generator của loader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    groups = M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật theo bước (iteration)."""
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class EMA:
    """W_ema <- d*W_ema + (1-d)*W. d hiệu dụng = min(decay, (1+n)/(10+n)) để EMA không bị kéo bởi trọng số
    khởi tạo ở đầu. Buffer (running_mean/var của BN) chép thẳng từ model."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.n = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for pe, p in zip(self.module.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.module.buffers(), model.buffers()):
            be.copy_(b)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    M.set_train_mode(model, frozen=(cfg.init == "frozen"))
    use_amp = cfg.amp and device.type == "cuda"
    tot_loss, tot_correct, n = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if cfg.mix:
            x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss = L.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        tot_loss += loss.item() * x.size(0)
        n += x.size(0)
        if not cfg.mix:
            tot_correct += (logits.argmax(1) == y).sum().item()
    return {"train_loss": tot_loss / n, "train_acc": float("nan") if cfg.mix else tot_correct / n,
            "lr": optimizer.param_groups[-1]["lr"]}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool = False):
    """(filenames, y_true[N], logits[N,9], loss). Giữ đúng thứ tự loader."""
    model.eval()
    names, ys, outs, tot, n = [], [], [], 0.0, 0
    for x, y, f in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        tot += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        names += list(f)
        ys.append(y.cpu())
        outs.append(logits.cpu())
    return names, torch.cat(ys).numpy(), torch.cat(outs).numpy(), tot / n


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    ep = h["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, h["train_loss"], "o-", label="train loss")
    ax[0].plot(ep, h["val_loss"], "s-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(ep, h["val_macro_f1"], "s-", label="val macro-F1")
    ax[1].plot(ep, h["val_top1"], "^--", label="val top-1")
    if h["train_acc"].notna().any():
        ax[1].plot(ep, h["train_acc"], "o:", label="train acc")
    ax[1].set(xlabel="epoch", ylabel="metric", title="Metric")
    ax[2].plot(ep, h["lr"], "o-", label="LR (head, cuối epoch)")
    ax[2].set(xlabel="epoch", ylabel="lr", title="Learning rate")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình, lưu config/history/checkpoint/logit/predictions/curves. Trả về dict tóm tắt."""
    out = run_dir(cfg)
    summary_file = out / "summary.json"
    if cfg.resume and summary_file.exists():
        cached = json.loads(summary_file.read_text())
        # Nếu cần predictions test mà lần trước chưa ghi thì chạy lại phần cuối; còn lại dùng kết quả cũ.
        if not cfg.save_test_predictions or pred_path(cfg, "test").exists():
            print(f"[{cfg.exp_id} seed{cfg.seed}] đã xong, dùng lại kết quả cũ")
            return cached

    set_seed(cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)

    model = M.build_model(cfg.backbone, True, D.NUM_CLASSES, cfg.drop_rate, cfg.init)
    dc = M.data_config(model)
    mean, std = dc["mean"], dc["std"]
    tf_train = D.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    tf_eval = D.build_transforms(False, cfg.img_size, "basic", mean, std)
    train_loader = D.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size, True, cfg.sampler,
                                 cfg.num_workers, cfg.seed, cfg.cache_images)
    val_loader = D.make_loader(val_df, cfg.images_dir, tf_eval, cfg.batch_size, False,
                               num_workers=cfg.num_workers, cache=cfg.cache_images)
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    kw = {}
    if cfg.loss == "ls":
        kw["smoothing"] = cfg.label_smoothing or 0.1
    elif cfg.loss == "focal":
        kw["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"].to_numpy(), minlength=D.NUM_CLASSES)  # chỉ dùng TRAIN
        kw["weight"] = L.class_weights(counts, cfg.class_weight_beta or 0.0)
    criterion = L.build_criterion(cfg.loss, **kw).to(device)
    val_criterion = nn.CrossEntropyLoss()  # val loss luôn là CE thường để so được giữa các thí nghiệm

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema else model

    history, best, start_epoch, train_time = [], {"f1": -1.0, "epoch": -1}, 0, 0.0
    ckpt_last, ckpt_best = out / "last.pt", out / "best.pt"
    if cfg.resume and ckpt_last.exists():
        s = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        optimizer.load_state_dict(s["optimizer"])
        scheduler.load_state_dict(s["scheduler"])
        scaler.load_state_dict(s["scaler"])
        if ema:
            ema.module.load_state_dict(s["ema"])
            ema.n = s["ema_n"]
        history, best, start_epoch, train_time = s["history"], s["best"], s["epoch"] + 1, s["train_time"]
        print(f"[{cfg.exp_id} seed{cfg.seed}] tiếp tục từ epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        names, y, logits, vloss = evaluate(eval_model, val_loader, val_criterion, device)
        probs = softmax_np(logits)
        m = ev.compute_metrics(y, probs.argmax(1), probs)
        row = {"epoch": epoch + 1, **tr, "val_loss": vloss, "val_macro_f1": m["macro_f1"],
               "val_top1": m["top1"], "val_ece": m["ece"]}
        history.append(row)
        print(f"[{cfg.exp_id} s{cfg.seed}] ep{epoch + 1:02d}/{cfg.epochs} train_loss={tr['train_loss']:.4f} "
              f"val_loss={vloss:.4f} val_F1={m['macro_f1']:.4f} val_top1={m['top1']:.4f}")
        if m["macro_f1"] > best["f1"]:  # hòa thì giữ epoch sớm hơn (không dùng >=)
            best = {"f1": m["macro_f1"], "epoch": epoch + 1}
            torch.save(eval_model.state_dict(), ckpt_best)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
                 "best": best, "epoch": epoch, "train_time": train_time, "ema_n": ema.n if ema else 0}
        if ema:
            state["ema"] = ema.module.state_dict()
        torch.save(state, ckpt_last)

    # --- cuối: nạp checkpoint tốt nhất, lưu val logits + predictions ---
    eval_model.load_state_dict(torch.load(ckpt_best, map_location=device))
    names, y, logits, _ = evaluate(eval_model, val_loader, val_criterion, device)
    probs = softmax_np(logits)
    np.save(out / "val_logits.npy", logits)
    ev.save_predictions(pred_path(cfg, "val"), names, y, probs)
    m = ev.compute_metrics(y, probs.argmax(1), probs)

    if cfg.save_test_predictions:  # đúng MỘT lần, chỉ ở Bước 4
        test_loader = D.make_loader(test_df, cfg.images_dir, tf_eval, cfg.batch_size, False,
                                    num_workers=cfg.num_workers, cache=cfg.cache_images)
        tn, ty, tl, _ = evaluate(eval_model, test_loader, val_criterion, device)
        np.save(out / "test_logits.npy", tl)
        ev.save_predictions(pred_path(cfg, "test"), tn, ty, softmax_np(tl))

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    plot_curves(history, curve_path(cfg), f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": "none (scratch)" if cfg.init == "scratch" else M.weight_tag(model),
        "best_epoch": best["epoch"], "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"],
        "train_time_per_epoch_s": train_time / max(1, len(history)),
        "params_m": M.count_params(model), "gmacs": M.count_gmacs(model, cfg.img_size),
        "run_dir": str(out), "device": str(device),
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict, ép kiểu theo field của Config."""
    types = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"'{p}' không có dạng KEY=VALUE")
        k, v = p.split("=", 1)
        if k not in types:
            raise KeyError(f"'{k}' không có trong Config. Hợp lệ: {sorted(types)}")
        t = types[k]
        if v.lower() in ("none", "null") and "None" in t:
            out[k] = None
        elif t.startswith("bool"):
            out[k] = v.lower() in ("1", "true", "yes")
        elif t.startswith("int"):
            out[k] = int(v)
        elif t.startswith("float"):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
