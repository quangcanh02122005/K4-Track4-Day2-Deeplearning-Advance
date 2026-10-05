"""final.py - Bước 4: dự đoán chung kết (độ phân giải test + temperature scaling) và chạy hàng loạt theo seed.

Quy tắc: test chỉ được mở MỘT lần cho mỗi seed. `write_final_predictions` tính logit test đúng một lần,
rồi ghi từ đó hai file (đã / chưa temperature scaling); nếu file test đã có thì bỏ qua (không mở test lại).
T khớp trên VAL (cùng độ phân giải), không dùng test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import dataset as D
import inference as I
import model as M
import train as TR
from train import Config, ev, pred_path, run, run_dir


def _pred_file(cfg: Config, split: str, exp_id: str) -> Path:
    return Path(cfg.pred_dir) / f"{exp_id}_seed{cfg.seed}_{split}.csv"


def write_final_predictions(cfg: Config, res: int = 288) -> dict:
    """Nạp best.pt của `cfg`, suy luận 1-view ở độ phân giải `res`, khớp T trên val, ghi:
        <exp_id>_seed<k>_val.csv / _test.csv           (đã temperature scaling)
        <exp_id>uncal_seed<k>_val.csv / _test.csv      (chưa temperature scaling)
    """
    out = run_dir(cfg)
    test_file = _pred_file(cfg, "test", cfg.exp_id)
    if test_file.exists():
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã có {test_file.name}, KHÔNG mở test lại")
        return json.loads((out / "final.json").read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.build_model(cfg.backbone, False, D.NUM_CLASSES, init="finetune").to(device)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    model.eval()
    dc = M.data_config(model)
    _, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    tf = D.build_transforms(False, res, "basic", dc["mean"], dc["std"])

    def loader(df):
        return D.make_loader(df, cfg.images_dir, tf, 64, False, num_workers=cfg.num_workers)

    vn, vy, vlog = I.predict_logits(model, loader(val_df), device)
    T = I.fit_temperature(vlog, vy)
    uncal = f"{cfg.exp_id}uncal"
    ev.save_predictions(_pred_file(cfg, "val", cfg.exp_id), vn, vy, I.apply_temperature(vlog, T))
    ev.save_predictions(_pred_file(cfg, "val", uncal), vn, vy, I._softmax(vlog))

    tn, ty, tlog = I.predict_logits(model, loader(test_df), device)  # test: đúng một lần
    np.save(out / "test_logits_final.npy", tlog)
    ev.save_predictions(test_file, tn, ty, I.apply_temperature(tlog, T))
    ev.save_predictions(_pred_file(cfg, "test", uncal), tn, ty, I._softmax(tlog))

    m = ev.compute_metrics(vy, vlog.argmax(1), I.apply_temperature(vlog, T))
    info = {"exp_id": cfg.exp_id, "seed": cfg.seed, "res": res, "T": T, "val_macro_f1": m["macro_f1"],
            "val_top1": m["top1"], "val_ece_calibrated": m["ece"],
            "val_ece_uncal": ev.compute_metrics(vy, vlog.argmax(1), I._softmax(vlog))["ece"]}
    (out / "final.json").write_text(json.dumps(info, indent=2))
    print(info)
    return info


def run_final(base: dict, final_overrides: dict, seeds=(0, 1, 2), res: int = 288, exp_id: str = "F01"):
    """Với mỗi seed: huấn luyện cấu hình cuối (KHÔNG ghi test trong run), ghi dự đoán chung kết ở `res` + T;
    rồi chạy mốc T00 (công thức nền, 1 view 224) có ghi test. Trả về list dict tóm tắt."""
    rows = []
    for seed in seeds:
        cfg = Config(**{**base, **final_overrides, "exp_id": exp_id, "seed": seed, "tag": "final",
                        "save_test_predictions": False})
        run(cfg)
        info = write_final_predictions(cfg, res)
        rows.append(info)
    for seed in seeds:
        cfg0 = Config(**{**base, "exp_id": "T00", "seed": seed, "tag": "baseline", "save_test_predictions": True})
        run(cfg0)
    return rows
