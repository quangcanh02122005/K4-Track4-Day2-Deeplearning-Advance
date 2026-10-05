"""Sinh kaggle_step4.ipynb: chỉ chạy phần còn lại của Bước 4 trên Kaggle (Colab hết hạn mức GPU).
Chạy: python make_kaggle_notebook.py

Cần: GPU (T4), Internet BẬT, và một Kaggle Dataset chứa best.pt của T00 seed 0 (train ở Colab).
"""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("# Lab Day 2 — Bước 4 trên Kaggle (F01 ×3 seed + mốc T00)\n"
   "Colab hết hạn mức GPU nên phần còn lại của Bước 4 chạy ở đây. Bước 1–3 đã xong ở Colab.\n"
   "Cần bật **GPU T4** và **Internet**. Dataset chứa `best.pt` của T00 seed 0 là tùy chọn.")
code('''
!pip -q install timm
import os, sys, glob, platform
REPO_URL = "https://github.com/quangcanh02122005/K4-Track4-Day2-Deeplearning-Advance.git"
REPO_DIR = "/kaggle/working/repo"
if not os.path.exists(REPO_DIR):
    !git clone -q -b lab-day2-code {REPO_URL} {REPO_DIR}
CODE_DIR = f"{REPO_DIR}/submissions/2A202602542_dao_quang_canh/code"
sys.path.insert(0, CODE_DIR)
WORK = "/kaggle/working/out"
os.makedirs(WORK, exist_ok=True)
import numpy as np, pandas as pd, torch, timm
print("python", platform.python_version(), "| torch", torch.__version__, "| timm", timm.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
ckpts = glob.glob("/kaggle/input/**/best.pt", recursive=True)
print("checkpoint T00 seed 0 (tùy chọn):", ckpts)
# Có checkpoint từ Colab thì dùng lại cho T00 seed 0; không có thì train T00 seed 0 ngay tại đây (cùng phần cứng với F01).
BASELINE_SEEDS = (1, 2) if ckpts else (0, 1, 2)
''')
md("### Tải dữ liệu (MD5 phải khớp)")
code('''
import hashlib
os.makedirs("data/labels", exist_ok=True)
if not os.path.exists("data/images.zip"):
    !wget -q -O data/images.zip "https://zenodo.org/records/7939060/files/images.zip?download=1"
h = hashlib.md5()
with open("data/images.zip", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
assert h.hexdigest() == "b7b30f96d466fba86016aa5a26606e0f", f"MD5 sai: {h.hexdigest()}"
!unzip -q -n data/images.zip -d data/
BASE_URL = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
for name in ["labels", "train_subset0", "val_subset0", "test_subset0"]:
    !wget -q -O data/labels/{name}.csv {BASE_URL}/{name}.csv
!rm -f data/images.zip
IMAGES_DIR = "data"
LABELS_DIR = "data/labels"
COMMON = dict(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, out_dir=f"{WORK}/runs",
              pred_dir=f"{WORK}/predictions", curves_dir=f"{WORK}/curves", num_workers=4)
import dataset as D
tr, va, te = D.load_split(LABELS_DIR, 0)
print(D.check_split(tr, va, te, IMAGES_DIR)["n"])
''')
md("### Chung kết: F01 (TrivialAugment + EMA 0.998, test ở 288 + temperature scaling) seed 0,1,2 và mốc T00 seed 1,2")
code('''
import final
FINAL = {"backbone": "convnext_tiny", "aug": "trivial", "ema_decay": 0.998}
final_rows = final.run_final(COMMON, FINAL, seeds=(0, 1, 2), res=288, exp_id="F01", baseline_seeds=BASELINE_SEEDS)
pd.DataFrame(final_rows)
''')
md("### Mốc T00 seed 0: nếu có checkpoint từ Colab thì chỉ ghi dự đoán val/test (không thì đã train ở ô trên)")
code('''
from train import Config
if ckpts:
    cfg0 = Config(**{**COMMON, "backbone": "convnext_tiny", "exp_id": "T00", "seed": 0, "tag": "baseline"})
    final.write_baseline_predictions(cfg0, ckpts[0])
''')
md("### Chấm bằng eval.py")
code('''
for tag in ("F01", "T00"):
    !python {REPO_DIR}/eval.py score --pred "{WORK}/predictions/{tag}_seed*_test.csv" --test-csv {LABELS_DIR}/test_subset0.csv --labels {LABELS_DIR}/labels.csv --tag {tag} --out {WORK}/eval_out
!python {REPO_DIR}/eval.py grade --final "{WORK}/predictions/F01_seed*_test.csv" --baseline "{WORK}/predictions/T00_seed*_test.csv" --test-csv {LABELS_DIR}/test_subset0.csv --labels {LABELS_DIR}/labels.csv --uncal "{WORK}/predictions/F01uncal_seed*_test.csv" --final-val "{WORK}/predictions/F01_seed*_val.csv" --out {WORK}/eval_out
''')
code('''
# Dọn checkpoint lớn (không cần mang về), giữ log/curves/predictions, rồi nén để tải về.
!find {WORK} -name "*.pt" -delete
!cd /kaggle/working && zip -qr step4_outputs.zip out && ls -la step4_outputs.zip
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("kaggle_step4.ipynb", "w", encoding="utf8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(len(cells), "cells")
