"""Sinh kaggle_baseline.ipynb: mốc T00 (ConvNeXt-tiny, công thức nền) 3 seed ghi dự đoán test + đo độ trễ.

Lý do: lần chạy Bước 4 đầu tiên trên Kaggle dựng nhầm mốc T00 với backbone mặc định resnet50 (lỗi trong
final.run_final, đã sửa), nên mốc đó không dùng được. Notebook này chạy lại mốc đúng backbone.
Chạy: python make_kaggle_baseline_notebook.py
"""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("# Mốc T00 đúng backbone (ConvNeXt-tiny) ×3 seed + độ trễ\n"
   "Chạy lại mốc vì lần trước dựng nhầm backbone mặc định (resnet50). Cần **GPU T4** và **Internet**.")
code('''
!pip -q install timm
import os, sys, platform
REPO_URL = "https://github.com/quangcanh02122005/K4-Track4-Day2-Deeplearning-Advance.git"
REPO_DIR = "/kaggle/working/repo"
if not os.path.exists(REPO_DIR):
    !git clone -q -b main {REPO_URL} {REPO_DIR}
CODE_DIR = f"{REPO_DIR}/submissions/2A202602542_dao_quang_canh/code"
sys.path.insert(0, CODE_DIR)
WORK = "/kaggle/working/out"
os.makedirs(WORK, exist_ok=True)
import numpy as np, pandas as pd, torch, timm
print("python", platform.python_version(), "| torch", torch.__version__, "| timm", timm.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
''')
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
md("### Mốc T00: ConvNeXt-tiny, công thức nền, 1-view 224, seed 0,1,2 (test mở đúng một lần mỗi seed)")
code('''
import final
final.run_baseline(COMMON, "convnext_tiny", seeds=(0, 1, 2))
''')
code('''
!python {REPO_DIR}/eval.py score --pred "{WORK}/predictions/T00_seed*_test.csv" --test-csv {LABELS_DIR}/test_subset0.csv --labels {LABELS_DIR}/labels.csv --tag T00 --out {WORK}/eval_out
''')
md("### Độ trễ ConvNeXt-tiny ở 224 và 288 (batch 1 và 32; warmup 10, synchronize, 100 lần)")
code('''
import json, benchmark as B, model as M
net = M.build_model("convnext_tiny", False, 9)
lat = []
for res in (224, 288):
    for bs in (1, 32):
        for dt in ("fp32", "amp", "fp16"):
            lat.append(B.latency_report(net, bs, res, dtype=dt))
json.dump(lat, open(f"{WORK}/latency_convnext_tiny.json", "w"), indent=1)
pd.DataFrame(lat)[["gpu", "dtype", "batch", "img_size", "p50", "p95", "p99", "images_per_s"]]
''')
code('''
# Dọn dữ liệu và checkpoint để output nhỏ (chỉ giữ log/curves/predictions), rồi nén.
!rm -rf data
!find {WORK} -name "*.pt" -delete
!cd /kaggle/working && zip -qr baseline_outputs.zip out && ls -la baseline_outputs.zip
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("kaggle_baseline.ipynb", "w", encoding="utf8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(len(cells), "cells")
