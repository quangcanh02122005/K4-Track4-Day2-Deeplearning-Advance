"""Sinh lab_day2.ipynb (notebook chạy trên Colab). Chạy: python make_notebook.py"""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("# Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện, suy luận\n"
   "Chạy trên Colab (GPU T4), từng ô theo thứ tự. Mọi thí nghiệm đi qua `train.run(Config(...))`, ghi ra Drive "
   "nên phiên bị ngắt vẫn chạy tiếp được (thí nghiệm đã xong được bỏ qua).")
md("## 0. Cài đặt")
code('''
!pip -q install timm openpyxl
from google.colab import drive
drive.mount("/content/drive")
import os, sys, platform
REPO_URL = "https://github.com/quangcanh02122005/K4-Track4-Day2-Deeplearning-Advance.git"
REPO_DIR = "/content/K4-Track4-Day2-Deeplearning-Advance"
if not os.path.exists(REPO_DIR):
    !git clone -q -b lab-day2-code {REPO_URL} {REPO_DIR}
else:
    !git -C {REPO_DIR} pull -q origin lab-day2-code
CODE_DIR = f"{REPO_DIR}/submissions/2A202602542_dao_quang_canh/code"
sys.path.insert(0, CODE_DIR)   # import dataset, model, train...; train.py tự tìm eval.py ở thư mục cha
WORK = "/content/drive/MyDrive/deepweeds_lab"   # log, checkpoint, predictions, curves nằm ở Drive
os.makedirs(WORK, exist_ok=True)
import numpy as np, pandas as pd, torch, timm
print("python", platform.python_version(), "| torch", torch.__version__, "| timm", timm.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
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
!ls data | head
''')
code('''
IMAGES_DIR = "data/images"   # xem `ls data` ở ô trên; đổi nếu ảnh nằm thư mục khác
LABELS_DIR = "data/labels"
COMMON = dict(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, out_dir=f"{WORK}/runs",
              pred_dir=f"{WORK}/predictions", curves_dir=f"{WORK}/curves", num_workers=2)
''')
md("## Bước 0 — EDA và kiểm tra pipeline")
code('''
import dataset as D
import matplotlib.pyplot as plt
train_df, val_df, test_df = D.load_split(LABELS_DIR, 0)
split_info = D.check_split(train_df, val_df, test_df, IMAGES_DIR)   # dừng nếu vi phạm S1-S4
pd.DataFrame(split_info["per_class"], index=D.CLASS_NAMES).plot.bar(figsize=(9, 3.5), title="Số ảnh theo lớp (fold 0)")
plt.tight_layout(); plt.show()
cnt = np.bincount(pd.concat([train_df, val_df, test_df]).Label, minlength=9)
print("Tổng mỗi lớp:", dict(zip(D.CLASS_NAMES, cnt)), "| lớn nhất / nhỏ nhất =", cnt.max() / cnt.min())
''')
code('''
# 3 ảnh mẫu mỗi lớp
from PIL import Image
fig, ax = plt.subplots(9, 3, figsize=(7, 20))
for c in range(9):
    for j, fn in enumerate(train_df[train_df.Label == c].Filename.iloc[:3]):
        ax[c, j].imshow(Image.open(f"{IMAGES_DIR}/{fn}")); ax[c, j].axis("off")
    ax[c, 0].set_title(D.CLASS_NAMES[c], fontsize=8, loc="left")
plt.tight_layout(); plt.show()
''')
code('''
# Kiểm tra pipeline (GUIDE 1.3): seed, loss ban đầu ~ ln 9, overfit 1 batch, ảnh sau augmentation
import math, torch.nn.functional as F
import model as M, train as TR
TR.set_seed(0)
m = M.build_model("resnet50", True, 9).cuda()
dc = M.data_config(m)
ld = D.make_loader(train_df, IMAGES_DIR, D.build_transforms(True, 224, "basic", dc["mean"], dc["std"]), 16, True, num_workers=2)
x, y, f = next(iter(ld)); x, y = x.cuda(), y.cuda()
m.eval()
with torch.no_grad():
    print("loss ban đầu =", F.cross_entropy(m(x), y).item(), "| ln 9 =", math.log(9))
m.train(); opt = torch.optim.AdamW(m.parameters(), 1e-4)
for i in range(40):
    opt.zero_grad(); loss = F.cross_entropy(m(x), y); loss.backward(); opt.step()
print("loss sau 40 bước trên 1 batch =", loss.item(), "(phải gần 0)")
mean = torch.tensor(dc["mean"]).view(3, 1, 1); std = torch.tensor(dc["std"]).view(3, 1, 1)
fig, ax = plt.subplots(2, 6, figsize=(14, 5))
for a, im, lb in zip(ax.flat, x.cpu(), y.cpu()):
    a.imshow((im * std + mean).clamp(0, 1).permute(1, 2, 0)); a.set_title(D.CLASS_NAMES[lb], fontsize=8); a.axis("off")
plt.show(); del m, opt
''')
md("## Bước 1 — So sánh backbone (cùng công thức nền, seed 0)")
code('''
import experiments as E
res_b = E.run_group(E.backbone_experiments(), COMMON, seed=0)
res_b[["exp_id", "backbone", "weight_tag", "params_m", "gmacs", "val_macro_f1", "val_top1", "train_time_per_epoch_s", "best_epoch"]]
''')
md("## Bước 2 — Công thức huấn luyện (backbone chọn từ Bước 1)")
code('''
BEST_BACKBONE = "resnet50"   # TODO: đổi theo kết quả Bước 1 (dựa trên val)
BASE = {**COMMON, "backbone": BEST_BACKBONE}
res_t = E.run_group(E.training_experiments(), BASE, seed=0)
t00 = res_t.set_index("exp_id").loc["T00", "val_macro_f1"]
res_t["delta_vs_T00"] = res_t["val_macro_f1"] - t00
res_t[["exp_id", "overrides", "val_macro_f1", "val_top1", "delta_vs_T00", "best_epoch"]]
''')
md("## Bước 3 — Suy luận (trên val, không huấn luyện lại)")
code('''
import inference as I, benchmark as B, eval as ev
dev = torch.device("cuda")
RUN = "T00"   # TODO: cấu hình cần phân tích suy luận
model = M.build_model(BEST_BACKBONE, False, 9).to(dev)
model.load_state_dict(torch.load(f"{WORK}/runs/{RUN}/seed0/best.pt", map_location=dev)); model.eval()
dc = M.data_config(model)
def val_loader(size):
    return D.make_loader(val_df, IMAGES_DIR, D.build_transforms(False, size, "basic", dc["mean"], dc["std"]), 64, False, num_workers=2)
rows = []
def score(name, probs, y):
    m = ev.compute_metrics(y, probs.argmax(1), probs)
    rows.append({"method": name, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"]})
names, y, lg = I.predict_logits(model, val_loader(224), dev)
score("I00 1-view", I._softmax(lg), y)
_, _, lg_f = I.predict_logits(model, val_loader(224), dev, view=I.view_hflip)
score("I01 TTA hflip (prob)", I.aggregate_views([lg, lg_f], "prob"), y)
score("I03 TTA hflip (logit)", I.aggregate_views([lg, lg_f], "logit"), y)
_, _, vs = I.predict_logits_views(model, val_loader(256), dev, lambda x: I.views_multicrop(x, 224))
score("I02 5-crop (prob)", I.aggregate_views(vs, "prob"), y)
for s in (224, 256, 288, 320):
    _, _, l = I.predict_logits(model, val_loader(s), dev); score(f"I04 test res {s}", I._softmax(l), y)
T = I.fit_temperature(lg, y)
score(f"I07 temperature T={T:.3f}", I.apply_temperature(lg, T), y)
pd.DataFrame(rows)
''')
code('''
# Độ trễ: batch 1 và 32; warmup 10, synchronize, 100 lần; fp32 / amp / fp16 / gộp BN / TTA
fused = I.fuse_conv_bn(model, check_input=torch.randn(2, 3, 224, 224, device=dev))
lat = []
for tag, mdl, kw in [("fp32", model, {}), ("amp", model, {"dtype": "amp"}), ("fp16", model, {"dtype": "fp16"}),
                     ("fp32+fusedBN", fused, {"fused_bn": True})]:
    for bs in (1, 32):
        lat.append({"config": tag, **B.latency_report(mdl, bs, 224, **kw)})
lat.append({"config": "TTA hflip K=2", **B.tta_latency(model, 2, batch_size=1, img_size=224)})
pd.DataFrame(lat)[["config", "gpu", "dtype", "batch", "fused_bn", "k_views", "p50", "p95", "p99", "images_per_s"]]
''')
md("## Bước 4 — Chung kết: ≥ 3 seed, test đúng một lần mỗi seed\n"
   "Chỉ chạy sau khi chốt cấu hình trên **val**. Không quay lại sửa sau khi đã xem test.")
code('''
FINAL = {"backbone": BEST_BACKBONE, "tag": "final"}   # TODO: gộp yếu tố tốt nhất ở Bước 2, vd {"mix": "cutmix", "loss": "ls", "ema_decay": 0.998}
for seed in (0, 1, 2):
    E.run_group({"F01": FINAL}, COMMON, seed=seed, save_test_predictions=True)
    E.run_group({"T00": {"backbone": BEST_BACKBONE, "tag": "baseline"}}, COMMON, seed=seed, save_test_predictions=True)
''')
code('''
for tag in ("F01", "T00"):
    !python {REPO_DIR}/eval.py score --pred "{WORK}/predictions/{tag}_seed*_test.csv" --test-csv {LABELS_DIR}/test_subset0.csv --labels {LABELS_DIR}/labels.csv --tag {tag} --out {WORK}/eval_out
!python {REPO_DIR}/eval.py grade --final "{WORK}/predictions/F01_seed*_test.csv" --baseline "{WORK}/predictions/T00_seed*_test.csv" --test-csv {LABELS_DIR}/test_subset0.csv --labels {LABELS_DIR}/labels.csv --out {WORK}/eval_out
''')
md("## Bước 5 — Sản phẩm\n`curves/`, `predictions/`, `runs/` đã nằm trong `WORK` trên Drive. "
   "Tải về và chép vào `submissions/<mssv>/`; dựng `results.xlsx` từ `runs/*/seed*/summary.json`.")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("lab_day2.ipynb", "w", encoding="utf8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(len(cells), "cells")
