"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu S1-S6: README.md mục 2.1. Giao diện giữ đúng như bộ khung starter/.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
TOTAL_IMAGES = 17509
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv nguyên bản (S1): không sửa, không lọc, không chia lại."""
    labels_dir = Path(labels_dir)
    dfs = [pd.read_csv(labels_dir / f"{s}_subset{fold}.csv") for s in ("train", "val", "test")]
    for name, df in zip(("train", "val", "test"), dfs):
        missing = {"Filename", "Label"} - set(df.columns)  # CSV chia sẵn của tác giả không có cột Species
        assert not missing, f"{name}_subset{fold}.csv thiếu cột {missing}"
    return tuple(dfs)


def check_split(train_df, val_df, test_df, images_dir, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md mục 2.1). Dừng ngay (assert) nếu vi phạm."""
    sets = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in sets.items()}
    total = sum(n.values())
    per_class = {k: np.bincount(v["Label"].to_numpy(), minlength=NUM_CLASSES).tolist() for k, v in sets.items()}
    names = {k: set(v["Filename"]) for k, v in sets.items()}
    overlap = {"train&val": len(names["train"] & names["val"]),
               "train&test": len(names["train"] & names["test"]),
               "val&test": len(names["val"] & names["test"])}
    union = len(names["train"] | names["val"] | names["test"])
    images_dir = Path(images_dir)
    missing = [f for k in sets for f in sets[k]["Filename"] if not (images_dir / f).exists()]
    ratio = {k: v / total for k, v in n.items()}

    if verbose:
        print("Số ảnh:", n, "| tổng", total)
        print("Tỉ lệ:", {k: round(v, 4) for k, v in ratio.items()})
        print(pd.DataFrame(per_class, index=CLASS_NAMES))
        print("Giao:", overlap, "| hợp:", union, "| file thiếu:", len(missing))

    assert all(v == 0 for v in overlap.values()), f"Rò rỉ dữ liệu giữa các tập: {overlap}"
    assert union == TOTAL_IMAGES and total == TOTAL_IMAGES, f"Hợp ba tập = {union}, tổng = {total}, kỳ vọng {TOTAL_IMAGES}"
    assert not missing, f"{len(missing)} file không có trong {images_dir}, ví dụ {missing[:3]}"
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(ratio[k] - target) < 0.01, f"Tỉ lệ {k} = {ratio[k]:.4f} lệch quá 1 điểm % so với {target}: báo giảng viên"
    return {"n": n, "ratio": ratio, "per_class": per_class, "overlap": overlap, "union": union, "missing": len(missing)}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Transform theo `train` và mức augmentation `aug`.

    Train (mọi mức): RandomResizedCrop(img_size, scale=(0.35, 1)) + lật NGANG. Không lật dọc (ảnh cỏ dại
    chụp từ trên xuống nên lật dọc/xoay vẫn hợp lệ về mặt vật lý; ở đây giữ cơ bản để công thức nền gọn, và
    thử xoay/lật dọc như một mức augmentation riêng: "geom").
      basic   : crop + hflip
      geom    : basic + vflip + xoay 90 độ ngẫu nhiên
      color   : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
      trivial : basic + TrivialAugmentWide
      randaug : basic + RandAugment(2, 9)
    Val/test: ảnh 256x256 -> CenterCrop(img_size) nếu img_size < 256; img_size >= 256 thì resize về img_size
    (không crop). Không có augmentation ngẫu nhiên.
    """
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        pre = [T.Resize(256)] if img_size <= 256 else [T.Resize(img_size)]
        if img_size < 256:
            pre.append(T.CenterCrop(img_size))
        return T.Compose(pre + norm)

    ops = [T.RandomResizedCrop(img_size, scale=(0.35, 1.0)), T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "geom":
        ops += [T.RandomVerticalFlip(), T.RandomApply([T.RandomRotation((90, 90))], p=0.5)]
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose(ops + norm)


class DeepWeedsDataset(Dataset):
    """__getitem__(i) -> (tensor, int(label), filename). `cache=True` giữ ảnh PIL trong RAM (~2 GB cho train)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: bool = False):
        self.files = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self._cache = [self._open(f) for f in self.files] if cache else None

    def _open(self, fname: str):
        with Image.open(self.images_dir / fname) as im:
            return im.convert("RGB")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = self._cache[i] if self._cache is not None else self._open(self.files[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.files[i]


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir, transform, batch_size: int, train: bool,
                sampler: str | None = None, num_workers: int = 2, seed: int = 0, cache: bool = False):
    """DataLoader. Val/test: không shuffle, giữ thứ tự df. Train: shuffle hoặc sampler "balanced"."""
    ds = DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=_worker_init, generator=g, persistent_workers=num_workers > 0)
    if not train:
        return DataLoader(ds, shuffle=False, drop_last=False, **kw)
    if sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        w = 1.0 / counts[np.asarray(ds.labels)]
        ws = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                   replacement=True, generator=g)
        return DataLoader(ds, sampler=ws, drop_last=True, **kw)
    if sampler is not None:
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(ds, shuffle=True, drop_last=True, **kw)
