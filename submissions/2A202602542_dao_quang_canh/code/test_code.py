"""Kiểm tra tự viết cho các phần dễ sai (chạy được trên CPU, không cần dữ liệu thật):
    python -m unittest test_code -v      (từ thư mục code/)
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import dataset as D
import inference as I
import losses as L
import model as M
import train as TR


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(32, 9)
        self.y = torch.randint(0, 9, (32,))

    def test_focal_gamma0_equals_ce(self):
        self.assertAlmostEqual(L.FocalLoss(gamma=0.0)(self.logits, self.y).item(),
                               F.cross_entropy(self.logits, self.y).item(), places=6)

    def test_focal_downweights_easy(self):
        self.assertLess(L.FocalLoss(gamma=2.0)(self.logits, self.y).item(),
                        F.cross_entropy(self.logits, self.y).item())

    def test_label_smoothing_matches_torch(self):
        for eps in (0.0, 0.1):
            self.assertAlmostEqual(L.LabelSmoothingCE(eps)(self.logits, self.y).item(),
                                   F.cross_entropy(self.logits, self.y, label_smoothing=eps).item(), places=6)

    def test_class_weights(self):
        counts = [100, 100, 100, 100, 100, 100, 100, 100, 1000]
        w = L.class_weights(counts, 0.0)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertGreater(w[0].item(), w[8].item())
        wb = L.class_weights(counts, 0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)
        self.assertGreater(wb[0].item(), wb[8].item())

    def test_cutmix_lambda_matches_area_and_labels(self):
        x = torch.zeros(8, 3, 32, 32)
        x += torch.arange(8).view(8, 1, 1, 1).float()  # mỗi ảnh một giá trị => biết ảnh nguồn
        y = torch.arange(8)
        np.random.seed(1)
        torch.manual_seed(1)
        xm, (ya, yb, lam) = L.mix_batch(x, y, 1.0, "cutmix")
        frac_b = (xm[:, 0] != x[:, 0]).float().mean(dim=(1, 2))  # phần diện tích lấy từ ảnh khác
        for i in range(8):
            if int(yb[i]) != i:
                self.assertAlmostEqual(frac_b[i].item(), 1 - lam, places=5)
        self.assertTrue(torch.equal(ya, y))

    def test_mixup_and_mixed_loss(self):
        x = torch.randn(4, 3, 8, 8)
        y = torch.arange(4)
        _, (ya, yb, lam) = L.mix_batch(x, y, 1.0, "mixup")
        logits = torch.randn(4, 9)
        crit = torch.nn.CrossEntropyLoss()
        expected = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
        self.assertAlmostEqual(L.mixed_loss(crit, logits, (ya, yb, lam)).item(), expected.item(), places=6)


class TestModel(unittest.TestCase):
    def test_initial_loss_near_ln9(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        m.eval()
        with torch.no_grad():
            loss = F.cross_entropy(m(torch.randn(32, 3, 64, 64)), torch.randint(0, 9, (32,)))
        self.assertLess(abs(loss.item() - math.log(9)), 0.6)

    def test_param_groups_and_wd(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        g = {x["name"]: x for x in M.param_groups(m, 1e-4, 1e-3, 0.05)}
        self.assertEqual(g["backbone_norm_bias"]["weight_decay"], 0.0)
        self.assertEqual(g["head"]["lr"], 1e-3)
        n = sum(len(x["params"]) for x in g.values())
        self.assertEqual(n, len(list(m.parameters())))

    def test_frozen_only_head_trains_and_bn_eval(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        M.freeze_backbone(m)
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        self.assertTrue(all(n.startswith("fc") for n in trainable) and trainable)
        M.set_train_mode(m, frozen=True)
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.fc.training)

    def test_overfit_one_batch(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        x, y = torch.randn(8, 3, 64, 64), torch.randint(0, 9, (8,))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        m.train()
        for _ in range(60):
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.1)

    def test_gmacs_resnet50(self):
        m = M.build_model("resnet50", pretrained=False, init="scratch")
        self.assertAlmostEqual(M.count_gmacs(m, 224), 4.1, delta=0.4)
        self.assertAlmostEqual(M.count_params(m), 23.5, delta=0.5)  # head 9 lớp: ít hơn 25,6M của 1000 lớp


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_resnet_and_effnet(self):
        for name in ("resnet18", "efficientnet_b0"):
            torch.manual_seed(0)
            m = M.build_model(name, pretrained=False, init="scratch").eval()
            for mod in m.modules():  # BN ngẫu nhiên để phép gộp không tầm thường
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.running_mean.normal_(0, 0.1)
                    mod.running_var.uniform_(0.5, 1.5)
            f = I.fuse_conv_bn(m, check_input=torch.randn(2, 3, 64, 64), tol=1e-3)
            self.assertGreater(f.n_fused, 5)
            self.assertFalse(any(isinstance(x, torch.nn.BatchNorm2d) for x in f.modules()))

    def test_temperature_recovers_scale(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 9, 4000)
        base = rng.normal(size=(4000, 9))
        base[np.arange(4000), y] += 2.0
        logits = base * 3.0  # quá tự tin
        T = I.fit_temperature(logits, y)
        self.assertGreater(T, 1.2)  # logit bị phóng đại 3 lần => T tối ưu phải > 1
        self.assertLess(I._nll(logits, y, T), I._nll(logits, y, 1.0))
        np.testing.assert_array_equal(I.apply_temperature(logits, T).argmax(1), logits.argmax(1))

    def test_aggregate_and_views(self):
        a, b = np.random.randn(5, 9), np.random.randn(5, 9)
        for sp in ("prob", "logit"):
            np.testing.assert_allclose(I.aggregate_views([a, b], sp).sum(1), 1.0, atol=1e-9)
        x = torch.randn(2, 3, 256, 256)
        self.assertEqual(len(I.views_multicrop(x, 224)), 5)
        self.assertEqual(len(I.views_multicrop(x, 224, flip=True)), 10)
        self.assertTrue(torch.equal(I.view_hflip(I.view_hflip(x)), x))
        self.assertEqual(I.views_multiscale(x, [224, 288])[1].shape[-1], 288)


class TestTrainUtils(unittest.TestCase):
    def test_parse_overrides(self):
        o = TR.parse_overrides(["seed=3", "loss=focal", "ema_decay=none", "lr_head=1e-3", "amp=false", "sampler=balanced"])
        self.assertEqual(o, {"seed": 3, "loss": "focal", "ema_decay": None, "lr_head": 1e-3, "amp": False,
                             "sampler": "balanced"})
        with self.assertRaises(KeyError):
            TR.parse_overrides(["nope=1"])

    def test_scheduler_warmup_then_cosine(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        cfg = TR.Config(epochs=10, warmup_epochs=1.0)
        opt = TR.build_optimizer(m, cfg)
        sch = TR.build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[-1]["lr"])
            opt.step()
            sch.step()
        self.assertLess(lrs[0], lrs[9])
        self.assertAlmostEqual(lrs[10], cfg.lr_head, delta=cfg.lr_head * 0.01)
        self.assertLess(lrs[-1], cfg.lr_head * 0.01)

    def test_ema_tracks_model(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        ema = TR.EMA(m, 0.9)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        for _ in range(200):
            ema.update(m)
        p0, pe = next(m.parameters()), next(ema.module.parameters())
        self.assertTrue(torch.allclose(p0, pe, atol=1e-3))


class TestDataset(unittest.TestCase):
    def _make(self, d):
        rows = []
        for i in range(12):
            Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(Path(d) / f"{i}.jpg")
            rows.append({"Filename": f"{i}.jpg", "Label": i % 9, "Species": "x"})
        return pd.DataFrame(rows)

    def test_dataset_loader_and_transforms(self):
        with tempfile.TemporaryDirectory() as d:
            df = self._make(d)
            for aug in ("basic", "geom", "color", "trivial", "randaug"):
                x, y, f = D.DeepWeedsDataset(df, d, D.build_transforms(True, 224, aug))[3]
                self.assertEqual(tuple(x.shape), (3, 224, 224))
                self.assertEqual((y, f), (3, "3.jpg"))
            ldr = D.make_loader(df, d, D.build_transforms(False, 224), 4, train=False, num_workers=0)
            names = [n for _, _, fs in ldr for n in fs]
            self.assertEqual(names, df["Filename"].tolist())  # giữ thứ tự
            self.assertEqual(tuple(D.build_transforms(False, 288)(Image.new("RGB", (256, 256))).shape), (3, 288, 288))
            bal = D.make_loader(df, d, D.build_transforms(True, 64), 4, train=True, sampler="balanced", num_workers=0)
            self.assertEqual(len(bal), 3)

    def test_check_split_catches_leak(self):
        with tempfile.TemporaryDirectory() as d:
            df = self._make(d)
            with self.assertRaises(AssertionError):
                D.check_split(df, df, df, d, verbose=False)


if __name__ == "__main__":
    unittest.main()
