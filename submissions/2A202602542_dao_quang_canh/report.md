# Báo cáo Lab Day 2: Backbone, công thức huấn luyện và suy luận trên DeepWeeds

Sinh viên: Đào Quang Cảnh (2A202602542). Mọi con số lấy từ log trong `logs/`, `eval_out/` và `results.xlsx`
(truy ngược theo `exp_id`). Số của bài báo gốc được ghi rõ là **trích dẫn**.

## 1. Tóm tắt

- **Bài toán:** phân loại 9 lớp ảnh cỏ dại DeepWeeds (17.509 ảnh), dùng đúng fold 0 chia sẵn (train/val/test = 10.501/3.501/3.507).
- **Đã làm:** 7 backbone (B01–B07), 17 thí nghiệm công thức huấn luyện (T00–T16), 7 phương pháp suy luận kèm độ trễ,
  chung kết 3 seed so với mốc 3 seed. Test mở đúng một lần cho mỗi cấu hình và seed (trừ một lỗi nêu ở mục 8).
- **Cấu hình tốt nhất (F01):** `convnext_tiny` (`in12k_ft_in1k`) + TrivialAugment + EMA 0,998, suy luận 1-view ở **288** rồi temperature scaling
  (T khớp trên val).
- **Kết quả test (3 seed, mean ± std):** top-1 **0,9799 ± 0,0027**, macro-F1 **0,9751 ± 0,0035**, ECE 0,0047 ± 0,0011.
  Mốc (cùng backbone, công thức nền, 224, 1-view): top-1 0,9751 ± 0,0007, macro-F1 0,9686 ± 0,0001, ECE 0,0172 ± 0,0012.
  Δ macro-F1 = **+0,0065**, lớn hơn std (0,0035) nhưng nhỏ hơn 0,01: cải thiện có nhưng khiêm tốn.
- **Kết luận chính:** yếu tố ảnh hưởng nhiều nhất là **backbone cùng trọng số tiền huấn luyện** (chênh tới 0,2 macro-F1 val ở Bước 1),
  sau đó là độ phân giải lúc test (+0,010 val) và hiệu chuẩn nhiệt độ (ECE 0,017 → 0,005). Gần như toàn bộ các yếu tố công thức huấn luyện
  (augmentation, loss, sampler, EMA) nằm trong khoảng nhiễu của 1 seed.
- **Độ trễ (Tesla T4, batch 1, fp32, p95):** 9,4 ms ở 224 và 12,7 ms ở 288, thấp hơn nhiều so với ngân sách 100 ms.

## 2. Dữ liệu và thiết lập

**Dữ liệu.** Ảnh từ Zenodo (MD5 `b7b30f96…` khớp), nhãn và fold từ GitHub của tác giả; không sửa, lọc hay chia lại (S1).
Kiểm tra bắt buộc (log trong notebook, `logs/kaggle_baseline_kernel.log`): 10.501 / 3.501 / 3.507 ảnh (59,97% / 20,00% / 20,03%),
giao từng cặp tập rỗng, hợp đúng 17.509, không thiếu file.

| Lớp | train | val | test | tổng | Table 1 bài báo (trích dẫn) |
|---|---|---|---|---|---|
| Chinee apple | 675 | 225 | 226 | 1.126 | 1.125 |
| Lantana | 637 | 213 | 213 | 1.063 | 1.064 |
| Parkinsonia | 618 | 206 | 207 | 1.031 | 1.031 |
| Parthenium | 613 | 204 | 205 | 1.022 | 1.022 |
| Prickly acacia | 637 | 212 | 213 | 1.062 | 1.062 |
| Rubber vine | 605 | 202 | 202 | 1.009 | 1.009 |
| Siam weed | 644 | 215 | 215 | 1.074 | 1.074 |
| Snake weed | 609 | 203 | 204 | 1.016 | 1.016 |
| Negative | 5.463 | 1.821 | 1.822 | 9.106 | 9.106 |

Số đếm khớp bài báo, chỉ lệch 1 ảnh ở hai lớp (Chinee apple và Lantana). Mất cân bằng: lớp `Negative` chiếm 52%, tỉ lệ lớp lớn nhất so với
nhỏ nhất là 9.106 / 1.009 ≈ 9,0. Vì vậy chỉ số chính là macro-F1.

**Kiểm tra pipeline trước khi chạy thật** (notebook Colab, ResNet-50 pretrained, head mới): loss ban đầu 2,184 (ln 9 = 2,197);
ảnh sau augmentation hiển thị cùng nhãn và khớp nhau. Overfit 1 batch (16 ảnh, 40 bước, LR 1e-4, có augmentation ngẫu nhiên): loss giảm
từ 2,18 xuống 0,44. Mức này chưa "gần 0" như GUIDE gợi ý vì chỉ chạy 40 bước; mình xem là đủ để thấy pipeline học được, không dùng nó làm bằng chứng mạnh.

**Công thức nền (T00).** `timm`, khởi tạo ImageNet, tinh chỉnh toàn bộ, head 9 lớp mới; AdamW, LR backbone 1e-4 / head 1e-3,
weight decay 0,05 (không áp dụng cho norm và bias), warmup 1 epoch rồi cosine; CE; batch 64; 12 epoch; AMP; clip gradient 1,0;
train: `RandomResizedCrop(224, scale 0,35–1)` + lật ngang; val/test: `Resize(256)` + `CenterCrop(224)`; chuẩn hoá theo cấu hình trọng số.
Checkpoint chọn theo macro-F1 val cao nhất (hòa: epoch sớm hơn).

**Môi trường.** Bước 1–3 chạy trên Google Colab (Tesla T4; Python 3.13.15, torch 2.11.0+cu130, timm 1.0.29). Bước 4 chạy trên Kaggle
(Tesla T4; torch 2.11.0+cu128, timm 1.0.29) vì Colab hết hạn mức GPU. Seed: 0 cho Bước 1–2; 0, 1, 2 cho Bước 4. Seed chỉ đổi khởi tạo head,
thứ tự batch và augmentation, không đổi cách chia (S5).

## 3. So sánh backbone (Bước 1)

Cùng công thức nền, cùng split, seed 0 (`results.xlsx`, sheet `Backbones`; biểu đồ `figures/B_backbones_val_f1.png`).

| exp_id | backbone (tag trọng số) | tham số (M) | GMAC | macro-F1 val | top-1 val | s/epoch | epoch tốt nhất |
|---|---|---|---|---|---|---|---|
| B01 | resnet50 (`a1_in1k`) | 23,53 | 4,09 | 0,8049 | 0,8578 | 46,0 | 12 |
| B02 | resnext50_32x4d (`a1h_in1k`) | 23,00 | 4,23 | 0,7874 | 0,8360 | 51,1 | 11 |
| **B03** | **convnext_tiny (`in12k_ft_in1k`)** | 27,83 | 4,45 | **0,9660** | **0,9737** | 53,3 | 11 |
| B04 | deit_small_patch16_224 (`fb_in1k`) | 21,67 | 4,24 | 0,9404 | 0,9560 | 45,9 | 12 |
| B05 | swin_tiny (`ms_in1k`) | 27,53 | 4,49 | 0,9614 | 0,9712 | 63,8 | 11 |
| B06 | efficientnet_b0 (`ra_in1k`) | 4,02 | 0,38 | 0,8223 | 0,8669 | 41,6 | 12 |
| B07 | mobilenetv3_large_100 (`ra_in1k`) | 4,21 | 0,22 | 0,7688 | 0,8309 | 39,1 | 11 |

GMAC đếm bằng `torch.utils.flop_counter` (FLOPs/2; chỉ tính conv, matmul, attention). Tham số tính với head 9 lớp nên nhỏ hơn số của
ImageNet-1k.

**Nhận xét.**
- Chênh lệch lớn nhất nằm giữa hai nhóm: ConvNeXt/Swin/DeiT (0,94–0,97) so với các CNN còn lại (0,77–0,82). Nhưng **đây không phải so sánh thuần kiến trúc**:
  các tag trọng số khác nhau về công thức huấn luyện và dữ liệu tiền huấn luyện (ConvNeXt dùng ImageNet-12k rồi tinh chỉnh ImageNet-1k;
  ResNet-50, ResNeXt, EfficientNet, MobileNet là các trọng số theo công thức `a1`/`a1h`/`ra`). Mình không tách riêng hai yếu tố này, nên không kết luận
  "ConvNeXt tốt hơn ResNet vì kiến trúc". ResNet-50 hội tụ chậm với công thức nền: epoch 1 mới đạt macro-F1 0,09 (gần như đoán toàn `Negative`).
- Thứ hạng khác thứ hạng ImageNet thông thường (ResNet-50 và ResNeXt-50 xếp cuối), phù hợp với giải thích về trọng số ở trên, nhưng giải thích này mình chưa kiểm chứng.
- **FLOPs không dự đoán được thời gian train:** MobileNetV3 chỉ có 0,22 GMAC nhưng 39 s/epoch, trong khi ResNet-50 (4,09 GMAC) là 46 s; Swin-T (4,49 GMAC) chậm nhất (63,8 s).
  Nhiều khả năng với mạng nhỏ, thời gian bị chi phối bởi nạp ảnh và tăng cường dữ liệu chứ không phải tính toán (mình chưa đo riêng điểm nghẽn).
- Mức chênh giữa ConvNeXt (0,9660) và Swin (0,9614) là 0,0046 với 1 seed, không phân biệt được với nhiễu (mục 4).
- **Lý do chọn:** `convnext_tiny` có macro-F1 val cao nhất, thời gian train hợp lý (53 s/epoch) và nhanh hơn Swin (63,8 s). Chỉ chọn 1 backbone cho Bước 2 và 3 do ngân sách GPU.

## 4. Công thức huấn luyện (Bước 2)

Backbone `convnext_tiny`, seed 0, mỗi thí nghiệm chỉ khác T00 đúng một yếu tố (trừ T16 là tổ hợp có chủ đích). Mình **không** dùng cách tham lam
(không cập nhật nền sau mỗi lần thắng). Sheet `Training`; biểu đồ `figures/T_delta_vs_T00.png`.

| exp_id | trục | khác T00 | macro-F1 val | Δ so với T00 |
|---|---|---|---|---|
| T00 | – | công thức nền | 0,9660 | 0 |
| T01 | A khởi tạo | đóng băng backbone, chỉ train head | 0,8567 | −0,1093 |
| T02 | A khởi tạo | từ đầu | 0,2920 | −0,6740 |
| T03 | B aug | + ColorJitter | 0,9647 | −0,0013 |
| T04 | B aug | TrivialAugmentWide | 0,9710 | **+0,0050** |
| T05 | B aug | + lật dọc, xoay 90° | 0,9664 | +0,0003 |
| T06 | B | CutMix (α=1) | 0,9641 | −0,0020 |
| T07 | B | Mixup (α=1) | 0,9667 | +0,0007 |
| T08 | C loss | label smoothing 0,1 | 0,9645 | −0,0015 |
| T09 | C loss | focal (γ=2) | 0,9630 | −0,0031 |
| T10 | C loss | CE có trọng số 1/n_c | 0,9649 | −0,0012 |
| T11 | D sampler | cân bằng lớp | 0,9636 | −0,0025 |
| T12 | E LR | LR head = LR backbone (1e-4) | 0,9657 | −0,0003 |
| T13 | F | EMA 0,998 | 0,9680 | +0,0019 |
| T14 | G | độ phân giải 256 | 0,9643 | −0,0017 |
| T15 | G | 20 epoch | 0,9684 | +0,0023 |
| T16 | tổ hợp | T04 + T13 | 0,9711 | +0,0050 |

**So với nhiễu.** Ba seed của mốc T00 trên Kaggle cho macro-F1 val 0,9703, 0,9673, 0,9668, std = 0,0019 (chỉ 3 seed nên ước lượng này thô; mốc Colab
seed 0 là 0,9660). Dải ±2 std ≈ ±0,0039. Chỉ T04 và T16 (+0,0050) vượt dải này, và vượt sát. Mọi thí nghiệm còn lại
(|Δ| ≤ 0,0031) **không phân biệt được với nhiễu**, nên mình không viết "tốt hơn" hay "tệ hơn" cho chúng.

**Nhận xét.**
- **Khởi tạo** là yếu tố duy nhất tác động lớn: đóng băng mất 0,109 và huấn luyện từ đầu mất 0,674 (ConvNeXt-T từ đầu không kịp học với 12 epoch và ~10 nghìn ảnh).
- **Augmentation:** TrivialAugment có lợi nhỏ; lật dọc và xoay 90° vô hại (Δ +0,0003), hợp lý vì ảnh chụp từ trên xuống; Mixup và CutMix không giúp.
  CutMix có thể cắt mất vật thể nhỏ trong ảnh nhưng mình chưa kiểm tra giả thuyết này.
- **Loss và sampler:** label smoothing, focal, CE có trọng số và sampler cân bằng đều không cải thiện macro-F1. Mình chỉ đo macro-F1, chưa đo F1 riêng từng lớp hiếm cho các thí nghiệm này,
  nên chưa trả lời được loss nào cải thiện lớp hiếm nhất và cái giá ở `Negative`.
- **EMA và 20 epoch:** +0,0019 và +0,0023, trong dải nhiễu.
- **Kết hợp T04 + T13 (T16)** cho +0,0050, bằng đúng T04 chạy riêng (+0,0050). EMA không cộng thêm: hiệu ứng **không cộng dồn** (lưu ý chỉ 1 seed).

## 5. Phương pháp suy luận và độ trễ (Bước 3)

Trên **val**, checkpoint T00 seed 0 của Colab, không huấn luyện lại (sheet `Inference` và `Latency`; `figures/inference_tradeoff.png`).
Các số của Bước 3 trên Colab chép từ output của notebook (không có file log riêng); độ trễ Kaggle đọc từ `logs/kaggle_final/latency_convnext_tiny.json`.

| exp_id | phương pháp | macro-F1 val | top-1 val | ECE val | p50 batch-1 (ms) |
|---|---|---|---|---|---|
| I00 | 1 view (mốc) | 0,9660 | 0,9737 | 0,0182 | 6,0 |
| I01 | TTA lật (gộp xác suất), K=2 | 0,9657 | 0,9732 | 0,0163 | 11,8 |
| I03 | TTA lật (gộp logit), K=2 | 0,9662 | 0,9734 | 0,0188 | 11,8 |
| I02 | 5-crop 224 từ 256 | 0,9650 | 0,9734 | 0,0157 | chưa đo |
| I04 | test ở 224 / 256 / 288 / 320 | 0,9660 / 0,9668 / **0,9760** / 0,9736 | 0,9737 / 0,9746 / 0,9811 / 0,9789 | 0,0182 / 0,0177 / 0,0141 / 0,0156 | – |
| I07 | temperature scaling (T = 2,107) | 0,9660 | 0,9737 | **0,0048** | không đổi |

- **TTA không giúp:** Δ nằm trong ±0,0005 (dưới nhiễu) và tốn đúng khoảng K lần độ trễ (11,8 ms so với 6,0 ms). Gộp xác suất hay logit không phân biệt được.
- **Độ phân giải test 288** tăng macro-F1 val +0,0099 mà không đổi tham số và chỉ tăng chi phí (p50 8,3 ms so với 5,7 ms ở Kaggle). Hợp với hiện tượng FixRes
  (train bằng `RandomResizedCrop` làm vật thể to hơn lúc test), nhưng mình chưa kiểm chứng nguyên nhân. 288 được chọn **trên val** rồi mới áp dụng cho F01.
- **Temperature scaling** giảm ECE từ 0,0182 xuống 0,0048 (T > 1: mô hình quá tự tin) và không đổi độ chính xác. T khớp trên val; nếu miền ảnh khi triển khai khác (mùa, ánh sáng) thì T này có thể không còn đúng.
- **Độ trễ (Tesla T4, 224, warmup 10, `synchronize`, 100 lần, chỉ tính forward):**

| dtype | batch 1: p50 / p95 / p99 (ms) | batch 32: p50 (ms) | ảnh/s (batch 32) |
|---|---|---|---|
| fp32 | 6,0 / 9,8 / 10,5 | 154 | 208 |
| AMP | 10,2 / 13,4 / 14,6 | 57 | 559 |
| fp16 | 5,6 / 6,3 / 9,4 | 45 | 708 |

  AMP **chậm hơn** fp32 ở batch 1 (10,2 so với 6,0 ms) nhưng nhanh gấp 2,7 lần ở batch 32, đúng như cảnh báo của GUIDE. Ở batch 1 độ lệch giữa các lần đo khá lớn:
  dòng "gộp BN" (thực chất là cùng model vì ConvNeXt dùng LayerNorm, gộp 0 cặp) đo 8,0 ms so với 6,0 ms của fp32, tức nhiễu cỡ 30%. Kaggle (batch 1, fp32): p50/p95/p99 = 5,7/9,4/9,4 ms ở 224 và
  8,3/12,7/12,7 ms ở 288.
- **Gộp BatchNorm** không áp dụng được với ConvNeXt; hàm `fuse_conv_bn` đã kiểm tra trên ResNet-18 và EfficientNet-B0 (sai số lớn nhất < 3e-7) nhưng không đo trên model chính.
- **Chưa làm:** ensemble nhiều mô hình, model soup, và độ chính xác riêng của fp16/AMP.
- **Offline và thời gian thực:** TTA tốn K lần mà không tăng độ chính xác nên không đáng. Các lựa chọn không tốn thêm khi suy luận (độ phân giải đã dò, temperature scaling, fp16) đều có ích.

## 6. Cấu hình tốt nhất và kết quả test (Bước 4)

**F01** = `convnext_tiny` (`in12k_ft_in1k`), TrivialAugment, EMA 0,998, 12 epoch, mọi thứ khác như T00; suy luận 1-view ở 288; T khớp trên val
(1,30 / 1,41 / 1,43 cho seed 0 / 1 / 2). Chạy 3 seed trên Kaggle, test đúng một lần mỗi seed.
Mốc **T00** = cùng backbone, công thức nền, 224, 1-view, 3 seed, trên Kaggle. Số liệu: `eval_out/`, tính bằng `eval.py` (không sửa) từ `predictions/`.

| Chỉ số (test, 3.507 ảnh) | T00 mốc | F01 |
|---|---|---|
| top-1 | 0,9751 ± 0,0007 | **0,9799 ± 0,0027** |
| macro-F1 | 0,9686 ± 0,0001 | **0,9751 ± 0,0035** |
| balanced accuracy | 0,9715 ± 0,0011 | 0,9693 ± 0,0039 |
| ECE (15 bin) | 0,0172 ± 0,0012 | **0,0047 ± 0,0011** (trước temperature scaling: 0,0102) |
| macro-F1 val | 0,9681 ± 0,0019 | 0,9760 ± 0,0013 |

Δ macro-F1 = +0,0065; s = 0,0035 (std lớn hơn trong hai nhóm) nên Δ > s nhưng Δ < 0,01. Chênh macro-F1 giữa val và test của F01 là 0,0009.

**Hai lớp khó (recall test, mean ± std):** Chinee apple **0,941 ± 0,003** (mốc T00 0,937; bài báo 0,885, trích dẫn); Snake weed **0,948 ± 0,012** (mốc T00 0,961 ± 0,008;
bài báo 0,888). F01 không tốt hơn mốc ở Snake weed (khác biệt −0,013, cỡ 1–2 std). Chi tiết theo lớp: sheet `PerClass`.

**Ma trận nhầm lẫn** (`figures/F01_confusion_matrix.png`, cộng 3 seed). Các nhầm lẫn nhiều nhất (`eval_out/F01_top_confusions.csv`):

| nhãn thật → dự đoán | số ảnh (3 seed) | tỉ lệ trong lớp thật |
|---|---|---|
| Chinee apple → Negative | 27 | 4,0% |
| Snake weed → Negative | 26 | 4,2% |
| Prickly acacia → Negative | 18 | 2,8% |
| Lantana → Negative | 17 | 2,7% |
| Negative → Prickly acacia | 16 | 0,3% |
| Rubber vine → Negative | 15 | 2,5% |
| Chinee apple → Snake weed | 10 | 1,5% |

**Phân tích lỗi.** Lỗi chủ yếu là **bỏ sót loài cỏ, đoán thành `Negative`**, chứ không phải nhầm giữa các loài. Cặp Chinee apple ↔ Snake weed mà bài báo nhấn mạnh
giờ chỉ còn 10 ảnh (1,5%) theo chiều Chinee apple → Snake weed và 2 ảnh theo chiều ngược lại (bài báo: 3,4% và 4,1%). Giả thuyết: `Negative` là lớp "nền" rất đa dạng
(đất, cỏ khô, bóng người, thực vật khác) và chiếm 52% dữ liệu, nên các ảnh cỏ dại chụp mờ hoặc có ít lá dễ bị kéo về lớp này. Giả thuyết này **chưa được kiểm chứng bằng cách xem
các ảnh bị đoán sai** vì ảnh không có sẵn trên máy phân tích (chỉ có trên Colab/Kaggle đã đóng); mình chỉ dựa vào ma trận nhầm lẫn. Thí nghiệm T10 và T11 (trọng số lớp, sampler) không cải thiện macro-F1,
nên đơn giản tăng trọng số lớp hiếm không phải cách khắc phục hiệu quả ở đây.

**So với bài báo (số trích dẫn, chỉ mang tính tham chiếu).** Bài báo: ResNet-50 95,7%, Inception-v3 95,1% (weighted accuracy, trung bình 5 fold, khoảng 100 epoch). Bài này:
top-1 98,0% với 12 epoch, 1 fold. Không thể kết luận mô hình tốt hơn vì khác định nghĩa chỉ số, khác số fold, khác trọng số tiền huấn luyện (ConvNeXt-T ImageNet-12k hiện đại hơn nhiều).

**Tự chấm phần I (`eval.py grade`, ngưỡng tạm thời):** I1 = 7/7, I2 = 4/5, I3 = 4/4, I4a = 1/1 (ECE 0,0102 → 0,0047), I4b = 1/1, I5 = 2/2
(p95 = 12,7 ms ở 288, batch 1, fp32, T4). Tổng **19/20**. Chi tiết: `eval_out/grade_I.md`.

## 7. Kết luận và khuyến nghị

1. **Cấu hình tốt nhất:** F01. Tốt hơn mốc về macro-F1 test +0,0065 (> std 0,0035, < 0,01) và hiệu chuẩn tốt hơn nhiều (ECE 0,017 → 0,005). Với 3 seed mỗi bên và khoảng chênh nhỏ, nên hiểu là "cải thiện nhẹ, có khả năng thật" chứ không phải chắc chắn.
2. **Yếu tố đóng góp nhiều nhất:** (a) backbone cùng trọng số tiền huấn luyện (chênh tới 0,2 macro-F1 val ở Bước 1; không tách được ảnh hưởng riêng của kiến trúc); (b) khởi tạo từ trọng số tiền huấn luyện
   thay vì từ đầu hay đóng băng (0,11–0,67); (c) độ phân giải test (+0,010 val); (d) temperature scaling (ECE). Các yếu tố công thức huấn luyện còn lại ≤ 0,005 và gần như nằm trong nhiễu.
3. **Triển khai trên robot (ngân sách 30–100 ms/khung):** ngân sách này rộng với ConvNeXt-T trên T4 (p95 12,7 ms ở 288 fp32), nên chọn luôn cấu hình chính xác nhất: **F01 (288, fp32, temperature scaling)**.
   Nếu cần nhanh hơn, fp16 ở batch 1 cho p95 khoảng 6 ms ở 224, nhưng độ chính xác fp16 chưa được đo riêng. Phần cứng thật của robot (ví dụ Jetson) sẽ chậm hơn T4, cần đo lại.

## 8. Hạn chế và trung thực

- **Lỗi của mình ở Bước 4 và cách xử lý.** Lần chạy đầu trên Kaggle dựng nhầm mốc T00 với backbone mặc định `resnet50` (hàm `final.run_final` không truyền backbone cho mốc). Mốc sai này đã được ghi ra file test và tính được Δ = +0,17 (sai vì so khác backbone).
  Mình phát hiện qua log (epoch 1 và 12 trùng với B01) và chạy lại mốc đúng bằng `convnext_tiny` (3 seed, Kaggle). Dự đoán và log của mốc sai được giữ riêng ở
  `predictions_discarded_resnet50_baseline/`, `logs/discarded_resnet50_baseline/`, `curves/discarded_resnet50_baseline/` và **không** dùng cho kết quả. Test của mốc sai đã bị mở một lần,
  nhưng ResNet-50 không phải ứng viên nào nên không ảnh hưởng đến việc chọn cấu hình. Không có cấu hình nào được chọn bằng test; F01 được chốt trên val trước khi chạy test.
- **Một seed ở Bước 1–2.** Các Δ nhỏ không phân biệt được với nhiễu; ước lượng nhiễu (std 0,0019) chỉ từ 3 seed.
- **Hai nơi chạy.** Bước 1–3 trên Colab, Bước 4 trên Kaggle. Mốc T00 ở Bước 2 (Colab, val 0,9660) và mốc T00 ở Bước 4 (Kaggle, val trung bình 0,9681) cùng cấu hình nhưng khác phần cứng
  và tính không tất định của CUDA; chênh 0,0021 nằm trong nhiễu. F01 và mốc ở Bước 4 chạy cùng nền (Kaggle) nên so sánh trực tiếp được.
- **Một fold, chia ngẫu nhiên, không theo địa điểm** nên điểm test có thể lạc quan so với khi gặp địa điểm hay mùa mới. Chưa đánh giá lệch miền.
- **Chưa làm:** ensemble và model soup; độ chính xác riêng của fp16/AMP; F1 theo lớp cho từng thí nghiệm Bước 2; xem ảnh bị đoán sai; các mục điểm thưởng; thử fold khác.
- **Giảm bớt do ngân sách GPU:** ablation Bước 2 chỉ trên 1 backbone và 1 seed; Colab hết hạn mức GPU nên Bước 4 chuyển sang Kaggle.
- **Nguồn số liệu Bước 3 (Colab):** chỉ có trong output notebook, mình chép vào `results.xlsx` và ghi rõ trong cột `source`.
- **Chuẩn bị dữ liệu:** bài báo ghi 1.125 ảnh Chinee apple và 1.064 Lantana, còn đếm thực là 1.126 và 1.063 (lệch 1 ảnh mỗi lớp); mình dùng đúng CSV của tác giả, không chỉnh.

## 9. Phụ lục

- **Mã thí nghiệm:** B01–B07 (backbone), T00–T16 (công thức; định nghĩa trong `code/experiments.py`), I00–I08 (suy luận), F01 (chung kết).
- **Cấu hình đầy đủ:** `logs/colab_runs/*/seed0/config.json` (Bước 1–2) và `logs/kaggle_final/*/seed*/config.json` (Bước 4).
- **Notebook:** `code/lab_day2.ipynb` (Colab, Bước 0–3), `code/kaggle_step4.ipynb` và `code/kaggle_baseline.ipynb` (Kaggle, Bước 4). Chi tiết chạy lại ở `README.md`.
- **Bảng và biểu đồ:** `results.xlsx` (Summary, Backbones, Training, Inference, Final, PerClass, Latency), `curves/`, `figures/`.
