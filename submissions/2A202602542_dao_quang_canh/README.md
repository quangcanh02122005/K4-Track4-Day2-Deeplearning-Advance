# Bài nộp Lab Day 2 - Đào Quang Cảnh (2A202602542)

Backbone, công thức huấn luyện và suy luận trên DeepWeeds (fold 0). Kết quả và phân tích: [`report.md`](report.md). Bảng so sánh: [`results.xlsx`](results.xlsx).

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `report.md` | Báo cáo kết luận |
| `results.xlsx` | 7 sheet: Summary, Backbones, Training, Inference, Final, PerClass, Latency |
| `curves/` | Ảnh đường cong huấn luyện của Bước 1–2 (`B01…B07`, `T00…T16`, Colab). `curves/final/`: F01 và mốc T00, 3 seed (Kaggle) |
| `predictions/` | Dự đoán **test** và val của chung kết `F01` (đã temperature scaling), `F01uncal` (chưa) và mốc `T00`, mỗi seed một file, đúng định dạng `eval.py` |
| `eval_out/` | Kết quả của `eval.py score` và `eval.py grade` (`grade_I.md`), ma trận nhầm lẫn, chỉ số theo lớp |
| `figures/` | Ma trận nhầm lẫn F01, đánh đổi độ chính xác - độ trễ, so sánh backbone, Δ của Bước 2 |
| `logs/` | `summary.json`, `config.json`, `history.csv` của từng lần chạy (`colab_runs/`, `kaggle_final/`), độ trễ Kaggle, log kernel |
| `code/` | Toàn bộ code (xem bên dưới) |
| `*_discarded_resnet50_baseline/` | Dự đoán, log, curves của mốc T00 **dựng nhầm với resnet50** ở lần chạy Bước 4 đầu tiên. **Không dùng** cho kết quả; giữ lại để minh bạch (xem `report.md` mục 8) |

## Mã nguồn (`code/`)

`dataset.py`, `model.py`, `losses.py`, `train.py` (một hàm `run(Config)` dùng chung), `inference.py`, `benchmark.py`, `experiments.py` (bảng B/T),
`final.py` (Bước 4), `build_results.py` (dựng `results.xlsx` và biểu đồ), `test_code.py` (19 kiểm tra tự viết: focal γ=0 ≡ CE, CutMix, gộp BN, EMA, scheduler, ...),
và các notebook: `lab_day2.ipynb` (Colab), `kaggle_step4.ipynb`, `kaggle_baseline.ipynb`. Không sửa `eval.py` (nằm ở thư mục gốc repo).

## Link chạy lại

- Repo: https://github.com/quangcanh02122005/K4-Track4-Day2-Deeplearning-Advance (nhánh `main`, thư mục `submissions/2A202602542_dao_quang_canh/code/`)
- Notebook Colab (Bước 0–3):
  https://colab.research.google.com/github/quangcanh02122005/K4-Track4-Day2-Deeplearning-Advance/blob/main/submissions/2A202602542_dao_quang_canh/code/lab_day2.ipynb
- Notebook Kaggle (Bước 4, chạy ở chế độ riêng tư, nên không có link công khai): `code/kaggle_step4.ipynb` (F01 3 seed) và `code/kaggle_baseline.ipynb`
  (mốc T00 3 seed + độ trễ). Nhập file `.ipynb` vào Kaggle, bật GPU T4 và Internet.

## Phiên bản và môi trường

| | Colab (Bước 1–3) | Kaggle (Bước 4) |
|---|---|---|
| GPU | Tesla T4 | Tesla T4 |
| Python | 3.13.15 | 3.13.15 |
| torch | 2.11.0+cu130 | 2.11.0+cu128 |
| timm | 1.0.29 | 1.0.29 |

Kiểm tra cục bộ (CPU): `cd code && python -m unittest test_code` (cần torch, timm, torchvision, pillow, pandas). Dựng bảng và biểu đồ: `python code/build_results.py`
(cần openpyxl và matplotlib).

## Thứ tự chạy

1. **Colab** (`lab_day2.ipynb`): cài đặt và gắn Drive, tải dữ liệu (kiểm tra MD5), EDA và kiểm tra split, kiểm tra pipeline, Bước 1 (`B01–B07`), Bước 2 (`T00–T16`),
   Bước 3 (suy luận và độ trễ). Kết quả lưu ở `MyDrive/deepweeds_lab/`. Cần `IMAGES_DIR = "data"` (ảnh giải nén thẳng vào `data/`).
2. **Kaggle** `kaggle_step4.ipynb`: F01 với seed 0, 1, 2 (`final.run_final`), rồi `eval.py score` và `grade`. (Mốc T00 trong notebook này chạy nhầm backbone, đã sửa trong `final.py`; kết quả đó bị loại.)
3. **Kaggle** `kaggle_baseline.ipynb`: mốc T00 `convnext_tiny` seed 0, 1, 2 và đo độ trễ ở 224 và 288.
4. Chấm: `python eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv <test_subset0.csv> --labels <labels.csv> --tag F01 --out eval_out`
   và `python eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" --uncal "predictions/F01uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --latency-p95-ms 12.7 ...`.

## Seed đã dùng

Bước 1–2: seed 0. Bước 4: seed 0, 1, 2 cho F01 và mốc T00. Seed không đổi cách chia dữ liệu (luôn là fold 0).

## Không có trong thư mục nộp

Dataset (`images.zip`, ảnh) và checkpoint (`*.pt`) không được commit (bị `.gitignore` chặn). Checkpoint nằm trên Google Drive của tác giả.
