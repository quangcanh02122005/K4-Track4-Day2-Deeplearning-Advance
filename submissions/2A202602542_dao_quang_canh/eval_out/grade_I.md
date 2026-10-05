## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 97.99% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 4 | 5 | final 0.9751, mốc 0.9686, Δ=+0.0065, s=0.0035 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 94.1% (mốc 88.5%), Snake Weed 94.8% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.0102, sau 0.0047 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9760, test 0.9751, chênh 0.0009 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 12.7 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 19 / 20** (phần I tối đa 20).

Ngưỡng điểm là TẠM THỜI (xem khối hằng số đầu file eval.py và RUBRIC.md mục I).
