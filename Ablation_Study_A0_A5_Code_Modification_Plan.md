# KẾ HOẠCH CHỈNH SỬA CODE ĐỂ THỰC HIỆN ABLATION STUDY A0–A5

## 1. Mục tiêu

Triển khai một thí nghiệm ablation có kiểm soát để đo đóng góp tương đối của:

1. Soft mask.
2. Boundary map.
3. Confidence map.
4. Skip-Attention.
5. Hierarchical M-CSAM (H-MCSAM).

Sáu cấu hình A0–A5 phải được huấn luyện lại từ đầu trên cùng một subset gồm 3.000 ảnh và được đánh giá trên cùng một test set.

Đây là **short controlled ablation under a fixed 12-epoch budget**, không phải thí nghiệm nhằm tìm hiệu năng hội tụ tối đa của từng mô hình.

---

## 2. Giao thức thí nghiệm đã chốt

### 2.1. Ngân sách huấn luyện

| Nhóm | Lịch huấn luyện |
|---|---|
| A0 — RAFI gốc | 12 epoch end-to-end theo pipeline gốc |
| A1–A5 — các biến thể RAFI++ | 3 epoch Stage 1 + 6 epoch Stage 2 + 3 epoch Stage 3 |

Đối với A1–A5:

```text
Epoch 1–3:   Stage 1 — huấn luyện SegNet++
Epoch 4–9:   Stage 2 — đóng băng SegNet++, huấn luyện RestoreNet và discriminator
Epoch 10–12: Stage 3 — joint fine-tuning SegNet++ và RestoreNet
```

A0 không có SegNet++ độc lập nên không được gán giả tạo vào ba stage. A0 sẽ chạy 12 epoch end-to-end, nhưng vẫn dùng cùng số lượt đi qua tập train.

### 2.2. Các yếu tố phải cố định

- Train subset: đúng 3.000 ảnh, dùng chung cho A0–A5.
- Validation split: giống nhau hoàn toàn.
- Test split: giống nhau hoàn toàn.
- Kích thước ảnh: `256 × 256`.
- Cùng masked image, ground truth và mask cho một tên ảnh.
- Cùng thứ tự tên file trong manifest.
- Cùng augmentation policy.
- Cùng effective batch size.
- Cùng số epoch và số optimizer step tương ứng.
- Cùng seed.
- Cùng cách tổng hợp metric theo số lượng ảnh.
- Cùng implementation của L1, PSNR, SSIM, LPIPS và Identity.
- Tất cả A0–A5 đều train from scratch trong bảng ablation.

Khuyến nghị ban đầu:

```text
image_size:       256
effective_batch:  2
num_workers:      0
epochs:           12
seeds:            1337, 2026, 3407
```

Nếu chỉ đủ tài nguyên cho một seed, dùng `1337` cho lần chạy chính và ghi rõ giới hạn này trong báo cáo.

### 2.3. Quy tắc checkpoint

Phương án chính cho bảng short ablation:

```text
Đánh giá checkpoint cuối epoch 12 của mọi cấu hình.
```

Lý do: quy tắc này loại bỏ việc chọn checkpoint khác nhau và tránh lỗi so sánh `best_score` giữa các stage.

Có thể báo cáo thêm checkpoint tốt nhất trong Stage 3, nhưng phải:

- Chỉ chọn trong epoch 10–12.
- Dùng cùng một tiêu chí, đề xuất validation L1 thấp nhất.
- Không chọn mỗi cấu hình theo một metric khác nhau.

Checkpoint A0 cũ và checkpoint A5 40 epoch không được đưa vào bảng ablation 12 epoch. Chúng chỉ được dùng trong một bảng “full-training reference” riêng.

---

## 3. Định nghĩa chính thức A0–A5

| ID | Guidance dùng bởi RestoreNet | Kênh đầu vào RestoreNet | Skip-Attention | Hierarchical routes |
|---|---|---:|---:|---:|
| A0 | Binary mask của RAFI gốc | 4 | Không | Không |
| A1 | Soft mask | 4 | Không | Không |
| A2 | Soft mask + boundary | 5 | Không | Không |
| A3 | Soft mask + boundary + confidence | 6 | Không | Không |
| A4 | Giống A3 | 6 | Có | Không |
| A5 | Giống A4 | 6 | Có | Có |

### 3.1. A0 — Original RAFI

Code nguồn:

```text
Face_Img_Inpainting/
```

Đặc điểm:

- Giữ nguyên kiến trúc RAFI gốc.
- Đầu vào restoration là RGB image và binary mask.
- Huấn luyện 12 epoch end-to-end.
- Không thêm SegNet++, boundary, confidence, Skip-Attention hoặc H-MCSAM.

### 3.2. A1 — RAFI++ backbone với soft mask

```python
x = torch.cat([image, mask_pred], dim=1)
```

- SegNet++ vẫn dự đoán cả mask, boundary và confidence để giữ nguyên supervision giữa A1–A5.
- RestoreNet chỉ nhận soft mask.
- Skip-Attention tắt.
- H-MCSAM tắt; ba M-CSAM chạy tuần tự.

### 3.3. A2 — A1 + boundary

```python
x = torch.cat([image, mask_pred, boundary_pred], dim=1)
```

- RestoreNet nhận soft mask và boundary.
- Confidence chưa được đưa vào RestoreNet.
- Skip-Attention tắt.
- H-MCSAM tắt.

### 3.4. A3 — A2 + confidence

```python
x = torch.cat(
    [image, mask_pred, boundary_pred, confidence_pred],
    dim=1,
)
```

- RestoreNet nhận đủ ba guidance map.
- Skip-Attention tắt.
- H-MCSAM tắt.

### 3.5. A4 — A3 + Skip-Attention

- Giữ đủ ba guidance map.
- Bật `SkipAttentionGate` ở ba skip connection.
- H-MCSAM vẫn tắt.

### 3.6. A5 — Full RAFI++

- Giữ đủ ba guidance map.
- Bật Skip-Attention.
- Bật hierarchical residual routes giữa các M-CSAM:

```python
z1 = mcsam1(e4)
z2 = mcsam2(z1) + alpha1 * z1
z3 = mcsam3(z2) + alpha2 * z2 + alpha3 * z1
```

### 3.7. Các phép so sánh cần diễn giải

```text
A1 − A0: chuyển từ RAFI gốc sang biến thể dùng soft mask
A2 − A1: đóng góp của boundary map
A3 − A2: đóng góp của confidence map
A4 − A3: đóng góp của Skip-Attention
A5 − A4: đóng góp của H-MCSAM
```

`A1 − A0` không phải phép cô lập hoàn toàn riêng soft mask vì A0 và A1 còn khác backbone. Trong bài báo, A0 nên được gọi là original/external RAFI baseline; chuỗi ablation nội bộ sạch nhất là A1–A5.

---

## 4. Nguyên tắc triển khai

### 4.1. Không tạo năm bản sao model

Không sao chép `networks.py` thành các file A1, A2, A3, A4, A5 riêng biệt.

Thay vào đó:

- Dùng một `config_id`.
- Ánh xạ `config_id` thành các cờ kiến trúc.
- Lưu `config_id` và các cờ vào checkpoint.
- Kiểm tra cấu hình khi load checkpoint.

### 4.2. SegNet++ phải giống nhau giữa A1–A5

SegNet++ luôn:

- Có ba head: mask, boundary và confidence.
- Tính đủ loss segmentation trong Stage 1 và Stage 3.
- Dùng cùng kiến trúc, loss weight và training schedule.

Chỉ thay đổi map nào được RestoreNet sử dụng. Cách này giúp:

```text
A2 − A1
A3 − A2
```

phản ánh việc bổ sung guidance cho restoration thay vì đồng thời thay đổi supervision của SegNet++.

### 4.3. Discriminator phải giữ cố định giữa A1–A5

Hiện `RAFIppSystem._disc_inputs()` dùng:

```python
torch.cat([image, mask_pred, boundary_pred], dim=1)
```

Giữ nguyên discriminator input và kiến trúc discriminator cho A1–A5. Không thay đổi số kênh discriminator theo từng cấu hình vì việc đó sẽ tạo thêm một biến gây nhiễu.

Trong báo cáo cần định nghĩa rõ rằng A1–A3 ablate **guidance consumed by RestoreNet**, không ablate supervision head hoặc discriminator conditioning.

### 4.4. Không giả lập ablation tại inference

Không được:

- Load checkpoint A5.
- Đặt boundary/confidence bằng zero.
- Tắt gate chỉ ở inference.
- Gọi kết quả đó là A1–A4.

Mỗi cấu hình phải có checkpoint được huấn luyện từ đầu với đúng kiến trúc tương ứng.

---

## 5. Kế hoạch chỉnh sửa code RAFI++ cho A1–A5

### 5.1. `Experiments/configs.py`

Thêm các tham số:

```text
--ablation_config {A1,A2,A3,A4,A5}
--train_manifest PATH
--train_subset_size 3000
--subset_seed 1337
--stage1_epochs 3
--stage2_epochs 6
--stage3_epochs 3
```

Thêm kiểm tra:

- Chỉ chấp nhận A1–A5 trong RAFI++.
- Tổng số epoch phải bằng 12 đối với protocol chính.
- `train_manifest` phải tồn tại.
- Manifest phải chứa đúng 3.000 tên file duy nhất.
- `run_name` nên chứa config và seed, ví dụ:

```text
ablation_A3_seed1337
```

Không âm thầm tạo subset khác nếu manifest đã được chỉ định.

### 5.2. `Model/networks.py`

#### Bước 1 — Tạo cấu hình kiến trúc

Thêm một cấu trúc ánh xạ bất biến:

```python
ABLATION_CONFIGS = {
    "A1": {
        "use_boundary": False,
        "use_confidence": False,
        "use_skip_attention": False,
        "use_hierarchical_mcsam": False,
    },
    "A2": {
        "use_boundary": True,
        "use_confidence": False,
        "use_skip_attention": False,
        "use_hierarchical_mcsam": False,
    },
    "A3": {
        "use_boundary": True,
        "use_confidence": True,
        "use_skip_attention": False,
        "use_hierarchical_mcsam": False,
    },
    "A4": {
        "use_boundary": True,
        "use_confidence": True,
        "use_skip_attention": True,
        "use_hierarchical_mcsam": False,
    },
    "A5": {
        "use_boundary": True,
        "use_confidence": True,
        "use_skip_attention": True,
        "use_hierarchical_mcsam": True,
    },
}
```

Yêu cầu:

- Reject config không hợp lệ.
- Không suy luận cấu hình từ số channel của checkpoint.
- A5 phải cho luồng tính toán giống code full RAFI++ hiện tại.

#### Bước 2 — Tham số hóa `RestoreNetPP`

Sửa constructor để nhận:

```python
RestoreNetPP(
    use_boundary,
    use_confidence,
    use_skip_attention,
    use_hierarchical_mcsam,
)
```

Tính số kênh đầu vào:

```python
in_channels = 3 + 1
if use_boundary:
    in_channels += 1
if use_confidence:
    in_channels += 1
```

Sau đó:

```python
self.enc1 = GatedConv2d(in_channels, 64, stride=1)
```

#### Bước 3 — Điều khiển multi-map trong `forward`

```python
guidance = [image, mask_pred]

if self.use_boundary:
    guidance.append(boundary_pred)

if self.use_confidence:
    guidance.append(confidence_pred)

x = torch.cat(guidance, dim=1)
```

Thêm assert rõ ràng:

- Tất cả tensor có batch size giống nhau.
- Map có một channel.
- Spatial size khớp với image.
- Không tự động broadcast tensor sai shape.

#### Bước 4 — Điều khiển H-MCSAM

Khi tắt:

```python
z1 = self.mcsam1(e4)
z2 = self.mcsam2(z1)
z3 = self.mcsam3(z2)
```

Khi bật:

```python
z1 = self.mcsam1(e4)
z2_prime = self.mcsam2(z1)
z2 = z2_prime + self.alpha1 * z1
z3_prime = self.mcsam3(z2)
z3 = z3_prime + self.alpha2 * z2 + self.alpha3 * z1
```

Các tham số `alpha` chỉ nên tham gia optimizer và checkpoint khi hierarchical mode được bật, hoặc phải được đánh dấu rõ là unused trong A1–A4.

#### Bước 5 — Điều khiển Skip-Attention

Khi bật:

```python
e3_skip, gate3 = self.skip3(...)
e2_skip, gate2 = self.skip2(...)
e1_skip, gate1 = self.skip1(...)
```

Khi tắt:

```python
e3_skip, gate3 = e3, None
e2_skip, gate2 = e2, None
e1_skip, gate1 = e1, None
```

Decoder topology và số channel sau phép `torch.cat` phải giữ nguyên giữa A3 và A4. Sự khác biệt duy nhất là encoder feature có hoặc không đi qua gate.

#### Bước 6 — Chuẩn hóa output

Output dict giữ cùng key cho mọi cấu hình:

```text
restored
isyn
gate1
gate2
gate3
z3
```

Khi Skip-Attention tắt, `gate1`, `gate2`, `gate3` có thể là `None`. Code visualization phải xử lý được `None`.

### 5.3. Class `RAFIpp` trong `Model/networks.py`

Sửa:

```python
RAFIpp(config_id="A5")
```

Luồng khởi tạo:

1. Validate `config_id`.
2. Khởi tạo một `SegNetPP` giống nhau cho A1–A5.
3. Lấy flags từ `ABLATION_CONFIGS`.
4. Truyền flags vào `RestoreNetPP`.
5. Lưu `self.config_id`.

`forward()` và `forward_with_guidance()` phải dùng cùng cấu hình RestoreNet.

### 5.4. `Model/RAFIpp.py`

Sửa:

```python
self.model = RAFIpp(config_id=args.ablation_config)
```

Checkpoint phải lưu thêm:

```text
ablation_config
architecture_flags
seed
subset_manifest
subset_manifest_sha256
image_size
stage schedule
```

Khi load:

- Nếu checkpoint có `ablation_config` khác CLI config, dừng với lỗi rõ ràng.
- Không dùng `strict=False` để che giấu mismatch giữa A1/A2/A3.
- Resume phải phục hồi optimizer đúng variant.

Giữ `_disc_inputs()` giống nhau giữa A1–A5.

### 5.5. `train.py`

Các thay đổi:

1. Load chính xác danh sách 3.000 ảnh từ manifest.
2. Không dùng `random.sample()` độc lập trong mỗi run.
3. Ghi config và hash manifest vào run directory.
4. Xác nhận tổng epoch:

```python
3 + 6 + 3 = 12
```

5. Tách best score theo stage:

```python
best_scores = {
    1: None,  # Dice, maximize
    2: None,  # L1, minimize
    3: None,  # L1, minimize
}
```

Không dùng cùng một scalar để so Dice với `-L1`.

6. Lưu tối thiểu:

```text
latest.pt
epoch_003.pt
epoch_009.pt
epoch_012.pt
best_stage1.pt
best_stage2.pt
best_stage3.pt
```

7. Ghi vào `history.json` và `metrics.csv`:

```text
config_id
seed
epoch
stage
train metrics
validation metrics
learning rates
number of samples
```

8. Khi resume:

- Kiểm tra config và manifest hash.
- Không resume A3 từ checkpoint A5.
- Không đổi schedule giữa chừng mà không báo lỗi.

### 5.6. `Dataset/datasets.py`

Thêm tùy chọn nhận manifest hoặc danh sách tên file:

```python
RAFIppCelebA(
    root,
    split="train",
    names=manifest_names,
    ...
)
```

Validation:

- Không có tên trùng.
- Mọi file `gt`, `masked`, `masks`, `boundaries` đều tồn tại.
- Không có tên train xuất hiện trong validation/test.

Augmentation phải được chốt chung với A0. Đề xuất:

```text
ablation_v1 = random horizontal flip với xác suất 0.5
```

Không để A0 xoay 0/90/180/270 trong khi A1–A5 chỉ horizontal flip.

Để tái lập chính xác:

- Seed Python, NumPy và PyTorch.
- Seed DataLoader generator.
- Có `worker_init_fn` nếu `num_workers > 0`.
- Protocol chính trên Windows dùng `num_workers=0`.

---

## 6. Kế hoạch chỉnh sửa code A0

### 6.1. `Face_Img_Inpainting/Experiments/configs.py`

Thêm cấu hình ablation riêng cho A0:

```text
ablation_id = A0
total_epochs = 12
seed
train_manifest
run_name
checkpoint_dir
metrics_path
```

Không thêm các cờ boundary, confidence, Skip-Attention hoặc H-MCSAM vào A0.

Đường dẫn output phải tách theo seed:

```text
checkpoints/ablation/A0/seed_1337/
logs/ablation/A0/seed_1337/
```

### 6.2. `Face_Img_Inpainting/train.py`

Thêm CLI rõ ràng:

```text
--seed
--epochs 12
--train-manifest
--run-name
--checkpoint-dir
--metrics-path
--resume-path
```

Thay cách suy ra số epoch từ:

```text
niter + niter_decay
```

bằng một `total_epochs` rõ ràng cho ablation để tránh sai lệch do cận trên của `range()`.

Loop yêu cầu:

```python
for epoch in range(start_epoch + 1, total_epochs + 1):
```

Như vậy epoch 12 chắc chắn được thực thi và lưu.

Giữ training end-to-end của RAFI:

- Generator được cập nhật ở mọi epoch.
- Discriminator được cập nhật theo pipeline A0.
- Không thêm Stage 1 giả.

### 6.3. `Face_Img_Inpainting/Dataset/datasets.py`

Sử dụng đúng manifest 3.000 ảnh chung.

Chuẩn hóa augmentation với A1–A5:

- Bỏ random rotation riêng của A0 trong protocol ablation.
- Áp dụng cùng horizontal flip 0.5, hoặc tắt augmentation cho toàn bộ A0–A5.
- Không thay đổi nội dung masked image hoặc mask ngoài augmentation chung.

Ưu tiên tạo một shared ablation dataset adapter ở cấp project để A0 và RAFI++ dùng cùng code đọc ảnh, resize và augment. A0 adapter chỉ cần đổi output dict thành tuple:

```text
(gt, masked, mask)
```

Cách này giảm nguy cơ khác biệt do:

- PIL so với skimage.
- Interpolation khác nhau.
- Chuẩn hóa tensor khác nhau.
- Random transform khác nhau.

### 6.4. Checkpoint A0

Checkpoint A0 cần chứa:

```text
epoch
seed
ablation_id = A0
train_manifest
manifest_sha256
generator
discriminators
optimizers
best validation L1
```

Lưu:

```text
epoch_012.pth
latest.pth
best.pth
```

Checkpoint dùng cho bảng chính là `epoch_012.pth`.

---

## 7. Tạo và khóa subset 3.000 ảnh

### 7.1. File mới đề xuất

```text
Dataset/CelebA/rafipp/splits/ablation_train_3000_seed1337.txt
```

Nếu chạy ba seed về khởi tạo model nhưng muốn cô lập variance do optimization, nên dùng **cùng một subset** cho cả ba seed. Khi đó tên manifest vẫn có thể là:

```text
ablation_train_3000.txt
```

và seed subset được ghi trong metadata.

### 7.2. Script tạo manifest

Tạo script:

```text
Dataset/create_ablation_subset.py
```

Chức năng:

1. Đọc `splits/train.txt`.
2. Kiểm tra không trùng tên.
3. Dùng seed cố định để lấy đúng 3.000 tên.
4. Sắp xếp hoặc lưu thứ tự deterministically.
5. Ghi manifest.
6. Ghi file metadata JSON:

```json
{
  "source_split": "splits/train.txt",
  "subset_size": 3000,
  "subset_seed": 1337,
  "sha256": "...",
  "overlap_with_val": 0,
  "overlap_with_test": 0
}
```

Script phải từ chối overwrite manifest đã tồn tại nếu nội dung khác, trừ khi người dùng truyền cờ xác nhận rõ ràng.

---

## 8. Chuẩn hóa evaluation A0–A5

### 8.1. Vấn đề hiện tại

Hai script test hiện chưa hoàn toàn đồng nhất:

- RAFI++ dùng `Utils.metrics.summarize_restoration`.
- A0 có implementation PSNR/SSIM/L1 riêng.
- A0 dùng `drop_last=True` ở test.
- Cách tính `data_range` của PSNR/SSIM trong A0 khác RAFI++.

Nếu giữ nguyên, chênh lệch metric có thể đến từ code đánh giá thay vì mô hình.

### 8.2. Thay đổi bắt buộc

Sửa `Face_Img_Inpainting/test.py` và `test.py` để:

- Cùng gọi functions trong `Utils/metrics.py`.
- `shuffle=False`.
- `drop_last=False`.
- Tính trung bình có trọng số theo số ảnh, không theo số batch.
- Cùng preprocessing cho LPIPS.
- Cùng Identity encoder và checkpoint.
- Cùng range tensor `[-1, 1]`.
- Cùng test file manifest và thứ tự file.

Mỗi kết quả cần lưu:

```json
{
  "config_id": "A3",
  "seed": 1337,
  "checkpoint": ".../epoch_012.pt",
  "checkpoint_epoch": 12,
  "num_images": 0,
  "l1": 0.0,
  "psnr": 0.0,
  "ssim": 0.0,
  "lpips": 0.0,
  "identity": 0.0
}
```

Không làm tròn metric trước khi ghi JSON.

### 8.3. Script tổng hợp

Tạo:

```text
analyze_ablation.py
```

Input:

```text
metrics JSON của A0–A5 và các seed
```

Output:

```text
outputs/ablation/summary.csv
outputs/ablation/summary.json
outputs/ablation/deltas.csv
outputs/ablation/ablation_metrics.png
```

`summary.csv`:

```text
config_id,metric,mean,std,n_seeds
```

`deltas.csv`:

```text
comparison,component,delta_psnr,delta_ssim,delta_lpips,delta_identity,delta_l1
```

Chiều tốt của metric:

```text
PSNR:     tăng
SSIM:     tăng
Identity: tăng
LPIPS:    giảm
L1:       giảm
```

Biểu đồ phải ghi rõ:

- Mean.
- Error bar là standard deviation giữa các seed.
- Số seed.
- Checkpoint epoch 12.

---

## 9. Cấu trúc output đề xuất

```text
checkpoints/
└── ablation/
    ├── A0/
    │   ├── seed_1337/
    │   ├── seed_2026/
    │   └── seed_3407/
    ├── A1/
    ├── A2/
    ├── A3/
    ├── A4/
    └── A5/

logs/
└── ablation/
    ├── A0/
    ├── A1/
    ├── A2/
    ├── A3/
    ├── A4/
    └── A5/

outputs/
└── ablation/
    ├── per_run/
    ├── summary.csv
    ├── summary.json
    ├── deltas.csv
    └── ablation_metrics.png
```

Mỗi run directory phải chứa:

```text
config.yaml hoặc config.json
manifest metadata
history.json
metrics.csv
checkpoint epoch 12
test metrics.json
```

---

## 10. Unit test và smoke test cần bổ sung

### 10.1. Test ma trận cấu hình

Tạo test xác nhận:

| Config | Expected RestoreNet input channels |
|---|---:|
| A1 | 4 |
| A2 | 5 |
| A3 | 6 |
| A4 | 6 |
| A5 | 6 |

### 10.2. Test forward

Với input `1 × 3 × 256 × 256`:

- A1–A5 đều trả `restored` và `isyn` kích thước `1 × 3 × 256 × 256`.
- Không có NaN/Inf.
- Output dict có cùng schema.

### 10.3. Test component enable/disable

- A1 không concatenate boundary/confidence.
- A2 concatenate boundary nhưng không confidence.
- A3 không gọi Skip-Attention.
- A4 gọi Skip-Attention nhưng không cộng hierarchical routes.
- A5 gọi cả Skip-Attention và hierarchical routes.

Không chỉ kiểm tra flag; dùng hook hoặc mock để xác nhận branch thực sự được/không được thực thi.

### 10.4. Test checkpoint

- A3 checkpoint load được vào A3.
- A3 checkpoint bị từ chối khi CLI chọn A5.
- Resume phục hồi đúng epoch, optimizer và config.
- Epoch 12 được lưu đúng.

### 10.5. Test subset

- Manifest có đúng 3.000 tên.
- Không trùng.
- Không overlap validation/test.
- A0 và A1–A5 trả cùng tên file ở cùng index khi không shuffle.

### 10.6. Test metric parity

Cho cùng một cặp tensor dự đoán/ground truth vào hai evaluation entry point:

- L1 phải bằng nhau.
- PSNR phải bằng nhau.
- SSIM phải bằng nhau.
- LPIPS và Identity phải dùng cùng model instance/config.

### 10.7. Smoke run

Trước khi train đầy đủ:

```text
2–8 ảnh
1 batch train cho từng stage
1 batch validation
save checkpoint
load checkpoint
1 batch test
```

Chạy smoke test cho cả A0–A5.

---

## 11. Trình tự triển khai

### Phase 1 — Data control

1. Tạo manifest 3.000 ảnh.
2. Kiểm tra overlap.
3. Chuẩn hóa augmentation và preprocessing.
4. Xác nhận A0 và RAFI++ đọc cùng sample.

### Phase 2 — RAFI++ variants

1. Thêm `ablation_config`.
2. Tham số hóa RestoreNet input.
3. Thêm cờ Skip-Attention.
4. Thêm cờ H-MCSAM.
5. Truyền config qua `RAFIppSystem`.
6. Gắn metadata vào checkpoint.
7. Viết unit test A1–A5.

### Phase 3 — Training control

1. Khóa schedule `3–6–3`.
2. Sửa best score theo từng stage.
3. Chuẩn hóa run directory.
4. Kiểm tra resume và epoch 12.

### Phase 4 — A0

1. Thêm CLI ablation.
2. Dùng manifest chung.
3. Chuẩn hóa augmentation/preprocessing.
4. Chốt loop đúng 12 epoch.
5. Giữ nguyên training end-to-end.

### Phase 5 — Evaluation

1. Dùng chung `Utils.metrics`.
2. Tắt `drop_last` ở validation/test.
3. Lưu metrics schema giống nhau.
4. Tạo script tổng hợp và biểu đồ.

### Phase 6 — Verification

1. Chạy unit tests.
2. Chạy smoke test A0–A5.
3. Kiểm tra manifest/checkpoint metadata.
4. Chạy một seed hoàn chỉnh.
5. Kiểm tra kết quả trước khi chạy hai seed còn lại.

---

## 12. Lệnh chạy dự kiến sau khi code được triển khai

Các lệnh dưới đây là giao diện mục tiêu của plan, chưa thể chạy trước khi hoàn thành các chỉnh sửa.

### 12.1. Tạo subset

```powershell
python Dataset/create_ablation_subset.py `
  --source Dataset/CelebA/rafipp/splits/train.txt `
  --output Dataset/CelebA/rafipp/splits/ablation_train_3000.txt `
  --size 3000 `
  --seed 1337
```

### 12.2. Train A1–A5

Ví dụ A3:

```powershell
python train.py `
  --ablation_config A3 `
  --train_manifest Dataset/CelebA/rafipp/splits/ablation_train_3000.txt `
  --stage1_epochs 3 `
  --stage2_epochs 6 `
  --stage3_epochs 3 `
  --seed 1337 `
  --batch_size 2 `
  --num_workers 0 `
  --run_name ablation_A3_seed1337
```

Lặp lại cho A1, A2, A4 và A5; sau đó lặp lại cho các seed còn lại.

### 12.3. Train A0

```powershell
cd Face_Img_Inpainting

python train.py `
  --epochs 12 `
  --train-manifest ../Dataset/CelebA/rafipp/splits/ablation_train_3000.txt `
  --seed 1337 `
  --run-name ablation_A0_seed1337
```

### 12.4. Test

Ví dụ RAFI++ A3:

```powershell
python test.py `
  --ablation_config A3 `
  --checkpoint checkpoints/ablation/A3/seed_1337/epoch_012.pt `
  --batch_size 2 `
  --num_workers 0 `
  --save_dir outputs/ablation/per_run/A3_seed1337
```

A0 phải dùng entry point tương ứng nhưng xuất cùng schema metrics.

### 12.5. Tổng hợp

```powershell
python analyze_ablation.py `
  --input_root outputs/ablation/per_run `
  --output_dir outputs/ablation
```

---

## 13. Tiêu chí hoàn thành

Plan được xem là triển khai thành công khi:

- [ ] Có đúng sáu cấu hình A0–A5.
- [ ] A1–A5 được điều khiển bằng config, không phải năm bản sao code.
- [ ] A5 tái tạo đúng luồng full RAFI++ hiện tại.
- [ ] Mỗi config có checkpoint train riêng từ đầu.
- [ ] Tất cả dùng đúng cùng manifest 3.000 ảnh.
- [ ] A0 chạy đúng 12 epoch end-to-end.
- [ ] A1–A5 chạy đúng schedule `3–6–3`.
- [ ] Checkpoint epoch 12 tồn tại cho mọi config/seed.
- [ ] Config mismatch khi load checkpoint bị từ chối.
- [ ] Validation/test không dùng `drop_last=True`.
- [ ] A0–A5 dùng cùng implementation metric.
- [ ] Có bảng mean ± std nếu chạy ba seed.
- [ ] Có bảng delta giữa các cấu hình liên tiếp.
- [ ] Không dùng checkpoint A5 để giả lập A1–A4 tại inference.
- [ ] Không trộn checkpoint 40 epoch vào bảng ablation 12 epoch.

---

## 14. Rủi ro và cách kiểm soát

### Rủi ro 1 — 12 epoch chưa hội tụ

Đây là short ablation nên chỉ diễn giải đóng góp dưới ngân sách cố định. Không tuyên bố đây là hiệu năng tối đa.

### Rủi ro 2 — A0 và A1 khác nhiều hơn soft mask

Gọi A0 là original RAFI baseline. Ưu tiên diễn giải quan hệ nhân quả trong chuỗi A1–A5.

### Rủi ro 3 — Khác preprocessing

Dùng shared dataset adapter hoặc viết parity test giữa hai loader.

### Rủi ro 4 — Metric không đồng nhất

Buộc cả hai test pipeline gọi chung `Utils.metrics`.

### Rủi ro 5 — Best checkpoint sai do khác thang điểm giữa stage

Theo dõi best score riêng cho từng stage; bảng chính dùng epoch 12.

### Rủi ro 6 — Variant checkpoint bị load nhầm

Lưu config ID và flags trong checkpoint, kiểm tra nghiêm ngặt khi load.

### Rủi ro 7 — Random variance trên subset nhỏ

Chạy ba seed và báo cáo mean ± standard deviation.

---

## 15. Kết luận

Thiết kế chính thức:

```text
A0: 12 epoch end-to-end
A1–A5: 3 epoch segmentation
         6 epoch restoration
         3 epoch joint fine-tuning
Train subset: 3.000 ảnh cố định
Main checkpoint: epoch 12
Recommended seeds: 1337, 2026, 3407
```

Các chỉnh sửa ưu tiên cao nhất trước khi chạy là:

1. Thêm config A1–A5 vào RAFI++.
2. Dùng chung manifest và preprocessing cho A0–A5.
3. Sửa quản lý best checkpoint theo stage.
4. Chuẩn hóa metric evaluation.
5. Thêm metadata config vào checkpoint.

Chỉ sau khi unit test và smoke test đều đạt mới bắt đầu chạy full 12 epoch cho cả sáu cấu hình.
