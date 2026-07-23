Tôi đã đọc kỹ toàn bộ [RAFI_RobustTest_Experiment_Protocol.md](F:/RAFIpp_Project/RAFI_RobustTest_Experiment_Protocol.md), đối chiếu với luồng inference, Dataset, metrics và checkpoint hiện tại.

Kết luận: protocol có thể triển khai trên code hiện tại mà không retrain và không thay đổi kiến trúc. Tuy nhiên, cần xây một đường inference riêng để chèn guidance giữa `SegNet++` và `RestoreNet++`; không thể dùng `test.py` hiện tại vì RAFI++ đang chỉ nhận `masked` rồi tự dự đoán toàn bộ guidance.

Tôi chưa chỉnh sửa, thêm hoặc xóa bất kỳ file nào.

## 1. Luồng thí nghiệm sẽ triển khai đúng theo protocol

Đối với mỗi ảnh test:

```text
GT image I_gt
GT mask M
Fixed masked input I_inp
       │
       ├─ SegNet++(I_inp)
       │      ├─ native mask prediction M_hat
       │      ├─ predicted boundary B_hat
       │      └─ predicted confidence C_hat
       │
       ├─ Perturb GT mask:
       │      M_tδ = T_tδ(M)
       │
       ├─ Recompute coherent boundary:
       │      B_tδ = Dilate(M_tδ) - Erode(M_tδ)
       │
       ├─ Select confidence:
       │      predicted: C_tδ = C_hat
       │      oracle:    C_tδ = exp(-β|M_tδ-M|)
       │
       └─ RestoreNet++(I_inp, M_tδ, B_tδ, C_tδ)
              └─ I_syn_tδ
```

Các nguyên tắc bắt buộc:

- `masked` luôn giữ nguyên ở tất cả perturbation.
- Perturb từ ground-truth mask `M`, không perturb `mask_pred`.
- Boundary luôn được tính lại từ `M_tδ`.
- Predicted confidence và oracle confidence là hai nhóm kết quả riêng.
- Không cập nhật trọng số, không gọi optimizer, không retrain.
- Vùng tính masked/boundary metric luôn lấy từ ground-truth `M`.
- RAFI và RAFI++ phải nhận cùng một `M_tδ` đến từng pixel.

## 2. Kế hoạch thay đổi code dự kiến

### Bước 1 — Thêm interface override guidance cho RAFI++

File dự kiến chỉnh sửa:

- [Model/networks.py](F:/RAFIpp_Project/Model/networks.py)
- [Model/RAFIpp.py](F:/RAFIpp_Project/Model/RAFIpp.py)

Hiện tại `RAFIpp.forward()` đang cố định:

```python
seg = self.segnet(image)
restore = self.restore(
    image,
    seg['mask_pred'],
    seg['boundary_pred'],
    seg['confidence_pred'],
)
```

Sẽ bổ sung một interface riêng, chẳng hạn:

```python
def restore_with_guidance(
    self,
    image,
    mask_guidance,
    boundary_guidance,
    confidence_guidance,
):
    return self.restore(
        image,
        mask_guidance,
        boundary_guidance,
        confidence_guidance,
    )
```

Và trong `RAFIppSystem`:

```python
@torch.no_grad()
def inference_with_guidance(
    self,
    batch,
    mask_guidance,
    boundary_guidance,
    confidence_guidance,
):
    self.eval()

    seg = self.model.segnet(batch['masked'])

    restore = self.model.restore_with_guidance(
        batch['masked'],
        mask_guidance,
        boundary_guidance,
        confidence_guidance,
    )

    return {
        **seg,
        **restore,
        'mask_guidance': mask_guidance,
        'boundary_guidance': boundary_guidance,
        'confidence_guidance': confidence_guidance,
    }
```

Interface inference mặc định sẽ được giữ nguyên để không ảnh hưởng training, test thông thường hoặc checkpoint cũ.

### Bước 2 — Xây module sinh perturbation

File dự kiến tạo mới:

```text
Utils/robustness.py
```

Các hàm chính:

```python
perturb_mask(mask, perturb_type, target_delta, image_name, seed)
compute_mask_error(mask_gt, mask_delta)
compute_mask_iou(mask_gt, mask_delta)
compute_boundary(mask_delta, kernel_size=3)
make_oracle_confidence(mask_delta, mask_gt, beta)
```

Năm perturbation đúng theo protocol:

1. `dilation`
2. `erosion`
3. `shift`
4. `deformation`
5. `corruption`

Tất cả mask đầu ra phải:

- Có shape `[B, 1, H, W]`.
- Nằm trong `[0,1]`.
- Là mask nhị phân.
- Không dùng circular shift/wrap-around.
- Sinh deterministic từ khóa `(image_name, perturbation, target_delta, seed)`.
- Không dùng Python `hash()` vì giá trị có thể thay đổi giữa các process.

Seed ổn định sẽ được tạo từ SHA-256 của chuỗi khóa.

### Bước 3 — Hiệu chỉnh perturbation theo actual mask error

Không quy đổi trực tiếp `5%` thành kernel bằng `5% × image_width`.

Đối với mỗi ảnh, tham số perturbation sẽ được tìm sao cho:

\[
e_{mask}
=
\frac{|M_{t,\delta}\triangle M|}{|M|}
\approx\delta
\]

Cách hiệu chỉnh:

- Dilation/erosion: thử hoặc binary-search bán kính morphology.
- Shift: tìm độ dịch pixel gần target delta nhất.
- Deformation: giữ cố định displacement field theo seed, điều chỉnh amplitude.
- Corruption: thêm/xóa tuần tự các connected blobs đến khi gần target.
- Chọn kết quả có `|actual_delta-target_delta|` nhỏ nhất.
- Luôn lưu cả `target_delta`, `actual_delta` và IoU.
- Với `δ=0`, trả về bản sao chính xác của `M`.

Nếu do tính rời rạc của pixel không thể đạt đúng target, evaluator sẽ lưu sai số hiệu chỉnh thay vì giả định đã đạt chính xác.

### Bước 4 — Tạo evaluator riêng cho RAFI++

File dự kiến tạo mới:

```text
robust_test.py
```

Không sửa `test.py` thành robust evaluator vì có nguy cơ ảnh hưởng pipeline đánh giá chuẩn.

Evaluator sẽ:

1. Khóa test manifest.
2. Load `best_stage3.pt`.
3. Load LPIPS và identity encoder đúng một lần.
4. Chạy `SegNet++` một lần cho mỗi batch.
5. Sinh `M_tδ` từ ground-truth `M`.
6. Tính `B_tδ` từ `M_tδ`.
7. Chạy hai chế độ confidence:

```text
predicted
oracle
```

8. Gọi `inference_with_guidance()`.
9. Tính metric theo từng ảnh.
10. Lưu output, mask, guidance, seed và metric.

CLI dự kiến:

```text
--checkpoint
--data_root
--output_dir
--perturbation
--target_delta
--robust_seed
--confidence_mode predicted|oracle
--beta_conf
--batch_size
```

### Bước 5 — Xử lý confidence đúng protocol

Predicted-confidence:

```python
confidence_delta = seg['confidence_pred']
```

Confidence này không thay đổi khi `M_tδ` thay đổi vì nó được dự đoán trực tiếp từ `I_inp`.

Oracle-confidence:

```python
confidence_delta = torch.exp(
    -beta_conf * torch.abs(mask_delta - mask_gt)
)
```

Sử dụng `beta_conf=5.0` mặc định để nhất quán với loss đã huấn luyện, đồng thời ghi giá trị beta vào run manifest.

Không sử dụng `mask_gt` trong predicted-confidence mode.

### Bước 6 — Mở rộng metric theo từng ảnh và từng vùng

File dự kiến chỉnh sửa:

- [Utils/metrics.py](F:/RAFIpp_Project/Utils/metrics.py)

Hiện tại nhiều hàm trả về batch mean, chưa đủ để xuất per-image CSV. Sẽ bổ sung phiên bản `reduction='none'` hoặc các hàm per-sample.

Các metric cần có:

- Full-image PSNR.
- Full-image SSIM.
- LPIPS.
- Identity similarity.
- Masked-region PSNR.
- Masked-region SSIM.
- Masked-region L1.
- Boundary error.
- Visible-region L1.
- False-negative residual error.
- Actual mask error.
- Mask IoU.

Boundary metric:

\[
E_B =
\frac{\left\|R_B\odot(I_{syn}-I_{gt})\right\|_1}
{\sum R_B+\epsilon}
\]

Trong đó `R_B` được tính từ ground-truth `M`, không dùng `B_tδ`.

False-negative region:

\[
R_{FN}=M\land(1-M_{t,\delta})
\]

Nếu `R_FN` rỗng, metric sẽ được lưu `NaN`, không ép thành 0.

Masked SSIM sẽ được tính bằng SSIM map rồi lấy trung bình có trọng số trong `M`; không zero vùng ngoài mask rồi gọi SSIM toàn ảnh.

### Bước 7 — Đảm bảo RAFI baseline dùng đúng cùng dữ liệu

File dự kiến tạo:

```text
Face_Img_Inpainting/robust_test.py
```

RAFI sẽ nhận:

```python
model.set_input(
    gt=gt,
    img=masked_fixed,
    mask=mask_delta,
    device=device,
)
```

Không dùng lại metric trong `Face_Img_Inpainting/test.py`, vì file đó đang có cách tính PSNR/SSIM khác RAFI++.

Cả RAFI và RAFI++ phải sử dụng chung:

- `splits/test.txt`.
- `gt`.
- `masked`.
- `M_tδ`.
- Module metric mới trong `Utils/metrics.py`.
- Tên ảnh và seed.
- Schema CSV.

Các multi-map variant chỉ được thêm nếu đã có checkpoint được huấn luyện tương ứng. Không được tạo “ablation” bằng cách zero channel tại inference rồi trình bày như một model đã train.

### Bước 8 — Cấu trúc dữ liệu đầu ra

Dự kiến:

```text
outputs/rafi_robusttest/<run_id>/
├── manifest.json
├── guidance/
│   └── <perturbation>/<level>/<seed>/
├── rafipp/
│   ├── predicted/
│   └── oracle/
├── rafi/
├── per_image.csv
├── summary.csv
├── curves/
└── qualitative/
```

`manifest.json` lưu:

- Checkpoint path và SHA-256.
- Test manifest SHA-256.
- Model.
- Confidence mode.
- Beta.
- Image size.
- Metric version.
- Perturbation config.
- Seeds.
- Ngày chạy.
- Số ảnh thực tế.

Per-image CSV tối thiểu:

```text
model
confidence_mode
image_name
seed
perturbation
target_delta
actual_mask_error
mask_iou
psnr
ssim
lpips
identity
masked_psnr
masked_ssim
masked_l1
boundary_error
visible_l1
false_negative_error
```

Protocol yêu cầu lưu từng ảnh output, vì vậy evaluator sẽ không chỉ lưu một qualitative subset. Một subset cố định sẽ được sao chép vào `qualitative/` để làm figure.

## 3. Kế hoạch tổng hợp thống kê

File dự kiến tạo:

```text
analyze_robustness.py
```

Script sẽ:

- Đọc `per_image.csv`.
- Nhóm theo model, confidence mode, perturbation và target level.
- Tính mean, standard deviation.
- Tính bootstrap 95% CI bằng resampling theo ảnh.
- Tính degradation so với `δ=0`.
- Fit robustness slope.
- Tính normalized AUC bằng trapezoidal rule.
- Vẽ absolute curve và degradation curve.
- Xuất bảng `Q(0)`, `Q(20%)`, degradation, slope, AUC.

Quy ước degradation:

- PSNR/SSIM/Identity:

\[
\Delta Q(\delta)=Q(0)-Q(\delta)
\]

- LPIPS/L1/Boundary error:

\[
\Delta Q(\delta)=Q(\delta)-Q(0)
\]

Slope và AUC sẽ dùng `actual_mask_error` trung bình trên trục X, không dùng nhãn target level làm giá trị thực.

Confidence claim sẽ dùng:

\[
\operatorname{Corr}
\left(
1-\hat C,\,
|\hat M-M|
\right)
\]

Ở đây `M_hat` là native prediction của SegNet++, không phải `M_tδ`. Chỉ số này được tính riêng một lần vì `SegNet++` và `I_inp` không thay đổi giữa các perturbation.

## 4. Kiểm thử trước khi chạy toàn bộ

Các test kỹ thuật bắt buộc:

- `δ=0` tạo mask giống `M` tuyệt đối.
- `masked` có cùng checksum ở mọi condition.
- Boundary được tái tạo từ đúng `M_tδ`.
- Perturbation cùng seed sinh cùng kết quả qua nhiều lần chạy.
- RAFI và RAFI++ đọc mask có cùng checksum.
- Actual delta nằm gần target delta.
- Không có gradient và không có optimizer step.
- Hash tham số model trước và sau run giống nhau.
- Không overwrite output giữa model/confidence/condition.
- Batch cuối không bị bỏ qua.
- Metric per-image tổng hợp lại phải khớp aggregate.
- Smoke test trên 20 ảnh xác nhận output thực sự thay đổi khi delta tăng.

## 5. Ma trận chạy

Protocol ghi `5 × 5 = 25` condition cho mỗi model. Tuy nhiên, năm condition `δ=0` là giống hệt nhau.

Để tránh đếm baseline năm lần:

- Số condition logic để biểu diễn: 25.
- Số inference condition duy nhất: `1 + 5×4 = 21` cho mỗi model/seed.
- Baseline `δ=0` chạy một lần rồi dùng chung cho năm đường perturbation.
- RAFI++ predicted-confidence: 21 condition/seed.
- RAFI++ oracle-confidence bổ sung: 21 condition/seed.
- RAFI: 21 condition/seed.

Cách này giữ nguyên ma trận khoa học nhưng không nhân trọng số baseline hoặc lãng phí forward pass.

## 6. Lịch trình 25–27/7

### Ngày 25/7

- Khóa `splits/test.txt` hiện có: 1.500 ảnh.
- Khóa RAFI++ `best_stage3.pt` và RAFI checkpoint.
- Cài interface override guidance.
- Cài năm perturbation và actual-delta calibration.
- Cài regional/per-image metrics.
- Chạy unit test và smoke test 20 ảnh.
- Kiểm tra checksum `masked` và guidance giữa hai model.

### Ngày 26/7

- Chạy RAFI.
- Chạy RAFI++ predicted-confidence.
- Chạy RAFI++ oracle-confidence.
- Chạy đủ 1 hoặc 3 seeds.
- Lưu toàn bộ output, mask và per-image CSV.
- Kiểm tra số dòng, NaN, delta thực tế và file thiếu.

### Ngày 27/7

- Bootstrap CI.
- Tính degradation, slope và AUC.
- Vẽ robustness curves.
- Tính confidence-error correlation.
- Chọn failure cases và qualitative grid.
- Đối chiếu claim trên nhiều perturbation, không dựa vào một condition.

## 7. Hai lưu ý khoa học cần ghi rõ

Thứ nhất, theo protocol mới, `δ=0` của RAFI++ sử dụng ground-truth mask `M`, không phải native predicted mask `M_hat`. Vì vậy đây là robustness của `RestoreNet++` dưới external imperfect guidance, không phải kết quả end-to-end mặc định của RAFI++.

Thứ hai, protocol hiện tại không yêu cầu fixed-composition. Do `RestoreNetPP` đang tạo:

\[
I_{syn}=(1-M_{t,\delta})I_{inp}+M_{t,\delta}I_r
\]

nên kết quả erosion bao gồm cả ảnh hưởng của generation lẫn compositing. Kế hoạch trên giữ nguyên điều này để đúng với protocol. Nếu mục tiêu thực sự là cô lập riêng năng lực sinh của RestoreNet++, protocol phải bổ sung output composite bằng ground-truth `M` trước khi triển khai; hiện tại tôi không tự thêm thí nghiệm đó.
