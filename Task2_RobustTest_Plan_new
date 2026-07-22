# Kế hoạch chi tiết: Xây dựng RAFI-RobustTest và chạy Mask Perturbation Experiments

**Thời gian thực hiện:** 25 - 27/07
**Mục tiêu:** Tạo ra một protocol đánh giá độ bền vững (robustness) của mô hình RAFI++ khi thông tin guidance (mask che khuất) bị sai lệch hoặc nhiễu. Từ đó, chứng minh rằng cơ chế "uncertainty-aware multi-map guidance" và "selective feature fusion" của RAFI++ giúp phục hồi ảnh tốt hơn so với RAFI gốc khi mask không hoàn hảo.

---

## 1. Tổng quan về RAFI-RobustTest

Thay vì kiểm thử trên ground-truth mask, chúng ta sẽ thêm nhiễu (perturbation) vào mask với các mức độ $\delta \in \{0\%, 5\%, 10\%, 15\%, 20\%\}$. Trục X của đồ thị biểu diễn độ nhiễu sẽ không dùng % kernel size mà dùng **lỗi mask thực tế** (Actual Mask Error):
$e_{\mathrm{IoU}} = 1 - \operatorname{IoU}(M_\delta, M)$

Các kiểu biến dạng mask (mask perturbations) cần cài đặt:
1. **Dilated mask (Phóng to mask):** Giả lập model dự đoán vùng che khuất lớn hơn thực tế.
2. **Eroded mask (Thu nhỏ mask):** Giả lập sót vùng bị che.
3. **Shifted mask (Dịch chuyển mask):** Trượt tọa độ mask theo 4 hướng cố định (để tránh nhiễu do sinh ngẫu nhiên 1 hướng).
4. **Irregular/Randomized corruption:** Làm biến dạng contour hoặc thêm các blob/hình tròn ngẫu nhiên (không dùng nhiễu muối tiêu pixel rời rạc vì không giống lỗi segmentation thực tế).

*(Lưu ý: Mức $\delta = 0\%$ tương ứng với mask ban đầu của thuật toán - Ground-truth đối với RAFI và Predicted Mask đối với RAFI++)*

---

## 2. Chi tiết các bước thực hiện

### Bước 1: Viết script sinh dữ liệu Mask Perturbation (`generate_robust_test.py`)

Cần viết các hàm perturb mang tính **deterministic** (cố định seed) để sinh ra mask nhiễu $M_\delta = T_\delta(M)$. Hàm này sẽ được gọi trực tiếp trong quá trình chạy Inference thay vì sinh sẵn ra ổ cứng, để dễ dàng can thiệp vào luồng dữ liệu của RAFI++.

### Bước 2: Can thiệp vào quá trình Inference (Inject Perturbation)

**Điểm khác biệt trọng tâm:** RAFI và RAFI++ có kiến trúc khác nhau, nên vị trí tiêm nhiễu phải khác nhau. Cả hai mô hình đều dùng chung ảnh đầu vào `masked` và `gt` cố định để đảm bảo mọi thay đổi metric đều đến từ guidance error.

**Đối với RAFI gốc (A0):**
Mô hình nhận mask trực tiếp. Ta áp dụng nhiễu lên ground-truth mask rồi đưa vào Generator.
$M_\delta = T_\delta(M)$
$\text{Input} = \text{masked} + M_\delta$

**Đối với RAFI++ (A5):**
RAFI++ tự dự đoán mask bằng SegNet++. Cần can thiệp vào giữa SegNet++ và RestoreNet++:
```python
seg = system.model.segnet(masked)

# Chỉ perturb predicted mask (Mask-channel corruption)
mask_base = seg['mask_pred']
mask_delta = perturb(mask_base, perturbation_type, level)

# Giữ nguyên Boundary và Confidence để kiểm tra tính hiệu quả của Multi-map
restore = system.model.restore(
    masked,
    mask_delta,
    seg['boundary_pred'],
    seg['confidence_pred'],
)
```

### Bước 3: Đánh giá bằng Fixed-Composition

RAFI++ dùng perturbed mask để ghép ảnh (compositing):
$I_{\mathrm{syn},\delta} = (1 - M_\delta)I_{\mathrm{inp}} + M_\delta I_{r,\delta}$
Nếu dùng $M_\delta$ (đặc biệt là eroded mask), pixel rác từ ảnh input sẽ bị lọt vào, làm sai lệch metric đo lường khả năng sinh ảnh của RestoreNet++.

Do đó, **chỉ tính metric dựa trên Fixed-composition output** (sử dụng ground-truth mask $M$ để ghép):
$I_{\mathrm{eval},\delta} = (1 - M)I_{\mathrm{inp}} + M I_{r,\delta}$
Cách này đo lường thuần túy sức mạnh của RestoreNet++ trước guidance lỗi.

### Bước 4: Chạy thử nghiệm và Lưu Data (Per-image CSV)

Thực hiện tổng cộng 34 lần chạy (conditions) cho mỗi model: 1 baseline 0% + 4 loại nhiễu $\times$ 4 mức độ $\times$ 2 loại mô hình (hoặc nhiều hơn nếu xét theo hướng shift).

Thay vì chỉ lưu `metrics.json` tổng hợp, cần xuất file CSV chứa metric cho **từng ảnh**:
```text
model, image_name, perturbation, target_level, actual_mask_error, mask_iou, psnr, ssim, lpips, identity
```

### Bước 5: Phân tích dữ liệu và vẽ đồ thị (Robustness curves)

Từ file CSV, sử dụng Python để:
1. Tính Mean, Standard Deviation, và 95% Bootstrap Confidence Interval.
2. Tính Relative degradation: $\Delta Q_\delta = Q_\delta - Q_0$ (vì baseline 0% của RAFI và RAFI++ khác nhau).
3. Tính Robustness AUC.
4. **Vẽ đồ thị:** 
   - Trục X: `actual_mask_error` (hoặc `1 - mask IoU`).
   - Trục Y: PSNR / SSIM / LPIPS / Identity Similarity (Cả absolute và relative drop).

---

## 3. Lịch trình hành động (25 - 27/07)

- **Ngày 25/7:**
  - Khóa checkpoint và test manifest.
  - Định nghĩa toán học cho 4 loại perturbation và viết các hàm perturb deterministic.
  - Viết evaluator can thiệp `M_\delta` vào RAFI và `\hat{M}_\delta` vào RAFI++.
  - Chạy thử nghiệm trên 20 ảnh để xác nhận output thay đổi chính xác.

- **Ngày 26/7:**
  - Chạy toàn bộ test protocol (baseline 0% và các mức nhiễu). Đảm bảo dùng cùng mẫu thử và seed cho cả hai model.
  - Sinh ra per-image CSV. Lưu ảnh qualitative cho một tập con nhỏ cố định.

- **Ngày 27/7:**
  - Viết script tổng hợp và vẽ đồ thị (Robustness curves).
  - Phân tích confidence intervals, relative drops và AUC.
  - Review lại các failure cases và chốt biểu đồ, bảng biểu cho paper.
