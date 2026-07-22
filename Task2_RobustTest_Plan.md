# Kế hoạch chi tiết: Xây dựng RAFI-RobustTest và chạy Mask Perturbation Experiments

**Thời gian thực hiện:** 25 - 27/07
**Mục tiêu:** Tạo ra một protocol đánh giá độ bền vững (robustness) của mô hình RAFI++ khi thông tin guidance (mask che khuất) bị sai lệch hoặc nhiễu. Từ đó, chứng minh rằng cơ chế "uncertainty-aware multi-map guidance" của RAFI++ giúp phục hồi ảnh tốt hơn so với RAFI gốc khi mask không hoàn hảo.

---

## 1. Tổng quan về RAFI-RobustTest

Thay vì chỉ kiểm thử trên một bộ mask cố định (mask lý tưởng/ground-truth mask), chúng ta sẽ cố tình thêm nhiễu (perturbation) vào mask gốc với các mức độ khác nhau. Giả sử ta định nghĩa mức độ nhiễu là $\delta \in \{0\%, 5\%, 10\%, 15\%, 20\%\}$.

Các kiểu biến dạng mask (mask perturbations) cần cài đặt:
1. **Dilated mask (Phóng to mask):** Giả lập trường hợp model phân vùng (segmentation) dự đoán vùng che khuất lớn hơn thực tế.
2. **Eroded mask (Thu nhỏ mask):** Giả lập trường hợp dự đoán vùng che khuất nhỏ hơn thực tế (sót vùng bị che).
3. **Shifted mask (Dịch chuyển mask):** Giả lập trường hợp mask bị trượt tọa độ theo các hướng (lên, xuống, trái, phải).
4. **Irregular/Randomized mask corruption (Nhiễu ngẫu nhiên):** Thêm các vùng che khuất ngẫu nhiên (nhiễu muối tiêu, đốm đen) hoặc xóa bớt ngẫu nhiên trên mask gốc.

*(Lưu ý: "Landmark-aligned mask" chính là bộ mask gốc ban đầu - tương đương với $\delta = 0\%$)*

---

## 2. Chi tiết các bước thực hiện

### Bước 1: Viết script sinh dữ liệu Mask Perturbation (`generate_robust_test.py`)

Cần tạo một file Python độc lập để tự động tạo ra các bộ test bị nhiễu từ bộ test gốc `f:\RAFIpp_Project\Dataset\CelebA\rafipp`.

**Yêu cầu kỹ thuật trong script:**
- **Thư viện sử dụng:** `OpenCV (cv2)`, `numpy`, `PIL`.
- **Input:** Thư mục `masks/` gốc của tập test (dựa trên file `splits/test.txt`).
- **Xử lý:** Đối với mỗi mask, thực hiện 4 loại biến đổi:
  - **Dilation:** Dùng `cv2.dilate` với kernel size tăng dần tương ứng với $\delta$.
  - **Erosion:** Dùng `cv2.erode` với kernel size tăng dần tương ứng với $\delta$.
  - **Shift:** Dùng `cv2.warpAffine` để dịch chuyển ngẫu nhiên một khoảng pixel (tính theo \% kích thước ảnh tương ứng với $\delta$).
  - **Irregular:** Thêm các hình vuông hoặc hình tròn ngẫu nhiên vào mask với tổng diện tích tương ứng $\delta$.
- **Output:** Lưu vào các thư mục mới:
  - `Dataset/CelebA/rafipp_robust/dilated_5/masks/`
  - `Dataset/CelebA/rafipp_robust/dilated_10/masks/`
  - (Tương tự cho các kiểu và mức độ khác).

*(Ảnh `gt`, `masked` giữ nguyên, vì ta chỉ đang test độ nhạy cảm của model với mask đầu vào sai lệch, còn ảnh thực tế bị che khuất không đổi)*

### Bước 2: Cập nhật DataLoader để hỗ trợ RobustTest

- Có thể dùng chung `Dataset.datasets.RAFIppCelebA` (cho RAFI++) và `Dataset.datasets.CelebA` (cho RAFI gốc), nhưng lúc khởi tạo Dataset, ta sẽ truyền tham số `--mask_dir` trỏ tới các thư mục mask đã bị làm nhiễu ở Bước 1 thay vì thư mục mask chuẩn.

### Bước 3: Chạy Inference và đo đạc chỉ số (Không cần train lại)

Viết một batch script (vd: `run_robust_test.ps1`) để tự động chạy file `test.py` trên cả 2 model (RAFI gốc A0 và RAFI++ full A5) qua tất cả các bộ mask nhiễu.

**Đối với RAFI++ (A5):**
```bash
# Ví dụ chạy với Dilated mask mức 5%
python test.py --data_root f:\RAFIpp_Project\Dataset\CelebA\rafipp --mask_dir f:\RAFIpp_Project\Dataset\CelebA\rafipp_robust\dilated_5\masks --checkpoint path\to\latest.pt --batch_size 4
```

**Đối với RAFI (A0):**
Sửa đường dẫn `self.mask_dir` trong `configs.py` (hoặc sửa code nhận parameter `--mask_dir` cho RAFI A0) để đọc mask nhiễu, rồi chạy:
```bash
python test.py
```

### Bước 4: Tổng hợp dữ liệu ra JSON / CSV

Thu thập kết quả `metrics.json` từ tất cả các lần chạy (mỗi model $\times$ 4 loại perturbation $\times$ 5 mức độ $\delta = 2 \times 4 \times 5 = 40$ lần test). Cần viết một script nhỏ (`summarize_robustness.py`) gộp các JSON này lại thành một bảng dữ liệu tổng (.csv) để dễ phân tích.

### Bước 5: Trực quan hóa dữ liệu (Vẽ đồ thị)

Sử dụng `matplotlib` hoặc `seaborn` để vẽ đồ thị:
- Trục hoành (X-axis): Perturbation Level $\delta$ (0%, 5%, 10%, 15%, 20%).
- Trục tung (Y-axis): Performance Metrics (PSNR, SSIM, LPIPS, Identity).
- Đường biểu diễn (Lines): So sánh RAFI gốc (thường sẽ giảm mạnh) và RAFI++ (kỳ vọng giảm từ từ và duy trì hiệu suất tốt hơn).

Các biểu đồ (Robustness curves) này sẽ là "Vũ khí cốt lõi" cho câu hỏi nghiên cứu số 3 (RQ3) của bài báo.

---

## 3. Các lưu ý quan trọng (Checklist)

- [ ] **Giữ nguyên ảnh gốc:** Chỉ mask (guidance) thay đổi, ảnh ground-truth (`gt/`) và ảnh đầu vào (`masked/`) phải hoàn toàn không thay đổi để metric so sánh được công bằng.
- [ ] **Mức độ $\delta$:** Cần định nghĩa rõ $\delta = 5\%$ nghĩa là gì (ví dụ: kernel size của dilation là 5% chiều rộng ảnh, hoặc diện tích thay đổi 5%). Càng chi tiết toán học, bài báo IEEE càng giá trị.
- [ ] **Khả năng sinh tự động:** Đảm bảo quá trình perturbation được set `seed` cố định, sao cho tạo đi tạo lại vẫn ra đúng một bộ mask đó, nhằm đảm bảo tính tái lặp (reproducibility) của thực nghiệm khoa học.
