# RAFI-RobustTest
## Quy trình thí nghiệm robustness under imperfect masks
*Bản mô tả cô đọng để triển khai và viết paper RIVF 2026*

### Controlled Guidance Stress Test

[ Ground truth $I_{gt}, M$ ] $\rightarrow$ [ Fixed masked input $I_{inp}$ ] $\rightarrow$ [ Perturb mask $M_{t, \delta}$ ] $\rightarrow$ [ Inference (no retraining) ] $\rightarrow$ [ Metrics + robustness curve ]

Giữ nguyên vùng che thật; chỉ làm sai guidance mask để cô lập độ nhạy với lỗi định vị.

> **Nguyên tắc cốt lõi:** giữ nguyên ảnh bị che theo mask chuẩn $M$; chỉ perturb guidance mask. Nếu tạo ảnh bị che mới bằng chính perturbed mask rồi cung cấp đúng mask đó, thí nghiệm chỉ đo generalization theo hình dạng occlusion, không đo robustness trước imperfect localization.

---

### 1. Mục tiêu và giả thuyết kiểm chứng
Mục tiêu là đo mức độ nhạy của RAFI++ và các baseline khi guidance về vị trí vùng che bị sai. Thí nghiệm không huấn luyện lại mô hình; chỉ thay đổi guidance trong giai đoạn inference.

$$I_{inp} = (1 - M) \odot I_{gt} + M \odot O$$
$$M_{occ} = M, \quad M_{guide} = M_{t, \delta} \neq M$$

**Giả thuyết khoa học:** khi mức lỗi guidance tăng, chất lượng của RAFI++ suy giảm chậm hơn RAFI hoặc các biến thể không có confidence-aware selective fusion.

---

### 2. Dữ liệu và các đại lượng cố định
* Dùng một test split CelebA-HQ cố định cho tất cả mô hình.
* Mỗi mẫu gồm ảnh ground truth $I_{gt}$, mask chuẩn $M$ và ảnh masked input $I_{inp}$.
* Ảnh $I_{inp}$, ground truth, seed, preprocessing và metric implementation phải giống nhau cho mọi mô hình.
* Vùng đánh giá luôn lấy từ mask chuẩn $M$, không lấy từ perturbed mask.

---

### 3. Sinh perturbed guidance

$$M_{t, \delta} = T_{t, \delta}(M)$$

Với mỗi loại biến đổi $t$ và mức lỗi mục tiêu $\delta$, tạo perturbed mask $M_{t, \delta}$. Năm loại perturbation đề xuất:

| Loại | Biến đổi | Lỗi mô phỏng | Quan sát chính |
| :--- | :--- | :--- | :--- |
| Dilation | Mở rộng mask | False positive: sửa cả vùng nhìn thấy | Visible-region preservation |
| Erosion | Thu nhỏ mask | False negative: bỏ sót vùng che | Residual occlusion |
| Boundary shift | Dịch mask theo ($\Delta x, \Delta y$) | Biên đúng kích thước nhưng sai vị trí | Boundary continuity |
| Contour deformation | Warp contour bằng displacement field trơn | Sai contour cục bộ, gần lỗi segmentation thực | Robustness với biên méo |
| Partial corruption | Xóa/thêm blob hoặc vùng liên thông | Mask thủng, đứt, thêm vùng giả | Khả năng chịu lỗi không cấu trúc |

$$e_{mask} = \frac{|M_{t, \delta} \bigtriangleup M|}{|M|} \approx \delta, \quad \delta \in \{0, 0.05, 0.10, 0.15, 0.20\}$$

Nên hiệu chỉnh tham số của từng phép biến đổi để symmetric-difference ratio xấp xỉ $\delta$. Đồng thời lưu $\text{IoU}(M, M_{t, \delta})$ thực tế để báo cáo.

$$B_{t, \delta} = \text{Dilate}(M_{t, \delta}) - \text{Erode}(M_{t, \delta})$$

---

### 4. Cách đưa guidance vào mô hình
Cần tách evaluation interface của RAFI++ để có thể override guidance trước RestoreNet++. Với mỗi condition $(t, \delta)$:

* RAFI nhận $I_{inp}$ và $M_{t, \delta}$.
* RAFI++ nhận $I_{inp}, M_{t, \delta}, B_{t, \delta}$ và confidence map $C_{t, \delta}$.
* Không thay trọng số và không retrain cho từng loại perturbation.

$$I_{syn}^{t, \delta} = F(I_{inp}, M_{t, \delta}, B_{t, \delta}, C_{t, \delta})$$

#### Xử lý confidence
* **Predicted confidence:** dùng confidence do SegNet++ dự đoán; phản ánh điều kiện inference thực tế.
* **Oracle confidence (thí nghiệm bổ sung):** $C_{oracle} = \exp(-\beta|M_{t, \delta} - M|)$, dùng để đo upper bound của cơ chế confidence-conditioned fusion.

Hai chế độ phải được ghi nhãn rõ; oracle confidence không được trình bày như kết quả end-to-end thực tế.

---

### 5. Ma trận thí nghiệm
Thiết lập tối thiểu:

| Thành phần | Thiết lập |
| :--- | :--- |
| **Perturbation types** | 5: dilation, erosion, shift, deformation, corruption |
| **Error levels** | $\delta = 0\%, 5\%, 10\%, 15\%, 20\%$ |
| **Models** | RAFI, multi-map variants cần thiết, full RAFI++ |
| **Randomness** | Cùng perturbed masks cho mọi model; ưu tiên 3 seeds nếu đủ tài nguyên |
| **Training** | Không retrain; chỉ inference với checkpoint hiện có |

Tổng số condition cho mỗi model là $5 \times 5 = 25$. Với $N$ ảnh test và $K$ seeds, số forward pass xấp xỉ $25NK$ cho mỗi model.

---

### 6. Metrics và vùng đánh giá
Báo cáo cả chất lượng tuyệt đối và độ suy giảm so với $\delta = 0$. Các metric chính: PSNR, SSIM, LPIPS và identity similarity.

$$S_{id} = \cos(\Psi(I_{syn}), \Psi(I_{gt}))$$

Ngoài full-image metrics, cần đánh giá trên vùng mask chuẩn $M$ và boundary chuẩn $R_B$ để tránh thay đổi vùng đo theo perturbation.

$$R_B = \text{Dilate}(M) - \text{Erode}(M), \quad E_B = \frac{\| R_B \odot (I_{syn} - I_{gt}) \|_1}{\sum R_B + \epsilon}$$

* Masked-region PSNR/SSIM/L1: tính trong $M$.
* Boundary error $E_B$: trực tiếp kiểm chứng boundary guidance.
* Visible-region error: đặc biệt hữu ích cho dilation.
* False-negative residual error: đặc biệt hữu ích cho erosion.

---

### 7. Định lượng robustness
Với metric $Q$ càng cao càng tốt (PSNR, SSIM, identity), độ suy giảm tại mức $\delta$ là:

$$\Delta Q(\delta) = Q(0) - Q(\delta), \quad Q(\delta) = a + b\delta$$

Hệ số $b$ là robustness slope. Với metric giảm khi perturbation tăng, slope càng gần $0$ thì mô hình càng robust.

$$|b_{\text{RAFI}++}| < |b_{\text{baseline}}|$$

Với LPIPS (càng thấp càng tốt), dùng $\text{LPIPS}(\delta) - \text{LPIPS}(0)$ và so sánh tốc độ tăng. Có thể tổng hợp toàn bộ đường cong bằng robustness AUC:

$$\text{AUC}_Q \approx \frac{1}{\delta_{\max}} \sum_k \frac{Q(\delta_k) + Q(\delta_{k+1})}{2} (\delta_{k+1} - \delta_k)$$

---

### 8. Kết quả đầu ra cần tạo
* Figure chính: perturbation level $\delta \rightarrow$ PSNR/SSIM/LPIPS/Identity; mỗi đường là một model.
* Bảng chi tiết theo từng perturbation: $Q(0), Q(20\%)$, degradation, slope và AUC.
* Một grid ảnh định tính: cùng một ảnh, cùng perturbation, so sánh RAFI và RAFI++.
* Báo cáo mask error thực tế: symmetric-difference ratio và IoU.
* Nếu có confidence claim: báo cáo $\text{Corr}(1-\hat{C}, |\hat{M}-M|)$ hoặc AUROC phát hiện pixel segmentation sai.

---

### 9. Checklist triển khai
- [ ] Test split và checkpoint được khóa cố định.
- [ ] Masked input $I_{inp}$ không thay đổi khi perturb guidance.
- [ ] Mỗi perturbed mask được dùng giống nhau cho mọi mô hình.
- [ ] Boundary được tái tạo từ perturbed mask để guidance nhất quán.
- [ ] Vùng đánh giá dùng ground-truth mask/boundary.
- [ ] Không retrain trong controlled stress test.
- [ ] Lưu từng ảnh output, mask, seed, actual $\delta$, IoU và metrics.
- [ ] Tách rõ predicted-confidence và oracle-confidence experiments.

---

### 10. Scientific claim chỉ được chấp nhận khi
RAFI++ vừa duy trì chất lượng tuyệt đối cạnh tranh, vừa có degradation/slope nhỏ hơn baseline trên nhiều loại perturbation. Kết luận không nên dựa vào một condition đơn lẻ.

> **Cách diễn đạt ngắn gọn cho paper:** We keep the masked input fixed and perturb only the occlusion guidance. This isolates sensitivity to localization errors from changes in the underlying image content.
