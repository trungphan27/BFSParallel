# KẾ HOẠCH XÂY DỰNG BÀI BÁO THAM GIA RIVF 2026

**Thời hạn (Deadline):** 15/08/2026
**Mục tiêu:** Phát triển báo cáo Project II (RAFI++) thành một bài báo chuẩn IEEE (tối đa 6 trang) nộp cho hội nghị RIVF 2026 (Track: AI Foundations and Big Data).

---

## 1. TÁI ĐỊNH VỊ BÀI BÁO (REPOSITIONING)

Thay vì trình bày "một kiến trúc có nhiều module cải tiến", bài báo cần được tái định vị tập trung vào một thông điệp khoa học rõ ràng: **Robust Face Inpainting under Occlusion Uncertainty**.

### 1.1. Đề xuất Tên bài báo
- **Lựa chọn 1 (Ưu tiên cao nhất):** RAFI++: Uncertainty-Aware Multi-Map Guidance and Selective Feature Fusion for Robust Face Inpainting.
- **Lựa chọn 2:** Robust Face Inpainting under Occlusion Uncertainty via Multi-Map Guidance and Selective Feature Fusion.

### 1.2. Thông điệp trung tâm
Việc phục hồi khuôn mặt (face inpainting) chịu ảnh hưởng trực tiếp của *sai số* và *sự bất định* trong việc xác định vùng bị che khuất. RAFI++ giải quyết vấn đề này bằng cách sử dụng **mask–boundary–confidence guidance** và **selective feature fusion** để giúp quá trình phục hồi ít nhạy cảm hơn với các guidance không hoàn hảo.

---

## 2. BA ĐÓNG GÓP KHOA HỌC CHÍNH (3 CONTRIBUTIONS)

### Contribution 1: Uncertainty-aware multi-map guidance
Đề xuất một biểu diễn (representation) mới cho guidance:
$$G = \{\widehat{M}, \widehat{B}, \widehat{C}\}$$
Trong đó:
- $\widehat{M}$: xác suất vùng occlusion (vùng bị che khuất).
- $\widehat{B}$: vùng boundary (đường biên) cần xử lý cẩn thận.
- $\widehat{C}$: độ tin cậy của guidance.

*Điểm cốt lõi:* Cần chứng minh rằng Confidence và Boundary information thực sự giúp face inpainting chống chịu tốt hơn trước sự bất định của phân vùng (segmentation uncertainty).

### Contribution 2: Guidance-conditioned selective feature fusion
Đề xuất cơ chế skip-attention:
$$A_l = \sigma(f(E_l, D_{l+1}, \widehat{M}_l, \widehat{B}_l, \widehat{C}_l))$$
$$\tilde{E}_l = A_l \odot E_l$$
*Ý tưởng khoa học:* Không phải mọi encoder feature đều đáng tin cậy khi input chứa vùng bị che khuất. Việc truyền đặc trưng (feature propagation) cần được điều khiển bởi vị trí che khuất, độ bất định của đường biên, và độ tin cậy.

### Contribution 3: Robustness-oriented evaluation protocol
Thay vì chỉ đánh giá trên một loại synthetic mask, bài báo chủ động tạo các mức độ nhiễu (perturbation) cho guidance:
$$M_\delta = T_\delta(M)$$
Với $\delta$ là các mức độ: dilation, erosion, boundary shift, random contour deformation, partial mask corruption.
Từ đó đánh giá:
$$\text{Performance} = f(\text{guidance error})$$
*Mục đích:* Chứng minh khi mask/guidance càng sai, hiệu suất của RAFI++ suy giảm chậm hơn so với RAFI gốc hoặc các phiên bản không có confidence-aware fusion.

---

## 3. BA CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS)

- **RQ1:** Does uncertainty-aware multi-map guidance improve face reconstruction compared with conventional single-mask guidance?
  *(So sánh: Binary Mask vs Binary Mask + Soft Mask + Boundary + Confidence)*
- **RQ2:** How much do guidance-conditioned skip-attention fusion and hierarchical multi-scale feature reuse contribute to reconstruction quality?
  *(Thực hiện Ablation Study)*
- **RQ3 (Quan trọng nhất):** How robust is RAFI++ to imperfect occlusion localization and mask uncertainty?
  *(Thiết kế các mức perturbation $\delta = 0, 5, 10, 15, 20\%$ và vẽ đồ thị `Perturbation Level \rightarrow PSNR/SSIM/LPIPS/Identity`)*

---

## 4. NÂNG CẤP BỘ CHỈ SỐ ĐÁNH GIÁ (METRICS)

Giữ lại các metrics hiện có: **PSNR, SSIM, L1**.
Cần bổ sung các metrics sau để phù hợp với bài toán face inpainting:

1. **LPIPS:** Đánh giá độ tương đồng về cảm nhận thị giác (perceptual similarity).
2. **Identity Similarity:** Sử dụng embedding của một pre-trained face-recognition model:
   $$S_{id} = \cos(\Psi(I_{syn}), \Psi(I_{gt}))$$
3. **Boundary-region error:** Chỉ tính toán metric trên một dải bao quanh đường biên để kiểm chứng trực tiếp sự đóng góp của boundary guidance:
   $$R_B = \text{Dilate}(M) - \text{Erode}(M)$$
4. **Confidence quality:** Kiểm chứng độ tin cậy của confidence map. Có thể dùng hệ số tương quan:
   $$Corr(1 - \widehat{C}, | \widehat{M} - M |)$$
   Hoặc đánh giá khả năng dùng $1 - \widehat{C}$ để phát hiện các pixel segmentation sai bằng chỉ số **AUROC**.

---

## 5. THỰC NGHIỆM CẦN THIẾT

### 5.1. Ablation Study Tối Thiểu (Short controlled ablation protocol)
Do giới hạn về thời gian huấn luyện (8 giờ/epoch trên RTX 4070), thiết kế giao thức đánh giá rút gọn: cùng subset, cùng seed, giảm số epoch, nhằm mục đích đo lường *relative contribution* giữa các module. 

| ID  | Cấu hình mô hình             | Multi-map | Skip-attention | H-MCSAM |
|:---:|:-----------------------------|:---------:|:--------------:|:-------:|
| A0  | Original RAFI                |     –     |       –        |    –    |
| A1  | RAFI + Soft Mask             |     ✓     |       –        |    –    |
| A2  | A1 + Boundary                |     ✓     |       –        |    –    |
| A3  | A2 + Confidence              |     ✓     |       –        |    –    |
| A4  | A3 + Skip-Attention          |     ✓     |       ✓        |    –    |
| A5  | Full RAFI++                  |     ✓     |       ✓        |    ✓    |

*(Có thể tách thêm cấu hình A6 cho Hierarchical M-CSAM nếu đủ tài nguyên. Full model vẫn sử dụng checkpoint 40 epochs hiện có)*

### 5.2. Robustness Benchmark Nhỏ (RAFI-RobustTest)
Tạo 5 kiểu occlusion mới từ test set của CelebA-HQ hiện tại:
1. Landmark-aligned mask
2. Dilated mask
3. Eroded mask
4. Shifted mask
5. Irregular/randomized mask
*(Chỉ cần chạy inference và đo đạc PSNR, SSIM, LPIPS, Identity. Không cần huấn luyện lại mô hình)*

---

## 6. CẤU TRÚC BÀI BÁO IEEE (6 TRANG)

| Phần | Nội dung chính | Ước lượng độ dài |
|:---|:---|:---|
| **I. Introduction** | Đặt vấn đề (Occluded Face $\rightarrow$ Imperfect Localization $\rightarrow$ Unreliable Features $\rightarrow$ Artifacts). Research gap. Câu hỏi nghiên cứu. | 0.7 trang |
| **II. Related Work** | Chỉ giữ 3 nhóm: face inpainting, segmentation-guided restoration, attention/gated feature propagation. Không tổng quan dài dòng. | 0.5 trang |
| **III. Proposed RAFI++** | Trọng tâm: Framework, Multi-map uncertainty guidance, Guidance-conditioned skip-attention, Hierarchical feature aggregation. Trình bày công thức tổng quát của Loss. | 1.3 trang |
| **IV. Experimental Protocol** | Dataset, implementation, baselines, metrics và mask perturbation protocol. | 0.8 trang |
| **V. Results** | Chứa Table I (Main comparison), Table II (Ablation), và Figure (Robustness curve - quan trọng nhất). | 1.5 trang |
| **VI. Discussion & Limitations** | Đề cập rõ: domain gap (synthetic-to-real), computational cost, inherent ambiguity, plausible reconstruction vs exact identity recovery. | 0.35 trang |
| **VII. Conclusion** | Tóm tắt đóng góp và kết luận. | 0.2 - 0.25 trang |

---

## 7. ABSTRACT DỰ KIẾN
Face inpainting under occlusion is challenging not only because the missing facial content is inherently ambiguous, but also because restoration quality depends strongly on the reliability of occlusion localization. Existing segmentation-guided approaches typically rely on deterministic binary masks and may propagate unreliable encoder features into the reconstruction process. This paper presents RAFI++, an uncertainty-aware face inpainting framework that introduces complementary soft-mask, boundary, and confidence guidance together with guidance-conditioned selective feature fusion. The proposed framework further employs hierarchical multi-scale channel-spatial attention to reuse contextual representations at different semantic levels. To evaluate robustness beyond standard reconstruction accuracy, we introduce a controlled occlusion-perturbation protocol that systematically varies mask boundaries, locations, and shapes. Experiments on CelebA-HQ evaluate reconstruction fidelity, perceptual quality, identity preservation, boundary consistency, and robustness under imperfect guidance. Results demonstrate that `[final experimental findings]`, showing that uncertainty-aware multi-map guidance provides a more robust alternative to conventional single-mask-guided face restoration.

---

## 8. LỊCH TRÌNH THỰC HIỆN ĐỀ XUẤT (KẾ HOẠCH HÀNH ĐỘNG)

- **21 - 25/07:** Chuẩn hóa test split và chạy thêm LPIPS + identity metric cho checkpoint hiện có.
- **25 - 27/07:** Xây dựng RAFI-RobustTest và chạy các experiments về mask perturbation (suy diễn model).
- **27 - 31/07:** Chạy ablation protocol rút gọn cho multi-map và skip-attention.
- **01/08:** Tổng hợp dữ liệu thành 2 bảng (So sánh chính + Ablation) và 1 biểu đồ Robustness.
- **02 - 05/08:** Viết lại toàn bộ nội dung thành bài báo IEEE 6 trang.
- **05 - 10/08:** Rà soát contribution, claims, kiểm tra tiếng Anh và nộp bài (submission).
