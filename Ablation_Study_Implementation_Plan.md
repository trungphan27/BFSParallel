# HƯỚNG DẪN CHI TIẾT THỰC HIỆN ABLATION STUDY (RAFI++)

Dựa trên cấu trúc code hiện tại của bạn, trong đó `f:\RAFIpp_Project\Face_Img_Inpainting` là model RAFI gốc (A0), và `f:\RAFIpp_Project\Model` chứa RAFI++ (A5), dưới đây là giải thích chi tiết và kế hoạch triển khai Ablation Study.

---

## 1. Giải thích các cấu hình A1, A2, A3

Trong báo cáo, model RAFI gốc (A0) chỉ sử dụng một **Binary Mask** để nối (concatenate) vào ảnh đầu vào, dùng các skip connection nối trực tiếp (dùng `torch.cat`), và các module attention ở bottleneck được xếp nối tiếp nhau đơn thuần.

Các cấu hình A1, A2, A3 nhằm mục đích kiểm chứng sức mạnh của từng bản đồ (map) sinh ra từ SegNet++ (chưa dùng Skip-Attention hay H-MCSAM):

- **A1 (RAFI + Soft Mask):** 
  - *SegNet++:* Vẫn xuất ra 3 map, nhưng ta chỉ lấy **Soft Mask** (`mask_pred`) để sử dụng.
  - *RestoreNet++:* Đầu vào chỉ gồm `torch.cat([image, mask_pred], dim=1)` (4 channels). Các skip connection là nối trực tiếp (như RAFI gốc). Bottleneck là M-CSAM nối tiếp, không có residual routes.
  
- **A2 (A1 + Boundary):**
  - Sử dụng **Soft Mask + Boundary Map**.
  - *RestoreNet++:* Đầu vào gồm `torch.cat([image, mask_pred, boundary_pred], dim=1)` (5 channels). Skip connection và bottleneck vẫn giữ nguyên cấu trúc đơn giản như A1.

- **A3 (A2 + Confidence):**
  - Sử dụng cả 3 bản đồ: **Soft Mask + Boundary Map + Confidence Map**.
  - *RestoreNet++:* Đầu vào gồm `torch.cat([image, mask_pred, boundary_pred, confidence_pred], dim=1)` (6 channels). Skip connection và bottleneck vẫn đơn giản như A1.

*Tiếp theo đó:*
- **A4 (A3 + Skip-Attention):** Giữ cấu hình A3, kích hoạt cơ chế `SkipAttentionGate` để lọc feature từ encoder trước khi đưa vào decoder.
- **A5 (Full RAFI++):** Giữ cấu hình A4, kích hoạt thêm các `alpha` (hierarchical residual routes) giữa các M-CSAM blocks.

---

## 2. Kế hoạch sửa code (Implementation Plan)

Thay vì viết 5 file model khác nhau, cách tốt nhất là **thêm các cờ (flags)** vào class `RestoreNetPP` trong `f:\RAFIpp_Project\Model\networks.py`.

### Bước 2.1: Chỉnh sửa hàm `__init__` của RestoreNetPP

Sửa lại để model nhận các biến điều khiển cấu hình:

```python
class RestoreNetPP(nn.Module):
    def __init__(self, use_boundary=True, use_confidence=True, 
                 use_skip_attention=True, use_hierarchical_mcsam=True):
        super().__init__()
        
        self.use_boundary = use_boundary
        self.use_confidence = use_confidence
        self.use_skip_attention = use_skip_attention
        self.use_hierarchical_mcsam = use_hierarchical_mcsam
        
        # Tính số channel đầu vào
        in_channels = 3 + 1 # Image (3) + Mask (1)
        if self.use_boundary: in_channels += 1
        if self.use_confidence: in_channels += 1
        
        # Lớp đầu tiên cần linh hoạt số channels
        self.enc1 = GatedConv2d(in_channels, 64, stride=1)
        self.enc2 = GatedConv2d(64, 128, stride=2)
        # ... (giữ nguyên các lớp khác) ...
```

### Bước 2.2: Chỉnh sửa luồng chạy `forward` của RestoreNetPP

```python
    def forward(self, image, mask_pred, boundary_pred, confidence_pred):
        # 1. Điều khiển Multi-map guidance
        inputs = [image, mask_pred]
        if self.use_boundary: inputs.append(boundary_pred)
        if self.use_confidence: inputs.append(confidence_pred)
        x = torch.cat(inputs, dim=1)
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 2. Điều khiển Hierarchical M-CSAM
        z1 = self.mcsam1(e4)
        z2_prime = self.mcsam2(z1)
        if self.use_hierarchical_mcsam:
            z2 = z2_prime + self.alpha1 * z1
        else:
            z2 = z2_prime
            
        z3_prime = self.mcsam3(z2)
        if self.use_hierarchical_mcsam:
            z3 = z3_prime + self.alpha2 * z2 + self.alpha3 * z1
        else:
            z3 = z3_prime

        # 3. Điều khiển Skip-Attention
        up4 = self.up_d4(z3, size=e3.shape[-2:])
        if self.use_skip_attention:
            e3_gated, gate3 = self.skip3(e3, up4, mask_pred, boundary_pred, confidence_pred)
        else:
            e3_gated, gate3 = e3, None  # Truyền thẳng không lọc
            
        d4 = self.dec4(torch.cat([up4, e3_gated], dim=1))
        d4 = self.refine4(d4)

        # Lặp lại tương tự cho các up3, up2...
        up3 = self.up_d3(d4, size=e2.shape[-2:])
        if self.use_skip_attention:
            e2_gated, gate2 = self.skip2(e2, up3, mask_pred, boundary_pred, confidence_pred)
        else:
            e2_gated, gate2 = e2, None
            
        d3 = self.dec3(torch.cat([up3, e2_gated], dim=1))
        d3 = self.refine3(d3)

        up2 = self.up_d2(d3, size=e1.shape[-2:])
        if self.use_skip_attention:
            e1_gated, gate1 = self.skip1(e1, up2, mask_pred, boundary_pred, confidence_pred)
        else:
            e1_gated, gate1 = e1, None
            
        d2 = self.dec2(torch.cat([up2, e1_gated], dim=1))
        d2 = self.refine2(d2)

        # ... (phần cuối giữ nguyên) ...
```

### Bước 2.3: Điều chỉnh RAFIpp.py để khởi tạo các model con

Trong class `RAFIpp`, bạn truyền các cấu hình ablation xuống:

```python
class RAFIpp(nn.Module):
    def __init__(self, config_id="A5"):
        super().__init__()
        self.segnet = SegNetPP() # Luôn tạo ra 3 map để tính loss cho SegNet, chỉ là RestoreNet có dùng hay không
        
        if config_id == "A1":
            self.restore = RestoreNetPP(use_boundary=False, use_confidence=False, use_skip_attention=False, use_hierarchical_mcsam=False)
        elif config_id == "A2":
            self.restore = RestoreNetPP(use_boundary=True, use_confidence=False, use_skip_attention=False, use_hierarchical_mcsam=False)
        elif config_id == "A3":
            self.restore = RestoreNetPP(use_boundary=True, use_confidence=True, use_skip_attention=False, use_hierarchical_mcsam=False)
        elif config_id == "A4":
            self.restore = RestoreNetPP(use_boundary=True, use_confidence=True, use_skip_attention=True, use_hierarchical_mcsam=False)
        elif config_id == "A5":
            self.restore = RestoreNetPP(use_boundary=True, use_confidence=True, use_skip_attention=True, use_hierarchical_mcsam=True)
```

---

## 3. Kế hoạch chạy thực tế (Execution Plan)

Vì RTX 4070 mất ~8 tiếng/epoch nếu train toàn bộ tập dữ liệu, ta cần làm một **Short Controlled Ablation Protocol**:

1. **Chuẩn bị 1 subset nhỏ hơn:** Thay vì dùng 24,000 ảnh train, lấy ngẫu nhiên 3,000 ảnh (cố định seed). 
2. **Train từ đầu (train from scratch) cho mỗi cấu hình (A1 $\rightarrow$ A4):** Train mỗi model khoảng **5-10 epochs** trên tập subset này. Do chung subset và seed, đồ thị Loss và PSNR/SSIM trên tập validation sẽ cho thấy chính xác module nào giúp model học nhanh hơn và hội tụ tốt hơn.
3. **Không cần train lại RAFI gốc (A0) và Full RAFI++ (A5):**
   - **A5** đã có checkpoint 40 epoch, cứ dùng thẳng để so sánh mức tối đa.
   - **A0** nằm ở thư mục `Face_Img_Inpainting`, bạn cũng lấy kết quả từ checkpoint cũ của nó.
4. **Lưu lại các chỉ số SSIM / PSNR của tập Dev** cho từng cấu hình A1-A4 ở epoch thứ 5 (hoặc thứ 10), và đưa vào bảng Bảng II trong báo cáo IEEE để chứng minh module nào mang lại bước nhảy lớn nhất về chất lượng.
