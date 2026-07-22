# KẾ HOẠCH CHI TIẾT MODULE 1 (21–25/07)
**Mục tiêu:** Chuẩn hóa test split và chạy LPIPS + identity metric cho checkpoint hiện có.

Dựa trên cấu trúc code hiện tại của project, dưới đây là danh sách chi tiết các công việc bạn cần làm (step-by-step).

---

## 1. Chuẩn hóa Test Split (Standardizing Test Split)

Hiện tại class `RAFIppCelebA` (trong `Dataset/datasets.py`) đọc danh sách file từ thư mục `splits/test.txt`. Việc "chuẩn hóa" nghĩa là bạn phải đảm bảo file `test.txt` này tuân thủ nguyên tắc:

- **Identity-disjoint:** Các subject trong `test.txt` tuyệt đối không được xuất hiện trong `train.txt` hay `val.txt`.
- **Số lượng cố định:** Chốt một số lượng ảnh test (theo báo cáo là 3000 ảnh).
- **Hành động cần làm:**
  1. Kiểm tra lại file `[data_root]/splits/test.txt`. Đếm số dòng (số lượng file).
  2. Nếu danh sách này chưa được phân tách cẩn thận theo identity (nghĩa là có khả năng bị trùng người với tập train), bạn cần tạo lại file `test.txt` bằng script lọc theo danh sách identity của CelebA, và đảm bảo file text đó không bị thay đổi trong suốt quá trình làm bài RIVF.

---

## 2. Cài đặt các thư viện bổ sung cho Metrics

Bạn cần cài thêm 2 thư viện chuẩn để chạy LPIPS và Identity metric:
Mở terminal và chạy lệnh:
```bash
pip install lpips facenet-pytorch
```
- `lpips`: Dùng để tính Perceptual Similarity.
- `facenet-pytorch`: Dùng để trích xuất đặc trưng khuôn mặt (face embedding) tính Identity Similarity. (Thư viện này đã được import sẵn trong class `IdentityEncoder` tại `Model/networks.py`).

---

## 3. Cập nhật file `Utils/metrics.py`

Mở file `f:\RAFIpp_Project\Utils\metrics.py` và bổ sung hai hàm tính LPIPS và Identity:

```python
import lpips
from Model.networks import IdentityEncoder # (Hoặc khởi tạo trực tiếp IdentityEncoder trong file test)

# Cần định nghĩa lpips_vgg model ở dạng global hoặc truyền vào hàm để tránh load lại model nhiều lần
_lpips_vgg = None

def get_lpips_model(device):
    global _lpips_vgg
    if _lpips_vgg is None:
        _lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        _lpips_vgg.eval()
    return _lpips_vgg

def lpips_metric(pred: torch.Tensor, target: torch.Tensor, device) -> torch.Tensor:
    loss_fn_vgg = get_lpips_model(device)
    # pred, target hiện đang nằm trong khoảng [-1, 1], phù hợp với đầu vào của LPIPS
    val = loss_fn_vgg(pred, target)
    return val.mean()

def identity_metric(pred: torch.Tensor, target: torch.Tensor, id_encoder: torch.nn.Module) -> float:
    """ Tính cosine similarity giữa đặc trưng danh tính của ảnh sinh và ảnh gốc """
    with torch.no_grad():
        feat_pred = id_encoder(pred)
        feat_target = id_encoder(target)
        # feat_pred, feat_target đã được normalize(dim=1) trong IdentityEncoder
        # cos_sim = (A * B) / (|A|*|B|) nhưng do đã norm, chỉ cần nhân vô hướng
        cos_sim = torch.sum(feat_pred * feat_target, dim=1)
    return cos_sim.mean().item()
```
*Ghi chú:* Hàm `summarize_restoration` hiện tại mới có l1, psnr, ssim. Bạn có thể giữ nguyên hàm này, và tính riêng LPIPS/Identity ở bên ngoài vòng lặp trong `test.py` để tiết kiệm bộ nhớ, vì các mô hình này cần tải lên GPU.

---

## 4. Cập nhật luồng chạy trong `test.py`

Mở file `f:\RAFIpp_Project\test.py`. Thêm code khởi tạo model LPIPS và Identity, sau đó đưa vào vòng lặp evaluation.

**Ví dụ sửa file `test.py`:**
```python
# Đầu file: thêm import
import lpips
from Model.networks import IdentityEncoder

# Trong hàm main(), sau khi khai báo device:
print("Loading metrics models...")
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
loss_fn_vgg.eval()

id_encoder = IdentityEncoder().to(device)
id_encoder.eval()

# Bổ sung key cho metrics
sums = {'l1': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'mask_dice': 0.0, 'mask_iou': 0.0, 'lpips': 0.0, 'identity': 0.0}

# Trong vòng lặp for batch in tqdm(loader, desc='Testing'):
# ... (code hiện tại)
outputs = system.inference(batch)
pred_img = outputs['isyn'] # Hoặc 'restored' tùy bạn chọn ảnh nào làm đại diện
gt_img = batch['gt']

# Thêm dòng tính LPIPS và ID:
lpips_val = loss_fn_vgg(pred_img, gt_img).mean().item()
sums['lpips'] += lpips_val

# feat_pred/feat_target đã norm nên tính Cosine Similarity là nhân vô hướng
feat_pred = id_encoder(pred_img)
feat_gt = id_encoder(gt_img)
cos_sim = torch.sum(feat_pred * feat_gt, dim=1).mean().item()
sums['identity'] += cos_sim
```

---

## 5. Chạy đánh giá cho các Checkpoint (Execution)

Sau khi code đã sẵn sàng, bạn cần chạy đánh giá thực tế cho 2 model chính:

1. **Cho RAFI++ (Checkpoint hiện có)**
   Mở terminal và chạy lệnh (thay đường dẫn cho đúng):
   ```bash
   python test.py --data_root path/to/dataset --checkpoint path/to/rafipp_epoch40.pth --batch_size 4
   ```
   Lưu lại file `metrics.json` thu được. Bạn sẽ có L1, PSNR, SSIM, **LPIPS**, và **Identity**.

2. **Cho RAFI gốc (A0)**
   Mô hình gốc nằm ở `f:\RAFIpp_Project\Face_Img_Inpainting`. Bạn cũng cần làm tương tự: copy đoạn code tính LPIPS và Identity sang file `test.py` bên trong folder RAFI gốc, và chạy bằng lệnh tương tự để đánh giá checkpoint của model A0. 
   Lưu ý so sánh kết quả để chứng minh RAFI++ làm tốt hơn không chỉ ở PSNR/SSIM mà còn ở LPIPS (độ chân thực) và Identity (giữ đặc trưng khuôn mặt gốc).
