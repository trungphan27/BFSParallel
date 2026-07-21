# Hướng Dẫn Chạy Inference PØDA Trực Tiếp Trên Ảnh Bất Kỳ

Tài liệu này hướng dẫn bạn chi tiết từng bước để chạy thử (inference) mô hình phân đoạn ngữ nghĩa của PØDA trên bất kỳ bức ảnh đường phố nào bạn tải về từ Internet (ví dụ: ảnh lái xe ban đêm, trời mưa, tuyết...).

---

## Bước 1: Cài đặt Môi trường (Environment Setup)

Để chạy được mã nguồn của PØDA, bạn cần thiết lập môi trường Python thông qua Conda (Anaconda/Miniconda) để cài đặt các thư viện phụ thuộc (PyTorch, TorchVision, Pillow,...).

Mở terminal (hoặc Anaconda Prompt) tại thư mục mã nguồn PØDA (`f:\PODA`) và chạy các lệnh sau:

```bash
# 1. Tạo môi trường conda từ file cấu hình environment.yml có sẵn
conda env create --file environment.yml

# 2. Kích hoạt môi trường vừa tạo
conda activate poda_env
```

## Bước 2: Tải Mô hình đã huấn luyện (Checkpoints)

PØDA cung cấp sẵn các mô hình đã được huấn luyện (có thể là Source-only hoặc đã được Adapt). 
1. Truy cập vào **[Google Drive Link (Source models)](https://drive.google.com/drive/folders/15-NhVItiVbplg_If3HJibokJssu1NoxL?usp=sharing)** được cung cấp trong file `README.md`.
2. Tải xuống một file checkpoint (có định dạng `.pth` hoặc `.pt`, ví dụ: `model_best.pth`).
3. Tạo một thư mục tên là `ckpts` trong thư mục gốc `f:\PODA\` và đặt file mô hình vừa tải vào đó.
   - Đường dẫn file của bạn sẽ trông giống như: `f:\PODA\ckpts\model_best.pth`.

## Bước 3: Chuẩn bị Dữ liệu Ảnh (Image Preparation)

Theo như thiết kế của file `predict.py`, mã nguồn sẽ tự động đọc tất cả các ảnh nằm trong một thư mục được gán cứng (hardcoded) có tên là `predict_test/`. Do đó, bạn BẮT BUỘC phải làm đúng theo cấu trúc này:

1. Tạo một thư mục mới có tên chính xác là `predict_test` nằm ngay trong thư mục gốc `f:\PODA\`.
2. Lên Internet tải về bất kỳ bức ảnh đường phố nào bạn muốn kiểm tra (ví dụ: `rainy_street.jpg`, `night_drive.png`).
3. Đưa tất cả các ảnh đó vào thư mục `predict_test`.

*Cấu trúc thư mục của bạn lúc này sẽ trông như sau:*
```text
f:\PODA\
├── predict.py
├── ...
├── ckpts\
│   └── model_best.pth      <-- (Mô hình tải ở Bước 2)
└── predict_test\
    ├── rainy_street.jpg    <-- (Ảnh tải trên mạng)
    └── night_drive.png     <-- (Ảnh tải trên mạng)
```

## Bước 4: Chạy Tập Lệnh Suy Luận (Run Inference)

Sau khi môi trường và dữ liệu đã sẵn sàng, bạn chạy script `predict.py` để tiến hành phân đoạn hình ảnh. 
Bạn cần chỉ định đường dẫn đến mô hình bằng tham số `--ckpt` và thư mục lưu kết quả bằng `--save_val_results_to`.

Chạy lệnh sau trong terminal:

```bash
python predict.py \
  --ckpt ckpts/model_best.pth \
  --save_val_results_to predict_results
```

**Quá trình thực thi sẽ diễn ra như sau:**
- Mã nguồn sẽ khởi tạo mô hình `DeepLabv3+` với backbone `ResNet-50` (sử dụng weights của CLIP).
- Nó sẽ load trọng số từ file `ckpts/model_best.pth`.
- Nó sẽ tự động quét tất cả các tệp tin trong thư mục `predict_test/`.
- Với mỗi ảnh, nó đưa qua mô hình, gán nhãn cho từng pixel thành 19 lớp đối tượng của Cityscapes (đường, xe, người, biển báo, v.v.).
- Kết quả được tô màu tương ứng với các lớp và lưu lại trong thư mục `predict_results/`.

## Bước 5: Xem Kết Quả (Visualization)

Sau khi terminal chạy xong (không báo lỗi), hãy mở thư mục `f:\PODA\predict_results\`.

Trong này, bạn sẽ thấy các ảnh kết quả phân đoạn có cùng tên với các ảnh gốc mà bạn đã đặt trong `predict_test/`. Ảnh kết quả là một "mặt nạ" (mask) đã được tô màu (ví dụ: màu tím là đường, màu xanh dương đậm là xe hơi, màu đỏ là người đi bộ...). Bạn có thể so sánh ảnh này với ảnh gốc để đánh giá chất lượng của mô hình PØDA trên miền dữ liệu của bạn.

---

> [!WARNING]
> **Lưu ý Quan Trọng Về Kích Thước Ảnh (OOM Error)**
> 
> Trong file `predict.py`, nếu bạn KHÔNG truyền tham số `--crop_val`, ảnh gốc sẽ được đưa nguyên bản kích thước (full resolution) vào mô hình. 
> - Nếu bạn tải một bức ảnh có độ phân giải quá lớn (như 4K), card đồ họa (GPU) của bạn có thể bị tràn bộ nhớ (Out of Memory - OOM).
> - Để khắc phục, bạn có thể truyền thêm tham số cắt/thu nhỏ ảnh, ví dụ: 
>   `python predict.py --ckpt ckpts/model_best.pth --save_val_results_to predict_results --crop_val --crop_size 768`
> - Mặc định `--crop_size` là 513 nếu gọi `--crop_val` mà không truyền `--crop_size`.
