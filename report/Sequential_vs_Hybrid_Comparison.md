# So sánh chi tiết BFS Tuần tự và BFS Hybrid (MPI + OpenMP)

Dựa trên mã nguồn của dự án BFSParallel, tài liệu này trình bày những sự khác biệt cốt lõi, các cải tiến đột phá của thuật toán BFS phân tán lai (Hybrid MPI + OpenMP) so với phiên bản BFS tuần tự truyền thống, cũng như những kết quả mà hệ thống phân tán này đạt được.

---

## 1. Khác biệt về Mô hình Thực thi và Kiến trúc

### 1.1. BFS Tuần tự (Sequential BFS)
- **Luồng xử lý:** Chạy trên một Process duy nhất, sử dụng 1 CPU Core duy nhất (Đơn luồng).
- **Bộ nhớ:** Sử dụng không gian bộ nhớ dùng chung, cục bộ trên 1 máy tính. Mọi thay đổi trên mảng khoảng cách `dist[]` có hiệu lực ngay lập tức.
- **Tính mở rộng:** Không có. Nếu đồ thị quá lớn, thuật toán sẽ mất rất nhiều thời gian chạy hoặc hết RAM cục bộ.

### 1.2. BFS Hybrid Phân tán (MPI + OpenMP)
- **Luồng xử lý đa tầng:** Chạy trên một Cluster nhiều máy tính (Ví dụ: 3 máy).
  - Tầng mạng LAN (MPI): Tạo ra nhiều Tiến trình (Processes) phân tán trên 3 máy.
  - Tầng cục bộ (OpenMP): Bên trong mỗi máy, tiến trình MPI lại tự động sinh ra nhiều Luồng (Threads) chạy song song trên các nhân CPU vật lý của máy đó.
- **Bộ nhớ phân tán & Phân vùng dữ liệu (1D Partitioning):** Mặc dù mảng đồ thị CSR được sao chép (Replicate) trên cả 3 máy để tránh chi phí truyền mạng liên tục, nhưng **khối lượng công việc thì được băm nhỏ**. 
  - Tập đỉnh $V$ được chia đều cho $P$ máy. Máy thứ $r$ chỉ tính toán trên dải đỉnh thuộc quyền sở hữu của nó.
  - OpenMP tiếp tục chia nhỏ dải đỉnh này cho các Thread bằng cơ chế cấp phát động (`schedule(dynamic)`), đảm bảo các nhân CPU không bị rảnh rỗi.

---

## 2. Khác biệt về Thuật toán Cốt lõi (Direction-Optimizing)

Đây là khác biệt lớn nhất tạo ra sự vượt trội về mặt thuật toán của bản Hybrid so với bản Tuần tự.

### 2.1. BFS Tuần tự: Chiến thuật Duyệt Thuận (Top-Down)
- Bản tuần tự sử dụng một Hàng đợi (Queue) tiêu chuẩn (FIFO).
- Thuật toán bốc từng đỉnh từ Queue ra, và xét tất cả các hàng xóm của nó.
- **Nhược điểm chí mạng:** Trên đồ thị R-MAT (cấu trúc mạng xã hội), số lượng đỉnh trong biên (Frontier) ở các level giữa sẽ bùng nổ lên rất lớn. BFS Tuần tự sẽ cắm đầu duyệt hàng trăm triệu cạnh một cách lãng phí, dù phần lớn các hàng xóm đó đã được thăm ở level trước.

### 2.2. BFS Hybrid: Tự động Đảo Hướng (Beamer et al. 2012)
Thuật toán phân tán có khả năng tự đánh giá và "chuyển số" linh hoạt giữa 2 hướng duyệt:
- **Top-Down (Duyệt Thuận):** Dùng khi biên nhỏ (lúc mới bắt đầu hoặc sắp kết thúc). Máy tính quét các đỉnh trong biên, tìm các lân cận chưa thăm và nạp vào biên mới.
- **Bottom-Up (Duyệt Ngược):** Dùng khi biên bùng nổ quá lớn (Level giữa). Thay vì lấy đỉnh trong biên đi tìm hàng xóm, máy tính làm ngược lại: Nó bốc các **Đỉnh Chưa Thăm**, kiểm tra lân cận của chúng xem **có ai đang nằm trong biên hiện tại không**. Nếu CÓ, nó lập tức cập nhật khoảng cách và **BREAK (Dừng ngay lập tức)**, không cần xét các hàng xóm còn lại nữa.
- **Sự cải tiến:** Nhờ lệnh `break` này, máy tính bỏ qua hàng triệu thao tác duyệt cạnh dư thừa, làm giảm đột ngột số lượng tính toán.

---

## 3. Khác biệt về Cấu trúc Dữ liệu và Xử lý Xung đột

### 3.1. Cấu trúc Frontier (Tập Biên)
- **Tuần tự:** Chỉ cần một mảng số nguyên đơn giản `queue` và 2 con trỏ `head`, `tail`.
- **Hybrid:** Thiết kế cấu trúc biên Kép (Dual Representation) cực kỳ phức tạp để phục vụ đảo hướng:
  - Dùng **Mảng Dense** (Mảng ID bình thường) để duyệt nhanh trong Top-Down.
  - Dùng **Mảng Bitmap** (Mảng Bit) để kiểm tra $O(1)$ xem một đỉnh có thuộc biên hay không trong Bottom-Up.
  - Các thao tác cập nhật bit được bảo vệ bằng cơ chế phần cứng `#pragma omp atomic`.

### 3.2. Chống ghi đè (Race Condition) và Đồng bộ
- **Tuần tự:** Không bao giờ bị xung đột ghi đè, vì chỉ có 1 luồng duy nhất thao tác.
- **Hybrid:** Hàng chục luồng trên 3 máy tính có thể đồng thời tìm ra một con đường ngắn nhất đến cùng 1 đỉnh.
  - **Tầng OpenMP (Trong 1 máy):** Dùng lệnh khóa phần cứng siêu nhanh C-A-S (`__sync_val_compare_and_swap`). CPU nào gán khoảng cách thành công trước sẽ đẩy đối thủ ra.
  - **Tầng MPI (Giữa 3 máy):** Vì 3 máy ở 3 nơi khác nhau, chúng chỉ ghi khoảng cách vào mảng RAM cục bộ. Ở cuối mỗi Level, cả 3 máy gọi lệnh **`MPI_Allreduce(MPI_MAX)`**. Mạng LAN sẽ tự động trộn 3 mảng `dist[]` lại với nhau, hàm MAX tự động vớt những giá trị đã được cập nhật đè lên số -1, làm cho cả 3 máy đồng bộ hóa kết quả hoàn hảo mà không dẫm chân lên nhau.

---

## 4. Kết quả đạt được của BFS Hybrid

Nhờ sự kết hợp của 3 yếu tố: Cụm máy chủ (MPI), Đa lõi (OpenMP) và Tối ưu thuật toán (Đảo hướng), hệ thống Hybrid mang lại các kết quả đột phá:

1. **Giảm thiểu khổng lồ lượng cạnh phải duyệt (Edge Traversal Reduction):**
   - Phiên bản tuần tự bắt buộc phải quét qua tất cả $E$ cạnh của đồ thị.
   - Phiên bản Hybrid (nhờ Bottom-Up) chỉ duyệt một phần rất nhỏ số lượng cạnh thực tế. Chỉ số "Visited Edges" ở output sẽ nhỏ hơn rất nhiều so với tổng số cạnh của đồ thị.
2. **Đạt chỉ số MTEPS cực cao:**
   - MTEPS (Mega Traversed Edges Per Second) là chỉ số chuẩn của Graph500 để đo sức mạnh xử lý đồ thị.
   - Nhờ việc cắt giảm thao tác thừa và chia việc cho nhiều CPU, bản Hybrid đẩy chỉ số MTEPS lên gấp nhiều lần so với thuật toán chạy tuần tự.
3. **Độ tăng tốc (Speedup):**
   - Cung cấp tốc độ giải quyết bài toán nhanh hơn rõ rệt (Speedup > 1).
   - Tuy nhiên, độ tăng tốc **không tuyến tính tuyệt đối**. Nguyên nhân là do mạng LAN xuất hiện điểm nghẽn (Overhead): Lệnh `MPI_Allreduce(MPI_MAX)` bắt buộc hệ thống phải chép đi chép lại toàn bộ mảng `dist[]` (kích thước bằng số lượng đỉnh) giữa 3 máy thông qua Switch mạng ở cuối mỗi một Level BFS, tiêu tốn thời gian truyền tải (Communication Time).

**Tổng kết:** Việc nâng cấp từ BFS Tuần tự lên Hybrid MPI+OpenMP không chỉ là nhồi thêm sức mạnh tính toán, mà còn là sự lột xác hoàn toàn về cấu trúc dữ liệu, cách phân công lao động, và đặc biệt là trí thông minh của thuật toán (Đảo hướng) để xử lý lượng dữ liệu khổng lồ một cách thông minh nhất.
