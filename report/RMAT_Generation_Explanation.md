# Giải thích Chi tiết Quá trình Sinh Đồ thị R-MAT (Recursive MATrix)

Đồ thị R-MAT (Recursive MATrix) là thuật toán tiêu chuẩn (được sử dụng trong benchmark Graph500) để sinh ra các đồ thị có cấu trúc mô phỏng mạng lưới trong thế giới thực (như mạng xã hội, mạng World Wide Web). Đặc trưng của nó là phân phối bậc theo **Power-law**: có một số rất ít các đỉnh làm "trung tâm" (hub) với hàng nghìn kết nối, trong khi phần lớn các đỉnh khác chỉ có vài kết nối.

Tài liệu này trình bày chi tiết từ đầu đến cuối quá trình hệ thống sinh ra một đồ thị R-MAT, dựa trên mã nguồn trong file `src/graph.c`.

---

## 1. Đầu vào (Inputs)
Quá trình sinh đồ thị bắt đầu từ hàm `graph_rmat_generate` nhận vào các tham số sau:
- `num_vertices` ($N$): Số đỉnh mong muốn.
- `scale_factor` ($K$): Bậc trung bình của mỗi đỉnh. Tổng số cạnh mục tiêu sẽ là $E = N \times K$.
- `seed`: Hạt giống khởi tạo ngẫu nhiên. Vì nhiều máy tính trong cụm (cluster) cùng sinh đồ thị song song, chúng phải dùng chung một `seed` để đảm bảo sinh ra đồ thị y hệt nhau.
- `params`: 4 xác suất $a, b, c, d$ ($a+b+c+d=1$). Mặc định dùng chuẩn Graph500: $a=0.57, b=0.19, c=0.19, d=0.05$.

---

## 2. Bước 1: Chuẩn bị & Cấp phát bộ nhớ
1. **Làm tròn số đỉnh:** Thuật toán R-MAT chia đôi ma trận đệ quy, nên số đỉnh $N$ **bắt buộc phải là lũy thừa của 2**. Hệ thống sẽ làm tròn $N$ lên lũy thừa của 2 gần nhất (VD: Nhập 1.000.000 sẽ làm tròn thành 1.048.576 đỉnh).
2. **Khởi tạo mảng cạnh tạm (Raw Edges):** Cấp phát một mảng `edges` khổng lồ để lưu trữ tạm thời các cạnh. Vì đồ thị vô hướng (mỗi cạnh $u-v$ phải lưu thành 2 chiều $u \rightarrow v$ và $v \rightarrow u$), mảng tạm này có kích thước bằng $2 \times E$.
3. **Khởi tạo Sinh số ngẫu nhiên:** Sử dụng thuật toán `xorshift64` siêu tốc để làm bộ sinh số ngẫu nhiên. Trạng thái (state) được khởi tạo bằng cách XOR `seed` với một hằng số.

---

## 3. Bước 2: Vòng lặp Sinh Cạnh Đệ Quy (Hàm `rmat_edge`)
Đây là trái tim của R-MAT. Hệ thống sẽ lặp lại $E$ lần, mỗi lần sinh ra một cạnh duy nhất $(u, v)$ bằng cách ném cạnh đó vào một ô trong ma trận kề $N \times N$.

Tuy nhiên, thay vì chọn ngẫu nhiên đều, máy tính dùng chiến thuật **chia để trị (đệ quy)**:
1. Xét ma trận kề kích thước $N \times N$. Tọa độ gốc ban đầu là $(u=0, v=0)$, bước nhảy `step = N`.
2. Máy tính chia ma trận làm 4 góc (Quadrant):
   - **Góc trên-trái (a = 57%):** Cạnh nối giữa 2 đỉnh chỉ số nhỏ (Rất dễ rớt vào đây).
   - **Góc trên-phải (b = 19%):** Cạnh nối từ đỉnh chỉ số nhỏ đến đỉnh chỉ số lớn.
   - **Góc dưới-trái (c = 19%):** Cạnh nối từ đỉnh chỉ số lớn đến đỉnh chỉ số nhỏ.
   - **Góc dưới-phải (d = 5%):** Cạnh nối giữa 2 đỉnh chỉ số lớn (Rất khó rớt vào đây).
3. Máy tính tung xúc xắc (`rand_double` ra số từ 0 đến 1). Dựa vào xác suất, nó quyết định ném cạnh vào 1 trong 4 góc.
4. Tùy vào góc được chọn, tọa độ $(u, v)$ được dịch chuyển:
   - Nếu vào ô **a**: Không dịch.
   - Nếu vào ô **b**: $v = v + step/2$.
   - Nếu vào ô **c**: $u = u + step/2$.
   - Nếu vào ô **d**: Cả $u$ và $v$ đều cộng thêm $step/2$.
5. Kích thước `step` được chia đôi. Lặp lại quá trình chia 4 ô nhỏ hơn bên trong ô vừa chọn. Quá trình dừng lại khi `step = 1` (đã khóa mục tiêu chính xác 1 ô $(u, v)$).

*Hệ quả:* Vì xác suất ô $a$ rất lớn, các đỉnh có ID nhỏ (nằm ở đầu) sẽ nhận được lượng cạnh dồn dập, tạo thành các siêu đỉnh (hub).

**Loại bỏ vòng lặp (Self-loop):** Nếu cạnh sinh ra có dạng $(u, u)$, nó sẽ bị ném bỏ. Nếu hợp lệ, hệ thống sẽ chèn 2 bản ghi $(u, v)$ và $(v, u)$ vào mảng tạm `edges`.

### 3.1. Ví dụ Minh họa Trực quan (Với N = 7, K = 4)
Giả sử người dùng yêu cầu sinh đồ thị với số đỉnh $N = 7$ và bậc trung bình $K = 4$.

**Khởi tạo:**
- $N = 7$ không phải là lũy thừa của 2, máy tính tự động làm tròn lên $N = 8 = 2^3$.
- Tổng số cạnh mục tiêu cần sinh: $E = 8 \times 4 = 32$ cạnh.
- Lưới ma trận kề sẽ có kích thước $8 \times 8$. Vì $8 = 2^3$, thuật toán sẽ chạy **đúng 3 vòng lặp** đệ quy để "khóa" được một cạnh $(u, v)$.
- Giả sử xác suất vẫn là $a=0.57, b=0.19, c=0.19, d=0.05$. 

**Trace (Theo vết) sinh cạnh thứ nhất:**
Khởi tạo gốc: $u = 0, v = 0$, biến `step = 8`.

* **Vòng lặp 1:**
  - `step` bị chia đôi: `step = 4`. Ma trận $8 \times 8$ bị bổ làm 4 góc, mỗi góc là $4 \times 4$.
  - Tung xúc xắc: Lấy ngẫu nhiên ra số $r = 0.8$.
  - Kiểm tra điều kiện: $r$ rơi vào khoảng từ $0.76$ đến $0.95$ ($a+b+c$), nên nó rớt trúng **góc c (dưới-trái)**.
  - Tọa độ dịch chuyển: Cập nhật $u = u + step = 0 + 4 = 4$. Biến $v$ giữ nguyên bằng 0.
  - Lúc này, ta đã khoanh vùng cạnh nằm trong khu vực hàng $4-7$, cột $0-3$.

* **Vòng lặp 2:**
  - `step` bị chia đôi tiếp: `step = 2`. Góc $4 \times 4$ hiện tại lại bị bổ làm 4 góc nhỏ $2 \times 2$.
  - Tung xúc xắc: Lần này $r = 0.3$.
  - Kiểm tra điều kiện: $r < 0.57$ ($a$), nên nó rớt trúng **góc a (trên-trái)** của khu vực hiện tại.
  - Tọa độ dịch chuyển: Vì là góc $a$, cả $u$ và $v$ đều giữ nguyên. $u = 4, v = 0$.
  - Khu vực khoanh vùng thu hẹp lại thành hàng $4-5$, cột $0-1$.

* **Vòng lặp 3:**
  - `step` tiếp tục chia đôi: `step = 1`. Góc $2 \times 2$ bị chia thành 4 ô đơn $1 \times 1$.
  - Tung xúc xắc: Được số $r = 0.65$.
  - Kiểm tra điều kiện: $r$ rơi vào khoảng $0.57$ đến $0.76$ ($a+b$), nên rớt trúng **góc b (trên-phải)**.
  - Tọa độ dịch chuyển: Cập nhật $v = v + step = 0 + 1 = 1$. Biến $u$ giữ nguyên bằng 4.
  - Khu vực khoanh vùng chỉ còn duy nhất 1 ô. Vòng lặp kết thúc (vì `step` không còn > 1).

**Kết quả:**
- Cạnh vừa sinh ra là $(u = 4, v = 1)$.
- Cạnh này hợp lệ (vì $u \neq v$).
- Máy tính sẽ thêm 2 cặp $(4, 1)$ và $(1, 4)$ vào mảng tạm `edges`.
- Quá trình này được lặp đi lặp lại đủ 32 lần để gom đủ số lượng 32 liên kết (thực tế sau khi loại bỏ trùng lặp, số cạnh sẽ ít hơn 32 một chút).

---

## 4. Bước 3: Sắp xếp và Khử Trùng lặp (Deduplication)
Do R-MAT sinh cạnh ngẫu nhiên và dồn tụ, khả năng máy tính tung xúc xắc và sinh ra **cùng một cạnh** nhiều lần là rất cao.

1. **Sắp xếp:** Máy tính dùng thuật toán QuickSort (`qsort`) để sắp xếp mảng `edges` tăng dần theo $u$, sau đó theo $v$. Khi đó, các cạnh bị trùng lặp sẽ nằm kề sát nhau trong mảng.
2. **Khử trùng (Dedup):** Dùng một vòng lặp quét qua mảng vừa sắp xếp. Máy tính duy trì một con trỏ `unique`. Nếu cạnh hiện tại khác cạnh liền trước nó, nó mới được giữ lại. Nếu giống hệt, nó bị ghi đè/bỏ qua. 
Sau bước này, số lượng cạnh thực tế sẽ bị "hao hụt" một chút so với số $E$ ban đầu do sự trùng lặp đã bị xóa.

---

## 5. Bước 4: Xây dựng cấu trúc CSR (Compressed Sparse Row)
Lúc này, chúng ta có một danh sách các liên kết 2 chiều đã sạch sẽ. Việc cuối cùng là nhồi nó vào định dạng CSR để tối ưu tốc độ cho BFS. Quá trình này trải qua 3 giai đoạn:

1. **Đếm Bậc (Degree Counting):**
   - Cấp phát mảng `row_ptr` có kích thước $N + 1$ toàn số 0.
   - Quét qua mảng cạnh. Với mỗi cạnh bắt đầu từ $u$, tăng `row_ptr[u + 1]` lên 1. 
   - Sau bước này, `row_ptr` chứa **bậc** của từng đỉnh.

2. **Cộng dồn Tiền tố (Prefix Sum):**
   - Máy tính chạy vòng lặp cộng dồn: `row_ptr[v+1] = row_ptr[v+1] + row_ptr[v]`.
   - Kết quả: `row_ptr` biến thành mảng chỉ mục. `row_ptr[v]` bây giờ chứa **vị trí bắt đầu** của hàng xóm đỉnh `v` trong mảng dữ liệu.

3. **Điền mảng hàng xóm (Fill `adj`):**
   - Cấp phát mảng `adj` bằng đúng tổng số cạnh thực tế còn lại.
   - Máy tính tạo một mảng tạm `tmp_pos` copy y hệt `row_ptr`.
   - Duyệt qua mảng cạnh một lần cuối. Khi thấy cạnh $(u, v)$, hệ thống làm 1 thao tác:
     `adj[tmp_pos[u]++] = v;`
     (Điền `v` vào danh sách hàng xóm của `u`, đồng thời tịnh tiến con trỏ của `u` lên 1 ô để dành cho hàng xóm tiếp theo).

---

## 6. Kết thúc
Bộ nhớ mảng tạm `edges` và `tmp_pos` được giải phóng. Hàm trả về cấu trúc `Graph` hoàn chỉnh dưới dạng CSR, sẵn sàng để thuật toán Hybrid BFS bắt đầu quá trình tìm kiếm.
