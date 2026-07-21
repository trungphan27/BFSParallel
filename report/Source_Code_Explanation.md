# Giải thích Chi tiết Mã Nguồn Project BFS Hybrid (MPI + OpenMP)

Tài liệu này giải thích chi tiết toàn bộ các file mã nguồn (C code) trong thư mục `src/` của dự án BFSParallel. Dự án cài đặt thuật toán BFS phân tán sử dụng mô hình lập trình lai (MPI cho đa máy, OpenMP cho đa nhân) kết hợp tối ưu hóa hướng (Direction-Optimizing theo Beamer et al. 2012) trên đồ thị R-MAT quy mô lớn.

## 1. Tổng quan cấu trúc thư mục `src/`

Thư mục `src/` gồm 10 file, chia làm 5 nhóm chính:
1. **Cấu trúc Đồ thị (Graph & R-MAT):** `graph.h`, `graph.c`
2. **Thuật toán BFS Tuần tự (Baseline):** `bfs_sequential.h`, `bfs_sequential.c`
3. **Thuật toán BFS Hybrid (MPI + OpenMP):** `bfs_hybrid.h`, `bfs_hybrid.c` (Đây là phần lõi, quan trọng nhất)
4. **Công cụ hỗ trợ (Utils & Metrics):** `utils.h`, `utils.c`
5. **Chương trình chính (Entry Points):** `main_seq.c`, `main_hybrid.c`

Dưới đây là giải thích luồng thực thi và logic của từng file từ lúc bắt đầu đến lúc kết thúc.

---

## 2. Cấu trúc Đồ thị & Sinh đồ thị R-MAT (`graph.h` & `graph.c`)

### 2.1. Cấu trúc dữ liệu CSR (`graph.h`)
Đồ thị vô hướng được lưu trữ dưới dạng **CSR (Compressed Sparse Row)**. CSR rất tối ưu cho BFS vì nó lưu bộ nhớ liên tục (cache-friendly) và tra cứu lân cận chỉ với O(1).
- `row_ptr`: Mảng lưu "con trỏ" (index offset). Hàng xóm của đỉnh `v` nằm trong khoảng từ `row_ptr[v]` đến `row_ptr[v+1] - 1`.
- `adj`: Mảng chứa liên tiếp danh sách hàng xóm của tất cả các đỉnh.

### 2.2. Sinh đồ thị R-MAT (`graph.c`)
Thuật toán R-MAT chia ma trận kề làm 4 góc với tỷ lệ xác suất $a, b, c, d$ (Mặc định Graph500: 0.57, 0.19, 0.19, 0.05). Do $a$ rất lớn, đồ thị sẽ có một số đỉnh làm "hub" (có cực kỳ nhiều kết nối - power-law), mô phỏng mạng xã hội thực tế.

**Hàm `graph_rmat_generate`:**
- Tính toán số đỉnh $N$ (làm tròn thành lũy thừa 2) và tổng số cạnh mục tiêu $E$.
- Sử dụng hàm tạo số ngẫu nhiên `xorshift64` để đảm bảo sinh số siêu nhanh.
- **Vòng lặp sinh cạnh:** Với mỗi cạnh cần tạo, gọi `rmat_edge` để ném cạnh đó ngẫu nhiên vào 1 trong 4 góc của ma trận kề đệ quy $\log_2(N)$ lần.
- **Loại bỏ trùng lặp (Dedup):** Lưu tất cả cạnh vào một mảng tạm thời `edges`, sau đó gọi `qsort` để sắp xếp và dùng một vòng lặp loại bỏ các cạnh trùng (self-loop đã được loại từ trước).
- **Xây dựng CSR:** 
  1. Đếm bậc (degree) của từng đỉnh.
  2. Tính tổng tiền tố (prefix sum) để tạo mảng `row_ptr`.
  3. Duyệt lại mảng cạnh đã dedup một lần nữa, dựa vào mảng offset tạm để tống thẳng các đỉnh kề vào mảng `adj`.

---

## 3. Thuật toán BFS Tuần tự (`bfs_sequential.h` & `bfs_sequential.c`)

File này cung cấp phiên bản BFS chạy trên 1 luồng duy nhất (tuần tự) bằng hàng đợi (queue) cổ điển. Dùng để (1) đo thời gian chạy baseline tính Speedup và (2) kiểm tra độ chính xác (Verify) của bản song song.

**Hàm `bfs_sequential`:**
- Khởi tạo mảng khoảng cách `dist` toàn `-1` (chưa thăm). `dist[source] = 0`.
- Khởi tạo hàng đợi `queue` kiểu mảng tĩnh và 2 con trỏ `head`, `tail`.
- Lan truyền **Top-Down** cổ điển:
  - Lấy đỉnh `u` ra khỏi hàng đợi (`head++`).
  - Duyệt tất cả các hàng xóm `v` của `u` bằng `row_ptr` và `adj` (CSR).
  - Nếu `dist[v] == -1`, cập nhật `dist[v] = dist[u] + 1` và đẩy `v` vào hàng đợi (`tail++`).
- Kết thúc, trả về số cạnh đã duyệt để tính chỉ số MTEPS.

---

## 4. Thuật toán BFS Hybrid Lõi (`bfs_hybrid.h` & `bfs_hybrid.c`)

Đây là **linh hồn** của project, kết hợp MPI (đa máy) và OpenMP (đa luồng) cùng tối ưu hướng Direction-Optimizing BFS.

### 4.1. Thiết kế dữ liệu trong `bfs_hybrid.c`
**Cấu trúc Frontier (Tập biên):**
Để duyệt Top-Down và Bottom-Up hiệu quả, biên $F$ được lưu dưới dạng kép (Dual Representation) trong `struct Frontier`:
- `bitmap`: Mảng bit. Đỉnh `v` có bit thứ $v$ bằng $1$ nếu nằm trong frontier. Hỗ trợ tra cứu O(1) cực nhanh (cho Bottom-Up).
- `dense`: Mảng chứa thẳng ID các đỉnh trong frontier (cho Top-Down quét qua các đỉnh cần xử lý).
- **Hàm `frontier_add_atomic`:** Sử dụng `#pragma omp atomic` để nhiều luồng OpenMP có thể OR bit vào `bitmap` cùng lúc mà không bị ghi đè dữ liệu.

**Phân rã dữ liệu (1D Block Partitioning):**
Sử dụng các hàm `partition_start` và `partition_end` trong `.h`. Chia $V$ đỉnh thành $P$ phần bằng nhau. Máy (Rank) thứ $r$ sẽ quản lý đỉnh từ $r \times \frac{V}{P}$ đến $(r+1) \times \frac{V}{P} - 1$.

### 4.2. Quá trình tính toán (`top_down_step` & `bottom_up_step`)
- **`top_down_step`:**
  - Lấy tập biên `curr` đang có, chia đều số lượng đỉnh trong tập biên này cho số tiến trình MPI. Mỗi tiến trình MPI dùng OpenMP chia công việc cho các thread nội bộ (`#pragma omp for schedule(dynamic, 64)` để cân bằng tải động do bậc của các đỉnh R-MAT không đều).
  - Duyệt các hàng xóm `v`. Nếu chưa thăm, dùng lệnh **Atomic Compare-And-Swap (CAS)** `__sync_val_compare_and_swap` để tranh quyền ghi. Thread nào ghi mức (level) vào `dist[v]` thành công đầu tiên sẽ được đưa `v` vào biên mới (`next`). CAS ở đây bảo vệ độ dài đường đi ngắn nhất khỏi Data Race.
- **`bottom_up_step`:**
  - Rank hiện tại chỉ quét các đỉnh trong vùng nhớ tĩnh tĩnh của nó: `[local_start, local_end)`. Dùng OpenMP dynamic để chia đỉnh cho các thread.
  - Với mỗi đỉnh `u` CHƯA THĂM trong vùng này, máy tính sẽ duyệt các lân cận `v`. Nếu phát hiện `v` đang nằm trong `curr` frontier (kiểm tra nhanh bằng hàm `frontier_has` trên `bitmap`), máy lập tức gán `dist[u] = level`, đẩy `u` vào `next` biên, và **`break` ngay lập tức**. Việc ngắt (break) sớm này tiết kiệm vô số vòng lặp so với việc kiểm tra toàn bộ hàng xóm.

### 4.3. Thuật toán lặp Level-Synchronous (`bfs_hybrid`)
Đây là thân hàm chính kết nối tất cả lại:
1. **Quyết định hướng (Heuristic Beamer):**
   - Đầu mỗi level, thuật toán đo tổng số cạnh của tập biên hiện tại (`fe_local`), sau đó gộp toàn cục bằng `MPI_Allreduce(SUM)`.
   - Nếu đang ở hướng Top-Down mà thấy tập biên quá lớn (`frontier_edges > unvisited_edges / ALPHA`), chuyển hướng sang **Bottom-Up**.
   - Nếu đang ở hướng Bottom-Up mà tập biên đã xẹp lại (`curr->size >= N / BETA`), quay về **Top-Down**.
2. **Thực thi:** Gọi `top_down_step` hoặc `bottom_up_step` như đã chọn.
3. **Đồng bộ Toàn cục (Cực kỳ quan trọng):**
   - Sau tính toán, bộ nhớ `dist[]` của máy tính sẽ có chỗ cập nhật, có chỗ không.
   - Gọi **`MPI_Allreduce(..., MPI_MAX, MPI_COMM_WORLD)`** trên mảng `dist[]`. Vì đỉnh chưa thăm là `-1`, và đỉnh đã thăm thì là số nguyên dương $\ge 0$, hàm MAX sẽ luôn lấy được giá trị đã thăm đúng đắn nhất đè lên `-1`. Từ đó mảng khoảng cách của tất cả các máy tính trên mạng LAN sẽ giống hệt nhau trở lại.
   - Gọi **`MPI_Allreduce(..., MPI_BOR, MPI_COMM_WORLD)`** trên mảng `bitmap`. Hàm OR sẽ chập tất cả các bit 1 ở các tiến trình lại thành 1 mảng tập biên hợp nhất toàn cục. Hàm `frontier_build_dense` sau đó sẽ tự động rebuild lại mảng `dense` từ cái `bitmap` đã gộp này.

---

## 5. Công cụ tiện ích (`utils.h` & `utils.c`)

Chứa các hàm phụ trợ phục vụ báo cáo và kiểm thử:
- `timer_start` & `timer_elapsed_ms`: Đo thời gian có độ chính xác cao bằng `clock_gettime(CLOCK_MONOTONIC)`.
- `compute_mteps`: Đánh giá hiệu suất duyệt đồ thị. Công thức: $MTEPS = \frac{\text{Tổng số cạnh đã duyệt}}{\text{Thời gian (giây)} \times 10^6}$. MTEPS càng cao hệ thống càng mạnh.
- `verify_bfs`: Chạy vòng lặp so sánh từng phần tử `dist[v]` của bản tuần tự (chuẩn) và bản Hybrid phân tán. Nếu mảng khớp nhau 100%, trả về 0 (PASSED). Nếu có lỗi do Data Race hay logic, hàm này sẽ báo Failed và in ra ID đỉnh lỗi.
- `pick_source`: Đảm bảo đỉnh nguồn luôn hợp lệ (có liên kết bậc > 0).

---

## 6. Entry Points (`main_seq.c` & `main_hybrid.c`)

**`main_seq.c`:**
- Hàm `main()` đơn giản, sinh graph, gọi `bfs_sequential` rồi tính và in MTEPS, thời gian chạy. 

**`main_hybrid.c`:**
- Đây là file sẽ được build thành binary `bfs_hybrid` và phân phát trên Cluster.
- **Khởi tạo MPI:** Gọi `MPI_Init_thread` với cờ `MPI_THREAD_FUNNELED` (báo cho MPI biết ứng dụng có dùng đa luồng, nhưng chỉ luồng chính mới được gọi các hàm MPI).
- **In môi trường:** Rank 0 in ra thông số hệ thống để tiện debug (Tổng tiến trình MPI, tổng số luồng OpenMP).
- **Sinh đồ thị (Replicated Graph):** Tất cả các Rank đều ĐỒNG LOẠT gọi `graph_rmat_generate` với cùng chung một Seed. Điều này đảm bảo mỗi máy trong Cluster đều ôm một bản sao giống hệt nhau của đồ thị R-MAT lên RAM cục bộ. (Thiết kế này nhằm tránh overhead truyền đồ thị siêu lớn qua cáp mạng, phù hợp cho cluster cỡ nhỏ).
- **Chạy giải thuật:** Gọi `bfs_hybrid`, sau đó dùng barrier để gom kết quả, đo tổng thời gian chạy (`result.time_ms`).
- **Xác minh (Verification):** Nếu có chạy thêm cờ `--verify`, Master (Rank 0) sẽ tự chạy thêm một bản tuần tự `bfs_sequential` nội bộ của nó, tính speedup thực tế, rồi so sánh 2 mảng `dist` để xem MPI+OpenMP kết hợp có ra kết quả đúng 100% hay không. Mọi object sau đó được dọn dẹp và kết thúc bằng `MPI_Finalize()`.
