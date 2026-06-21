# Guideline viết báo cáo — BFS Hybrid (MPI + OpenMP, Direction-Optimizing)

> File này KHÔNG phải báo cáo hoàn chỉnh. Đây là khung chi tiết + ghi chú nội dung
> cho từng phần, dựa trên việc đọc trực tiếp source code của project (`bfs_hybrid.c`,
> `bfs_hybrid.h`, `graph.c`, `main_hybrid.c`, `utils.c`...). Người viết lý thuyết
> (bạn của bạn) chỉ cần bám theo khung này, điền lý thuyết + số liệu thực nghiệm vào.
>
> Độ dài mục tiêu: 10–20 trang (tối đa 20). Phân bổ gợi ý ở cuối file.

---

## 0. Trước khi viết — 2 tài liệu tham khảo cốt lõi cần trích dẫn đúng chỗ

Code dựa trên 2 bài báo, cần trích dẫn (citation) đúng vị trí trong báo cáo,
KHÔNG cần mục "Related Work" riêng — chỉ cần trích khi giải thích thuật toán
tương ứng:

1. **Beamer, S., Asanović, K., & Patterson, D. (2012). "Direction-Optimizing
   Breadth-First Search."** *SC '12: Proceedings of the International Conference
   on High Performance Computing, Networking, Storage and Analysis.*
   → Trích dẫn khi giải thích cơ chế **top-down/bottom-up switching** (mục 2.3,
   3.b, pseudo-code).

2. Bài báo về **Parallel BFS on Distributed Memory Systems** mà bạn dùng làm
   nền tảng (ghi rõ tên tác giả + năm bạn đang dùng — ví dụ phổ biến là Yoo et al.
   "A Scalable Distributed Parallel Breadth-First Search Algorithm on BlueGene/L",
   hoặc Buluç & Madduri "Parallel Breadth-First Search on Distributed Memory
   Systems"). → Trích dẫn khi giải thích **1D vertex partitioning** và **mô hình
   giao tiếp bằng Allreduce theo level** (mục 3.a, 3.b).

   ⚠️ Bạn cần xác nhận lại chính xác bài báo thứ 2 bạn dùng (tên, tác giả, năm)
   để trích dẫn đúng — guideline này dùng tên tạm để minh hoạ vị trí trích dẫn.

**Lưu ý quan trọng khi đối chiếu với bài báo gốc:** Buluç & Madduri trong bài báo
gốc dùng **2D partitioning** của ma trận kề (`√p × √p` block) để giảm chi phí
giao tiếp khi graph được phân tán thật sự. Project của bạn **không dùng 2D
partition** — lý do là vì đồ thị được **replicate toàn bộ** trên mọi rank (xem
comment trong `bfs_hybrid.c` dòng 13: "Graph được replicate đầy đủ trên mọi
rank... Tránh MPI scatter lớn, phù hợp với cluster nhỏ"). Đây là một **lựa chọn
thiết kế có chủ đích, đơn giản hoá so với bài báo gốc**, không phải thiếu sót.
Báo cáo nên nêu rõ điểm khác biệt này như một phần của "thiết kế" — vừa thể hiện
hiểu bài báo gốc, vừa giải thích tại sao nhóm chọn cách khác (phù hợp quy mô
cluster nhỏ 3-4 máy thay vì hệ thống lớn hàng nghìn node).

---

## 1. Trang bìa + Mục lục (không tính vào số trang nội dung)

- Tên đề tài: "Song song hóa thuật toán Breadth-First Search (BFS) trên hệ
  thống bộ nhớ phân tán sử dụng MPI + OpenMP"
- Môn học, giảng viên, nhóm/thành viên, ngày nộp.
- Mục lục tự động (heading 1/2/3).

---

## 2. Giới thiệu (Introduction) — ~1 trang

Nội dung cần có:

- **Bài toán BFS là gì**, tại sao cần song song hóa (BFS là building block của
  rất nhiều ứng dụng: social network analysis, web crawling, shortest path
  trên đồ thị không trọng số, benchmark Graph500...).
- **Thách thức cố hữu của BFS song song**: BFS có **tỷ lệ tính toán/giao tiếp
  thấp** (low arithmetic intensity) — mỗi cạnh chỉ cần vài phép so sánh, nên
  overhead đồng bộ hóa giữa các tiến trình dễ áp đảo phần tính toán thật. Đây là
  lý do BFS được xem là bài toán "khó song song hóa hiệu quả" hơn nhiều so với
  các bài toán dense linear algebra.
- **Mục tiêu của project**: cài đặt BFS song song hybrid (MPI cho phân tán
  liên-máy + OpenMP cho song song trong-máy) kết hợp kỹ thuật
  **direction-optimizing** (Beamer et al., 2012) để giảm số cạnh phải duyệt.
- Loại đồ thị thử nghiệm: đồ thị tổng hợp theo mô hình **R-MAT** (Recursive
  MATrix), chuẩn dùng trong benchmark Graph500, có phân phối bậc kiểu
  power-law (giống đồ thị thực tế: mạng xã hội, web graph...) — khác hẳn so với
  đồ thị ngẫu nhiên đều (Erdős–Rényi), điều này ảnh hưởng trực tiếp đến tải
  giữa các tiến trình (nêu sơ ở đây, phân tích sâu ở phần Load Balancing).
- Cấu trúc các phần còn lại của báo cáo (roadmap 1 đoạn).

---

## 3. Cơ sở lý thuyết (Background) — ~2-3 trang

### 3.1. Biểu diễn đồ thị: CSR (Compressed Sparse Row)

- Giải thích cấu trúc `row_ptr[]` + `adj[]`: đỉnh `v` có hàng xóm nằm trong
  `adj[row_ptr[v] .. row_ptr[v+1]-1]`.
- Ưu điểm CSR cho BFS song song: truy cập hàng xóm O(1) offset, bộ nhớ liên
  tục (cache-friendly), dễ chia theo dải đỉnh liên tục cho 1D partitioning.
- Có thể vẽ 1 hình minh họa CSR đơn giản (ma trận kề thưa → row_ptr/adj).

### 3.2. Sinh đồ thị thử nghiệm: mô hình R-MAT

- Giải thích thuật toán R-MAT: chia đệ quy ma trận kề `n × n` thành 4 góc với
  xác suất `a, b, c, d` (`a+b+c+d=1`), lặp `log2(n)` lần để xác định 1 cạnh.
- Tham số mặc định dùng trong code (chuẩn Graph500): `a=0.57, b=0.19, c=0.19,
  d=0.05`.
- **Hệ quả quan trọng cho phần Load Balancing**: vì `a` (góc trên-trái, ứng với
  đỉnh chỉ số nhỏ) lớn hơn hẳn `d` (góc dưới-phải, đỉnh chỉ số lớn), đồ thị
  R-MAT có **phân phối bậc lệch mạnh theo chỉ số đỉnh** — các đỉnh có chỉ số
  nhỏ có xu hướng bậc cao hơn (hub), đỉnh chỉ số lớn bậc thấp hơn. Đây là điểm
  mấu chốt giải thích tại sao 1D partition đều theo SỐ ĐỈNH (không phải đều
  theo SỐ CẠNH) gây mất cân bằng tải — sẽ dùng lại ý này ở mục Load Balancing
  và mục Results.

### 3.3. BFS tuần tự (baseline)

- Thuật toán BFS cổ điển dùng queue (FIFO), độ phức tạp O(V+E).
- Đây chính là baseline dùng để (a) verify tính đúng đắn, (b) tính speedup.

### 3.4. Direction-Optimizing BFS (Beamer et al., 2012)

**[Trích dẫn bài báo Beamer ở đây]**

- Giải thích vấn đề của BFS top-down thuần túy: ở các level giữa của BFS trên
  đồ thị "small-world" (đường kính nhỏ, bậc trung bình cao — đặc trưng của
  R-MAT/mạng xã hội), frontier có thể chứa hàng trăm nghìn đỉnh. Duyệt top-down
  nghĩa là với MỌI đỉnh trong frontier, xét MỌI hàng xóm — kể cả hàng xóm ĐÃ
  được thăm rồi → lãng phí.
- Giải pháp: **bottom-up** — với mỗi đỉnh CHƯA thăm, chỉ cần tìm 1 hàng xóm
  đang ở frontier hiện tại là đủ (không cần xét hết hàng xóm, dừng ngay khi
  tìm thấy 1 cái) → giảm số cạnh phải kiểm tra khi frontier lớn.
- Tiêu chí chuyển hướng (heuristic, dùng đúng các hằng số trong code):
  - Top-down → Bottom-up khi: `frontier_edges > unvisited_edges / ALPHA`
    (với `ALPHA = 14`)
  - Bottom-up → Top-down khi: `frontier_size < num_vertices / BETA`
    (với `BETA = 24`)
- Nêu rõ đây chính là 2 hằng số `BFS_ALPHA`/`BFS_BETA` định nghĩa trong
  `bfs_hybrid.h` — bám sát đúng giá trị Beamer đề xuất trong bài báo gốc (có
  thể nêu Beamer cũng đề xuất giá trị tương tự, tùy bạn đối chiếu lại bài báo).
- Vẽ 1 hình/bảng minh hoạ ví dụ output thực tế của chương trình (log mẫu trong
  README mục 8 đã có sẵn — ví dụ Level 1-2 top-down, Level 3-4 bottom-up,
  Level 5-6 quay lại top-down) để minh hoạ trực quan việc chuyển hướng.

### 3.5. Mô hình lập trình song song lai (Hybrid MPI + OpenMP)

- Giải thích ngắn gọn: MPI cho phân tán bộ nhớ giữa các máy (distributed
  memory), OpenMP cho chia sẻ bộ nhớ trong 1 máy (shared memory) — kết hợp 2
  tầng để tận dụng cả cluster nhiều máy lẫn nhiều core/máy.
- Đây là phần lý thuyết nền cho mục 4 (Cách thức song song hóa).

---

## 4. Thiết kế song song (Parallel Design) — đây là phần TRỌNG TÂM, ~5-7 trang

> Đây là phần trả lời trực tiếp các câu hỏi đề bài yêu cầu. Cấu trúc gợi ý theo
> đúng thứ tự câu hỏi.

### 4.1. Mức độ song song: Data Parallelism (Song song dữ liệu)

Nội dung cần trình bày:

- Khẳng định: project sử dụng **song song dữ liệu (data parallelism)**, theo
  mô hình **SPMD (Single Program, Multiple Data)** — mọi tiến trình MPI và mọi
  thread OpenMP chạy **cùng một đoạn mã nguồn**, chỉ khác nhau ở **phần dữ liệu
  (tập đỉnh) mà chúng xử lý**.
- Phân biệt rõ với task parallelism: không có khái niệm "tác vụ A khác tác vụ
  B" chạy song song (như pipeline hay master phân công việc khác loại) — toàn
  bộ rank/thread đều gọi cùng hàm `top_down_step()` / `bottom_up_step()`.
- Song song xảy ra ở **2 tầng lồng nhau**:
  1. Tầng liên-tiến-trình (MPI, giữa các máy/process): chia tập đỉnh.
  2. Tầng trong-tiến-trình (OpenMP, giữa các thread cùng máy): chia tiếp phần
     dữ liệu (frontier hoặc dải đỉnh local) cho các thread.
- Lưu ý đặc thù: **đồ thị (CSR) được replicate (sao chép) toàn bộ trên mọi
  rank**, không bị chia nhỏ. Vì vậy điểm "dữ liệu được chia" không phải là cấu
  trúc đồ thị, mà là **tập đỉnh cần xử lý ở mỗi bước BFS** (xem code
  `bfs_hybrid.c`, biến `local_start`/`local_end`, và việc chia `curr->dense[]`
  theo `rank_s`/`rank_e` trong `top_down_step`).

### 4.2. Kỹ thuật phân rã: Data Decomposition (Domain/Vertex Decomposition)

- Khẳng định: kỹ thuật phân rã dùng là **data decomposition** — phân chia tập
  dữ liệu lớn nhất (tập đỉnh V) thành các phần độc lập, áp dụng cùng một phép
  toán lên từng phần.
- Giải thích vì sao KHÔNG phải các loại khác (mỗi gạch đầu dòng nên có 1-2 câu
  giải thích "tại sao không"):
  - *Không phải exploratory decomposition*: BFS không có khái niệm "nhánh tìm
    kiếm có thể bỏ qua" — mọi đỉnh trong frontier đều bắt buộc phải xử lý.
  - *Không phải recursive decomposition*: BFS xử lý tuần tự theo level
    (level-synchronous), level sau phụ thuộc hoàn toàn kết quả level trước,
    không chia được thành các bài toán con độc lập kiểu divide-and-conquer.
  - *Không phải speculative decomposition*: không có việc đoán trước nhánh nào
    sẽ xảy ra rồi tính song song nhiều khả năng.
- Cụ thể hoá: đây là phân rã theo **owner-computes rule** ở tầng MPI — rank sở
  hữu dải đỉnh `[local_start, local_end)` chịu trách nhiệm **tính** `dist[]`
  cho các đỉnh đó trong bước bottom-up (xem `bottom_up_step`, tham số
  `local_start, local_end`).
- Ở tầng OpenMP, phân rã dữ liệu xảy ra lần 2 (lồng trong lần 1): chia tiếp
  phần tử frontier / dải đỉnh local cho từng thread.
- Kết luận mục này: đây là **data decomposition áp dụng 2 tầng** (không phải
  hybrid giữa các *loại* phân rã khác nhau).

### 4.3. Mapping Technique — Phân bổ tiến trình/bộ xử lý

**Đây là phần đề bài hỏi rõ "1D hay 2D n/√p × n/√p" — trả lời dứt khoát: 1D.**

- Trình bày công thức 1D block partition theo đỉnh, trích nguyên văn từ
  `bfs_hybrid.h`:
  ```c
  static inline vertex_t partition_start(vertex_t n, int num_ranks, int rank)
  { return (vertex_t)((long long)n * rank / num_ranks); }
  static inline vertex_t partition_end(vertex_t n, int num_ranks, int rank)
  { return (vertex_t)((long long)n * (rank + 1) / num_ranks); }
  ```
- Rank `r` trong tổng `P` rank sở hữu dải đỉnh liên tục:
  `[⌊n·r/P⌋, ⌊n·(r+1)/P⌋)`.
- Vẽ sơ đồ minh hoạ (đơn giản, dạng thanh ngang chia đoạn):
  ```
  |---- Rank 0 ----|---- Rank 1 ----|---- Rank 2 ----|---- Rank 3 ----|
  0              n/4              2n/4             3n/4              n
  ```
- **Giải thích tại sao KHÔNG dùng 2D `n/√p × n/√p`**: 2D block partitioning
  (chia ma trận kề theo cả hàng lẫn cột) chỉ có ý nghĩa khi **bản thân ma trận
  kề/CSR được phân tán thật sự** giữa các rank (mỗi rank chỉ giữ 1 block con
  của ma trận kề — đây là cách tiếp cận trong bài báo gốc về Parallel BFS on
  Distributed Memory khi target hệ thống lớn). Trong project này, đồ thị được
  **replicate toàn bộ** trên mọi rank để đơn giản hoá triển khai trên cluster
  nhỏ (3-4 máy) — do đó 2D partition không cần thiết và không được áp dụng;
  thay vào đó chỉ cần 1D partition tập đỉnh để **phân công ai tính `dist[]`
  cho đỉnh nào**, không phải để giảm dung lượng lưu trữ đồ thị trên từng rank.
- Mapping tầng 2 (OpenMP, **dynamic/runtime mapping**, không phải static):
  - Top-down: `#pragma omp for schedule(dynamic, 64) nowait` — chia phần tử
    frontier theo chunk 64, gán động cho thread rảnh.
  - Bottom-up: `#pragma omp parallel for schedule(dynamic, 256)` — chia dải
    đỉnh local theo chunk 256.
  - Giải thích lý do dùng `dynamic` thay vì `static`: vì bậc đỉnh phân phối
    không đều (R-MAT, power-law) → khối lượng công việc trên mỗi đỉnh/chunk
    không đều → cần cân bằng tải động ở tầng thread (liên kết với mục 4.5).

### 4.4. Communication Strategy and Topology — Chiến lược & cấu trúc giao tiếp

**Trả lời rõ từng câu hỏi con của đề bài:**

| Câu hỏi | Trả lời (theo code) |
|---|---|
| Blocking hay non-blocking? | **Blocking hoàn toàn.** Toàn bộ giao tiếp dùng `MPI_Allreduce` và `MPI_Barrier` — đều là hàm blocking. Không có `MPI_Isend`/`MPI_Irecv` hay bất kỳ giao tiếp non-blocking nào trong code. |
| Master-slave hay không? | **Không phải master-slave.** Rank 0 chỉ "đặc biệt" ở việc in log tiến trình và giữ kết quả `dist[]` cuối cùng để xuất ra — nhưng **mọi rank đều tính toán ngang hàng** (đều chạy `top_down_step`/`bottom_up_step`/tham gia `Allreduce`). Đây là mô hình **SPMD ngang hàng (peer collective)**, rank 0 không phân công việc khác loại cho rank khác. |
| Topology: tree / ring / hypercube? | Code **không tự cài đặt** topology tường minh (không dùng `MPI_Cart_create` hay tự code ring/tree). Toàn bộ giao tiếp đi qua `MPI_COMM_WORLD` bằng `MPI_Allreduce` — topology vật lý/logic của quá trình reduce (binomial tree, recursive doubling, ring-allreduce...) là do **runtime OpenMPI tự chọn nội bộ** dựa theo kích thước message và số rank, không phải do nhóm tự thiết kế. Báo cáo nên nêu rõ: *"Topology giao tiếp ở mức logic ứng dụng là all-to-all/flat (mọi rank tham gia bình đẳng vào mỗi `Allreduce`); topology vật lý của thuật toán reduction bên dưới do thư viện MPI runtime quyết định."* Có thể tham khảo thêm tài liệu OpenMPI để nêu cụ thể thuật toán Allreduce nào được chọn (tuỳ ngưỡng kích thước message, OpenMPI thường dùng recursive-doubling cho message nhỏ, hoặc Rabenseifner's algorithm cho message lớn) — phần này có thể ghi là "tìm hiểu thêm" nếu muốn đào sâu, không bắt buộc. |
| Các hàm MPI cụ thể được dùng | `MPI_Init_thread` (khởi tạo, `MPI_THREAD_FUNNELED`), `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Barrier`, `MPI_Allreduce` (3 biến thể: `MPI_MAX` merge `dist[]`, `MPI_BOR` merge frontier bitmap, `MPI_SUM` tính tổng `frontier_edges`), `MPI_Abort`, `MPI_Finalize`. |
| Tần suất đồng bộ | **Đồng bộ mỗi level BFS** (level-synchronous BFS) — sau mỗi bước top-down/bottom-up, có 2 lần `Allreduce` bắt buộc (đồng bộ `dist[]` và đồng bộ `frontier` bitmap), cộng thêm 1 lần `Allreduce` đầu mỗi level để tính tổng `frontier_edges` quyết định hướng duyệt → tổng **3 lần Allreduce/level**. |
| Khối lượng dữ liệu mỗi lần đồng bộ | `Allreduce(dist, MPI_MAX)` luôn truyền **toàn bộ mảng `n` số nguyên** (kích thước O(n)), bất kể frontier hiện tại lớn hay nhỏ. Đây là điểm **không scale tốt** theo lý thuyết — chi phí giao tiếp không giảm dù ở các level frontier rất nhỏ (ví dụ level đầu/cuối BFS, frontier chỉ vài chục đỉnh nhưng vẫn phải Allreduce toàn bộ `n`). Đây là nội dung quan trọng cần phân tích định lượng ở phần Results (so sánh thời gian có/không communication). |

- Vẽ sơ đồ giao tiếp (gợi ý dạng vòng lặp theo level):
  ```
  ┌─────────────────────────────────────────────────────┐
  │  Mỗi LEVEL của BFS:                                  │
  │                                                       │
  │  1. [Tính fe_local song song OpenMP, mỗi rank riêng] │
  │  2. MPI_Allreduce(SUM)  → frontier_edges toàn cục    │
  │  3. [Top-down HOẶC Bottom-up step, song song OpenMP] │
  │  4. MPI_Allreduce(MAX)  → đồng bộ dist[] toàn cục    │
  │  5. MPI_Allreduce(BOR)  → đồng bộ frontier bitmap    │
  │  6. swap(curr, next) ; level++                       │
  └─────────────────────────────────────────────────────┘
  ```

### 4.5. Load Balancing — Cân bằng tải

**Trả lời rõ: CÓ áp dụng, nhưng chỉ ở 1 trong 2 tầng.**

- **Tầng OpenMP (trong từng rank): CÓ cân bằng tải động.**
  - `schedule(dynamic, 64)` (top-down) và `schedule(dynamic, 256)` (bottom-up)
    — đây là cơ chế **runtime load balancing**: thread nào xử lý xong chunk
    của mình sẽ tự động lấy chunk tiếp theo, không cần biết trước workload.
  - Lý do cần thiết: đồ thị R-MAT có phân phối bậc power-law → một số đỉnh
    (hub) có hàng nghìn cạnh trong khi phần lớn đỉnh khác chỉ có vài cạnh →
    nếu chia tĩnh đều số lượng đỉnh/thread (`static`) sẽ gây mất cân bằng
    nghiêm trọng giữa các thread.

- **Tầng MPI (giữa các rank): KHÔNG có cân bằng tải động — chỉ static
  partition đều theo SỐ LƯỢNG ĐỈNH.**
  - Đây là điểm yếu cố hữu của thiết kế hiện tại: `partition_start/end` chia
    đều `n/P` đỉnh cho mỗi rank, **không quan tâm đến tổng số cạnh (tổng bậc)**
    của dải đỉnh đó.
  - Liên hệ với mục 3.2 (R-MAT có `a=0.57` lớn hơn hẳn `d=0.05`): các đỉnh có
    chỉ số nhỏ (do cách sinh R-MAT) có xu hướng bậc cao hơn → **dải đỉnh đầu**
    (rank giữ `[0, n/P)`) khả năng cao có **tổng số cạnh lớn hơn** dải đỉnh
    cuối, dù số lượng đỉnh bằng nhau tuyệt đối → rank giữ dải đỉnh đầu phải
    làm nhiều việc hơn ở bước bottom-up (`bottom_up_step` duyệt theo
    `[local_start, local_end)`).
  - **Đây chính là giả thuyết cần kiểm chứng bằng thực nghiệm** ở phần Results
    (mục Granularity/Load Balancing) — đo thời gian tính toán thực tế của
    từng rank để xác nhận có lệch tải hay không, và lệch bao nhiêu %.
  - Nếu kết quả thực nghiệm cho thấy lệch tải > 25% giữa 2 rank bất kỳ (theo
    yêu cầu đề bài), báo cáo nên đề xuất hướng cải tiến cụ thể, ví dụ:
    **edge-balanced partitioning** — chia dải đỉnh sao cho tổng bậc
    (`row_ptr[end] - row_ptr[start]`) xấp xỉ bằng nhau giữa các rank, thay vì
    chia đều số lượng đỉnh. Có thể trình bày sơ đồ thuật toán đề xuất (dùng
    prefix-sum của bậc đỉnh để tìm điểm chia) như một phần "hướng phát triển",
    không bắt buộc phải cài đặt nếu đề bài chỉ yêu cầu phân tích.

### 4.6. Pseudo-code thuật toán song song

> Có thể dùng nguyên bản pseudo-code dưới đây (đã được rút gọn, bám sát code
> thật trong `bfs_hybrid.c`), chỉnh format theo chuẩn báo cáo (thường dùng môi
> trường `algorithm`/`algorithmic` trong LaTeX nếu báo cáo dùng LaTeX, hoặc
> trình bày dạng code-block có đánh số dòng).

```
ALGORITHM: Hybrid Direction-Optimizing BFS (MPI + OpenMP)
INPUT:  Graph G(V,E) dạng CSR, replicated trên mọi rank; đỉnh nguồn s
OUTPUT: dist[] (đầy đủ tại rank 0; rank khác chỉ giữ phần tính toán local)

1.  my_rank, P  <- MPI_Comm_rank(), MPI_Comm_size()
2.  [local_start, local_end)  <- 1D-PARTITION(|V|, P, my_rank)   // owner-computes
3.  dist[0..n-1] <- -1 ;  dist[s] <- 0
4.  curr_frontier <- { s } ;  level <- 1
5.  unvisited_edges <- |E| ;  use_bottom_up <- FALSE
6.
7.  WHILE curr_frontier != rong DO
8.      // ----- (a) Thống kê toàn cục để quyết định hướng -----
9.      fe_local <- SUM song song (OpenMP) deg(v), forall v in curr_frontier (phần local)
10.     frontier_edges <- MPI_ALLREDUCE(fe_local, SUM)        // O(n) comm: 1 số/rank
11.
12.     // ----- (b) Heuristic Beamer chuyển hướng -----
13.     IF (NOT use_bottom_up) AND (frontier_edges > unvisited_edges / ALPHA) THEN
14.         use_bottom_up <- TRUE
15.     ELSE IF use_bottom_up AND (|curr_frontier| < |V| / BETA) THEN
16.         use_bottom_up <- FALSE
17.
18.     IF NOT use_bottom_up THEN                              // ===== TOP-DOWN =====
19.         [rank_s, rank_e) <- chia curr_frontier cho rank này (theo P)
20.         PARALLEL FOR (OpenMP, dynamic chunk=64) fi in [rank_s, rank_e) DO
21.             u <- curr_frontier[fi]
22.             FOR EACH v in Neighbors(u) DO
23.                 IF CompareAndSwap(dist[v], -1, level) = -1 THEN  // tránh race
24.                     next_frontier.ADD_ATOMIC(v)
25.     ELSE                                                    // ===== BOTTOM-UP =====
26.         PARALLEL FOR (OpenMP, dynamic chunk=256) u in [local_start, local_end) DO
27.             IF dist[u] = -1 THEN
28.                 FOR EACH v in Neighbors(u) DO
29.                     IF v in curr_frontier THEN
30.                         dist[u] <- level ; next_frontier.ADD_ATOMIC(u) ; BREAK
31.
32.     // ----- (c) Đồng bộ toàn cục cuối level (BLOCKING) -----
33.     dist[]              <- MPI_ALLREDUCE(dist[],  MPI_MAX)   // O(n) comm
34.     next_frontier.bitmap <- MPI_ALLREDUCE(bitmap, MPI_BOR)   // O(n/64) comm
35.
36.     unvisited_edges <- unvisited_edges - frontier_edges
37.     curr_frontier, next_frontier <- next_frontier, curr_frontier   // swap
38.     level <- level + 1
39.
40. RETURN dist[]   // valid đầy đủ tại rank 0
```

- Sau pseudo-code, nên thêm 1 đoạn phân tích độ phức tạp giao tiếp ngắn:
  mỗi level tốn **O(n)** dữ liệu trao đổi qua `Allreduce(dist)` bất kể kích
  thước frontier, cộng **O(n/64)** qua `Allreduce(bitmap)` — tổng chi phí
  giao tiếp toàn chương trình xấp xỉ **O(L × n)** với `L` là số level BFS
  (thường `L = O(log n)` hoặc `O(diameter)` với đồ thị small-world). Đây là cơ
  sở lý thuyết để giải thích kết quả đo thực nghiệm ở phần 5 (communication
  overhead, đặc biệt khi `P` tăng nhưng `n` cố định).

---

## 5. Thực nghiệm và Kết quả (Results) — đây là phần dài nhất, ~6-9 trang

> Phần này map trực tiếp 4 yêu cầu con của đề bài. Mỗi mục nên có: (1) mục
> tiêu đo, (2) cách đo (script/lệnh chạy), (3) bảng số liệu, (4) biểu đồ,
> (5) nhận xét/kết luận.

### 5.0. Môi trường thực nghiệm (mục con bắt buộc, viết đầu phần 5)

- Liệt kê: số máy, CPU model, số core vật lý/máy, RAM/máy, hệ điều hành,
  version OpenMPI/GCC, kết nối mạng (LAN, băng thông).
- Bảng tổng số core: ví dụ "3 máy × 4 nhân = tổng 12 tiến trình tối đa" (đúng
  ví dụ đề bài cho).
- Tham số cố định trong toàn bộ thực nghiệm (trừ khi đang test chính tham số
  đó): `scale_factor` (bậc trung bình), `seed`, compiler flags (`-O2`).
- **Lưu ý kỹ thuật quan trọng cần nêu**: code hiện tại đo `time_ms` (biến
  trong `BFSResult`) **chỉ tính thời gian chạy vòng lặp BFS**
  (`timer_start`/`timer_elapsed_ms` quanh `while (curr->size > 0)` trong
  `bfs_hybrid.c`), KHÔNG bao gồm thời gian sinh đồ thị R-MAT (`gen_ms` được đo
  riêng trong `main_hybrid.c`). Báo cáo cần làm rõ: khi đề bài nói "thời gian
  chạy của chương trình (từ lúc bắt đầu đến lúc kết thúc toàn bộ chương
  trình)", người viết cần **chủ động định nghĩa rõ mốc đo** — khuyến nghị tách
  riêng 2 thành phần (`gen_ms` và `bfs_time_ms`) và báo cáo cả hai, vì sinh đồ
  thị **chạy tuần tự, lặp lại giống hệt trên mọi rank** (không hưởng lợi gì từ
  song song hóa — đây là chi phí cố định, không phải phần thuật toán đang được
  đánh giá) sẽ làm sai lệch số liệu nếu gộp chung vào "thời gian chương trình"
  khi so sánh speedup.

### 5.1. Kiểm tra tính đúng đắn (Correctness)

- Mục tiêu: xác nhận `dist[]` do bản hybrid tính ra giống hệt bản tuần tự.
- Cách đo: dùng cờ có sẵn `--verify` (đã cài đặt sẵn trong code, hàm
  `verify_bfs()` trong `utils.c`, so từng `dist[v]`).
- Lệnh chạy mẫu (đã có sẵn trong Makefile/README):
  ```bash
  OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 --verify
  ```
- Trình bày: chạy verify với **ít nhất 3-4 cấu hình khác nhau** (số rank, số
  thread, kích thước đồ thị khác nhau) để chứng minh tính đúng đắn ổn định,
  không phải ăn may với 1 cấu hình. Chụp lại output `[HYBRID] Verify : PASSED ✓`
  cho từng lần chạy, đưa vào bảng hoặc phụ lục.
- Nên thử cả trường hợp **graph nhỏ** (dễ debug nếu sai) và **graph lớn** (gần
  với kích thước N sẽ dùng ở các thí nghiệm sau).

### 5.2. Xác định kích thước N để chương trình chạy 2-3 phút

- Mục tiêu: tìm `N` (số đỉnh) sao cho thời gian chạy nằm trong khoảng 2-3
  phút, dùng số tiến trình = tổng số nhân CPU thật (ví dụ 12 nếu 3 máy × 4
  nhân).
- Cách đo: chạy với `-np = 12` (cố định), quét nhiều giá trị N tăng dần (ví
  dụ: 500K, 1M, 2M, 4M, 8M, 16M... — tăng theo cấp số nhân để nhanh tìm khoảng
  phù hợp, sau đó tinh chỉnh quanh khoảng đó).
- **2 trục cần vẽ theo đúng yêu cầu đề bài**: trục hoành = kích thước N, trục
  tung = thời gian chạy chương trình — và yêu cầu vẽ **2 đường/2 trường hợp**:
  - (1) Có thời gian truyền thông (tổng `time_ms` thực tế, bao gồm cả
    `Allreduce`).
  - (2) Không có thời gian truyền thông (`time_ms` trừ đi tổng thời gian các
    lệnh `Allreduce` đo riêng).
  - ⚠️ Code GỐC hiện KHÔNG tách riêng được 2 thời gian này — cần bổ sung
    instrumentation (đo bằng `MPI_Wtime()` hoặc `Timer` có sẵn quanh từng lệnh
    `MPI_Allreduce` trong `bfs_hybrid.c`, cộng dồn vào biến đếm
    `comm_time_ms`). Đây là phần code cần sửa thêm trước khi chạy thí nghiệm
    này (có thể nhờ hỗ trợ riêng nếu cần).
- Biểu đồ: line chart, trục X = N (có thể dùng thang log nếu N trải rộng
  nhiều bậc), trục Y = thời gian (giây hoặc phút), 2 đường (có comm / không
  comm), đánh dấu vùng "2-3 phút" trên trục Y để dễ xác định N phù hợp.
- Kết luận mục này: chốt 1 giá trị N cụ thể (gọi là N₀) — N₀ này sẽ được dùng
  làm baseline cho mục 5.3, và `2×N₀` sẽ dùng cho mục 5.4 (đúng yêu cầu đề
  bài).

### 5.3. Kiểm tra tính mịn — Granularity & Load Balancing

- Mục tiêu: với N₀ cố định (chọn ở mục 5.2) và số tiến trình = tổng số nhân
  CPU (ví dụ 12), kiểm tra xem tải có cân bằng giữa các tiến trình không.
- **Granularity ở đây là gì** (cần định nghĩa rõ trong báo cáo, đúng yêu cầu
  đề bài: "kích thước dữ liệu trên mỗi tiến trình nếu song song hóa dựa trên
  dữ liệu"): vì đây là data parallelism, granularity = **số đỉnh (và quan
  trọng hơn, tổng số cạnh) mà mỗi rank phải xử lý** = `local_end - local_start`
  (số đỉnh) và tổng bậc của dải đỉnh đó (số cạnh thực tế).
- Cách đo: cần bổ sung code để **mỗi rank tự đo thời gian tính toán cục bộ
  (compute time) và thời gian chờ/giao tiếp (comm time) của riêng nó**, sau đó
  gom về rank 0 bằng `MPI_Gather` để xuất ra (hiện code CHƯA có — cần sửa
  thêm, có thể đo bằng cách bọc `MPI_Wtime()` quanh `top_down_step`/
  `bottom_up_step` riêng cho compute, và quanh các `MPI_Allreduce` riêng cho
  comm, cộng dồn theo từng rank suốt toàn bộ vòng lặp BFS).
- **Biểu đồ bắt buộc theo đề bài**: stacked bar chart — **1 cột = 1 tiến
  trình**, mỗi cột chia 2 phần màu khác nhau (compute time, communication
  time) xếp chồng lên nhau, trục Y = thời gian (ms hoặc s), trục X = rank ID
  (0, 1, 2, ..., 11).
- **Tiêu chí đánh giá cân bằng tải (theo đúng đề bài)**: tính "thời gian rảnh"
  của mỗi tiến trình (rảnh = chờ ở `Allreduce`/`Barrier` trong khi rank khác
  còn tính toán — tương đương phần comm time nếu đo đúng cách, hoặc đo riêng
  bằng thời gian chờ tại barrier). So sánh **lệch giữa 2 tiến trình bất kỳ**:
  nếu lệch > 25% → **không cân bằng tải**.
- **Hướng xử lý nếu mất cân bằng** (liên hệ trực tiếp mục 4.5 đã phân tích lý
  thuyết): do 1D partition hiện tại chia đều theo SỐ ĐỈNH chứ không theo SỐ
  CẠNH, và R-MAT có phân phối bậc lệch (mục 3.2) → khả năng cao sẽ thấy lệch
  tải thật giữa rank giữ đỉnh chỉ số nhỏ (nhiều cạnh hơn) và rank giữ đỉnh chỉ
  số lớn (ít cạnh hơn). Báo cáo nên:
  - Nếu lệch < 25%: kết luận hệ thống cân bằng tải tương đối tốt ở granularity
    hiện tại (số đỉnh chia đều theo `n/P`) — có thể do bottom-up scan toàn bộ
    dải đỉnh (kể cả đỉnh không có cạnh) làm tải cân bằng hơn dự kiến.
  - Nếu lệch > 25%: đề xuất chuyển sang **edge-balanced 1D partition** (chia
    theo tổng số cạnh thay vì số đỉnh, dùng prefix-sum trên `row_ptr` để tìm
    điểm cắt sao cho mỗi rank nhận ≈ `|E|/P` cạnh) — coi đây là "điều chỉnh độ
    mịn", đúng tinh thần câu hỏi đề bài ("mịn hơn hoặc thô hơn"). Nói rõ:
    chia theo cạnh = **giảm độ thô (coarse) hiện tại của granularity ở chiều
    "công việc thực", tăng độ mịn theo nghĩa cân bằng workload**, dù số lượng
    đỉnh/rank không còn đều nhau về mặt số lượng.

### 5.4. Kiểm tra độ tăng tốc (Speedup)

- Mục tiêu: với kích thước đầu vào cố định = **2×N₀** (gấp đôi N₀ chọn ở mục
  5.2, đúng yêu cầu đề bài), biến đổi số lượng tiến trình `P` = 1, 2, 4, 8,
  ..., đến 2X (X = tổng số nhân vật lý của cluster, ví dụ X=12 → chạy đến
  P=24, có thể cần `--oversubscribe` khi P > số core vật lý).
- 2 biểu đồ cần vẽ:
  1. **Thời gian chạy theo P** (2 đường: có/không communication time, tương tự
     cách làm ở mục 5.2 nhưng trục X giờ là P thay vì N).
  2. **Speedup theo P**: `Speedup(P) = Time(P=1) / Time(P)`. Vẽ kèm đường lý
     tưởng (linear speedup, `Speedup = P`) để so sánh trực quan mức độ "gần
     tuyến tính" của hệ thống thực tế.
- Phân tích kỳ vọng (để định hướng viết nhận xét, không phải số liệu thật —
  bạn cần điền số liệu thật sau khi chạy):
  - Speedup khả năng **không tuyến tính hoàn toàn**, đặc biệt khi `P` lớn, vì:
    (a) chi phí `Allreduce(dist)` là O(n) **không giảm** theo P (ngược lại,
    với P lớn, thời gian tính toán/rank giảm nhưng chi phí communication
    không giảm tương ứng → tỷ trọng comm/tổng tăng dần, đúng theo phân tích lý
    thuyết ở mục 4.6); (b) mất cân bằng tải (mục 5.3) càng rõ khi P lớn vì mỗi
    rank có ít việc hơn nhưng vẫn phải chờ rank chậm nhất; (c) khi vượt qua số
    core vật lý (oversubscribe), hiệu năng có thể **giảm** thay vì tăng.
  - Đây chính là minh hoạ thực nghiệm cho **định luật Amdahl** — nên trích dẫn
    công thức `Speedup ≤ 1/(s + (1-s)/P)` với `s` là tỷ lệ phần tuần tự không
    song song hóa được (ở đây phần "tuần tự" tương đương phần communication
    không scale + phần sinh đồ thị nếu tính gộp) làm khung lý thuyết giải
    thích độ lệch so với tuyến tính lý tưởng.
- Nên có thêm 1 bảng tổng hợp: cột P | Time(s) | Speedup | Efficiency
  (`Speedup/P`) — Efficiency giảm dần khi P tăng là hiện tượng bình thường,
  cần giải thích bằng overhead communication.

---

## 6. Thảo luận / Hạn chế (Discussion) — ~1 trang

Gợi ý nội dung (tổng hợp lại các điểm đã phân tích rải rác ở trên thành 1 mục
nhìn lại tổng thể):

- **Graph replication** (đồ thị được sao chép toàn bộ trên mọi rank) là điểm
  đánh đổi chính: đơn giản hoá cài đặt + tránh communication phức tạp khi build
  graph, nhưng **giới hạn khả năng mở rộng theo kích thước đồ thị** — bộ nhớ
  mỗi rank phải đủ chứa TOÀN BỘ đồ thị (không scale theo P), khác với cách
  tiếp cận distributed-CSR (2D partition) trong bài báo gốc Parallel BFS on
  Distributed Memory — phù hợp khi target là cluster nhỏ, đồ thị vừa phải
  (đúng phạm vi đề bài/project này), nhưng sẽ là nút thắt nếu áp dụng cho đồ
  thị thật sự lớn (tỷ đỉnh).
- **Allreduce(dist) toàn cục mỗi level** là nguồn overhead chính giới hạn
  speedup khi P lớn — như đã phân tích lý thuyết (mục 4.6) và nên đối chiếu
  lại với số liệu thực nghiệm đo được (mục 5.4) ở đây.
- **1D partition theo số đỉnh** (không theo số cạnh) là điểm có thể cải tiến
  nếu thực nghiệm 5.3 cho thấy mất cân bằng tải rõ rệt.
- Hướng phát triển tiếp theo (gợi ý, chọn lọc tuỳ thời gian còn lại): (a)
  edge-balanced partitioning, (b) non-blocking communication
  (`MPI_Iallreduce`) để overlap tính toán/giao tiếp, (c) phân tán thật sự CSR
  (2D partition) thay vì replicate để scale lên đồ thị lớn hơn.

---

## 7. Kết luận (Conclusion) — ~0.5 trang

- Tóm tắt lại: đã cài đặt thành công BFS song song hybrid MPI+OpenMP với
  direction-optimizing, đã verify đúng đắn, đã đo được tốc độ và phân tích
  cân bằng tải trên cluster thật.
- Nêu lại 1-2 con số ấn tượng nhất từ thực nghiệm (ví dụ: speedup tối đa đạt
  được, MTEPS đạt được so với baseline tuần tự).
- 1 câu chốt về hạn chế lớn nhất + hướng cải thiện nếu có thêm thời gian.

---

## 8. Tài liệu tham khảo (References)

Tối thiểu 2 mục bắt buộc (định dạng theo chuẩn môn yêu cầu, ví dụ IEEE hoặc
APA):

1. Beamer, S., Asanović, K., & Patterson, D. (2012). Direction-optimizing
   breadth-first search. In *SC '12: Proceedings of the International
   Conference on High Performance Computing, Networking, Storage and
   Analysis*.
2. [Điền chính xác bài báo "Parallel BFS on Distributed Memory System" bạn
   dùng — tên tác giả, hội nghị/tạp chí, năm].
3. (Tuỳ chọn) Chakrabarti, D., Zhan, Y., & Faloutsos, C. (2004). R-MAT: A
   recursive model for graph mining — nếu muốn trích nguồn gốc mô hình R-MAT
   dùng để sinh đồ thị thử nghiệm.
4. (Tuỳ chọn) Tài liệu OpenMPI/OpenMP chính thức nếu trích dẫn chi tiết hành
   vi của `MPI_Allreduce` hoặc `schedule(dynamic)`.

---

## 9. Phụ lục (Appendix, không tính vào 20 trang nếu môn cho phép)

- Toàn bộ log output các lần chạy `--verify`.
- Bảng số liệu thô (raw CSV) của tất cả thí nghiệm 5.2–5.4 (không chỉ số liệu
  đã vẽ biểu đồ).
- (Nếu cần) đoạn code đã chỉnh sửa thêm để đo comm time / per-rank time, kèm
  giải thích ngắn đã thay đổi gì so với code gốc.

---

## 10. Phân bổ số trang gợi ý (tổng ≈ 20 trang)

| Phần | Số trang gợi ý |
|---|---|
| 1. Bìa + mục lục | (không tính) |
| 2. Giới thiệu | 1 |
| 3. Cơ sở lý thuyết | 2.5 |
| 4. Thiết kế song song | 6 |
| 5. Thực nghiệm & kết quả | 8 |
| 6. Thảo luận | 1 |
| 7. Kết luận | 0.5 |
| 8. Tài liệu tham khảo | 0.5 |
| 9. Phụ lục | (không tính, hoặc tính phần dư) |
| **Tổng** | **~19.5** |

Nếu thiếu trang: ưu tiên thêm hình ảnh/biểu đồ minh hoạ ở mục 3 (CSR, R-MAT) và
mục 4 (sơ đồ mapping, sơ đồ giao tiếp) — không nên "độn chữ" ở mục 5
(Results), vì đây là phần giám khảo thường chấm kỹ nhất do gắn trực tiếp với
yêu cầu đề bài.

---

## 11. Checklist cuối cùng — đối chiếu lại với từng yêu cầu gốc của đề bài

- [ ] Đã nêu rõ: song song dữ liệu (không phải tác vụ) — mục 4.1
- [ ] Đã nêu rõ: data decomposition (không phải exploratory/recursive/
      speculative) — mục 4.2
- [ ] Đã trả lời: mapping 1D (không phải 2D √p×√p), kèm công thức + sơ đồ —
      mục 4.3
- [ ] Đã trả lời: blocking, không master-slave, topology flat/runtime-managed,
      liệt kê đủ hàm MPI — mục 4.4
- [ ] Đã trả lời: có load balancing (OpenMP dynamic) nhưng không có ở tầng MPI
      (static 1D theo đỉnh) — mục 4.5, kiểm chứng lại bằng số liệu ở mục 5.3
- [ ] Đã có pseudo-code đầy đủ — mục 4.6
- [ ] Đã verify correctness bằng thực nghiệm thật (không chỉ nói suông) —
      mục 5.1
- [ ] Đã tìm được N để chạy 2-3 phút, có biểu đồ 2 đường (có/không comm time)
      theo N — mục 5.2
- [ ] Đã đo granularity/load balancing bằng biểu đồ cột chồng theo từng tiến
      trình (compute + comm 2 màu), có kết luận cân bằng hay không (ngưỡng
      25%), có điều chỉnh granularity nếu cần — mục 5.3
- [ ] Đã đo speedup với 2×N, quét P từ 1 đến 2X, có 2 biểu đồ (thời gian +
      speedup) — mục 5.4
- [ ] Tổng số trang trong khoảng 10-20 trang
