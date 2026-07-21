# Hướng dẫn toàn tập: Thiết lập Cluster 3 máy và Triển khai Parallel BFS Hybrid

Tài liệu này tổng hợp chi tiết quy trình từ đầu đến cuối để xây dựng một cụm (cluster) gồm 3 máy tính, kết nối chúng và chạy thuật toán Breadth-First Search (BFS) song song. Tài liệu được biên soạn và chắt lọc từ `README.md`, `outline_bao_cao_BFS_hybrid.md` và các ghi chú cấu hình thực tế trong `setup.txt`.

---

## 1. Các Kỹ thuật Parallel (Song song hóa) đã sử dụng

Dự án này cài đặt thuật toán BFS phân tán với nhiều kỹ thuật tối ưu song song tiên tiến:

### 1.1. Mô hình song song lai (Hybrid MPI + OpenMP)
- **MPI (Message Passing Interface):** Giải quyết bài toán phân tán bộ nhớ (Distributed Memory) giữa 3 máy vật lý trong cluster giao tiếp qua mạng LAN.
- **OpenMP:** Khai thác đa luồng chia sẻ bộ nhớ (Shared Memory) trên từng máy vật lý để tận dụng tối đa các nhân CPU cục bộ của máy đó.

### 1.2. Mức độ song song (Data Parallelism)
Sử dụng mô hình **SPMD (Single Program, Multiple Data)**. Dự án hoàn toàn không dùng mô hình Master-Slave (nơi một máy làm chủ phân phát việc cho các máy khác). Ở đây, mọi tiến trình MPI trên 3 máy đều chạy cùng một mã nguồn và thực hiện cùng các bước của thuật toán, điểm khác biệt duy nhất là **phần dữ liệu (tập đỉnh đồ thị)** mà chúng chịu trách nhiệm xử lý.

### 1.3. Kỹ thuật phân rã dữ liệu (Data Decomposition & Partitioning)
- **1D Block Partitioning (Phân rã tĩnh 1 chiều):** Đồ thị CSR không bị chia nhỏ thành block ma trận 2D mà được nhân bản (replicated) toàn bộ trên mọi node. Điều này giảm thiểu giao tiếp phức tạp lúc nạp dữ liệu. Tập đỉnh lớn `V` được chia thành các dải liên tục. Nếu có `P` tiến trình, tiến trình $r$ sẽ quản lý và làm chủ dải đỉnh từ $\lfloor n \cdot r / P \rfloor$ đến $\lfloor n \cdot (r+1) / P \rfloor$.
- **Owner-Computes Rule:** Tiến trình nào sở hữu đỉnh thì tiến trình đó có trách nhiệm tính toán và cập nhật giá trị `dist[]` (khoảng cách) của đỉnh đó.

### 1.4. Cân bằng tải (Load Balancing)
Vì đồ thị được sinh bằng mô hình R-MAT có sự phân bố bậc cực kỳ lệch (power-law degree skew), lượng công việc tính toán giữa các vùng đỉnh là không đồng đều.
- **Tầng MPI:** Phân rã theo tĩnh (Static mapping) dựa trên số đỉnh chia đều `n/P`. Tầng này không có cân bằng tải động.
- **Tầng OpenMP:** Sử dụng **cân bằng tải động (Dynamic Runtime Mapping)** với chỉ thị `#pragma omp for schedule(dynamic, chunk_size)`. Các luồng (threads) xử lý xong sớm chunk của mình sẽ tự động bốc chunk tiếp theo, giúp bù trừ sự chênh lệch thời gian do khác biệt số cạnh (bậc) của đỉnh.

### 1.5. Chiến lược giao tiếp (Communication Strategy)
- Sử dụng giao tiếp tập thể, đồng bộ và chặn (Peer Collective & Blocking).
- Toàn bộ 3 máy trao đổi với nhau bằng `MPI_Allreduce`. Thuật toán duyệt đồng bộ từng cấp (level-synchronous). Cuối mỗi mức duyệt đồ thị, tất cả các tiến trình phải hội quân ở Synchronization Barrier (`MPI_Allreduce`) để gộp chung mảng khoảng cách `dist[]` và mảng bitmap của tập biên (frontier).

### 1.6. Direction-Optimizing Heuristic (Theo Beamer et al.)
- **Tối ưu hướng duyệt:** Thuật toán đếm và theo dõi lượng đỉnh trên frontier để linh hoạt hoán đổi giữa duyệt **Top-Down** (truyền thống, hiệu quả khi frontier nhỏ) và **Bottom-Up** (ngược hướng, hiệu quả lúc frontier bùng nổ quá lớn). Việc này cắt giảm triệt để số lượng cạnh cần kiểm tra dư thừa.

---

## 2. Quy trình thiết lập Cluster 3 máy chi tiết

Giả định Cluster gồm 3 máy: `node01` (đóng vai trò Master/Host), `node02` (Worker 1), `node03` (Worker 2). Tất cả chung Subnet mạng LAN.

### Bước 1: Kiểm tra IP và cấu hình `/etc/hosts`
Trên **tất cả** 3 máy thực hiện các bước sau:
1. Gõ lệnh `ip a` để xem và ghi nhận lại địa chỉ IP mạng nội bộ (subnet) của từng máy.
2. Mở file cấu hình tên miền cục bộ: `sudo nano /etc/hosts`
3. Thêm IP và tên máy của 3 node vào file để MPI phân giải tên miền:
   ```text
   192.168.1.101   node01
   192.168.1.102   node02
   192.168.1.103   node03
   ```
4. **Xác nhận kết nối:** Chạy lệnh `ping node02` hoặc `ping node03` từ `node01` (và làm ngược lại) để đảm bảo mạng thông suốt.

### Bước 2: Cài đặt phần mềm cơ bản
Cài đặt môi trường bắt buộc trên cả 3 máy (Lưu ý: Mọi máy đều phải sử dụng chung một phiên bản OpenMPI và GCC):
```bash
sudo apt update && sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev openssh-server nfs-common gcc make
```

### Bước 3: Tạo User chung và Cấu hình SSH (Không mật khẩu)
Quá trình MPI spawn các process từ Master sang Worker đều đi ngầm qua SSH. Do đó, việc thiết lập SSH trust (bỏ qua mật khẩu) là điều bắt buộc.
1. Khởi tạo một user giống hệt nhau trên 3 máy (ví dụ `mpiuser`): `sudo adduser mpiuser`
2. Đăng nhập vào user `mpiuser` trên máy Master (`node01`), sau đó tạo khóa SSH public/private:
   ```bash
   su - mpiuser
   ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa   # Bỏ trống phần passphrase (Enter 3 lần)
   ```
3. Copy khóa Public sang tất cả các máy, bao gồm cả chính máy Master vì MPI thỉnh thoảng gọi lại local node:
   ```bash
   ssh-copy-id mpiuser@node01
   ssh-copy-id mpiuser@node02
   ssh-copy-id mpiuser@node03
   ```
4. Xác minh lại bằng cách SSH thử: `ssh mpiuser@node02 "hostname"`. Nếu trả về tên máy mà không cần nhập mật khẩu là đạt.

### Bước 4: Cấu hình chia sẻ mã nguồn với NFS (Network File System)
Chia sẻ thư mục mã nguồn từ Master cho 2 máy Worker đọc/ghi chung. Khi biên dịch code 1 lần trên Master, các Worker tự động cập nhật được file chạy.
1. **Trên Master (`node01`):** 
   - Cài server chia sẻ file: `sudo apt install nfs-kernel-server`
   - Cấu hình chia sẻ thư mục của mpiuser, thêm dòng sau vào file `sudo nano /etc/exports`:
     ```text
     /home/mpiuser 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)
     ```
   - Xác nhận thiết lập với `sudo exportfs -a` và khởi động lại dịch vụ NFS: `sudo systemctl start nfs-kernel-server`.
2. **Trên các Worker (`node02`, `node03`):**
   - Đảm bảo thư mục mount (ổ đĩa mạng) có sẵn: `sudo mkdir -p /home/mpiuser`
   - Mở file `sudo nano /etc/fstab` và thêm vào dòng sau để Worker tự mount khi khởi động máy:
     ```text
     node01:/home/mpiuser  /home/mpiuser  nfs  defaults,_netdev  0  0
     ```
3. **Kiểm tra NFS:**
   Trên tất cả các node, gõ lệnh `sudo mount -a` để tải lại trạng thái mount. Dùng lệnh `df -h | grep mpiuser` để xác nhận phân vùng mạng đã được gắn thành công.

### Bước 5: Cấu hình Hostfile và Test MPI Cluster
1. Trên Master (`node01`), tạo file text có tên `hostfile` trong thư mục gốc của project chứa nội dung sau (lưu ý Master có 4 core, 2 Slaves mỗi máy 2 core):
   ```text
   node01 slots=4
   node02 slots=2
   node03 slots=2
   ```
   *(Thuộc tính `slots` chỉ định số lượng MPI process tối đa được cấp cho máy, ở đây tổng cộng là 8).*
2. **Tiến hành Test toàn hệ thống:** Từ Master gõ lệnh gọi tên máy theo thiết lập hostfile.
   ```bash
   mpirun -np 8 --hostfile hostfile hostname
   ```
   Nếu màn hình in ra kết quả đan xen đầy đủ 8 dòng hiển thị tên 3 máy chủ thì **Cluster của bạn đã sẵn sàng**.

---

## 3. Chạy thuật toán Parallel BFS

### 3.1. Biên dịch và Sửa đổi Bash Script (Dựa trên setup.txt)
- Chuyển vào thư mục code và gọi `make` (việc này chỉ thực hiện trên Master).
- Có sự khác biệt trong cờ CLI để load/save graph trong script benchmark. Cần thực hiện các lệnh sau để patch `scripts/run_benchmark.sh`:
  ```bash
  sed -i 's/echo "--graph $path"/echo "--load $path"/g' scripts/run_benchmark.sh
  sed -i 's/--save-graph/--save/g' scripts/run_benchmark.sh
  ```

### 3.2. Chạy tay (Manual Run)
Chạy trực tiếp thuật toán BFS lai (Hybrid). Mẫu sau mô phỏng chạy 8 tiến trình (rank), mỗi tiến trình 1 luồng OpenMP. Chúng ta sử dụng `--oversubscribe` và `--allow-run-as-root` theo tình huống cụ thể (Ví dụ chạy trên tài khoản root và ép chạy nhiều tiến trình hơn số lượng core):
```bash
OMP_NUM_THREADS=1 mpirun --oversubscribe --allow-run-as-root -np 8 \
    --hostfile hostfile \
    ./bfs_hybrid 500000 16 42 \
    --csv results/test_manual.csv
```
Đoạn lệnh trên chạy đồ thị R-MAT 500.000 đỉnh, độ thưa 16, mã seed ngẫu nhiên là 42 và xuất lưu metric vào file csv.

### 3.3. Chạy bằng Tool Benchmark tự động
Sử dụng script để chạy hàng loạt thực nghiệm với các bộ tham số khác nhau:
```bash
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh
```
Hệ thống sẽ chạy đồng loạt và bạn có thể theo dõi kết quả tự lưu tại thư mục `results/`.

### 3.4. Mô tả chi tiết Luồng Thuật toán Parallel BFS (Theo Lý thuyết Toán học và Đồ thị)

Dưới đây là mô tả thuật toán BFS lai phân tán (Hybrid MPI + OpenMP) theo dạng lý thuyết toán học (pseudocode diễn giải bằng lời):

**Định nghĩa và Ký hiệu Toán học:**
- Cho đồ thị $G = (V, E)$, trong đó $V$ là tập hợp các đỉnh ($|V| = n$) và $E$ là tập hợp các cạnh.
- Đỉnh nguồn (source vertex) được ký hiệu là $s \in V$.
- Hệ thống phân tán bao gồm một tập các tiến trình $P = \{p_0, p_1, ..., p_{k-1}\}$.
- Tập đỉnh $V$ được phân hoạch (partition) thành $k$ tập con không giao nhau $V_0, V_1, ..., V_{k-1}$ sao cho $\bigcup_{i=0}^{k-1} V_i = V$ và $V_i \cap V_j = \emptyset, \forall i \neq j$. Tiến trình $p_i$ quản lý và chịu trách nhiệm tính toán cho tập đỉnh $V_i$.
- Hàm $N(u) = \{v \in V | (u, v) \in E \text{ hoặc } (v, u) \in E\}$ là tập các đỉnh kề của $u$.
- Hàm khoảng cách $d: V \rightarrow \mathbb{N} \cup \{-1\}$ lưu độ dài đường đi ngắn nhất từ $s$ đến $v \in V$ (quy ước $-1$ là chưa thăm).
- Tập $F_l \subseteq V$ (Frontier) là tập hợp các đỉnh biên đang được xét tại cấp độ (level) $l$.

**Bước 1: Khởi tạo (Initialization)**
- Tại mỗi tiến trình $p_i \in P$:
  - Khởi tạo khoảng cách: $d(v) \leftarrow -1, \forall v \in V$.
  - Gán khoảng cách cho đỉnh nguồn: $d(s) \leftarrow 0$.
  - Khởi tạo tập biên cấp độ 0: $F_0 \leftarrow \{s\}$.
  - Khởi tạo cấp độ duyệt $l \leftarrow 0$.

**Bước 2: Vòng lặp duyệt đồ thị (Level-synchronous Traversal)**
Trong khi tập biên $F_l \neq \emptyset$, thực hiện các bước sau đồng thời tại mọi tiến trình $p_i$:

- **2.1. Đánh giá Heuristic (Direction-Optimizing):**
  - Tiến trình $p_i$ tính tổng số cạnh xuất phát từ các đỉnh thuộc $F_l$ nằm trong phần dữ liệu của nó: $c_i = \sum_{u \in (F_l \cap V_i)} |N(u)|$.
  - Thực hiện phép toán gộp toàn cục (Global Reduction): $C = \sum_{i=0}^{k-1} c_i$ (thông qua `MPI_Allreduce(MPI_SUM)`).
  - Nếu $C > \alpha \times \text{số cạnh chưa thăm}$, thuật toán chuyển sang chiến lược duyệt ngược (**Bottom-Up**). Ngược lại, sử dụng chiến lược duyệt xuôi (**Top-Down**).

- **2.2. Tính toán Cục bộ (Local Computation) sinh ra $F_{l+1}^{(i)}$:**
  Khởi tạo tập biên tiếp theo của tiến trình $i$ là $F_{l+1}^{(i)} \leftarrow \emptyset$.
  - **Trường hợp Top-Down (Khai thác tính song song của OpenMP trên $F_l$):**
    Với mỗi đỉnh $u \in (F_l \cap V_i)$:
      - Với mỗi đỉnh $v \in N(u)$:
        - Nếu $d(v) = -1$:
          - $d(v) \leftarrow l + 1$
          - $F_{l+1}^{(i)} \leftarrow F_{l+1}^{(i)} \cup \{v\}$
  - **Trường hợp Bottom-Up (Khai thác tính song song của OpenMP trên $V_i$):**
    Với mỗi đỉnh $u \in V_i$ thỏa mãn $d(u) = -1$:
      - Nếu $\exists v \in N(u)$ sao cho $v \in F_l$:
        - $d(u) \leftarrow l + 1$
        - $F_{l+1}^{(i)} \leftarrow F_{l+1}^{(i)} \cup \{u\}$
        - Dừng việc xét các $v$ còn lại của $u$ (Symmetry Breaking).

- **2.3. Đồng bộ hóa Toàn cục (Global Synchronization & Reduction):**
  - Tiến hành đồng nhất hàm khoảng cách $d(v)$ trên toàn hệ thống bằng phép gộp toàn cục:
    $d(v) \leftarrow \max_{p_j \in P} d^{(j)}(v), \forall v \in V$ (sử dụng `MPI_Allreduce(MPI_MAX)`).
  - Hợp nhất tập biên tiếp theo từ tất cả các tiến trình:
    $F_{l+1} \leftarrow \bigcup_{i=0}^{k-1} F_{l+1}^{(i)}$ (sử dụng `MPI_Allreduce(MPI_BOR)` trên biểu diễn bitmap của $F$).
  - Cập nhật cấp độ: $l \leftarrow l + 1$.

**Bước 3: Kết thúc (Termination)**
- Thuật toán dừng lại tại cấp độ $l_{max}$ khi $F_{l_{max}} = \emptyset$.
- Đầu ra của thuật toán là mảng $d(v)$ hoàn chỉnh được đồng bộ trên tất cả các tiến trình, chứa độ dài đường đi ngắn nhất từ $s$ đến mọi $v \in V$.

### 3.5. Ví dụ Minh họa Chạy từng bước (Đa đường đi & Tránh xung đột)

Để chứng minh sức mạnh của thuật toán khi đối phó với đồ thị phức tạp (nơi một đỉnh có nhiều đường đi từ nguồn đến), hãy xem ví dụ trên đồ thị $N = 8$ đỉnh ($V = \{0, 1, ..., 7\}$) chạy trên **$k = 4$ tiến trình** ($p_0, p_1, p_2, p_3$).

**1. Cấu trúc Đồ thị (Nhiều đường đi chéo nhau):**
- $N(0) = \{1, 2, 3\}$ (0 là đỉnh nguồn)
- $N(1) = \{0, 3, 4\}$
- $N(2) = \{0, 4, 5\}$
- $N(3) = \{0, 1, 6\}$
- $N(4) = \{1, 2, 6, 7\}$
- $N(5) = \{2\}$
- $N(6) = \{3, 4\}$
- $N(7) = \{4\}$

*Lưu ý:* 
- Đỉnh `6` có 2 đường đi cách biệt: Một đường ngắn $0 \rightarrow 3 \rightarrow 6$ (dài 2) và một đường dài $0 \rightarrow 1 \rightarrow 4 \rightarrow 6$ (dài 3).
- Đỉnh `4` có 2 đường đi cùng độ dài: $0 \rightarrow 1 \rightarrow 4$ (dài 2) và $0 \rightarrow 2 \rightarrow 4$ (dài 2).

**2. Phân chia dữ liệu (1D Block Partitioning):**
- $p_0$ quản lý $V_0 = \{0, 1\}$
- $p_1$ quản lý $V_1 = \{2, 3\}$
- $p_2$ quản lý $V_2 = \{4, 5\}$
- $p_3$ quản lý $V_3 = \{6, 7\}$

**3. Theo vết Thuật toán (Nguồn $s = 0$):**

**[KHỞI TẠO] (Level $l = 0$)**
- Cả 4 tiến trình khởi tạo mảng $d$: $[-1, -1, -1, -1, -1, -1, -1, -1]$
- Gán đỉnh nguồn: $d(0) = 0$
- Tập biên ban đầu: $F_0 = \{0\}$

**[VÒNG LẶP 1] (Cấp độ $l = 0 \rightarrow 1$)**
- **Duyệt xuôi (Top-Down):** Tiến trình $p_0$ xử lý đỉnh $0 \in F_0$.
- **Tính toán:** $p_0$ xét $N(0) = \{1, 2, 3\}$. Tất cả đều có $d=-1$, nên $p_0$ gán $d(1)=1, d(2)=1, d(3)=1$ và đưa vào $F_1^{(0)} = \{1, 2, 3\}$.
- **Đồng bộ Toàn cục:** 
  - `MPI_MAX` mảng $d \rightarrow$ Toàn hệ thống có: $d = [0, \mathbf{1, 1, 1}, -1, -1, -1, -1]$.
  - `MPI_BOR` bitmap $\rightarrow$ $F_1 = \{1, 2, 3\}$.

**[VÒNG LẶP 2] (Cấp độ $l = 1 \rightarrow 2$)**
- Giả sử thuật toán vẫn chọn **Top-Down**. Các tiến trình chia nhau xử lý tập $F_1 = \{1, 2, 3\}$.
  - $p_0$ xử lý đỉnh 1: Dò $N(1) = \{0, 3, 4\}$.
    - Đỉnh 0 và 3 đã thăm ($d \neq -1$). Bỏ qua.
    - Đỉnh 4 chưa thăm. Gán $d^{(0)}(4) = 2$, thêm 4 vào $F_2^{(0)}$.
  - $p_1$ xử lý đỉnh 2: Dò $N(2) = \{0, 4, 5\}$.
    - Đỉnh 4 chưa thăm. Gán $d^{(1)}(4) = 2$, thêm 4 vào $F_2^{(1)}$.
    - Đỉnh 5 chưa thăm. Gán $d^{(1)}(5) = 2$, thêm 5 vào $F_2^{(1)}$.
  - $p_1$ xử lý đỉnh 3: Dò $N(3) = \{0, 1, 6\}$. 
    - Đỉnh 6 chưa thăm. Gán $d^{(1)}(6) = 2$, thêm 6 vào $F_2^{(1)}$. *(Đường đi ngắn $0 \rightarrow 3 \rightarrow 6$ đã được chốt ở đây!)*
- **Đồng bộ Toàn cục:**
  - Nhận thấy đỉnh `4` được CẢ HAI tiến trình $p_0$ và $p_1$ phát hiện đồng thời ở 2 máy khác nhau.
  - $p_0$ có mảng nội bộ: $d = [0, 1, 1, 1, \mathbf{2}, -1, -1, -1]$
  - $p_1$ có mảng nội bộ: $d = [0, 1, 1, 1, \mathbf{2}, 2, 2, -1]$
  - Khi gọi `MPI_Allreduce(MPI_MAX)`, mạng sẽ gộp: $d(4) = \max(-1, 2, 2, -1) = 2$.
  - **Kết quả:** Xung đột được giải quyết hoàn hảo. Mảng toàn cục: $d = [0, 1, 1, 1, \mathbf{2, 2, 2}, -1]$. Biên toàn cục $F_2 = \{4, 5, 6\}$.

**[VÒNG LẶP 3] (Cấp độ $l = 2 \rightarrow 3$)**
- Biên $F_2 = \{4, 5, 6\}$. Thuật toán kích hoạt **Top-Down**.
- **Tính toán cục bộ:** Các tiến trình xử lý $F_2$.
  - $p_2$ xử lý đỉnh 4: Xét $N(4) = \{1, 2, 6, 7\}$.
    - Đỉnh 1, 2 đã thăm. Bỏ qua.
    - **Xét đỉnh 6:** Đỉnh 6 đã có $d(6) = 2 \neq -1 \rightarrow$ **Bỏ qua ngay lập tức!** *(Đây chính là cơ chế để máy tính TỪ CHỐI con đường dài $0 \rightarrow 1 \rightarrow 4 \rightarrow 6$ dài 3 bước, bảo vệ nguyên vẹn con đường 2 bước đã tìm thấy ở Vòng 2).*
    - Xét đỉnh 7: Chưa thăm. Gán $d^{(2)}(7) = 3$, thêm 7 vào $F_3^{(2)}$.
  - $p_2$ xử lý đỉnh 5: $N(5)=\{2\}$. Đã thăm.
  - $p_3$ xử lý đỉnh 6: $N(6)=\{3, 4\}$. Cả 2 đều đã thăm. Bỏ qua.
- **Đồng bộ Toàn cục:** Toàn hệ thống cập nhật $d(7) = 3$. $F_3 = \{7\}$.

**[VÒNG LẶP 4] (Cấp độ $l = 3 \rightarrow 4$)**
- Xử lý $F_3 = \{7\}$. Lân cận duy nhất là $4$ (đã thăm). Không tìm ra đỉnh nào mới $\rightarrow F_4 = \emptyset$.

**[KẾT THÚC]**
- Vòng lặp dừng. Mảng $d$ cuối cùng: $[0, 1, 1, 1, 2, 2, 2, 3]$. 
- **Bài học:** 
  1. Câu lệnh `if d(v) == -1` đóng vai trò khiên bảo vệ, giúp đồ thị dù có hàng chục đường đi ngoằn ngoèo thì BFS luôn giữ lại con đường ngắn nhất (được phát hiện sớm nhất).
  2. Lệnh `MPI_MAX` xử lý triệt để hiện tượng nhiều máy cùng lúc tìm ra một đỉnh, loại bỏ hoàn toàn lỗi Data Race (ghi đè hỗn loạn) khi tính toán phân tán.

---

## 4. Các Tiêu chí Đánh giá và Phân tích Kết quả Thực nghiệm

Trong báo cáo, hiệu năng và hành vi của hệ thống Cluster 3 máy được đánh giá toàn diện qua 4 khía cạnh chính sau:

### 4.1. Đánh giá Tính đúng đắn (Correctness Verification)
- **Mục tiêu:** Xác nhận thuật toán phân tán (Hybrid MPI + OpenMP) tính toán chính xác đường đi ngắn nhất giống hệt như thuật toán tuần tự (Sequential) cơ bản.
- **Cách thực hiện:** Chương trình chạy song song thuật toán BFS trên phiên bản tuần tự và phiên bản Hybrid, sau đó so sánh mảng `dist[]` từng phần tử một (thông qua cờ `--verify`).
- **Đánh giá rút ra:** Bất kể cấu hình phân bổ thay đổi (kích thước đồ thị N, số tiến trình P và số luồng Thread/Rank), phiên bản song song luôn luôn tính đúng và trả về `PASSED`. Báo cáo cũng ghi nhận ở đồ thị rất nhỏ (như N = 16,384), tốc độ thực thi (speedup) của phiên bản song song có thể < 1 do thời gian thiết lập mạng và giao tiếp lấn át hoàn toàn lợi ích của việc chia nhỏ tính toán.

### 4.2. Khảo sát Kích thước Đầu vào N (Determining Input Size)
- **Mục tiêu:** Quét để xác định kích thước đồ thị (số đỉnh $N_0$) sao cho hệ thống chạy hết công suất mất khoảng 2-3 phút, thiết lập mốc baseline cho các bài test tải phía sau.
- **Đánh giá & Phân tích:** 
  - Qua thử nghiệm quét (Scan-N), báo cáo bóc tách thời gian thực thi tổng ($T_{total}$) thành 2 thành phần độc lập: **Thời gian giao tiếp qua mạng ($T_{comm}$)** và **Thời gian CPU tính toán thuần ($T_{comp}$)**.
  - Phân tích chỉ ra một hiện tượng thắt cổ chai cực kỳ nghiêm trọng: $T_{comm}$ (thời gian chờ các hàm `MPI_Allreduce`) chiếm tới 99.9% tổng thời gian, trong khi $T_{comp}$ chỉ tiêu tốn vài chục mili-giây.
  - Nguyên nhân cốt lõi là do thiết kế `MPI_Allreduce` phải gộp (broadcast) mảng `dist[]` có kích thước $O(N)$ (tương đương hàng chục Megabyte dữ liệu) qua hệ thống mạng LAN giữa 3 máy ở mỗi level. Hệ thống tính toán rất nhanh nhưng phải mất hàng chục giây chỉ để chờ gửi và nhận mảng dữ liệu khổng lồ này qua mạng.

### 4.3. Phân tích Cân bằng tải và Độ mịn (Granularity & Load Balancing)
- **Mục tiêu:** Kiểm tra xem khối lượng công việc có được chia đều giữa các tiến trình hay không. Mức độ lệch tải lý tưởng phải $\le 25\%$. Khái niệm độ mịn (granularity) ở đây chính là số lượng đỉnh và số lượng cạnh thực tế mà mỗi Rank phụ trách.
- **Đánh giá & Phân tích:**
  - Nhờ việc đồ thị R-MAT có đặc thù lệch bậc (power-law degree skew), tiến trình Rank 0 (quản lý vùng đỉnh chỉ số nhỏ) phải gánh lượng cạnh lớn gấp hàng chục lần tiến trình Rank 7 (vùng đỉnh chỉ số lớn) – Tỷ lệ workload lên tới 30:1.
  - Nghịch lý thay, tổng thời gian thực thi (Tổng $T_{total}$) giữa các tiến trình lại gần như bằng nhau chằn chặn (chênh lệch chỉ $\approx 0.05\%$).
  - Báo cáo phân tích rằng: Hệ thống đạt chỉ số cân bằng tải **hoàn toàn không phải vì công việc được chia đều**, mà do thời gian chờ mạng (`MPI_Allreduce` blocking) ở các máy là quá lớn (hàng chục giây) che lấp hoàn toàn sự chênh lệch tính toán (vài mili-giây). Qua đó kết luận độ mịn (granularity) tính toán hiện tại quá nhỏ so với chi phí giao tiếp (ratio Tính toán / Giao tiếp là quá thấp).

### 4.4. Đánh giá Độ Tăng tốc (Speedup Evaluation)
- **Mục tiêu:** Kiểm tra khả năng mở rộng (scalability) của thuật toán khi tăng số lượng tiến trình và số core làm việc trên 3 máy.
- **Cách thực hiện:** Khảo sát trên đồ thị kích thước $2 \times N_0$, chạy với số tiến trình $P$ tăng dần. Báo cáo đo lường 2 loại Speedup: Speedup tổng thể $S(P)$ và Speedup tính toán thuần $S'(P)$ (sau khi đã trừ đi chi phí giao tiếp mạng).
- **Đánh giá & Phân tích:**
  - **Tăng tốc tính toán thuần ($S'(P)$):** Đạt mức độ tăng tốc gần như tuyến tính tuyệt vời. Điều này chứng minh rằng thuật toán Hybrid, OpenMP và heuristic Direction-Optimizing hoàn toàn chuẩn xác về mặt toán học và thiết kế trong việc phân tán khối lượng công việc.
  - **Tăng tốc tổng thể ($S(P)$):** Sụp đổ gần về 0 (crashing to near zero) khi số tiến trình P tăng cao. Kết luận chỉ ra thủ phạm là kiến trúc Cluster phân tán (Distributed-Memory): Băng thông vật lý và độ trễ (latency) của mạng LAN hoàn toàn không đủ khả năng gánh vác lượng traffic khổng lồ (lên tới 200 Megabyte mỗi lần chạy) sinh ra do cơ chế đồng bộ tập thể dữ liệu toàn cục. Kiến trúc sao chép toàn đồ thị (Graph Replication) bộc lộ nhược điểm rõ rệt khi đối mặt với thắt cổ chai mạng.
