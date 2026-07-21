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
1. Trên Master (`node01`), tạo file text có tên `hostfile` trong thư mục gốc của project chứa nội dung sau:
   ```text
   node01 slots=4
   node02 slots=4
   node03 slots=4
   ```
   *(Thuộc tính `slots` chỉ định số tối đa MPI process được cấp cho máy, thông thường tương đương số nhân vật lý).*
2. **Tiến hành Test toàn hệ thống:** Từ Master gõ lệnh gọi tên máy theo thiết lập hostfile.
   ```bash
   mpirun -np 12 --hostfile hostfile hostname
   ```
   Nếu màn hình in ra kết quả đan xen đầy đủ 12 dòng hiển thị tên 3 máy chủ thì **Cluster của bạn đã sẵn sàng**.

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
