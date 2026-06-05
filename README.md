# BFS Hybrid — MPI + OpenMP + Direction-Optimizing

Project **Parallel & Distributed Programming** implement thuật toán
**Breadth-First Search song song** trên đồ thị R-MAT quy mô lớn, chạy
phân tán trên nhiều máy (cluster).

---

## Mục lục

1. [Cấu trúc project](#1-cấu-trúc-project)
2. [Thuật toán](#2-thuật-toán)
3. [Cài đặt dependencies](#3-cài-đặt-dependencies)
4. [Build project](#4-build-project)
5. [Chạy trên 1 máy (local)](#5-chạy-trên-1-máy-local)
6. [Thiết lập Cluster — Hướng dẫn chi tiết](#6-thiết-lập-cluster--hướng-dẫn-chi-tiết)
   - [Yêu cầu phần cứng / mạng](#61-yêu-cầu-phần-cứng--mạng)
   - [Cài OpenMPI trên tất cả nodes](#62-cài-openmpi-trên-tất-cả-nodes)
   - [Tạo user chung và cấu hình SSH](#63-tạo-user-chung-và-cấu-hình-ssh-không-password)
   - [Cấu hình NFS chia sẻ code](#64-cấu-hình-nfs-chia-sẻ-thư-mục-code)
   - [Cấu hình /etc/hosts](#65-cấu-hình-etchosts)
   - [Tạo hostfile](#66-tạo-hostfile)
   - [Test kết nối cluster](#67-test-kết-nối-cluster)
   - [Chạy chương trình trên cluster](#68-chạy-chương-trình-trên-cluster)
7. [Tham số và biến môi trường](#7-tham-số-và-biến-môi-trường)
8. [Kết quả mẫu](#8-kết-quả-mẫu)
9. [Xử lý lỗi thường gặp](#9-xử-lý-lỗi-thường-gặp)
10. [Các hàm MPI được dùng](#10-các-hàm-mpi-được-dùng)

---

## 1. Cấu trúc project

```
bfs-project/
├── src/
│   ├── main_seq.c          ← Entry point BFS tuần tự (baseline)
│   ├── main_hybrid.c       ← Entry point BFS hybrid (MPI + OpenMP)
│   ├── bfs_sequential.c/h  ← BFS queue-based tiêu chuẩn
│   ├── bfs_hybrid.c/h      ← Direction-optimizing + MPI + OpenMP
│   ├── graph.c/h           ← R-MAT generator, định dạng CSR
│   └── utils.c/h           ← Timer, MTEPS, verify
├── scripts/
│   └── run_benchmark.sh    ← Benchmark tự động, xuất CSV
├── results/                ← Output CSV (tự tạo khi chạy benchmark)
├── hostfile                ← Danh sách nodes cluster
├── Makefile
└── README.md
```

---

## 2. Thuật toán

### R-MAT Graph Generator

Sinh đồ thị ngẫu nhiên theo mô hình **R-MAT** (Recursive MATrix),
chuẩn Graph500. Đồ thị vô hướng, lưu dạng **CSR** (Compressed Sparse Row).

```
Ma trận kề được chia đệ quy thành 4 góc với xác suất a, b, c, d:
  ┌──────┬──────┐
  │  a   │  b   │    a=0.57, b=0.19
  │      │      │    c=0.19, d=0.05
  ├──────┼──────┤    (a+b+c+d = 1)
  │  c   │  d   │
  └──────┴──────┘
Mỗi cạnh được đặt vào một góc ngẫu nhiên, lặp lại log2(n) lần.
→ Tạo ra đồ thị có phân phối bậc power-law (thực tế hơn random uniform)
```

### Direction-Optimizing BFS (Beamer et al. 2012)

BFS thông thường luôn duyệt **top-down** (từ frontier ra hàng xóm).
Khi frontier rất lớn, cách này lãng phí vì kiểm tra nhiều cạnh đã thăm.

Thuật toán tự động chuyển hướng theo từng level:

```
TOP-DOWN  : frontier → xét hàng xóm của mỗi đỉnh trong frontier
BOTTOM-UP : mỗi đỉnh CHƯA THĂM → tìm 1 hàng xóm trong frontier
            (dừng ngay khi tìm thấy → tiết kiệm hơn khi frontier rộng)

Chuyển sang BOTTOM-UP khi: frontier_edges > unvisited_edges / ALPHA (=14)
Quay lại TOP-DOWN khi:     frontier_size  < num_vertices    / BETA  (=24)
```

### Kiến trúc song song

```
┌─────────────────────────────────────────────────────┐
│                  MPI RANK 0 (Master)                 │
│  ┌──────────┐  ┌──────────┐                          │
│  │ Thread 0 │  │ Thread 1 │  ← OpenMP threads        │
│  └──────────┘  └──────────┘                          │
├─────────────────────────────────────────────────────┤
│                  MPI RANK 1                          │
│  ┌──────────┐  ┌──────────┐                          │
│  │ Thread 0 │  │ Thread 1 │                          │
│  └──────────┘  └──────────┘                          │
├─────────────────────────────────────────────────────┤
│                  MPI RANK 2, 3, ...                  │
└─────────────────────────────────────────────────────┘

Phân vùng đỉnh (Vertex Partition):
  Rank r chịu trách nhiệm đỉnh [n*r/P, n*(r+1)/P)

Đồng bộ cuối mỗi level:
  MPI_Allreduce (MAX) → merge dist[]
  MPI_Allreduce (OR)  → merge frontier bitmap
```

---

## 3. Cài đặt dependencies

Thực hiện trên **tất cả** các máy tham gia cluster:

```bash
sudo apt update && sudo apt install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    openssh-server \
    gcc \
    make
```

Kiểm tra version — phải **giống nhau** trên tất cả nodes:

```bash
mpirun --version
# Ví dụ: mpirun (Open MPI) 4.1.6

mpicc --version
# Ví dụ: gcc (Ubuntu 13.2.0) 13.2.0

# Kiểm tra OpenMP (có sẵn trong GCC >= 4.9)
echo "#include <omp.h>
int main(){return 0;}" | gcc -fopenmp -x c - -o /tmp/t && echo "OpenMP OK"
```

> ⚠️ **Quan trọng:** Version OpenMPI phải **giống hệt nhau** trên tất cả nodes.
> Nếu khác version, chương trình có thể crash hoặc treo.

---

## 4. Build project

```bash
make          # Build cả bfs_seq và bfs_hybrid
make clean    # Xóa binary và results/*.csv
```

Makefile cũng có các target tiện dụng để test nhanh:

```bash
make test_seq    # Chạy sequential 500k đỉnh
make test_local  # Chạy hybrid 1 rank + 4 thread + verify
make verify      # Verify correctness với graph nhỏ (100k đỉnh)
```

---

## 5. Chạy trên 1 máy (local)

### BFS tuần tự (baseline)

```bash
./bfs_seq <num_vertices> <scale_factor> <seed>

# Ví dụ: 1 triệu đỉnh, bậc trung bình 16, seed 42
./bfs_seq 1000000 16 42
```

### BFS hybrid — 1 máy

```bash
OMP_NUM_THREADS=<threads> mpirun -np <ranks> \
    ./bfs_hybrid <num_vertices> <scale_factor> <seed> [--verify]

# Ví dụ: 4 rank × 2 thread = 8 cores tổng
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42

# Thêm --verify để so sánh kết quả với sequential
OMP_NUM_THREADS=2 mpirun -np 4 ./bfs_hybrid 1000000 16 42 --verify
```

> `ranks × threads ≤ số core vật lý` để tránh oversubscribe.
> Nếu máy ít core, thêm `--oversubscribe` để ép chạy (kết quả đúng, hiệu năng thấp hơn).

### Benchmark tự động

```bash
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh
# Kết quả lưu tại: results/benchmark_<timestamp>.csv
```

---

## 6. Thiết lập Cluster — Hướng dẫn chi tiết

Phần này hướng dẫn cấu hình **mỗi VM chạy trên 1 máy vật lý riêng**,
kết nối qua mạng LAN, để chạy BFS phân tán thật sự.

### Topology trong hướng dẫn này

```
┌────────────────────────────────────────────────────────────┐
│                     LAN / Switch / Router                  │
│                                                            │
│  ┌─────────────────┐   ┌──────────────┐  ┌──────────────┐ │
│  │  Máy vật lý 1   │   │ Máy vật lý 2 │  │ Máy vật lý 3 │ │
│  │  ┌───────────┐  │   │ ┌──────────┐ │  │ ┌──────────┐ │ │
│  │  │   VM      │  │   │ │   VM     │ │  │ │   VM     │ │ │
│  │  │  node01   │  │   │ │  node02  │ │  │ │  node03  │ │ │
│  │  │.1.101     │  │   │ │ .1.102   │ │  │ │ .1.103   │ │ │
│  │  │ (master)  │  │   │ │ (worker) │ │  │ │ (worker) │ │ │
│  │  └───────────┘  │   │ └──────────┘ │  │ └──────────┘ │ │
│  └─────────────────┘   └──────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────────┘
```

> Thay IP và hostname theo thực tế của bạn.
> Mỗi VM phải dùng **Bridged Adapter** trong VirtualBox để lấy IP thật trên LAN.

---

### 6.1. Yêu cầu phần cứng / mạng

- Tất cả VM phải **cùng subnet** (ping được nhau)
- OS: **Ubuntu 22.04+ / 24.04** (khuyến nghị)
- Mỗi VM: tối thiểu 2 vCPU, 1GB RAM, 15GB disk
- Network Adapter trong VirtualBox: **Bridged Adapter** (không phải NAT)
- Cổng cần mở nếu có firewall: **SSH (22)**, **NFS (2049)**

Kiểm tra ping trước khi làm bất cứ thứ gì:

```bash
# Trên node01, ping sang các node khác
ping -c 3 192.168.1.102
ping -c 3 192.168.1.103

# Nếu không ping được → kiểm tra lại Bridged Adapter trong VirtualBox
```

---

### 6.2. Cài OpenMPI trên tất cả nodes

Chạy lệnh sau trên **TỪNG VM** (node01, node02, node03, ...):

```bash
sudo apt update && sudo apt install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    nfs-common \
    openssh-server \
    gcc \
    make

# Kiểm tra version — phải giống nhau trên tất cả
mpirun --version
# Ví dụ: mpirun (Open MPI) 4.1.6
```

---

### 6.3. Tạo user chung và cấu hình SSH không password

MPI dùng SSH để spawn process trên các node worker.
Phải cấu hình để **node01 SSH sang các node khác mà không cần nhập password**.

#### Bước 1 — Tạo user `mpiuser` trên TẤT CẢ các VM

```bash
# Chạy trên TỪNG VM
sudo adduser mpiuser
# Nhập password, Enter bỏ qua các trường còn lại

# Tuỳ chọn: thêm sudo để tiện cài gói
sudo usermod -aG sudo mpiuser
```

> Tên user phải **giống nhau** trên tất cả VM.

#### Bước 2 — Tạo SSH key trên node01 (master)

```bash
# Chuyển sang user mpiuser trên node01
su - mpiuser

# Tạo SSH key — nhấn Enter 3 lần, không đặt passphrase
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa

# Kiểm tra
ls ~/.ssh/
# Phải thấy: id_rsa  id_rsa.pub
```

#### Bước 3 — Copy public key sang tất cả nodes (kể cả chính node01)

```bash
# Vẫn trong session mpiuser trên node01
ssh-copy-id mpiuser@node01   # Localhost cũng cần — MPI đôi khi SSH vào chính nó
ssh-copy-id mpiuser@node02
ssh-copy-id mpiuser@node03
# Lệnh này hỏi password mpiuser trên node đích — đây là lần cuối cùng cần nhập
```

#### Bước 4 — Kiểm tra SSH không password

```bash
# Từ node01, SSH sang từng node — KHÔNG được hỏi password
ssh mpiuser@node02 "hostname && echo SSH OK"
ssh mpiuser@node03 "hostname && echo SSH OK"

# Kết quả mong đợi:
# node02
# SSH OK
```

Nếu vẫn hỏi password, kiểm tra trên node worker:

```bash
# Trên node02, node03, ...
ls -la ~/.ssh/authorized_keys
# Phải tồn tại và chứa public key của node01

chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

#### Bước 5 (tuỳ chọn) — Bỏ qua host key check

Thêm vào `~/.ssh/config` trên node01 để tránh hỏi "Are you sure?" lần đầu:

```bash
cat >> ~/.ssh/config << 'EOF'
Host node01 node02 node03 node04
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
```

---

### 6.4. Cấu hình NFS chia sẻ thư mục code

NFS cho phép tất cả worker đọc binary từ node01, không cần copy thủ công.
Build 1 lần trên node01 → tất cả nodes dùng được ngay.

#### Trên node01 — Cài và cấu hình NFS server

```bash
sudo apt install -y nfs-kernel-server

# Tạo thư mục project
mkdir -p /home/mpiuser/bfs-project
sudo chown -R mpiuser:mpiuser /home/mpiuser

# Thêm vào /etc/exports
echo "/home/mpiuser  192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)" \
    | sudo tee -a /etc/exports

# Apply và khởi động
sudo exportfs -a
sudo exportfs -v    # Kiểm tra đang export gì
sudo systemctl enable nfs-kernel-server
sudo systemctl start nfs-kernel-server
sudo systemctl status nfs-kernel-server  # Phải thấy "active (running)"
```

Giải thích các option trong `/etc/exports`:
- `rw` — cho phép đọc và ghi
- `sync` — ghi đồng bộ, an toàn hơn
- `no_subtree_check` — tắt subtree check, tăng performance
- `no_root_squash` — root trên client giữ quyền root

#### Trên node02, node03, ... — Mount NFS

```bash
# Tạo mount point cùng đường dẫn với node01
sudo mkdir -p /home/mpiuser

# Mount thử
sudo mount 192.168.1.101:/home/mpiuser /home/mpiuser

# Kiểm tra
df -h | grep mpiuser
ls /home/mpiuser/bfs-project/   # Phải thấy file từ node01
```

#### Cấu hình mount tự động khi khởi động

```bash
# Thêm vào /etc/fstab trên từng worker
echo "192.168.1.101:/home/mpiuser  /home/mpiuser  nfs  defaults,_netdev  0  0" \
    | sudo tee -a /etc/fstab

# Test fstab
sudo umount /home/mpiuser
sudo mount -a               # Mount lại theo fstab
df -h | grep mpiuser        # Phải mount được
```

---

### 6.5. Cấu hình /etc/hosts

Thêm hostname của tất cả nodes vào `/etc/hosts` trên **TỪNG VM**
để dùng hostname thay IP:

```bash
sudo tee -a /etc/hosts << 'EOF'
192.168.1.101   node01
192.168.1.102   node02
192.168.1.103   node03
192.168.1.104   node04
EOF
```

Kiểm tra:

```bash
ping -c 3 node02    # Từ node01
ping -c 3 node01    # Từ node02
```

---

### 6.6. Tạo hostfile

Trên **node01**, tạo file `~/bfs-project/hostfile`:

```bash
su - mpiuser
nano ~/bfs-project/hostfile
```

Nội dung cho 2 node:

```
node01 slots=4
node02 slots=4
```

Nội dung cho 4 node:

```
node01 slots=4
node02 slots=4
node03 slots=4
node04 slots=4
```

Giải thích:
- `slots=N` — số MPI process tối đa được spawn trên node đó
- Thường đặt bằng số vCPU của VM: `nproc` để kiểm tra

Ví dụ nếu các máy có số core khác nhau:

```
node01 slots=8    # máy 8-core
node02 slots=4    # máy 4-core
node03 slots=4    # máy 4-core
# Tổng: -np tối đa 16
```

---

### 6.7. Test kết nối cluster

Trước khi chạy BFS, kiểm tra MPI có spawn đúng trên các node không:

```bash
# Test: mỗi process in hostname của nó
mpirun -np 6 --hostfile hostfile hostname

# Kết quả mong đợi (thứ tự có thể khác):
# node01
# node01
# node02
# node02
# node03
# node03
```

```bash
# Test hello world MPI + OpenMP
cat > /tmp/hello_mpi.c << 'EOF'
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size; char host[256];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(host, sizeof(host));
    #pragma omp parallel
    {
        #pragma omp critical
        printf("Rank %d/%d | Host: %s | Thread %d/%d\n",
               rank, size, host,
               omp_get_thread_num(), omp_get_num_threads());
    }
    MPI_Finalize();
    return 0;
}
EOF
mpicc -fopenmp /tmp/hello_mpi.c -o /tmp/hello_mpi

OMP_NUM_THREADS=2 mpirun -np 6 --hostfile hostfile /tmp/hello_mpi

# Kết quả mong đợi:
# Rank 0/6 | Host: node01 | Thread 0/2
# Rank 0/6 | Host: node01 | Thread 1/2
# Rank 2/6 | Host: node02 | Thread 0/2
# ...
```

Nếu thấy tất cả nodes xuất hiện → cluster sẵn sàng.

---

### 6.8. Chạy chương trình trên cluster

#### Quy trình đầy đủ từ đầu

```bash
# 1. Trên node01 — vào thư mục project
su - mpiuser
cd ~/bfs-project

# 2. Build (chỉ cần làm trên node01, NFS tự sync sang worker)
make

# 3. Chạy sequential lấy baseline
./bfs_seq 4000000 16 42

# 4. Chạy BFS hybrid trên cluster
# 2 node, 8 rank (4/node), mỗi rank 2 thread → 16 core tổng
OMP_NUM_THREADS=2 mpirun -np 8 --hostfile hostfile \
    ./bfs_hybrid 4000000 16 42

# 5. Kèm --verify để kiểm tra kết quả đúng/sai
OMP_NUM_THREADS=2 mpirun -np 8 --hostfile hostfile \
    ./bfs_hybrid 1000000 16 42 --verify

# 6. Scale lên 4 node, graph lớn hơn
OMP_NUM_THREADS=2 mpirun -np 16 --hostfile hostfile \
    ./bfs_hybrid 16000000 16 42
```

#### Các tuỳ chọn hữu ích của mpirun

```bash
# 1 process mỗi node (test topology)
mpirun -np 3 --hostfile hostfile \
    --map-by node ./bfs_hybrid 1000000 16 42

# Bind process theo core (tăng performance)
mpirun -np 8 --hostfile hostfile \
    --bind-to core ./bfs_hybrid 4000000 16 42

# Xem chi tiết process được spawn ở đâu
mpirun -np 8 --hostfile hostfile \
    --display-map ./bfs_hybrid 1000000 16 42

# Verbose output của MPI runtime
mpirun -np 8 --hostfile hostfile \
    -v ./bfs_hybrid 1000000 16 42

# Ép chạy dù không đủ slot (khi test oversubscribe)
mpirun -np 8 --hostfile hostfile \
    --oversubscribe ./bfs_hybrid 1000000 16 42
```

#### Benchmark tự động

```bash
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh
# Kết quả lưu tại: results/benchmark_<timestamp>.csv
```

---

## 7. Tham số và biến môi trường

### Tham số chương trình

| Tham số | Ý nghĩa | Ví dụ |
|---------|---------|-------|
| `num_vertices` | Số đỉnh đồ thị (làm tròn lên lũy thừa 2) | `1000000` → thực tế `1048576` |
| `scale_factor` | Bậc trung bình mỗi đỉnh (~mật độ cạnh) | `16` |
| `seed` | Random seed để tái hiện kết quả | `42` |
| `--verify` | So sánh kết quả với BFS tuần tự | (flag) |

### Biến môi trường

| Biến | Ý nghĩa | Gợi ý |
|------|---------|-------|
| `OMP_NUM_THREADS` | Số OpenMP thread mỗi MPI rank | `= số vCPU / số rank trên node` |

### Hằng số thuật toán (trong `bfs_hybrid.h`)

| Hằng | Giá trị | Ý nghĩa |
|------|---------|---------|
| `BFS_ALPHA` | `14` | Ngưỡng chuyển sang bottom-up |
| `BFS_BETA`  | `24` | Ngưỡng quay lại top-down |

---

## 8. Kết quả mẫu

### BFS tuần tự

```
[SEQ] Generating R-MAT graph: 500000 vertices, scale=16, seed=42
[SEQ] Graph generated in 4868.78 ms

[SEQ] BFS from source vertex 42
[SEQ] Graph: 524288 vertices, 15480484 edges
[SEQ] Visited: 335349 vertices, 15480274 edges
[SEQ] Time: 42.34 ms
[SEQ] MTEPS: 365.65
```

### BFS hybrid (4 rank × 2 thread = 8 cores)

```
[HYBRID] ============================================
[HYBRID] BFS Hybrid (MPI + OpenMP + Direction-Opt)
[HYBRID] MPI ranks   : 4
[HYBRID] OMP threads : 2 per rank
[HYBRID] Total cores : 8
[HYBRID] Alpha       : 14
[HYBRID] Beta        : 24
[HYBRID] ============================================

[HYBRID] Generating R-MAT graph: 500000 vertices, scale=16, seed=42
[HYBRID] BFS from source vertex 42

[HYBRID]   Level  1: top-down  (frontier=1,      fe=9888,     ue=15480484)
[HYBRID]   Level  2: bottom-up (frontier=2472,   fe=10882620, ue=15470596)
[HYBRID]   Level  3: bottom-up (frontier=224067, fe=50015036, ue=2966837)
[HYBRID]   Level  4: bottom-up (frontier=108000, fe=1010272,  ue=2714269)
[HYBRID]   Level  5: top-down  (frontier=808,    fe=3276,     ue=2713450)
[HYBRID]   Level  6: top-down  (frontier=1,      fe=4,        ue=2710174)

[HYBRID] ============================================
[HYBRID] Graph    : 524288 vertices, 15480484 edges
[HYBRID] Visited  : 335349 vertices, 12770314 edges
[HYBRID] Levels   : 6
[HYBRID] Time     : 40.62 ms
[HYBRID] MTEPS    : 314.35

[HYBRID] Sequential : 69.14 ms | MTEPS: 223.90
[HYBRID] Speedup    : 1.70x
[HYBRID] Verify     : PASSED ✓
[HYBRID] ============================================
```

### Giải thích output

| Trường | Ý nghĩa |
|--------|---------|
| `top-down` / `bottom-up` | Hướng duyệt BFS của level đó |
| `frontier` | Số đỉnh đang trong frontier |
| `fe` | Frontier edges — tổng số cạnh của các đỉnh frontier |
| `ue` | Unvisited edges — số cạnh chưa được duyệt còn lại |
| `MTEPS` | Mega Traversed Edges Per Second — thước đo chuẩn Graph500 |
| `Speedup` | Thời gian sequential / thời gian hybrid |
| `Verify` | PASSED = kết quả hybrid giống hệt sequential |

---

## 9. Xử lý lỗi thường gặp

**`ssh: connect to host node02 port 22: Connection refused`**
```bash
# Kiểm tra SSH service trên node worker
sudo systemctl status ssh
sudo systemctl enable --now ssh
```

**`Permission denied (publickey)`**
```bash
# Key chưa được copy đúng, làm lại bước 6.3
ssh-copy-id mpiuser@node02

# Hoặc kiểm tra trực tiếp trên worker
cat ~/.ssh/authorized_keys   # Phải thấy public key của node01

# Kiểm tra quyền thư mục
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

**`There are not enough slots available`**
```bash
# Tăng slots trong hostfile, hoặc giảm -np
# Ví dụ: đang chạy -np 8 nhưng hostfile chỉ có 2 node × slots=2 = 4 slots

# Cách 1: Sửa hostfile
node01 slots=4    # tăng lên
node02 slots=4

# Cách 2: Thêm --oversubscribe (dùng khi test, không dùng production)
mpirun -np 8 --hostfile hostfile --oversubscribe ./bfs_hybrid ...
```

**`No such file or directory` khi chạy trên worker**
```bash
# NFS chưa mount hoặc mount sai đường dẫn
# Kiểm tra trên worker:
ls ~/bfs-project/bfs_hybrid

# Nếu không thấy → mount lại NFS (bước 6.4)
sudo mount 192.168.1.101:/home/mpiuser /home/mpiuser
```

**IP VM thay đổi sau khi restart**
```bash
# Đặt IP tĩnh trong /etc/netplan/00-installer-config.yaml
sudo nano /etc/netplan/00-installer-config.yaml
```
```yaml
network:
  version: 2
  ethernets:
    enp0s3:                        # Tên card — dùng `ip addr` để xem
      dhcp4: no
      addresses: [192.168.1.101/24]  # Đổi theo từng node
      routes:
        - to: default
          via: 192.168.1.1         # Gateway router
      nameservers:
        addresses: [8.8.8.8]
```
```bash
sudo netplan apply
```

**Verify FAILED — kết quả sai**
```bash
# Chạy với graph nhỏ để debug
OMP_NUM_THREADS=1 mpirun -np 2 ./bfs_hybrid 10000 8 42 --verify
```

**Kiểm tra version OpenMPI nhất quán**
```bash
# Chạy trên từng node — phải ra cùng version
mpirun --version
# mpirun (Open MPI) 4.1.6   ← phải giống nhau
```

---

## 10. Các hàm MPI được dùng

| Hàm | Mục đích |
|-----|----------|
| `MPI_Init_thread` | Khởi tạo MPI với hỗ trợ thread (`MPI_THREAD_FUNNELED`) |
| `MPI_Comm_rank` | Lấy rank (ID) của process hiện tại |
| `MPI_Comm_size` | Lấy tổng số MPI process đang chạy |
| `MPI_Barrier` | Đồng bộ — tất cả rank phải đến đây mới tiếp tục |
| `MPI_Allreduce (MAX)` | Merge dist[] toàn cục cuối mỗi level BFS |
| `MPI_Allreduce (BOR)` | Merge frontier bitmap giữa các rank |
| `MPI_Abort` | Dừng khẩn cấp tất cả process khi có lỗi nghiêm trọng |
| `MPI_Finalize` | Kết thúc MPI, giải phóng tài nguyên |