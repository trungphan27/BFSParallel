# Hướng Dẫn Setup Cluster MPI + OpenMP cho BFS Project
## VirtualBox + Ubuntu 24.04 + Bridge Adapter

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Tạo và cấu hình VM](#2-tạo-và-cấu-hình-vm)
3. [Cài đặt phần mềm trên mỗi VM](#3-cài-đặt-phần-mềm-trên-mỗi-vm)
4. [Cấu hình mạng Bridge Adapter](#4-cấu-hình-mạng-bridge-adapter)
5. [Cấu hình SSH không cần mật khẩu](#5-cấu-hình-ssh-không-cần-mật-khẩu)
6. [Cấu hình MPI hostfile](#6-cấu-hình-mpi-hostfile)
7. [Kiểm tra cluster hoạt động](#7-kiểm-tra-cluster-hoạt-động)
8. [Chạy chương trình BFS trên 1 máy](#8-chạy-chương-trình-bfs-trên-1-máy)
9. [Chạy chương trình BFS trên nhiều máy](#9-chạy-chương-trình-bfs-trên-nhiều-máy)
10. [Cấu trúc project và Makefile](#10-cấu-trúc-project-và-makefile)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Yêu cầu hệ thống

### Máy host
- VirtualBox 7.x trở lên
- RAM tối thiểu 8GB (để chạy 2–4 VM cùng lúc)
- Kết nối mạng LAN (Ethernet hoặc WiFi)

### Mỗi VM
| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| OS         | Ubuntu 24.04 LTS | Ubuntu 24.04 LTS |
| RAM        | 1GB | 2GB |
| CPU        | 2 vCPU | 4 vCPU |
| Disk       | 15GB | 20GB |
| Network    | Bridge Adapter | Bridge Adapter |

### Quy ước đặt tên trong guide này
| VM | Hostname | IP (ví dụ) | Vai trò |
|----|----------|------------|---------|
| VM1 | `node01` | `192.168.1.101` | Master (rank 0) |
| VM2 | `node02` | `192.168.1.102` | Worker |
| VM3 | `node03` | `192.168.1.103` | Worker |
| VM4 | `node04` | `192.168.1.104` | Worker |

> **Lưu ý:** IP thực tế phụ thuộc vào mạng LAN của bạn. Dùng `ip addr` để kiểm tra sau khi cài xong.

---

## 2. Tạo và cấu hình VM

### 2.1. Tạo VM đầu tiên (node01)

1. Mở VirtualBox → **New**
2. Cấu hình:
   - Name: `node01`
   - Type: `Linux`, Version: `Ubuntu (64-bit)`
   - RAM: `2048 MB`
   - vCPU: `2–4` (Settings → System → Processor)
   - Disk: `20 GB` (VDI, dynamically allocated)

3. Cài Ubuntu 24.04 LTS từ ISO:
   - Trong quá trình cài: chọn **Minimal installation**
   - Đặt username: `mpiuser` (dùng thống nhất trên tất cả node)
   - Đặt hostname: `node01`

### 2.2. Nhân bản VM (Clone) cho node02, node03, node04

Sau khi cài xong node01 và cấu hình xong phần mềm (mục 3):

1. Tắt VM node01
2. Chuột phải vào node01 → **Clone**
   - Name: `node02`
   - MAC Address Policy: **Generate new MAC addresses for all network adapters** ✅
   - Clone type: **Full clone**
3. Lặp lại cho node03, node04

4. Sau khi clone, bật từng VM và đổi hostname:
```bash
# Trên node02
sudo hostnamectl set-hostname node02
sudo reboot

# Trên node03
sudo hostnamectl set-hostname node03
sudo reboot
```

---

## 3. Cài đặt phần mềm trên mỗi VM

Thực hiện trên **tất cả** các VM (node01 → node04):

```bash
# Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y

# Cài OpenMPI và development tools
sudo apt install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    openssh-server \
    openssh-client \
    gcc \
    make \
    git \
    htop \
    net-tools

# Kiểm tra phiên bản MPI
mpirun --version
mpicc --version

# Kiểm tra OpenMP (đã có sẵn với GCC)
gcc --version
echo "#include <omp.h>
int main() { return 0; }" | gcc -fopenmp -x c - -o /tmp/test_omp && echo "OpenMP OK"
```

### Cài đặt NFS để chia sẻ thư mục project (tùy chọn nhưng tiện)

**Trên node01 (NFS server):**
```bash
sudo apt install -y nfs-kernel-server

# Tạo thư mục chia sẻ
mkdir -p /home/mpiuser/bfs-project
sudo chown mpiuser:mpiuser /home/mpiuser/bfs-project

# Cấu hình export
echo "/home/mpiuser/bfs-project *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

**Trên node02, node03, node04 (NFS client):**
```bash
sudo apt install -y nfs-common

# Mount thư mục từ node01
mkdir -p /home/mpiuser/bfs-project
sudo mount 192.168.1.101:/home/mpiuser/bfs-project /home/mpiuser/bfs-project

# Mount tự động khi khởi động (thêm vào /etc/fstab)
echo "192.168.1.101:/home/mpiuser/bfs-project /home/mpiuser/bfs-project nfs defaults 0 0" | sudo tee -a /etc/fstab
```

> Nếu không dùng NFS, bạn cần copy source code lên từng node thủ công hoặc dùng `scp`.

---

## 4. Cấu hình mạng Bridge Adapter

### 4.1. Cài đặt trong VirtualBox

1. Chọn VM → **Settings** → **Network**
2. Adapter 1:
   - **Enable Network Adapter**: ✅
   - Attached to: **Bridged Adapter**
   - Name: chọn card mạng vật lý đang dùng (VD: `Intel(R) Ethernet...` hoặc `Wi-Fi`)
3. Nhấn OK → Khởi động VM

### 4.2. Kiểm tra IP sau khi boot

```bash
ip addr show
# Tìm dòng có inet, ví dụ: inet 192.168.1.101/24
```

### 4.3. Đặt IP tĩnh (khuyến nghị để tránh IP thay đổi)

Chỉnh file `/etc/netplan/00-installer-config.yaml`:

```yaml
network:
  version: 2
  ethernets:
    enp0s3:           # Tên card mạng, có thể khác (dùng `ip addr` để xem)
      dhcp4: no
      addresses:
        - 192.168.1.101/24   # Đổi thành IP tương ứng mỗi node
      routes:
        - to: default
          via: 192.168.1.1   # Gateway của router
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

```bash
sudo netplan apply
```

> Đặt IP tương ứng: node01=`.101`, node02=`.102`, node03=`.103`, node04=`.104`

### 4.4. Thêm vào /etc/hosts trên tất cả các node

```bash
sudo tee -a /etc/hosts << 'EOF'
192.168.1.101   node01
192.168.1.102   node02
192.168.1.103   node03
192.168.1.104   node04
EOF
```

Kiểm tra ping giữa các node:
```bash
ping -c 3 node02
ping -c 3 node03
```

---

## 5. Cấu hình SSH không cần mật khẩu

MPI cần SSH để spawn process trên các node. Phải cấu hình **passwordless SSH** từ node01 đến tất cả node còn lại.

### 5.1. Tạo SSH key trên node01

```bash
# Trên node01, đăng nhập với user mpiuser
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
```

### 5.2. Copy public key đến tất cả các node

```bash
# Từ node01, copy key đến từng node (nhập mật khẩu lần đầu)
ssh-copy-id mpiuser@node01   # Cả node01 cũng cần (localhost)
ssh-copy-id mpiuser@node02
ssh-copy-id mpiuser@node03
ssh-copy-id mpiuser@node04
```

### 5.3. Kiểm tra SSH không cần mật khẩu

```bash
# Từ node01, SSH vào từng node - không được hỏi mật khẩu
ssh mpiuser@node02 hostname
ssh mpiuser@node03 hostname
ssh mpiuser@node04 hostname
```

Output mong đợi:
```
node02
node03
node04
```

### 5.4. Cấu hình SSH client (tùy chọn, bỏ qua host key check)

Thêm vào `~/.ssh/config` trên node01:

```
Host node01 node02 node03 node04
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

---

## 6. Cấu hình MPI hostfile

Tạo file `~/bfs-project/hostfile` trên node01:

### Chạy trên 2 node (node01 + node02)
```
node01 slots=4
node02 slots=4
```

### Chạy trên 4 node
```
node01 slots=4
node02 slots=4
node03 slots=4
node04 slots=4
```

> `slots` = số MPI process tối đa trên mỗi node. Đặt bằng số vCPU của VM.

---

## 7. Kiểm tra cluster hoạt động

### 7.1. Test MPI hello world

Tạo file `hello_mpi.c`:

```c
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        #pragma omp critical
        printf("MPI rank %d/%d | Host: %s | Thread %d/%d\n",
               rank, size, hostname, tid, nthreads);
    }

    MPI_Finalize();
    return 0;
}
```

```bash
mpicc -fopenmp hello_mpi.c -o hello_mpi
```

### 7.2. Chạy trên 1 node (local)

```bash
# 4 MPI process, mỗi process 2 OpenMP thread
OMP_NUM_THREADS=2 mpirun -np 4 ./hello_mpi
```

Output mong đợi:
```
MPI rank 0/4 | Host: node01 | Thread 0/2
MPI rank 0/4 | Host: node01 | Thread 1/2
MPI rank 1/4 | Host: node01 | Thread 0/2
...
```

### 7.3. Chạy trên nhiều node

```bash
# 8 MPI process trên 2 node (4 process/node), mỗi process 2 thread
OMP_NUM_THREADS=2 mpirun -np 8 --hostfile hostfile ./hello_mpi
```

Output mong đợi:
```
MPI rank 0/8 | Host: node01 | Thread 0/2
MPI rank 4/8 | Host: node02 | Thread 0/2
...
```

---

## 8. Chạy chương trình BFS trên 1 máy

### 8.1. Compile

```bash
cd ~/bfs-project
make
```

Makefile sẽ biên dịch ra 2 binary:
- `bfs_seq` – BFS tuần tự (baseline)
- `bfs_hybrid` – Full hybrid (MPI + OpenMP + direction-optimizing)

### 8.2. Chạy BFS sequential (baseline)

```bash
# Cú pháp: ./bfs_seq <số_đỉnh> <scale_factor> <seed>
./bfs_seq 1000000 16 42
```

Output:
```
[SEQ] Graph: 1000000 vertices, ~16000000 edges
[SEQ] BFS from source 0
[SEQ] Time: 1234.56 ms
[SEQ] MTEPS: 12.96
```

### 8.3. Chạy BFS hybrid trên 1 node (shared memory only)

```bash
# 1 MPI process, 4 OpenMP thread
OMP_NUM_THREADS=4 mpirun -np 1 ./bfs_hybrid 1000000 16 42
```

Output:
```
[HYBRID] Graph: 1000000 vertices, ~16000000 edges
[HYBRID] Nodes: 1, Threads/node: 4
[HYBRID] BFS from source 0
[HYBRID] Levels: 23
[HYBRID]   Level  0: top-down  (frontier=1)
[HYBRID]   Level  1: top-down  (frontier=12)
[HYBRID]   Level  5: SWITCH -> bottom-up (frontier=189432)
[HYBRID]   Level 18: SWITCH -> top-down  (frontier=521)
[HYBRID] Time: 234.12 ms
[HYBRID] Speedup vs sequential: 5.27x
[HYBRID] MTEPS: 68.34
```

---

## 9. Chạy chương trình BFS trên nhiều máy

### 9.1. Đảm bảo source code có mặt trên tất cả node

**Cách 1 (khuyến nghị): Dùng NFS** – chỉ cần compile trên node01, tất cả node đều thấy binary qua NFS mount.

**Cách 2: Copy thủ công:**
```bash
# Từ node01, copy binary đến các node
scp bfs_hybrid mpiuser@node02:~/bfs-project/
scp bfs_hybrid mpiuser@node03:~/bfs-project/
scp bfs_hybrid mpiuser@node04:~/bfs-project/
```

### 9.2. Chạy trên 2 node

```bash
# 8 MPI process (4/node), mỗi process 2 OpenMP thread
OMP_NUM_THREADS=2 mpirun -np 8 \
    --hostfile hostfile \
    ./bfs_hybrid 4000000 16 42
```

### 9.3. Chạy trên 4 node

```bash
# 16 MPI process (4/node), mỗi process 2 OpenMP thread
OMP_NUM_THREADS=2 mpirun -np 16 \
    --hostfile hostfile \
    ./bfs_hybrid 16000000 16 42
```

### 9.4. Script benchmark tự động

Tạo file `scripts/run_benchmark.sh`:

```bash
#!/bin/bash
# run_benchmark.sh - Chạy benchmark và xuất kết quả

BINARY=./bfs_hybrid
SEQ_BINARY=./bfs_seq
HOSTFILE=./hostfile
SCALE=16
SEED=42
OUTPUT=results/benchmark_$(date +%Y%m%d_%H%M%S).csv

mkdir -p results
echo "mode,nodes,mpi_procs,omp_threads,vertices,time_ms,mteps,speedup" > $OUTPUT

echo "=== Chạy Sequential baseline ==="
for V in 500000 1000000 2000000; do
    TIME=$($SEQ_BINARY $V $SCALE $SEED | grep "Time:" | awk '{print $2}')
    MTEPS=$($SEQ_BINARY $V $SCALE $SEED | grep "MTEPS:" | awk '{print $2}')
    echo "sequential,1,1,1,$V,$TIME,$MTEPS,1.0" >> $OUTPUT
    echo "  V=$V | Time=${TIME}ms | MTEPS=$MTEPS"
done

echo ""
echo "=== Chạy Hybrid (1 node) ==="
for V in 500000 1000000 2000000; do
    for T in 1 2 4; do
        TIME=$(OMP_NUM_THREADS=$T mpirun -np 1 $BINARY $V $SCALE $SEED | grep "Time:" | awk '{print $2}')
        MTEPS=$(OMP_NUM_THREADS=$T mpirun -np 1 $BINARY $V $SCALE $SEED | grep "MTEPS:" | awk '{print $2}')
        echo "hybrid,1,1,$T,$V,$TIME,$MTEPS,-" >> $OUTPUT
        echo "  V=$V | Threads=$T | Time=${TIME}ms | MTEPS=$MTEPS"
    done
done

echo ""
echo "=== Chạy Hybrid (multi-node) ==="
for NODES in 2 4; do
    NP=$((NODES * 4))
    for V in 1000000 4000000; do
        TIME=$(OMP_NUM_THREADS=2 mpirun -np $NP --hostfile $HOSTFILE $BINARY $V $SCALE $SEED | grep "Time:" | awk '{print $2}')
        MTEPS=$(OMP_NUM_THREADS=2 mpirun -np $NP --hostfile $HOSTFILE $BINARY $V $SCALE $SEED | grep "MTEPS:" | awk '{print $2}')
        echo "hybrid,$NODES,$NP,2,$V,$TIME,$MTEPS,-" >> $OUTPUT
        echo "  Nodes=$NODES | V=$V | Time=${TIME}ms | MTEPS=$MTEPS"
    done
done

echo ""
echo "=== Kết quả lưu tại: $OUTPUT ==="
cat $OUTPUT
```

```bash
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh
```

---

## 10. Cấu trúc project và Makefile

```
bfs-project/
├── src/
│   ├── main_seq.c          # Entry point BFS sequential
│   ├── main_hybrid.c       # Entry point BFS hybrid
│   ├── bfs_sequential.c    # Thuật toán BFS tuần tự
│   ├── bfs_sequential.h
│   ├── bfs_hybrid.c        # Direction-optimizing + MPI + OpenMP
│   ├── bfs_hybrid.h
│   ├── graph.c             # R-MAT generator, CSR format
│   ├── graph.h
│   ├── utils.c             # Timer, metrics (MTEPS)
│   └── utils.h
├── scripts/
│   └── run_benchmark.sh
├── results/                # CSV output
├── hostfile                # MPI hostfile
└── Makefile
```

### Makefile

```makefile
CC      = mpicc
CFLAGS  = -O2 -fopenmp -Wall
LDFLAGS = -lm

SEQ_SRC  = src/main_seq.c src/bfs_sequential.c src/graph.c src/utils.c
HYB_SRC  = src/main_hybrid.c src/bfs_hybrid.c src/graph.c src/utils.c

all: bfs_seq bfs_hybrid

bfs_seq: $(SEQ_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bfs_hybrid: $(HYB_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f bfs_seq bfs_hybrid results/*.csv

.PHONY: all clean
```

---

## 11. Troubleshooting

### Lỗi: "ssh: connect to host node02 port 22: Connection refused"
```bash
# Kiểm tra SSH service trên node02
sudo systemctl status ssh
sudo systemctl start ssh
sudo systemctl enable ssh
```

### Lỗi: "There are not enough slots available"
```bash
# Tăng slots trong hostfile, hoặc thêm --oversubscribe
mpirun -np 8 --oversubscribe --hostfile hostfile ./bfs_hybrid ...
```

### Lỗi: MPI process không tìm thấy binary trên node khác
```bash
# Kiểm tra binary có mặt trên tất cả node (nếu không dùng NFS)
ssh node02 ls ~/bfs-project/bfs_hybrid
# Nếu không có → copy lại (mục 9.1)
```

### Lỗi: IP VM thay đổi sau khi restart
```bash
# Dùng IP tĩnh (mục 4.3), hoặc tạm thời update /etc/hosts
ip addr show   # Xem IP hiện tại
sudo nano /etc/hosts   # Sửa lại IP mới
```

### Kiểm tra OpenMPI version nhất quán giữa các node
```bash
# Chạy trên từng node
mpirun --version
# Phải giống nhau, ví dụ: "Open MPI 4.1.x"
```

### Xem log chi tiết khi MPI gặp lỗi
```bash
mpirun -np 8 --hostfile hostfile --mca btl_base_verbose 30 ./bfs_hybrid ...
```

---

## Tóm tắt nhanh

| Tình huống | Lệnh |
|------------|------|
| Compile | `make` |
| Test 1 node sequential | `./bfs_seq 1000000 16 42` |
| Test 1 node hybrid | `OMP_NUM_THREADS=4 mpirun -np 1 ./bfs_hybrid 1000000 16 42` |
| Test 2 node hybrid | `OMP_NUM_THREADS=2 mpirun -np 8 --hostfile hostfile ./bfs_hybrid 1000000 16 42` |
| Test 4 node hybrid | `OMP_NUM_THREADS=2 mpirun -np 16 --hostfile hostfile ./bfs_hybrid 4000000 16 42` |
| Benchmark đầy đủ | `./scripts/run_benchmark.sh` |