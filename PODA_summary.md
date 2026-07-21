# PØDA: Prompt-driven Zero-shot Domain Adaptation - Hướng Dẫn Chi Tiết

Tài liệu này trình bày lại một cách toàn diện và chi tiết nhất về bài báo "PØDA: Prompt-driven Zero-shot Domain Adaptation". Nội dung bao gồm việc phân tích vấn đề, phương pháp tiếp cận, công thức toán học, thuật toán, kết quả thực nghiệm và các phân tích chuyên sâu.

## 1. Giới thiệu (Introduction)

Những năm gần đây, các phương pháp phân đoạn ngữ nghĩa (semantic segmentation) có giám sát đã đạt được thành công to lớn. Tuy nhiên, khi áp dụng các mô hình này vào môi trường thực tế với dữ liệu nằm ngoài phân phối huấn luyện (out-of-distribution data) - hay còn gọi là sự thay đổi miền (domain-shift), hiệu suất thường sụt giảm nghiêm trọng.

**Unsupervised Domain Adaptation (UDA)** nổi lên như một giải pháp, nhưng nó yêu cầu dữ liệu có nhãn từ miền nguồn (source domain) và dữ liệu không nhãn từ miền đích (target domain). Mặc dù có vẻ đơn giản, nhưng việc thu thập dữ liệu không nhãn ở một số điều kiện là rất khó khăn (ví dụ: lái xe trong bão tuyết, bão cát). Hơn nữa, trong môi trường công nghiệp, việc sử dụng dữ liệu công cộng từ Internet thường bị giới hạn.

Để giảm bớt gánh nặng thu thập dữ liệu miền đích, các nghiên cứu gần đây hướng tới **One-shot UDA** (chỉ sử dụng một hình ảnh miền đích). 

Bài báo này tiến xa hơn một bước bằng cách giới thiệu một nhiệm vụ hoàn toàn mới và đầy thách thức: **Prompt-driven Zero-shot Domain Adaptation (PØDA)**. 

### Mục tiêu của PØDA

PØDA nhằm mục đích điều chỉnh (adapt) một mô hình phân đoạn đã được huấn luyện trên miền nguồn sang một miền đích mà **KHÔNG CẦN BẤT KỲ HÌNH ẢNH NÀO** từ miền đích đó. Thay vào đó, nó chỉ sử dụng một **mô tả bằng văn bản tự nhiên** (prompt) về miền đích (ví dụ: "driving at night" - lái xe vào ban đêm, "driving in snow" - lái xe trong tuyết).

**Figure 1** trong bài báo minh họa rõ ràng mục tiêu này: PØDA cho phép một mô hình phân đoạn (DeepLabv3+ huấn luyện trên Cityscapes) thích ứng với các điều kiện chưa từng thấy như ban đêm, lửa, sương mù, bão cát chỉ thông qua các prompt. Kết quả phân đoạn tốt hơn đáng kể so với mô hình chỉ dùng nguồn (source-only).

### Đóng góp chính (Contributions)

1.  **Nhiệm vụ mới**: Giới thiệu prompt-driven zero-shot domain adaptation, một thiết lập thiết thực hơn UDA truyền thống.
2.  **Phương pháp mới**: Trực tiếp điều chỉnh/thao tác (manipulate) trên các đặc trưng sâu (deep features) thay vì ở không gian pixel. Đề xuất lớp **Prompt-driven Instance Normalization (PIN)** đơn giản nhưng hiệu quả để tăng cường (augment) đặc trưng nguồn sao cho biểu diễn của chúng trong không gian CLIP khớp với biểu diễn của prompt miền đích.
3.  **Khả năng thích ứng đa dạng**: Chứng minh sự thành công trên nhiều điều kiện (thời tiết khắc nghiệt, chuyển đổi synthetic-to-real và ngược lại). PØDA thậm chí vượt trội hơn các phương pháp one-shot UDA hiện đại nhất (SOTA) mà không cần hình ảnh đích.
4.  **Tính linh hoạt**: Áp dụng thành công cho cả phát hiện đối tượng (object detection) và phân loại hình ảnh (image classification).

---

## 2. Các công trình liên quan (Related Works)

*   **Unsupervised Domain Adaptation (UDA)**: Các phương pháp truyền thống như adversarial learning, self-training, entropy minimization... thường giảm thiểu khoảng cách miền ở mức đầu vào, đặc trưng hoặc đầu ra. PØDA tập trung vào mức đặc trưng nhưng ở môi trường zero-shot.
*   **One-Shot Unsupervised Domain Adaptation (OSUDA)**: Các phương pháp như ASM (Adversarial Style Mining) hay SM-PPM (Style Mixing and Patch-wise Prototypical Matching) sử dụng 1 hình ảnh duy nhất để trích xuất phong cách. PØDA thách thức hơn khi không có hình ảnh nào cả.
*   **Zero-shot Domain Adaptation**: Lengyel et al. đề xuất một lớp tích chập không biến đổi màu sắc (CIConv) dựa trên các tiên đề vật lý. Cách tiếp cận này bị giới hạn ở một loại thay đổi miền cụ thể (ví dụ: ngày-đêm).
*   **Text-driven image synthesis**: Các công trình sử dụng CLIP để chỉnh sửa hình ảnh bằng văn bản (ví dụ: StyleCLIP, CLIPstyler). Tuy nhiên, chúng thường thay đổi trên không gian pixel và sử dụng quá trình tạo hình (generative process) phức tạp. PØDA khác biệt ở chỗ nó **trực tiếp thay đổi trên không gian đặc trưng**.

---

## 3. Thích ứng Zero-shot dựa trên Prompt (Prompt-driven Zero-shot Adaptation)

### 3.1. Tổng quan phương pháp (Method Overview)

Phương pháp của PØDA, được minh họa chi tiết trong **Figure 2**, dựa trên mô hình ngôn ngữ-hình ảnh CLIP. CLIP học một không gian biểu diễn chung mạnh mẽ giữa hai phương thức này.

**Ý tưởng cốt lõi**: Sử dụng CLIP để "lái" (steer) các đặc trưng từ bất kỳ hình ảnh miền nguồn nào hướng tới miền đích trong không gian tiềm ẩn (latent space) của CLIP, với sự chỉ dẫn từ một prompt ngẫu nhiên mô tả miền đích.

**Quy trình (Figure 2)**:
1.  *(Left)*: Học cách chuyển đổi phong cách đặc trưng cấp thấp (low-level feature) từ miền nguồn sang miền đích thông qua PIN. Quá trình này được hướng dẫn bằng cách đối chiếu (cosine distance) embedding của đặc trưng đã chuyển đổi với embedding của prompt.
2.  *(Middle)*: Thích ứng (Adaptation). Tinh chỉnh (fine-tune) mô hình phân đoạn (segmenter $M$) trên miền nguồn đã được tăng cường bằng các đặc trưng chuyển đổi (feature-augmented source domain).
3.  *(Right)*: Suy luận trên các miền chưa từng thấy bằng mô hình đã được điều chỉnh.

### 3.2. Trình bày bài toán (Problem Formulation)

Bài toán chính là phân đoạn ngữ nghĩa (phân loại pixel).
*   **Mô hình ($M$)**: Cấu trúc DeepLabv3+ với backbone $E_{img}$ là bộ mã hóa hình ảnh CLIP (ví dụ: ResNet-50) được đóng băng.
*   $M = (M_{feat}, M_{cls})$. Trong đó $M_{feat}$ là phần trích xuất đặc trưng (frozen) và $M_{cls}$ là phần phân loại (sẽ được fine-tune).
*   **Miền nguồn ($D_s$)**: Tập dữ liệu huấn luyện, $D_s = \{(x_s, y_s) \mid x_s \in \mathbb{R}^{H\times W \times 3}, y_s \in \{0,1\}^{H\times W \times K}\}$.
*   **Miền đích ($D_t$)**: Tập dữ liệu chưa từng thấy, chỉ được mô tả qua một prompt (TrgPrompt). Mục tiêu là cải thiện hiệu suất trên $D_t = \{x_t \mid x_t \in \mathbb{R}^{H\times W \times 3}\}$.

### 3.3. Tăng cường Đặc trưng Zero-shot (Zero-shot Feature Augmentation)

Thay vì chuyển đổi ở cấp độ pixel như các phương pháp khác, PØDA tập trung vào việc **khai phá phong cách (style mining)** trực tiếp trên đặc trưng.

PØDA định nghĩa tập đặc trưng cấp thấp từ ảnh nguồn: $\mathcal{F}_s = \{\mathbf{f}_s \mid \mathbf{f}_s = \text{feat-ext}(M_{feat}, \mathbf{x}_s)\}$. Trong bài báo, lớp `Layer1` của ResNet-50 thường được sử dụng.
Embedding của prompt mục tiêu: $\text{TrgEmb} = E_{txt}(\text{TrgPrompt})$.

**Prompt-driven Instance Normalization (PIN)**:

Lấy cảm hứng từ AdaIN (Adaptive Instance Normalization) - một phương pháp tinh tế để truyền các thành phần đặc tả phong cách qua các đặc trưng sâu. 
Công thức AdaIN cơ bản:
$$ \text{AdaIN}(\mathbf{f}_s, \mathbf{f}_t) = \sigma(\mathbf{f}_t) \left( \frac{\mathbf{f}_s - \mu(\mathbf{f}_s)}{\sigma(\mathbf{f}_s)} \right) + \mu(\mathbf{f}_t) \quad (1) $$
Với $\mu(\cdot)$ và $\sigma(\cdot)$ là giá trị trung bình (mean) và độ lệch chuẩn (standard deviation) theo kênh. $\mathbf{f}_t$ là đặc trưng phong cách mục tiêu.

Tuy nhiên, trong Zero-shot, ta không có hình ảnh mục tiêu nên không có $\mathbf{f}_t$. Do đó, PØDA đề xuất lớp **PIN**:

$$ \text{PIN}(\mathbf{f}_s, \boldsymbol{\mu}, \boldsymbol{\sigma}) = \boldsymbol{\sigma} \left( \frac{\mathbf{f}_s - \mu(\mathbf{f}_s)}{\sigma(\mathbf{f}_s)} \right) + \boldsymbol{\mu} \quad (2) $$

Trong đó, $\boldsymbol{\mu}$ và $\boldsymbol{\sigma}$ là **các biến số có thể tối ưu hóa** được dẫn dắt bởi prompt.

**Tối ưu hóa (Optimization)**:

Quá trình tối ưu hóa (khai phá phong cách) được mô tả trong **Algorithm 1** và **Figure 3**.

1.  Khởi tạo các biến phong cách: $\boldsymbol{\mu}^0 \leftarrow \text{mean}(\mathbf{f}_s)$ và $\boldsymbol{\sigma}^0 \leftarrow \text{std}(\mathbf{f}_s)$.
2.  Lặp lại $N$ lần (optimization steps):
    *   Tạo đặc trưng phong cách giả định: $\mathbf{f}^i_{s\rightarrow t} = \text{PIN}(\mathbf{f}_s, \boldsymbol{\mu}^{i-1}, \boldsymbol{\sigma}^{i-1})$
    *   Lấy embedding của đặc trưng giả định trong không gian CLIP: $\bar{\mathbf{f}}^i_{s\rightarrow t} = \text{get-embedding}(\mathbf{f}^i_{s\rightarrow t})$. Hàm `get-embedding` sử dụng lớp attention pooling của $E_{img}$.
    *   Tính toán hàm mất mát (loss): Tính theo khoảng cách cosine trong không gian latent của CLIP giữa embedding của đặc trưng giả định và embedding của prompt đích.
        $$ \mathcal{L}_{\boldsymbol{\mu}, \boldsymbol{\sigma}}(\bar{\mathbf{f}}_{s\rightarrow t}, \text{TrgEmb}) = 1 - \frac{\bar{\mathbf{f}}_{s\rightarrow t} \cdot \text{TrgEmb}}{\|\bar{\mathbf{f}}_{s\rightarrow t}\| \|\text{TrgEmb}\|} \quad (3) $$
    *   Cập nhật $\boldsymbol{\mu}$ và $\boldsymbol{\sigma}$ thông qua quá trình hạ gradient (gradient descent) để giảm thiểu hàm mất mát trên.

Kết quả cuối cùng của quá trình lặp này là một cặp $(\boldsymbol{\mu}_t, \boldsymbol{\sigma}_t)$ đại diện cho phong cách mục tiêu được khai phá từ ảnh nguồn tương ứng. Tập hợp tất cả các cặp này tạo thành $\mathcal{S}_{s\rightarrow t}$.

### 3.4. Tinh chỉnh để Thích ứng (Fine-tuning for Adaptation)

Quy trình tổng thể được mô tả trong **Algorithm 2**.

Sau khi có tập các phong cách mục tiêu $\mathcal{S}_{s\rightarrow t}$ (có kích thước bằng với kích thước tập dữ liệu nguồn $|\mathcal{D}_s|$), quá trình huấn luyện bắt đầu.

Ở mỗi vòng lặp huấn luyện:
1.  Lấy một ảnh nguồn $\mathbf{x}_s$ và trích xuất đặc trưng $\mathbf{f}_s$.
2.  Chọn ngẫu nhiên một phong cách $(\boldsymbol{\mu}_t, \boldsymbol{\sigma}_t)$ từ $\mathcal{S}_{s\rightarrow t}$.
3.  Áp dụng PIN để tạo đặc trưng tăng cường: $\mathbf{f}_{s\rightarrow t} = \text{PIN}(\mathbf{f}_s, \boldsymbol{\mu}_t, \boldsymbol{\sigma}_t)$.
4.  Truyền $\mathbf{f}_{s\rightarrow t}$ qua các lớp còn lại của $M_{feat}$ và sau đó vào $M_{cls}$.
5.  Do thao tác điều chỉnh phong cách (PIN) bảo toàn nội dung ngữ nghĩa (semantic-content), nhãn nguồn $\mathbf{y}_s$ vẫn có thể được sử dụng để huấn luyện. Hàm mất mát phân đoạn thông thường (Cross Entropy Loss) được áp dụng.
6.  Chỉ trọng số của bộ phân loại $M_{cls}$ được cập nhật trong bước truyền ngược (backward pass). $M_{feat}$ giữ nguyên (frozen).

Mô hình đã fine-tune $M'$ sẽ có khả năng hoạt động tốt trên các điều kiện (phong cách) miền đích chưa từng thấy trong quá trình huấn luyện.

---

## 4. Thực nghiệm Phân đoạn Ngữ nghĩa (Semantic Segmentation Experiments)

### 4.1. Chi tiết cài đặt (Implementation Details)

*   **Kiến trúc**: DeepLabv3+ với backbone là ResNet-50 (đã khởi tạo bằng trọng số hình ảnh của CLIP).
*   **Huấn luyện Source-only**: Huấn luyện $M_{cls}$ 200k vòng lặp trên ảnh cắt ngẫu nhiên 768x768. `Layer1` bị đóng băng. Sử dụng Data Augmentation tiêu chuẩn. (Việc giữ lại $M_{feat}$ là CLIP pre-trained weights được chứng minh là có ích, giúp chống over-fitting, theo **Table 1**).
*   **Tăng cường đặc trưng**: Đặc trưng của `Layer1` (kích thước $192\times192\times256$) được chọn để augment. Các biến $\mu$ và $\sigma$ là vector 256 chiều (bằng số kênh của `Layer1`).
*   **Fine-tuning**: Bộ phân loại $M_{cls}$ được fine-tune trong 2000 vòng lặp trên các đặc trưng đã được chuyển đổi.
*   **Tập dữ liệu**: Cityscapes (miền nguồn). Đánh giá trên ACDC (các điều kiện khắc nghiệt như Night, Snow, Rain) và GTA5 (chuyển đổi real->synthetic).

### 4.2. Kết quả chính (Main Results)

PØDA được so sánh với hai baseline SOTA:
1.  **CLIPstyler**: Phương pháp zero-shot style transfer. Sử dụng hình ảnh được tạo ra bằng văn bản.
2.  **SM-PPM**: Phương pháp One-shot UDA, cần 1 hình ảnh từ miền đích để trích xuất phong cách.

**Table 2** trình bày kết quả chi tiết cho nhiều loại prompt (Ví dụ: "driving at night", "driving in snow", "driving under rain", "driving in a game", "driving").

*   PØDA vượt qua baseline nguồn (Source-only) trên tất cả các kịch bản. Ví dụ: +6.72 mIoU trên ACDC Night, +4.62 trên ACDC Snow.
*   PØDA liên tục đánh bại CLIPstyler. CLIPstyler tạo ra các hình ảnh với phong cách mong muốn nhưng chứa nhiều lỗi (artifacts) làm giảm hiệu suất phân đoạn. **Figure 4** minh họa các artifacts này (ví dụ: mưa xuất hiện dưới dạng các đốm trắng bất thường, hay trò chơi Atari xuất hiện trên các tòa nhà). Việc trực tiếp thay đổi đặc trưng ở PØDA tránh được sự trôi dạt (drifts) lớn trên đa tạp đặc trưng (feature manifold).
*   PØDA có tính hiệu quả về chi phí tính toán: Augment 1 đặc trưng mất 0.3s, trong khi CLIPstyler mất 65s để xử lý 1 hình ảnh (trên RTX 2080Ti).

**Table 3** so sánh với SM-PPM (One-shot UDA). PØDA (0 hình ảnh đích) cho kết quả tương đương hoặc tốt hơn SM-PPM (cần 1 hình ảnh đích làm điểm neo để trích xuất phong cách).

**Figure 5** cung cấp kết quả trực quan chất lượng cao, chứng minh PØDA có khả năng phân đoạn tốt hơn ở điều kiện đêm, mưa, và trên dữ liệu giả lập.

**Figure 6** mô tả các kết quả phân đoạn ở các tình huống không phổ biến (uncommon conditions) hiếm gặp hoặc không có trong dữ liệu huấn luyện: Lái xe xuyên qua lửa (driving through fire), trong bão cát (sandstorm), hoặc trong phim cũ (old movie). PØDA cũng mang lại cải thiện rõ rệt so với source-only.

### 4.3. Nghiên cứu sâu (Ablation Studies)

*   **Sự lựa chọn Prompt (Prompt selection)**: Bảng **Table 4** so sánh các prompt mang nghĩa tương tự do ChatGPT sinh ra (Relevant prompt) và các prompt hoàn toàn không liên quan (Irrelevant prompt). PØDA mạnh mẽ với các prompt mang nghĩa đúng (dù cách diễn đạt khác nhau), trong khi các prompt sai nghĩa sẽ không giúp ích.
*   **Chọn lớp để augment (Choice of features)**: **Table 5** xác nhận rằng việc augment đặc trưng cấp thấp nhất (`Layer1`) đem lại hiệu suất tốt nhất. Augment các lớp sâu hơn (Layer2, 3, 4) mang lại kết quả kém đi vì điều đó phá vỡ tính nhất quán giữa thông tin đầu vào.
*   **Đóng băng backbone**: **Table 6** chỉ ra rằng việc đóng băng toàn bộ backbone (`Layer1-4`) tốt hơn so với chỉ đóng băng `Layer1`.
*   **Số lượng phong cách được khai phá (Number of mined styles)**: Thực nghiệm chỉ ra rằng số lượng phong cách $|\mathcal{S}_{s\rightarrow t}|$ cần xấp xỉ bằng số lượng ảnh nguồn $|\mathcal{D}_s|$ để mang lại độ bao phủ phân phối (distribution coverage) tốt nhất. Quá ít phong cách sẽ dẫn đến phương sai cao.
*   **Khởi tạo khai phá phong cách (Style mining initialization)**: **Table 12** trong phụ lục chứng minh rằng việc khởi tạo $\boldsymbol{\mu}^0, \boldsymbol{\sigma}^0$ bằng mean và std của *chính bức ảnh nguồn* (thay vì khởi tạo bằng 0 hoặc nhiễu Gaussian ngẫu nhiên) đóng vai trò như một hình thức chính quy hóa (regularization). Điều này giúp giữ lại nội dung ngữ nghĩa (semantic preservation).
*   **Số bước tối ưu hóa (Optimization steps)**: **Figure 7** trong phụ lục cho thấy đồ thị hiệu suất theo số bước tối ưu hóa. Mức khoảng 80-100 vòng lặp là tối ưu, thực hiện quá nhiều vòng lặp dẫn đến lỗi "over-stylization", làm suy giảm hiệu năng.

### 4.4. Thảo luận thêm (Further Discussion)

*   **Tính tổng quát hóa (Generalization with PØDA)**: **Table 7** cho thấy PØDA có thể được kết hợp với phương pháp nhiễu ngẫu nhiên (Gaussian noises shifting). Phương pháp `Source-only-G` (thêm nhiễu ngẫu nhiên vào đặc trưng với SNR=20dB) mang lại sự cải thiện tổng quát. PØDA áp dụng trên `Source-only-G` mang lại kết quả còn cao hơn, và khi kết hợp thêm kỹ thuật style-mixing từ SM-PPM thì càng xuất sắc hơn.
*   **Sử dụng Priors khác**: **Table 8** so sánh giữa việc dùng prompt (textual priors) của PØDA so với priors vật lý của CIConv, cho thấy PØDA rất cạnh tranh mà không cần kiến trúc chuyên biệt.

---

## 5. PØDA cho các tác vụ khác (PØDA for other tasks)

Do hoạt động ở cấp độ đặc trưng (feature level), phương pháp PIN của PØDA không phụ thuộc vào tác vụ cụ thể (task-agnostic).

### 5.1. Phát hiện Đối tượng (Object Detection)

*   Sử dụng Faster-RCNN.
*   Miền nguồn: Cityscapes hoặc Day-Clear. Miền đích: Các kịch bản sương mù (Foggy), đêm (Night), chạng vạng (Dusk), mưa (Rainy).
*   Kết quả ở **Table 10** cho thấy PØDA có thể cải thiện kết quả đáng kể so với baseline source-only, thậm chí tiệm cận hoặc vượt qua các phương pháp UDA (Domain Generalization Object Detection).

### 5.2. Phân loại Hình ảnh (Image Classification)

*   Khởi tạo bộ phân loại tuyến tính (linear probe) trên các đặc trưng CLIP-RN50.
*   Thực nghiệm 1 (Real-to-Painting): Huấn luyện trên ảnh chim thật (CUB-200), kiểm tra trên tranh vẽ chim (CUB-200-Paintings) với prompt "Painting of a bird".
*   Thực nghiệm 2 (Giải quyết color bias): Colored MNIST. Chữ số chẵn có màu đỏ, chữ số lẻ có màu xanh lam (trong tập huấn luyện). Tập test có màu ngẫu nhiên. Sử dụng prompt "Blue digit" và "Red digit" để augment đặc trưng (ngăn ngừa sự rò rỉ phong cách bằng cách áp dụng "blue" cho chẵn và "red" cho lẻ).
*   Kết quả ở **Table 11** chứng minh PØDA có khả năng khắc phục domain shift trong phân loại ảnh một cách hiệu quả.

---

## 6. Kết luận (Conclusion)

Bài báo đã tiên phong đưa ra nhiệm vụ "Prompt-driven zero-shot domain adaptation" đầy tiềm năng. Việc khai phá và thao tác trực tiếp trên số liệu thống kê của các đặc trưng thông qua lớp **Prompt-driven Instance Normalization (PIN)** cho phép điều chỉnh mô hình sang các miền đích chỉ được mô tả bằng văn bản. PØDA hoạt động hiệu quả, chi phí tính toán thấp, loại bỏ sự cần thiết của hình ảnh miền đích và giải quyết được vấn đề mất nét (artifacts) trong các phương pháp dịch phong cách trên pixel. Hơn nữa, tính phổ quát của cơ chế này cho phép áp dụng sang Object Detection và Image Classification.

PØDA là một bước tiến hứa hẹn trong việc khai thác các "foundation models" lớn (như CLIP) để giải quyết những thách thức trong nhận thức môi trường thực tế với dữ liệu hạn chế. Mọi kỹ thuật, từ PIN đến các bước Optimization, đều được cung cấp công khai tại [GitHub Repository của nhóm nghiên cứu](https://github.com/astra-vision/PODA).
