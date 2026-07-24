# SciLens: Kiến trúc mô hình Deep Learning end-to-end

> Đặc tả đầy đủ các thành phần, bài toán, kiến trúc Deep Learning, dữ liệu, hàm loss, chiến lược huấn luyện và quy trình suy luận của project **Scientific Paper Summarization**.
>
> Phiên bản tài liệu: **24/07/2026**
>
> Tài liệu định nghĩa kiến trúc đề xuất để nghiên cứu và triển khai; không có nghĩa mọi module đều phải được huấn luyện từ đầu trong phiên bản đầu tiên.

---

## 0. Câu trả lời ngắn gọn

SciLens **không nên là một API wrapper thuần túy**. Kiến trúc hợp lý là một hệ thống hybrid gồm ba lớp:

1. **Pretrained document intelligence**: dùng model có sẵn cho layout, OCR, equation transcription và table structure.
2. **Scientific understanding models**: fine-tune encoder cho rhetorical role, entity, claim, equation/table role và evidence linking.
3. **Lõi mô hình mới của SciLens**: tự xây Heterogeneous Scientific Document Graph Transformer, multi-task heads, evidence-aware content planner, ablation extractor, typed verifier và confidence/abstention.

LLM chỉ là một thành phần sinh ngôn ngữ ở cuối pipeline. LLM không được tự quyết định source cell, tự tính delta, tự khôi phục LaTeX hoặc tự gán provenance.

### 0.1. Phần nào dùng lại, phần nào tự xây?

| Thành phần | Vai trò | Chiến lược |
|---|---|---|
| GROBID/Docling/LaTeX parser | Khôi phục cấu trúc ban đầu | Dùng lại, không huấn luyện trong SciLens |
| LayoutLMv3 hoặc layout detector | Page/layout representation | Dùng pretrained, fine-tune tùy dữ liệu |
| Nougat/MathNet | PDF/formula image → markup/LaTeX | Dùng pretrained; chỉ fine-tune khi equation OCR là đóng góp chính |
| Table Transformer | Table detection/structure | Dùng pretrained hoặc fine-tune nhẹ |
| SciBERT/Longformer | Scientific text encoding | Fine-tune |
| Semantic equation encoder | Hiểu vai trò công thức và ký hiệu | Tự xây trên pretrained encoders |
| Scientific table encoder | Hiểu header, cell, metric, result | Tự xây/fine-tune |
| Scientific Document Graph | Biểu diễn typed objects và relations | Tự thiết kế |
| Heterogeneous Graph Transformer | Truyền thông tin xuyên text–equation–table | **Mô hình nghiên cứu cốt lõi, tự xây** |
| Salience/evidence/ablation heads | Chọn và liên kết thông tin | **Tự xây và huấn luyện** |
| Evidence-aware planner | Lập dàn ý có kiểm soát | **Tự xây và huấn luyện** |
| Generator | Structured plan → summary | Fine-tune model mở hoặc gọi LLM API |
| Typed verifier | Kiểm chứng claim theo modality | **Tự xây; kết hợp neural và deterministic rules** |
| Calibrator/abstention | Ước lượng độ tin cậy | Tự xây và fit trên validation set |

### 0.2. Đóng góp Deep Learning cốt lõi

Đóng góp mô hình nên được phát biểu như sau:

> SciLens xây dựng một **Heterogeneous Scientific Document Graph Transformer** học biểu diễn liên kết giữa section, claim, equation, symbol, table cell, experiment và result; sau đó dùng multi-task learning để đồng thời chọn nội dung quan trọng, truy tìm bằng chứng, trích xuất result/ablation và lập content plan có thể kiểm chứng trước khi sinh summary.

---

## 1. Định nghĩa bài toán

### 1.1. Input

Một paper $D$ có thể gồm:

$$
D = \{P, S, X, M\}
$$

trong đó:

- $P$: PDF pages và page images;
- $S$: source có cấu trúc nếu tồn tại, ví dụ LaTeX, XML/JATS hoặc HTML;
- $X$: metadata, citation và supplementary material;
- $M$: thông tin modality gồm text, layout, equation, table và figure.

Phiên bản đầu nên giới hạn:

- paper tiếng Anh;
- lĩnh vực ML/NLP/CV;
- PDF born-digital hoặc arXiv source;
- paper có phần Method và Experiments;
- ưu tiên paper có equation, result table và ablation.

### 1.2. Output

Hệ thống tạo:

$$
Y = \{Y_{\text{summary}},Y_{\text{equation}},
Y_{\text{result}},Y_{\text{ablation}},
Y_{\text{contribution}},Y_{\text{limitation}},
Y_{\text{evidence}},Y_{\text{confidence}}\}
$$

Cụ thể:

- TL;DR và executive summary;
- tóm tắt theo section;
- công thức quan trọng dưới dạng LaTeX;
- giải thích công thức và bảng ký hiệu;
- experimental setup;
- bảng kết quả chính và các comparison có số liệu;
- ablation study theo component–intervention–outcome;
- đóng góp kỹ thuật;
- ưu điểm có đối tượng so sánh;
- hạn chế do tác giả nêu và hạn chế do hệ thống suy luận;
- claim–evidence map;
- confidence và parsing warnings.

### 1.3. Bài toán học máy tổng quát

Thay vì học trực tiếp:

$$
p(Y\mid D),
$$

SciLens phân rã thành:

$$
p(G\mid D)
\;p(Z\mid G)
\;p(Y\mid Z,G)
\;p(V\mid Y,G),
$$

trong đó:

- $G$: Scientific Document Graph;
- $Z$: structured content plan;
- $Y$: summary;
- $V$: verification labels và confidence.

Phân rã này giúp:

- dễ huấn luyện theo module;
- có supervision trung gian;
- xác định lỗi phát sinh ở parsing, retrieval hay generation;
- kiểm tra factuality trước khi trả output;
- thực hiện ablation rõ ràng.

---

## 2. Các task con

| Mã | Task | Input | Output |
|---|---|---|---|
| T1 | Source routing | PDF/LaTeX/XML | đường xử lý và confidence |
| T2 | Layout analysis | page image + PDF tokens | block, bbox, reading order |
| T3 | Section reconstruction | blocks/headings | section tree |
| T4 | Equation detection/transcription | image region | normalized LaTeX |
| T5 | Symbol grounding | equation + local text | symbol–definition links |
| T6 | Equation salience/role | equation object | importance + role |
| T7 | Table detection/structure | page/table image | rows, columns, cells, headers |
| T8 | Table semantic encoding | canonical table | method/dataset/metric/result |
| T9 | Result extraction | text + tables | typed result tuples |
| T10 | Ablation extraction | experiment graph | control–variant–component–delta |
| T11 | Rhetorical zoning | sentence/paragraph | background, method, result... |
| T12 | Scientific entity extraction | full text | task, method, dataset, metric... |
| T13 | Claim extraction | sentences | atomic claims |
| T14 | Contribution/limitation extraction | claims + context | typed contributions/limitations |
| T15 | Evidence linking | claim + document objects | supporting object IDs |
| T16 | Salience ranking | graph nodes | importance ranking |
| T17 | Content planning | salient grounded nodes | ordered summary plan |
| T18 | Grounded generation | plan + evidence packets | structured summary |
| T19 | Typed verification | output claim + evidence | supported/unsupported/conflict |
| T20 | Confidence/abstention | model scores | calibrated confidence/abstain |

---

## 3. Nguyên tắc thiết kế

### 3.1. Structure before generation

Không sinh summary trực tiếp từ raw PDF. Tài liệu phải được chuyển thành typed objects và relations trước.

### 3.2. Retrieval before explanation

Trước khi giải thích một claim hoặc equation, hệ thống phải retrieve được:

- source span;
- định nghĩa ký hiệu;
- table cell;
- caption hoặc experimental setting liên quan.

### 3.3. Neural semantics, deterministic arithmetic

Deep Learning dùng để:

- hiểu nghĩa;
- phân loại;
- liên kết;
- xếp hạng;
- sinh ngôn ngữ.

Code xác định dùng để:

- parse số;
- tính delta;
- kiểm tra metric direction;
- giữ row/column path;
- validate JSON;
- compile LaTeX;
- kiểm tra source ID tồn tại.

### 3.4. Typed verification

Text claim, numerical claim và equation explanation không được kiểm tra bằng một verifier giống nhau.

### 3.5. Selective prediction

Nếu confidence thấp, hệ thống được quyền:

- bỏ field;
- dùng extractive wording;
- thêm cảnh báo;
- yêu cầu kiểm tra thủ công.

### 3.6. Không huấn luyện mọi thứ từ đầu

Việc tự pretrain OCR, VLM hoặc LLM từ đầu không phải lựa chọn hiệu quả. Novelty nên đặt ở scientific structure, graph learning, evidence grounding và verification.

---

## 4. Kiến trúc tổng thể

```text
┌───────────────────────────────────────────────────────────────────┐
│                    INPUT: PDF / LaTeX / XML                       │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M0. Source Router & Quality Assessor                              │
│ Chọn source-native path, PDF path hoặc OCR fallback               │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M1. Multimodal Document Parser                                   │
│ Layout ─ reading order ─ sections ─ spans ─ bbox ─ cross-ref      │
└───────────────┬───────────────────────┬───────────────────────────┘
                │                       │
                ▼                       ▼
┌───────────────────────────┐  ┌────────────────────────────────────┐
│ M2. Equation Branch       │  │ M3. Table/Result Branch            │
│ detect → LaTeX → symbols  │  │ detect → cells → semantics         │
│ → role → salience         │  │ → results → ablations              │
└───────────────┬───────────┘  └──────────────────┬─────────────────┘
                │                                 │
                └───────────────┬─────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M4. Scientific Text Understanding                                │
│ hierarchical encoder → roles → entities → claims → contributions │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M5. Scientific Document Graph Builder                            │
│ typed nodes + typed relations + provenance                        │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M6. Heterogeneous Scientific Document Graph Transformer          │
│ relation-aware message passing across modalities                  │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M7. Multi-task Scientific Understanding Heads                    │
│ salience │ evidence │ equation │ table │ result │ ablation        │
│ contribution │ limitation │ contradiction                         │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M8. Evidence-Aware Content Planner                               │
│ select → budget → order → deduplicate → evidence packets          │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M9. Grounded Structured Generator                                │
│ plan + typed evidence → Markdown/JSON with citations              │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M10. Typed Claim Verifier & Repair                               │
│ text NLI │ numeric │ table-cell │ equation-symbol │ consistency   │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ M11. Confidence Calibration & Abstention                         │
│ calibrated score → accept / repair / extractive fallback / omit   │
└───────────────────────────────┬───────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│ OUTPUT: structured summary + evidence + confidence + warnings     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 5. Biểu diễn dữ liệu trung gian

### 5.1. Document Object Model

Trước graph, mỗi paper được chuẩn hóa thành:

```json
{
  "paper_id": "paper_001",
  "metadata": {},
  "pages": [],
  "sections": [],
  "spans": [],
  "equations": [],
  "symbols": [],
  "tables": [],
  "figures": [],
  "citations": [],
  "parser_warnings": []
}
```

Mỗi object bắt buộc có:

- stable ID;
- source page;
- bounding box nếu lấy từ PDF;
- source character/token offsets nếu có;
- parser name/version;
- confidence;
- provenance tới raw source.

### 5.2. Scientific Document Graph

Graph:

$$
G=(\mathcal{V},\mathcal{E},\tau,\phi),
$$

với:

- $\mathcal{V}$: tập node;
- $\mathcal{E}$: tập edge;
- $\tau(v)$: node type;
- $\phi(e)$: relation type.

Node types:

```text
Paper, Section, Paragraph, Sentence, Claim,
Equation, Symbol, Table, Cell, Figure,
Method, Component, Task, Dataset, Metric,
Experiment, Result, Contribution, Limitation
```

Edge types:

```text
CONTAINS, NEXT, REFERENCES, DEFINES, USES,
MENTIONS, PART_OF, MEASURES, EVALUATED_ON,
HAS_SCORE, COMPARES_WITH, SUPPORTED_BY,
CONTRADICTS, REMOVES, REPLACES, ADDS,
QUALIFIES, DERIVED_FROM
```

### 5.3. Tại sao không dùng vector database làm biểu diễn chính?

Vector database chỉ trả về các đoạn gần nghĩa. Nó không bảo toàn chắc chắn:

- header path của table cell;
- relation giữa equation và symbol;
- control–variant trong ablation;
- provenance chính xác;
- section hierarchy;
- typed relation.

Vector index vẫn hữu ích cho candidate retrieval, nhưng graph và schema mới là nguồn sự thật.

---

## 6. M0 — Source Router và Quality Assessor

### 6.1. Vấn đề cần giải quyết

Cùng một paper có thể có:

- LaTeX source tốt;
- publisher XML;
- PDF born-digital;
- PDF scan;
- hai phiên bản preprint/camera-ready.

Chọn sai source có thể làm giảm chất lượng toàn pipeline.

### 6.2. Task

Phân loại:

```text
native_structured
born_digital_pdf
scanned_pdf
hybrid_pdf
unsupported_or_corrupted
```

Đồng thời dự đoán quality vector:

$$
q=[q_{\text{text}},q_{\text{layout}},q_{\text{equation}},
q_{\text{table}},q_{\text{scan}}].
$$

### 6.3. Kiến trúc

Phiên bản đầu dùng rule:

- PDF text coverage;
- font/object statistics;
- ratio image/text;
- số token trên mỗi page;
- khả năng compile/parse LaTeX;
- metadata agreement.

Phiên bản học máy có thể dùng:

- page thumbnail encoder: nhỏ gọn CNN/ViT;
- metadata feature MLP;
- late fusion classifier.

$$
\mathbf{h}_{\text{route}}
=
\operatorname{MLP}([\mathbf{h}_{\text{page}};\mathbf{x}_{\text{meta}}]).
$$

### 6.4. Huấn luyện

- weak labels từ source availability và OCR quality;
- human labels cho tập nhỏ;
- cross-entropy cho route;
- regression loss cho quality dimensions.

$$
\mathcal{L}_{\text{route}}
=
\mathcal{L}_{\text{CE}}
+\lambda_q\mathcal{L}_{\text{Huber}}.
$$

### 6.5. Khuyến nghị

Không coi M0 là novelty chính. Dùng rules trước; chỉ huấn luyện khi routing error thực sự ảnh hưởng đáng kể.

---

## 7. M1 — Multimodal Document Parser

### 7.1. Vấn đề

PDF lưu các glyph theo tọa độ, không lưu đầy đủ cây section hay reading order. Paper nhiều cột còn chứa caption, footnote, equation và table chen vào luồng text.

### 7.2. Task

- block detection;
- block classification;
- reading-order prediction;
- section heading detection;
- section tree reconstruction;
- figure/table/equation region detection;
- cross-reference linking;
- coordinate preservation.

### 7.3. Kiến trúc đề xuất

M1 gồm ba đường.

#### Đường A — Source-native parser

Nếu có LaTeX/XML:

- parse AST;
- giữ section hierarchy;
- lấy equation source;
- lấy table source;
- resolve label–reference.

Đây chủ yếu là deterministic parsing, không phải Deep Learning.

#### Đường B — Layout-aware PDF encoder

Dùng pretrained [LayoutLMv3](https://arxiv.org/abs/2204.08387) hoặc layout model trong [Docling](https://arxiv.org/abs/2408.09869).

Input token:

$$
\mathbf{x}_i =
\mathbf{e}_{\text{text},i}
+\mathbf{e}_{\text{bbox},i}
+\mathbf{e}_{\text{page},i}
+\mathbf{e}_{\text{visual},i}.
$$

LayoutLMv3 có unified text/image masking và word–patch alignment, phù hợp để tạo biểu diễn kết hợp text, layout và image patch.

Các head:

- token/block classification head;
- bounding-box regression/detection head;
- pairwise reading-order head;
- heading-level classification head.

Reading order có thể được học như edge prediction:

$$
p(i\rightarrow j)
=
\sigma(\operatorname{MLP}[
\mathbf{h}_i;\mathbf{h}_j;
\Delta\mathbf{b}_{ij}]).
$$

Sau đó tìm ordering hợp lệ bằng graph decoding có ràng buộc.

#### Đường C — OCR/image-to-markup fallback

Dùng [Nougat](https://arxiv.org/abs/2308.13418), một visual encoder–text decoder Transformer cho academic document image → markup. Không nên chạy Nougat trên mọi page nếu source-native hoặc PDF text đã tốt; chỉ chạy khi quality thấp hoặc page chứa math/layout phức tạp.

### 7.4. Chiến lược huấn luyện

#### Giai đoạn 1: freeze parser

- dùng pretrained weights;
- benchmark trên 50–100 paper nội bộ;
- tập trung xây downstream graph.

#### Giai đoạn 2: domain fine-tuning tùy chọn

- fine-tune layout head trên page annotations của SciLensBench;
- giữ phần lớn backbone frozen trong pilot;
- unfreeze vài layer cuối nếu domain shift lớn.

Loss:

$$
\mathcal{L}_{\text{layout}}
=
\lambda_{\text{cls}}\mathcal{L}_{\text{block-cls}}
+\lambda_{\text{box}}\mathcal{L}_{\text{box}}
+\lambda_{\text{ord}}\mathcal{L}_{\text{order}}
+\lambda_{\text{head}}\mathcal{L}_{\text{heading}}.
$$

### 7.5. Output

```json
{
  "block_id": "b_52",
  "type": "paragraph",
  "page": 4,
  "bbox": [74, 122, 518, 281],
  "reading_order": 37,
  "section_id": "sec_3_2",
  "text": "...",
  "confidence": 0.96
}
```

### 7.6. Evaluation

- block mAP/F1;
- reading-order pairwise accuracy;
- section-title F1;
- section-tree edit distance;
- paragraph boundary F1;
- cross-reference link F1.

---

## 8. M2 — Equation Understanding Branch

M2 không chỉ OCR công thức. Nó giải quyết năm bài toán: detection, transcription, normalization, symbol grounding và semantic salience.

### 8.1. M2.1 — Equation detection

#### Task

Tìm inline/display equation và bounding box.

#### Kiến trúc

- object detector kiểu DETR/LayoutLMv3 detection head;
- class labels: inline equation, display equation, equation number.

#### Huấn luyện

- pretrain/fine-tune trên vùng equation từ source–PDF alignment;
- Hungarian matching như object detection;
- classification + L1/GIoU box loss.

#### Output

Equation crop, bbox, type, equation number và confidence.

### 8.2. M2.2 — Formula image to LaTeX

#### Kiến trúc

Hai lựa chọn:

1. Dùng pretrained Nougat cho page/region-to-markup.
2. Dùng [MathNet](https://arxiv.org/abs/2404.13667) hoặc mô hình MER encoder–decoder chuyên biệt.

Kiến trúc khái quát:

```text
formula crop
    ↓
CNN/ViT/Swin visual encoder
    ↓
visual patch embeddings
    ↓
autoregressive Transformer decoder
    ↓
LaTeX tokens
```

$$
p(L\mid I)
=
\prod_{t=1}^{T}
p(l_t\mid l_{<t},\operatorname{Enc}_{\text{vision}}(I)).
$$

#### Loss

$$
\mathcal{L}_{\text{LaTeX}}
=
-\sum_t\log p(l_t^\star\mid l_{<t}^\star,I),
$$

kèm:

- label smoothing;
- token masking/augmentation;
- optional render consistency loss.

#### Normalization

Nhiều chuỗi LaTeX khác nhau render giống nhau. Sau decoding cần:

- bỏ formatting không mang nghĩa;
- chuẩn hóa macro;
- chuẩn hóa whitespace;
- parse thành syntax tree khi có thể;
- compile và render lại.

### 8.3. M2.3 — Dual-view Equation Encoder

#### Vấn đề

Chuỗi LaTeX giữ cấu trúc ký hiệu; image giữ bố cục 2D. Chỉ dùng một modality có thể bỏ mất thông tin.

#### Kiến trúc SciLens đề xuất

```text
Equation image ─► ViT encoder ─────┐
                                   ├─► gated cross-attention ─► h_eq
Normalized LaTeX ─► Transformer ───┘
Local context ─► SciBERT ──────────┘
```

$$
\mathbf{h}_{\text{eq}}
=
g_v\mathbf{h}_{\text{vision}}
+g_l\mathbf{h}_{\text{LaTeX}}
+g_c\mathbf{h}_{\text{context}},
$$

với:

$$
[g_v,g_l,g_c]
=
\operatorname{softmax}(\operatorname{MLP}
[\mathbf{h}_{\text{vision}};\mathbf{h}_{\text{LaTeX}};\mathbf{h}_{\text{context}}]).
$$

Mô hình có thể giảm trọng số image nếu LaTeX source đáng tin và tăng trọng số image khi source thiếu.

### 8.4. M2.4 — Symbol grounding

#### Task

Với mỗi symbol $s$, tìm definition span $d$:

```text
\lambda → regularization coefficient
N → number of samples
h_i → representation of token i
```

#### Candidate generation

Tìm ở:

- cùng câu;
- câu trước/sau equation;
- đoạn chứa “where”, “denotes”, “is defined as”;
- notation section;
- equation được tham chiếu.

#### Kiến trúc

Bi-encoder để retrieve candidates:

$$
s_{\text{retr}}(s,d)
=
\cos(\mathbf{h}_s,\mathbf{h}_d).
$$

Cross-encoder reranker:

$$
p(d\mid s)
=
\operatorname{softmax}\!\left(
\operatorname{MLP}(\operatorname{CrossEnc}[s;\text{eq};d])
\right).
$$

#### Huấn luyện

- positive: gold symbol–definition;
- hard negative: định nghĩa của symbol giống hình thức ở section khác;
- in-batch negatives;
- contrastive InfoNCE + cross-entropy reranking.

### 8.5. M2.5 — Equation role và salience

#### Labels

```text
model_definition
training_objective
loss_function
inference_rule
theoretical_statement
evaluation_metric
auxiliary_derivation
hyperparameter_definition
unimportant
```

#### Kiến trúc

MLP multi-label head trên $\mathbf{h}_{eq}$ đã được graph-contextualize:

$$
\mathbf{p}_{\text{role}}
=
\sigma(\mathbf{W}_{\text{role}}\mathbf{h}_{\text{eq}}+\mathbf{b}).
$$

Salience:

$$
s_{\text{eq}}
=
\operatorname{MLP}
[\mathbf{h}_{\text{eq}};
\mathbf{x}_{\text{references}};
\mathbf{x}_{\text{section}};
\mathbf{x}_{\text{position}}].
$$

#### Huấn luyện

- role: focal loss hoặc weighted BCE do mất cân bằng;
- salience: pairwise/listwise ranking;
- positive: equation được expert chọn;
- negative: auxiliary/appendix equations.

### 8.6. M2.6 — Equation explanation

Không sinh trực tiếp từ image. Evidence packet phải gồm:

- normalized LaTeX;
- role;
- symbol definitions;
- surrounding method claims;
- equations referenced trước/sau;
- source IDs.

Generator tạo:

- ý nghĩa;
- input/output;
- trực giác;
- vai trò trong phương pháp;
- điều kiện/giả định.

Verifier kiểm tra mọi symbol trong explanation có definition hoặc source evidence.

### 8.7. Loss tổng M2

$$
\mathcal{L}_{\text{eq}}
=
\lambda_d\mathcal{L}_{\text{detect}}
+\lambda_l\mathcal{L}_{\text{LaTeX}}
+\lambda_g\mathcal{L}_{\text{ground}}
+\lambda_r\mathcal{L}_{\text{role}}
+\lambda_s\mathcal{L}_{\text{salience}}
+\lambda_a\mathcal{L}_{\text{alignment}}.
$$

### 8.8. Phần nào nên huấn luyện?

- Detection/transcription: dùng pretrained trước.
- Dual-view encoder, symbol grounding, role và salience: tự xây/fine-tune.
- Explanation generator: SFT/QLoRA hoặc API.
- Symbol/equation verifier: tự xây.

---

## 9. M3 — Table, Result và Ablation Branch

### 9.1. M3.1 — Table detection và structure recognition

#### Vấn đề

PDF table không đảm bảo có cấu trúc hàng/cột. Merged headers, projected row headers, footnotes và style mang ý nghĩa.

#### Kiến trúc

Dùng [Table Transformer](https://github.com/microsoft/table-transformer), dựa trên DETR và được huấn luyện với PubTables-1M:

```text
page image
   ↓
CNN backbone
   ↓
Transformer encoder-decoder + object queries
   ↓
table/row/column/header/cell boxes
```

Các object query dự đoán:

- table bbox;
- row;
- column;
- column header;
- projected row header;
- spanning cell.

#### Huấn luyện

- dùng pretrained PubTables-1M;
- fine-tune trên SciLensBench nếu table style khác;
- DETR classification + box loss;
- đánh giá thêm GriTS/TEDS thay vì chỉ detection.

### 9.2. M3.2 — Canonical Table Builder

Đây là module deterministic:

- OCR/PDF text alignment vào cell;
- xây header tree;
- tạo full row path và column path;
- parse bold/underline;
- parse `mean ± std`, percentage, range;
- gắn caption và footnote;
- giữ cell source bbox.

Ví dụ:

```json
{
  "cell_id": "t4-r3-c5",
  "row_path": ["Ours", "Large"],
  "column_path": ["WMT14 En-De", "BLEU ↑"],
  "raw_value": "31.4 ± 0.2",
  "value": 31.4,
  "std": 0.2,
  "style": ["bold"],
  "bbox": [310, 402, 371, 423]
}
```

### 9.3. M3.3 — Scientific Table Encoder

#### Vấn đề

Linearize table thành một chuỗi dài dễ làm mất quan hệ hàng/cột và header.

#### Kiến trúc đề xuất

Mỗi cell:

$$
\mathbf{x}_{\text{cell}}
=
\mathbf{e}_{\text{text}}
+\mathbf{e}_{\text{row}}
+\mathbf{e}_{\text{col}}
+\mathbf{e}_{\text{row-path}}
+\mathbf{e}_{\text{col-path}}
+\mathbf{e}_{\text{style}}
+\mathbf{e}_{\text{numeric}}.
$$

Encoder gồm:

1. row-wise self-attention;
2. column-wise self-attention;
3. header-to-cell cross-attention;
4. caption/context cross-attention;
5. table-level pooling.

Có thể khởi tạo text encoder từ TAPAS, vì [TAPAS](https://research.google/pubs/tapas-weakly-supervised-table-parsing-via-pre-training/) đưa row/column positional information vào BERT và học chọn cell/aggregation. Tuy nhiên, SciLens phải mở rộng cho scientific header paths, caption và experimental context.

### 9.4. M3.4 — Table role classification

Labels:

```text
main_result
ablation
efficiency
robustness
hyperparameter
dataset_statistics
qualitative
error_analysis
architecture
other
```

Classifier:

$$
p(r\mid T)
=
\operatorname{softmax}(\mathbf{W}_r\mathbf{h}_T).
$$

Huấn luyện bằng weighted cross-entropy hoặc focal loss.

### 9.5. M3.5 — Scientific result tuple extraction

#### Output schema

```text
(method, variant, task, dataset, split, metric,
 score, uncertainty, metric_direction,
 experimental_setting, source_cell)
```

#### Kiến trúc

Kết hợp:

- cell classification;
- span/entity linking;
- relation extraction.

Các head:

- cell type: method/dataset/metric/score/setting;
- header linking;
- entity normalization;
- tuple assembly.

Score một tuple candidate:

$$
s(\text{method},\text{dataset},\text{metric},\text{cell})
=
\operatorname{MLP}
[\mathbf{h}_m;\mathbf{h}_d;\mathbf{h}_{metric};\mathbf{h}_{cell}].
$$

Negative samples:

- đúng score nhưng sai metric;
- đúng metric nhưng sai dataset;
- cell cùng cột nhưng sai row;
- giá trị trong footnote;
- value từ setting không so sánh được.

### 9.6. M3.6 — Ablation extraction

#### Task

Tìm:

```text
full model/control
variant
component
intervention: remove/add/replace/change
dataset
metric
control score
variant score
delta
conclusion
```

#### Kiến trúc

1. Table-role classifier xác định ablation table.
2. Row/cell encoder nhận diện control và variants.
3. Text encoder đọc caption, row labels và surrounding paragraphs.
4. Relation head dự đoán:

```text
variant REMOVES component
variant ADDS component
variant REPLACES component
result BELONGS_TO variant
result COMPARED_WITH control
```

5. Deterministic engine tính delta.

#### Relation classifier

$$
p(r_{ij})
=
\operatorname{softmax}\!\left(
\operatorname{MLP}
[\mathbf{h}_i;\mathbf{h}_j;
\mathbf{h}_i\odot\mathbf{h}_j;
\mathbf{e}_{\text{table-position}}]
\right).
$$

#### Delta

Với metric higher-is-better:

$$
\Delta_{\text{impact}}
=
s_{\text{full}}-s_{\text{variant}}.
$$

Với metric lower-is-better:

$$
\Delta_{\text{impact}}
=
s_{\text{variant}}-s_{\text{full}}.
$$

$\Delta_{\text{impact}}>0$ nghĩa là bỏ/thay component làm hiệu năng xấu đi.

#### Causal strength

Classifier hoặc rules gắn:

```text
controlled_single_change
multiple_changes
unclear_control
descriptive_only
```

Summary không được dùng ngôn ngữ nhân quả mạnh khi `multiple_changes` hoặc `unclear_control`.

### 9.7. M3.7 — Main-result salience

Không lấy mọi số trong paper. Result salience dùng:

- table role;
- text claim references;
- bold/best indicators;
- method ownership;
- abstract/conclusion overlap;
- statistical evidence;
- novelty alignment.

Pairwise ranker học:

$$
\mathcal{L}_{\text{rank}}
=
-\log\sigma(s_{\text{positive}}-s_{\text{negative}}).
$$

### 9.8. Loss tổng M3

$$
\mathcal{L}_{\text{table}}
=
\lambda_{\text{det}}\mathcal{L}_{\text{TATR}}
+\lambda_{\text{role}}\mathcal{L}_{\text{table-role}}
+\lambda_{\text{cell}}\mathcal{L}_{\text{cell-type}}
+\lambda_{\text{link}}\mathcal{L}_{\text{header-link}}
+\lambda_{\text{tuple}}\mathcal{L}_{\text{result-tuple}}
+\lambda_{\text{abl}}\mathcal{L}_{\text{ablation}}
+\lambda_{\text{rank}}\mathcal{L}_{\text{result-rank}}.
$$

### 9.9. Evaluation

- table detection mAP;
- GriTS/TEDS;
- header-to-cell link F1;
- table-role macro-F1;
- result tuple exact/micro-F1;
- numeric cell accuracy;
- control–variant pairing accuracy;
- intervention/component relation F1;
- delta exact accuracy;
- ablation conclusion faithfulness.

---

## 10. M4 — Scientific Text Understanding

### 10.1. Vấn đề

Paper dài, có cấu trúc phân cấp và các loại câu mang chức năng khác nhau. Cùng một cụm từ có thể là background, contribution claim hoặc kết quả tùy section.

### 10.2. Kiến trúc hierarchical scientific encoder

M4 sử dụng ba tầng:

```text
Tokens
  ↓
Scientific token/sentence encoder
  ↓
Sentence/paragraph representations
  ↓
Section-aware long-document encoder
  ↓
Contextualized span/claim representations
```

#### Tầng 1 — Scientific token encoder

Dùng [SciBERT](https://aclanthology.org/D19-1371/) hoặc một encoder đã pretrain trên scientific text.

$$
\mathbf{H}^{\text{token}}
=
\operatorname{SciEnc}(x_1,\ldots,x_n).
$$

#### Tầng 2 — Sentence/paragraph pooling

$$
\mathbf{h}_{\text{sent}}
=
\operatorname{AttentionPool}
(\mathbf{H}^{\text{token}}_{\text{sent}}).
$$

Có thể thêm:

- section type embedding;
- relative position in section;
- citation density;
- equation/table reference features.

#### Tầng 3 — Long-document encoder

Dùng [Longformer](https://arxiv.org/abs/2004.05150) với local-window attention và global tokens hoặc một hierarchical Transformer.

Global attention đặt tại:

- section heading;
- paragraph start;
- `[CLAIM]`;
- equation/table reference;
- abstract/conclusion sentences.

Nếu full paper quá dài, xử lý theo section rồi dùng document-level Transformer trên các sentence/paragraph embeddings.

### 10.3. M4.1 — Rhetorical zoning

#### Labels

```text
background
problem
gap
motivation
objective
method
experiment_setup
result
analysis
contribution
limitation
future_work
other
```

Đây có thể là multi-label vì một câu vừa mô tả method vừa nêu contribution.

#### Head

$$
\mathbf{p}_{\text{rhet}}
=
\sigma(\mathbf{W}_{\text{rhet}}\mathbf{h}_{\text{sent}}+\mathbf{b}).
$$

#### Huấn luyện

- warm-start từ MuLMS-AZ hoặc argumentative zoning datasets;
- map taxonomy nguồn sang taxonomy SciLens;
- fine-tune bằng SciLensBench;
- weighted BCE/focal loss;
- hard example mining giữa `method`, `contribution` và `result`.

### 10.4. M4.2 — Scientific entity extraction

#### Entity types

```text
TASK, METHOD, COMPONENT, DATASET, SPLIT,
METRIC, SCORE, HYPERPARAMETER, RESOURCE,
ASSUMPTION, BASELINE
```

#### Kiến trúc

- encoder + BIO/span classification;
- biaffine span scorer hoặc span-based NER;
- entity normalization head để liên kết các alias.

$$
s_{\text{span}}(i,j,t)
=
\operatorname{MLP}_t
[\mathbf{h}_i;\mathbf{h}_j;\mathbf{e}_{\text{width}}].
$$

SciREX, SciER và AxCell có thể dùng cho warm-start một phần taxonomy.

### 10.5. M4.3 — Relation extraction

Relations:

```text
METHOD_HAS_COMPONENT
METHOD_EVALUATED_ON_DATASET
METHOD_MEASURED_BY_METRIC
SCORE_FOR_METHOD
CLAIM_ABOUT_METHOD
RESULT_SUPPORTS_CLAIM
```

Kiến trúc:

- pairwise biaffine classifier;
- graph-aware relation classifier sau khi tạo candidate nodes;
- negative sampling theo entity type constraints.

### 10.6. M4.4 — Atomic claim extraction

#### Task

Biến một câu phức thành các proposition có thể kiểm chứng:

```text
Input:
"Our model improves BLEU by 1.5 and reduces inference time by 20%."

Claims:
1. The proposed model improves BLEU by 1.5.
2. The proposed model reduces inference time by 20%.
```

#### Kiến trúc

Hai lựa chọn:

1. span/semantic-role based claim segmenter;
2. seq2seq claim decomposer.

Trong nghiên cứu, nên kết hợp:

- encoder classifier tìm claim-bearing sentences;
- constrained seq2seq tách atomic claims;
- entity preservation checker.

#### Huấn luyện

- human annotated atomic claims;
- synthetic split/merge augmentation;
- hard negatives là câu mô tả background hoặc speculation;
- SFT loss cho decomposition;
- auxiliary entity-copy loss.

### 10.7. M4.5 — Contribution extraction

Contribution không chỉ là câu chứa “we propose”.

Schema:

```text
problem
artifact_or_method
technical_difference
novelty_claim
supporting_result
evidence
```

Kiến trúc:

- claim-type classifier;
- relation head liên kết contribution với method/result;
- evidence linking head;
- graph aggregation từ Introduction, Method, Experiments và Conclusion.

Training:

- warm-start từ contribution/context summary data;
- positive claims do expert gắn;
- hard negative là câu related work có từ “propose” nhưng nói về paper khác;
- multi-label classification + relation loss.

### 10.8. M4.6 — Limitation extraction

Labels:

```text
author_stated
evidence_supported
analyst_hypothesis
unsupported
```

Phân loại thêm scope:

```text
data
method
evaluation
generalization
efficiency
reproducibility
ethics
theory
```

Training phải đặc biệt chú ý `unsupported` để mô hình học không tự bịa limitation.

### 10.9. Loss tổng M4

$$
\mathcal{L}_{\text{text}}
=
\lambda_{\text{rhet}}\mathcal{L}_{\text{rhet}}
+\lambda_{\text{ner}}\mathcal{L}_{\text{NER}}
+\lambda_{\text{rel}}\mathcal{L}_{\text{relation}}
+\lambda_{\text{claim}}\mathcal{L}_{\text{claim}}
+\lambda_{\text{contrib}}\mathcal{L}_{\text{contribution}}
+\lambda_{\text{limit}}\mathcal{L}_{\text{limitation}}.
$$

### 10.10. Phần nào được huấn luyện?

- SciBERT/Longformer backbone: fine-tune, không pretrain từ đầu.
- Pooling, section embeddings và task heads: tự xây.
- Claim decomposition: fine-tune seq2seq model nhỏ hoặc LLM adapter.
- Contribution/limitation heads: tự xây.

---

## 11. M5 — Scientific Document Graph Builder

### 11.1. Vấn đề

Các branch trước tạo objects riêng lẻ. M5 phải biến chúng thành graph nhất quán và loại các edge không hợp lệ.

### 11.2. Candidate graph construction

Rules tạo edge chắc chắn:

- Section `CONTAINS` Paragraph;
- Table `HAS_CELL` Cell;
- Equation `USES` Symbol;
- source callout `REFERENCES` Table/Equation;
- Result `DERIVED_FROM` Cell.

Neural models dự đoán edge ngữ nghĩa:

- Claim `SUPPORTED_BY` Cell/Span/Equation;
- Result `SUPPORTS` Contribution;
- Variant `REMOVES` Component;
- Limitation `QUALIFIES` Claim.

### 11.3. Graph consistency constraints

Ví dụ:

- `HAS_SCORE` chỉ nối Result với numeric Cell;
- `DEFINES` phải nối Span với Symbol/Method;
- `REMOVES` phải nối Variant với Component;
- mỗi Result phải có Dataset và Metric nếu output không abstain;
- source cell phải thuộc table tồn tại.

Constraint decoding có thể loại edge impossible trước và sau neural scoring.

### 11.4. Có cần huấn luyện M5?

Builder bản thân chủ yếu là schema + rule. Edge prediction được huấn luyện trong M4/M6/M7. Đây là quyết định có chủ ý: không dùng neural network cho quan hệ cấu trúc đã biết chắc.

---

## 12. M6 — Heterogeneous Scientific Document Graph Transformer

Đây là mô hình trung tâm của SciLens.

### 12.1. Vấn đề

Một contribution claim ở Introduction có thể được hỗ trợ bởi:

- equation ở Method;
- table ở Experiments;
- ablation ở subsection khác;
- limitation ở Discussion.

Flat chunking không tạo một representation thống nhất cho các loại object này.

### 12.2. Khởi tạo node embedding

Mỗi node $v$:

$$
\mathbf{h}_v^{(0)}
=
\mathbf{W}_{\tau(v)}
[\mathbf{e}_{\text{content}};
\mathbf{e}_{\text{type}};
\mathbf{e}_{\text{section}};
\mathbf{e}_{\text{position}};
\mathbf{e}_{\text{layout}};
\mathbf{e}_{\text{confidence}}].
$$

Trong đó:

- `content`: từ M2/M3/M4;
- `type`: node type embedding;
- `section`: section role/level;
- `position`: page và relative order;
- `layout`: bbox/style;
- `confidence`: parser/model confidence.

Node-specific content:

| Node | Content encoder |
|---|---|
| Sentence/Claim | SciBERT/Longformer |
| Equation | dual-view equation encoder |
| Symbol | LaTeX token + definition span encoder |
| Table | scientific table encoder |
| Cell/Result | cell + header path + numeric encoder |
| Method/Dataset/Metric | entity span + normalized name |
| Section | attention pooling các child nodes |

### 12.3. Relation-aware attention

SciLens dựa trên ý tưởng [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332), dùng tham số phụ thuộc node/edge type.

Với edge $e=(s,r,t)$:

$$
\mathbf{q}_t
=
\mathbf{W}^{\mathrm{Q}}_{\tau(t)}\mathbf{h}_t,
$$

$$
\mathbf{k}_s^{r}
=
\mathbf{W}^{\mathrm{K}}_{\tau(s),r}\mathbf{h}_s,
\qquad
\mathbf{v}_s^{r}
=
\mathbf{W}^{\mathrm{V}}_{\tau(s),r}\mathbf{h}_s.
$$

Attention:

$$
\alpha_{s\rightarrow t}^{r}
=
\operatorname{softmax}_{s\in\mathcal{N}(t)}
\left(
\frac{\mathbf{q}_t^\top
\mathbf{W}^{\mathrm{ATT}}_{r}\mathbf{k}_s^r}
{\sqrt{d}}
+b_r
\right).
$$

Message:

$$
\mathbf{m}_{s\rightarrow t}^{r}
=
\alpha_{s\rightarrow t}^{r}
\mathbf{W}^{\mathrm{MSG}}_{r}\mathbf{v}_s^r.
$$

Update:

$$
\mathbf{h}_t^{(l+1)}
=
\operatorname{LayerNorm}
\left(
\mathbf{h}_t^{(l)}
+
\operatorname{FFN}_{\tau(t)}
\left(
\sum_{r}\sum_{s\in\mathcal{N}_r(t)}
\mathbf{m}_{s\rightarrow t}^{r}
\right)
\right).
$$

### 12.4. Global paper node

Thêm node `Paper` kết nối tới:

- section nodes;
- top-ranked claims;
- tables/equations;
- contributions/limitations.

Nó cung cấp document-level representation nhưng không thay thế local edges.

### 12.5. Graph sparsification

Không nối mọi node với mọi node.

Giữ:

- structural edges;
- explicit cross-references;
- top-$k$ semantic candidate edges;
- section-local edges;
- learned evidence edges trên threshold.

Mục tiêu là tránh $O(|V|^2)$ và giảm noisy message passing.

### 12.6. Graph pretraining objectives

#### Masked node attribute modeling

Che node type/role/entity label rồi dự đoán:

$$
\mathcal{L}_{\text{MNM}}
=
-\sum_{v\in\mathcal{M}}\log p(a_v\mid G_{\setminus\mathcal{M}}).
$$

#### Edge prediction

$$
\mathcal{L}_{\text{edge}}
=
-\sum_{(u,r,v)}
\log p(r\mid \mathbf{h}_u,\mathbf{h}_v).
$$

#### Claim–evidence contrastive alignment

$$
\mathcal{L}_{\text{CE-align}}
=
-\log
\frac{\exp(\operatorname{sim}(c,e^+)/\tau)}
{\exp(\operatorname{sim}(c,e^+)/\tau)+
\sum_{e^-}\exp(\operatorname{sim}(c,e^-)/\tau)}.
$$

#### Equation–context alignment

Positive là equation với đoạn định nghĩa/giải thích thật; negative là equation cùng paper nhưng section khác.

#### Cell–header alignment

Dự đoán full header path của cell hoặc phân biệt đúng/sai header path.

#### Cross-modal matching

Phân biệt:

- đúng equation crop–LaTeX;
- đúng table image–canonical table;
- đúng claim–result cell.

### 12.7. Supervised multi-task fine-tuning

Sau pretraining, HGT được fine-tune cùng M7 heads. Ban đầu freeze các modality encoders để graph học ổn định, sau đó unfreeze một số layer cuối.

### 12.8. Sampling

Một paper có thể có hàng nghìn sentence/cell nodes. Training dùng:

- section-based subgraph;
- target-centered $k$-hop sampling;
- giữ toàn bộ evidence positives;
- hard negative nodes;
- global paper/section nodes.

### 12.9. Novelty thực sự

Chỉ “dùng HGT” chưa đủ mới. Novelty nằm ở:

- scientific multimodal node/edge schema;
- equation/table/ablation relations;
- cross-modal pretraining objectives;
- typed evidence grounding;
- planner và verifier sử dụng chung graph representation.

---

## 13. M7 — Multi-task Scientific Understanding Heads

M7 nhận contextualized node embeddings từ M6.

### 13.1. Salience head

Dự đoán node có cần vào summary:

$$
p_v^{\text{sal}}
=
\sigma(\operatorname{MLP}_{\text{sal}}(\mathbf{h}_v)).
$$

Có head riêng cho:

- claims;
- equations;
- tables;
- results;
- ablations;
- limitations.

Loss nên dùng listwise/pairwise ranking thay vì chỉ binary classification vì content budget hữu hạn.

### 13.2. Evidence linking head

Hai tầng:

1. bi-encoder retrieval lấy top-$k$;
2. cross-encoder/graph scorer rerank.

$$
s(c,e)
=
\operatorname{MLP}
[\mathbf{h}_c;\mathbf{h}_e;
\mathbf{h}_c\odot\mathbf{h}_e;
\mathbf{e}_{\text{rel-path}}].
$$

Output có thể là multi-evidence set.

### 13.3. Equation heads

- equation role;
- salience;
- symbol-definition link;
- equation–claim support.

### 13.4. Table/result heads

- table role;
- cell type;
- result tuple validity;
- main-result rank;
- comparable/not-comparable.

### 13.5. Ablation heads

- control detection;
- variant detection;
- component extraction;
- intervention relation;
- control–variant pairing;
- causal-strength label.

### 13.6. Contribution head

Dự đoán contribution tuple completeness:

```text
problem present?
artifact/method present?
technical difference present?
evidence present?
reported effect present?
```

### 13.7. Limitation head

Dự đoán:

- limitation scope;
- evidence strength;
- author-stated vs analyst-inferred;
- safe-to-present status.

### 13.8. Contradiction head

Phát hiện:

- abstract/body inconsistency;
- claim/table conflict;
- same metric nhưng different setting;
- contribution claim không có supporting result.

### 13.9. Multi-task loss

$$
\mathcal{L}_{\text{M7}}
=
\sum_{k=1}^{K}\lambda_k\mathcal{L}_k.
$$

Không nên cố định $\lambda_k=1$ mà không kiểm tra gradient. Ba lựa chọn:

1. manual weights dựa trên validation;
2. uncertainty weighting;
3. gradient normalization/dynamic task sampling.

Khuyến nghị ban đầu:

- train task heads riêng để kiểm tra;
- sau đó joint fine-tune theo nhóm task liên quan;
- cuối cùng mới train multi-task toàn cục.

### 13.10. Nhóm task để tránh negative transfer

```text
Group A: rhetorical + claim + contribution + limitation
Group B: equation role + symbol grounding + equation salience
Group C: table role + result + ablation
Group D: evidence + contradiction + factuality
Group E: global salience + planning
```

Nếu joint training làm một nhóm giảm mạnh, dùng task-specific adapters hoặc alternating updates.

---

## 14. M8 — Evidence-Aware Content Planner

### 14.1. Vấn đề

Generator dễ:

- ưu tiên đầu paper;
- bỏ sót ablation;
- lặp contribution;
- dùng evidence không đủ;
- vượt length budget.

Planner tách “chọn nội dung” khỏi “diễn đạt”.

### 14.2. Input

Các grounded candidates:

```text
(content node, type, salience, evidence set,
 confidence, section, estimated token cost)
```

### 14.3. Output

```json
{
  "problem": ["claim_12"],
  "gap": ["claim_19"],
  "method": ["claim_33", "eq_4"],
  "main_results": ["result_6"],
  "ablations": ["ablation_2"],
  "limitations": ["limit_1"],
  "order": ["problem", "gap", "method", "main_results", "ablations", "limitations"]
}
```

### 14.4. Kiến trúc

#### Candidate encoder

Dùng node embeddings từ HGT.

#### Budget-aware selector

Pointer Transformer hoặc set-to-sequence decoder:

$$
p(z_t=v\mid z_{<t},G,B)
=
\operatorname{softmax}
(\operatorname{score}(\mathbf{q}_t,\mathbf{h}_v)),
$$

trong đó $B$ là remaining token budget.

#### Coverage state

Planner theo dõi:

- section roles đã được phủ;
- content types đã được chọn;
- entities/results đã xuất hiện;
- redundancy với selected nodes.

#### Constraints

- main result phải có source cell/span;
- equation explanation phải có LaTeX + symbol grounding;
- ablation phải có control–variant pair;
- limitation phải có source type;
- mỗi field có budget tối thiểu/tối đa.

### 14.5. Huấn luyện

#### Teacher plan

Sinh gold plan từ expert summary và claim-evidence annotations.

#### Loss

$$
\mathcal{L}_{\text{plan}}
=
\mathcal{L}_{\text{select}}
+\lambda_o\mathcal{L}_{\text{order}}
+\lambda_c\mathcal{L}_{\text{coverage}}
+\lambda_r\mathcal{L}_{\text{redundancy}}
+\lambda_b\mathcal{L}_{\text{budget}}.
$$

#### Curriculum

1. train selection độc lập;
2. train ordering với teacher forcing;
3. joint selection–ordering;
4. optional policy optimization theo human usefulness, chỉ sau khi supervised model ổn định.

### 14.6. Không dùng LLM làm planner duy nhất

LLM có thể đề xuất plan nhưng plan phải được:

- map về valid node IDs;
- kiểm tra evidence;
- áp constraints;
- so với learned planner baseline.

---

## 15. M9 — Grounded Structured Generator

### 15.1. Nhiệm vụ

Chuyển structured plan và evidence packets thành:

- JSON đúng schema;
- Markdown dễ đọc;
- citations/source IDs;
- LaTeX giữ nguyên;
- wording phù hợp confidence.

### 15.2. Hai lựa chọn kiến trúc

#### Lựa chọn A — Open-weight encoder–decoder

Dùng LED/T5-like model:

```text
typed evidence serialization
       ↓
long-context encoder
       ↓
autoregressive decoder
       ↓
structured summary
```

[Longformer](https://arxiv.org/abs/2004.05150) giới thiệu LED cho long-document seq2seq và đánh giá trên arXiv summarization.

Ưu điểm:

- kiểm soát huấn luyện;
- reproducible;
- có thể fine-tune end-to-end.

Nhược điểm:

- khả năng giải thích và tiếng Việt có thể yếu hơn LLM lớn.

#### Lựa chọn B — Decoder-only LLM

Serialize plan thành typed prompt:

```text
<FIELD=METHOD>
<CLAIM id=claim_33>...</CLAIM>
<EQUATION id=eq_4>...</EQUATION>
<SYMBOL ... />
<EVIDENCE id=span_231>...</EVIDENCE>
```

Fine-tune bằng [LoRA](https://arxiv.org/abs/2106.09685) hoặc [QLoRA](https://arxiv.org/abs/2305.14314), hoặc gọi API nếu chưa có compute.

### 15.3. Grounded generation constraints

- mọi generated claim phải có `<cite:id>`;
- không cho model tự sinh source ID ngoài candidate list;
- equations được copy từ normalized object;
- numeric values được copy hoặc canonical formatter chèn;
- output JSON được constrained decode theo schema;
- limitation cần source-type token;
- system inference cần hedge marker.

### 15.4. Training data

Một training example:

```text
Input:
  content plan + evidence packets + target language + detail level

Target:
  structured fields + atomic claims + source IDs
```

Targets đến từ:

- expert summaries;
- section summaries;
- important equation explanations;
- result/ablation annotations;
- verified synthetic examples;
- teacher-generated drafts đã được human sửa.

### 15.5. Supervised fine-tuning loss

$$
\mathcal{L}_{\text{gen}}
=
-\sum_t w_t
\log p(y_t^\star\mid y_{<t}^\star,Z,G).
$$

Tăng trọng số cho:

- citation tokens;
- numbers;
- entity names;
- LaTeX delimiters;
- field boundaries.

### 15.6. Auxiliary grounding loss

Nếu decoder cross-attend evidence:

$$
\mathcal{L}_{\text{ground-gen}}
=
\operatorname{CE}
(a_t,e_t^\star),
$$

trong đó $a_t$ là evidence attribution của generated claim.

### 15.7. Preference optimization tùy chọn

Tạo cặp:

- preferred: faithful, complete, concise, cited;
- rejected: number swap, wrong cell, unsupported limitation, missing ablation.

Chỉ làm sau SFT. Không dùng LLM judge duy nhất để tạo preference labels; cần rules và human review.

### 15.8. Multilingual output

Để sinh tiếng Việt:

1. generate grounded semantic representation bằng tiếng Anh;
2. verbalize sang tiếng Việt nhưng giữ entity, number, LaTeX và source IDs;
3. chạy verifier lại sau dịch.

Không nên dịch raw paper trước khi trích xuất equation/table/result.

---

## 16. M10 — Typed Claim Verifier và Repair

### 16.1. Tại sao cần typed verifier?

Một textual NLI model có thể thấy chuỗi “31.4” ở đâu đó nhưng không biết đó là score của method nào. Equation verifier lại phải quan tâm symbol và phép toán. Vì vậy M10 gồm nhiều verifier.

### 16.2. Claim decomposition

Mỗi generated sentence được tách thành atomic claims:

$$
Y \rightarrow \{c_1,c_2,\ldots,c_m\}.
$$

Mỗi claim được phân loại:

```text
textual
numerical
table_comparison
equation_explanation
contribution
limitation
ablation
mixed
```

### 16.3. Textual entailment verifier

Dùng NLI cross-encoder:

$$
p(label\mid e,c)
=
\operatorname{softmax}\!\left(
\operatorname{CrossEncoder}(e,c)
\right),
$$

labels:

```text
entailed
contradicted
insufficient
```

[FENICE](https://aclanthology.org/2024.findings-acl.841/) là tham chiếu quan trọng vì dùng atomic claims và NLI alignment cho factuality evaluation.

Training:

- LongSciVerify/FENICE-style data;
- SciLens claim-evidence pairs;
- domain hard negatives;
- adversarial paraphrases.

### 16.4. Numerical verifier

Deterministic checks:

- value tồn tại trong source object;
- unit/percent đúng;
- variance đúng;
- delta tính đúng;
- rounding nằm trong tolerance;
- metric direction đúng.

Neural model chỉ dùng để map phrase sang typed result object; phép toán do code thực hiện.

### 16.5. Table-cell verifier

Kiểm tra:

- source cell ID;
- row path;
- column path;
- method;
- dataset;
- metric;
- setting;
- comparability.

Claim chỉ được accepted khi tất cả critical fields phù hợp.

### 16.6. Equation verifier

Kiểm tra:

- LaTeX compile;
- symbol coverage;
- symbol meanings có evidence;
- operator/term không bị đổi;
- explanation role phù hợp equation role;
- image-render similarity nếu OCR từ PDF.

Neural cross-encoder đánh giá equation-context consistency, nhưng rules bảo vệ symbol và LaTeX.

### 16.7. Contribution verifier

Tách:

- author novelty claim;
- technical difference;
- empirical support.

Không cho chuyển “authors claim X is novel” thành “X is definitively first” nếu không có literature-wide evidence.

### 16.8. Limitation verifier

- `author_stated`: phải có direct span;
- `evidence_supported`: phải có result/error/coverage evidence;
- `analyst_hypothesis`: phải dùng ngôn ngữ thận trọng;
- unsupported: xóa hoặc abstain.

### 16.9. Repair policy

```text
SUPPORTED
  → giữ claim

MINOR_ERROR + unambiguous source
  → deterministic repair

AMBIGUOUS
  → regenerate from restricted evidence

UNSUPPORTED
  → delete hoặc abstain

CONTRADICTED
  → delete + warning
```

Giới hạn số vòng repair, ví dụ tối đa hai vòng, để tránh feedback loop vô hạn.

### 16.10. Loss verifier

$$
\mathcal{L}_{\text{verify}}
=
\lambda_{\text{nli}}\mathcal{L}_{\text{NLI}}
+\lambda_{\text{type}}\mathcal{L}_{\text{claim-type}}
+\lambda_{\text{attr}}\mathcal{L}_{\text{attribution}}
+\lambda_{\text{conf}}\mathcal{L}_{\text{confidence}}.
$$

---

## 17. M11 — Confidence Calibration và Abstention

### 17.1. Confidence decomposition

Không dùng một probability duy nhất từ generator.

$$
C(c)
=
f(
C_{\text{parse}},
C_{\text{extract}},
C_{\text{evidence}},
C_{\text{generate}},
C_{\text{verify}},
C_{\text{agreement}}).
$$

Ví dụ:

- parser confidence;
- evidence retrieval margin;
- verifier entailment probability;
- agreement giữa source-native và PDF parse;
- number/equation rule checks;
- ensemble disagreement.

### 17.2. Calibrator

Dùng:

- temperature scaling;
- isotonic regression;
- small MLP calibration model.

Fit trên validation set, không fit trên test.

### 17.3. Decision policy

$$
\operatorname{action}(c)=
\begin{cases}
\text{accept}, & C(c)\ge \tau_{\text{high}},\\
\text{repair}, & \tau_{\text{low}}\le C(c)<\tau_{\text{high}},\\
\text{abstain}, & C(c)<\tau_{\text{low}}.
\end{cases}
$$

Threshold có thể khác theo claim type; numerical claim nên yêu cầu confidence cao hơn prose background.

### 17.4. Evaluation

- Expected Calibration Error;
- Brier score;
- risk–coverage curve;
- unsupported claim rate tại các coverage levels;
- abstention precision/recall.

---

## 18. Quy trình suy luận end-to-end

### Bước 1 — Ingest

Nhận PDF/LaTeX/XML, tính checksum, xác định version và license.

### Bước 2 — Route

M0 chọn source-native, born-digital hoặc OCR fallback.

### Bước 3 — Parse

M1 tạo page objects, section tree, spans, reading order và coordinates.

### Bước 4 — Equation branch

M2:

1. detect;
2. transcribe LaTeX;
3. normalize;
4. encode image/LaTeX/context;
5. ground symbols;
6. classify role;
7. rank salience.

### Bước 5 — Table branch

M3:

1. detect table;
2. recover rows/columns/cells;
3. build header paths;
4. classify table role;
5. extract result tuples;
6. detect control/variants;
7. compute ablation deltas.

### Bước 6 — Text understanding

M4:

1. encode full paper hierarchically;
2. classify rhetorical roles;
3. extract entities/relations;
4. decompose claims;
5. identify contribution/limitation candidates.

### Bước 7 — Build graph

M5 tạo typed graph và constraints.

### Bước 8 — Graph reasoning

M6 contextualizes mọi object qua relation-aware attention.

### Bước 9 — Predict structured information

M7 tạo:

- salience;
- evidence links;
- result/ablation relations;
- contribution/limitation labels;
- contradiction warnings.

### Bước 10 — Plan

M8 chọn nội dung theo output schema và token budget.

### Bước 11 — Generate

M9 sinh JSON/Markdown có source IDs.

### Bước 12 — Verify

M10 tách claim, kiểm tra theo type, repair/delete.

### Bước 13 — Calibrate

M11 gắn confidence và abstention.

### Bước 14 — Render

UI render LaTeX, tables, highlights và click-to-source.

---

## 19. Chiến lược huấn luyện tổng thể

Không nên bật gradient xuyên toàn bộ pipeline ngay từ đầu. Training end-to-end trực tiếp sẽ:

- tốn memory;
- khó debug;
- làm parser bị catastrophic forgetting;
- gây negative transfer giữa task;
- không xác định được module nào gây lỗi.

SciLens nên dùng curriculum nhiều giai đoạn.

### 19.1. Phase 0 — Chuẩn hóa dữ liệu và parser benchmark

Mục tiêu:

- xây unified schema;
- align PDF với LaTeX/XML;
- đo parser quality;
- tạo training examples có provenance.

Không huấn luyện model mới.

Deliverables:

- Document Object Model;
- source alignment;
- table/equation crops;
- parser confidence labels;
- error taxonomy.

### 19.2. Phase 1 — Domain-adapt modality encoders

#### Text encoder

- khởi tạo SciBERT/Longformer;
- continued pretraining bằng masked language modeling trên paper trong domain nếu cần;
- không dùng test papers cho domain-adaptive training nếu temporal leakage là vấn đề.

#### Equation encoder

- dùng pretrained OCR;
- train dual-view alignment bằng equation image–LaTeX–context triples;
- freeze OCR decoder trong bản đầu.

#### Table encoder

- dùng TATR cho structure;
- pretrain semantic encoder với masked cell/header prediction;
- train cell–header alignment.

Loss:

$$
\mathcal{L}_{\text{phase-1}}
=
\mathcal{L}_{\text{MLM}}
+\mathcal{L}_{\text{eq-align}}
+\mathcal{L}_{\text{cell-header}}
+\mathcal{L}_{\text{cross-modal}}.
$$

### 19.3. Phase 2 — Huấn luyện các extractor độc lập

Huấn luyện:

- rhetorical zoning;
- NER/relation;
- claim extraction;
- equation role/symbol grounding;
- table role/result tuples;
- ablation relations;
- contribution/limitation.

Mục tiêu:

- có baseline cho từng task;
- chọn encoder và hyperparameters;
- phát hiện label noise;
- không để HGT che giấu lỗi extractor.

### 19.4. Phase 3 — Graph pretraining

Tạo graph từ extractor predictions và gold annotations.

Training mixture:

- ban đầu dùng nhiều gold edges để graph học;
- dần thay bằng predicted edges để giảm train–test mismatch;
- edge dropout để tăng robustness;
- node masking;
- hard-negative evidence edges.

Loss:

$$
\mathcal{L}_{\text{graph-pre}}
=
\lambda_1\mathcal{L}_{\text{MNM}}
+\lambda_2\mathcal{L}_{\text{edge}}
+\lambda_3\mathcal{L}_{\text{claim-evidence}}
+\lambda_4\mathcal{L}_{\text{eq-context}}
+\lambda_5\mathcal{L}_{\text{cell-header}}.
$$

### 19.5. Phase 4 — Supervised multi-task HGT

#### Bước A

Freeze modality encoders, train HGT + M7 heads.

#### Bước B

Unfreeze layer cuối của text/equation/table encoders với learning rate thấp hơn.

Ví dụ:

```text
HGT/task heads LR:       2e-4
modality adapters LR:   5e-5
pretrained backbone LR: 1e-5
```

Đây chỉ là điểm khởi đầu; giá trị phải được tune trên validation.

#### Batch schedule

Alternating task batches:

```text
text → table → equation → evidence → mixed-graph
```

Oversample ablation/equation examples nếu chúng hiếm, nhưng báo cáo calibration trên distribution thật.

### 19.6. Phase 5 — Planner training

1. train node salience;
2. train selection;
3. train ordering;
4. train coverage/budget constraints;
5. evaluate plan trước khi nối generator.

Planner gold được tạo từ:

- expert content units;
- evidence annotations;
- alignment giữa gold summary và graph nodes;
- manual correction cho ambiguous cases.

### 19.7. Phase 6 — Generator SFT

#### Dataset format

```json
{
  "instruction": "Generate a structured paper summary.",
  "plan": {},
  "evidence_packets": [],
  "target": {},
  "language": "vi",
  "detail_level": "technical"
}
```

#### Training

- SFT với LoRA/QLoRA nếu dùng decoder-only LLM;
- curriculum từ field đơn đến toàn report;
- bắt đầu với extractive/low-abstraction targets;
- sau đó thêm reader-friendly explanations;
- weighted tokens cho citation/numbers/entities.

#### Scheduled corruption

Đưa một số evidence candidates gây nhiễu để model học không copy sai nguồn:

- cùng metric nhưng sai dataset;
- cùng value nhưng sai method;
- equation gần giống;
- limitation của paper khác.

### 19.8. Phase 7 — Verifier training

Positive:

- gold claims;
- verified generated claims.

Negative corruption taxonomy:

```text
entity_swap
dataset_swap
metric_swap
number_swap
row_swap
column_swap
direction_flip
unit_error
symbol_swap
operator_change
scope_exaggeration
causal_overclaim
unsupported_limitation
missing_qualifier
```

Hard negatives phải gần nghĩa với evidence; random negatives quá dễ không tạo verifier mạnh.

### 19.9. Phase 8 — Closed-loop evaluation, không phải joint backprop toàn bộ

Pipeline:

```text
plan → generate → verify → repair
```

Tối ưu theo:

- factual claim rate;
- evidence correctness;
- coverage;
- human usefulness;
- risk–coverage.

Trong giai đoạn đầu, không backprop qua parser và deterministic modules. Nếu cần joint refinement, chỉ fine-tune planner/generator với verifier-derived rewards đã được kiểm chứng bằng human correlation.

---

## 20. Hàm loss toàn hệ thống

Biểu diễn:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{parse}}
+\mathcal{L}_{\text{eq}}
+\mathcal{L}_{\text{table}}
+\mathcal{L}_{\text{text}}
+\mathcal{L}_{\text{graph-pre}}
+\mathcal{L}_{\text{M7}}
+\mathcal{L}_{\text{plan}}
+\mathcal{L}_{\text{gen}}
+\mathcal{L}_{\text{verify}}.
$$

Không tính tất cả trong cùng một batch. Đây là ký hiệu tổng quát cho training curriculum.

### 20.1. Loss–module mapping

| Loss | Model nhận gradient | Phase |
|---|---|---|
| $\mathcal{L}_{parse}$ | layout heads/parser adapters | tùy chọn |
| $\mathcal{L}_{eq}$ | equation adapters/heads | 1–2 |
| $\mathcal{L}_{table}$ | table encoder/heads | 1–2 |
| $\mathcal{L}_{text}$ | scientific text encoder/heads | 1–2 |
| $\mathcal{L}_{graph-pre}$ | HGT | 3 |
| $\mathcal{L}_{M7}$ | HGT + task heads | 4 |
| $\mathcal{L}_{plan}$ | planner | 5 |
| $\mathcal{L}_{gen}$ | generator adapters | 6 |
| $\mathcal{L}_{verify}$ | verifier | 7 |

### 20.2. Loss weighting

Theo dõi:

- scale của từng loss;
- gradient norm;
- task validation metric;
- negative transfer.

Không tăng weight một task chỉ vì dataset của nó lớn. Sampling rate và loss weight cần được tách biệt.

---

## 21. Dữ liệu huấn luyện

### 21.1. Dữ liệu có sẵn

| Dataset/resource | Module sử dụng |
|---|---|
| S2ORC | full-text structure, scientific pretraining |
| arXiv/PubMed summarization | long-document summarization |
| SciTLDR | short scientific summary |
| LongSumm | extended summary |
| What’s New? contribution/context | contribution classification/planning |
| MuLMS-AZ | rhetorical zoning |
| PubTables-1M | table detection/structure |
| AxCell | result extraction |
| SciREX/SciER | task/method/dataset/metric entities và relations |
| Nougat training resources | page-to-markup |
| MathNet/realFormula | formula recognition |
| LongSciVerify/FENICE annotations | factuality verifier |
| TracSum | traceable aspect summary |
| AbGen | taxonomy và reasoning quanh ablation |

Không dataset nào cung cấp toàn bộ label cần thiết. SciLensBench vẫn cần thiết.

### 21.2. SciLensBench labels

Mỗi paper:

#### Document labels

- section tree;
- blocks/read order;
- page/bbox;
- parser quality.

#### Text labels

- rhetorical role;
- entities/relations;
- atomic claims;
- contribution;
- limitation;
- key content units.

#### Equation labels

- bbox;
- normalized LaTeX;
- role;
- important/not important;
- symbol definitions;
- explanation key points.

#### Table labels

- structure/header paths;
- table role;
- result tuples;
- main results;
- comparability.

#### Ablation labels

- control;
- variants;
- components;
- interventions;
- metrics/scores;
- causal strength;
- supported conclusion.

#### Grounding labels

- claim–span;
- claim–cell;
- claim–equation;
- contribution–result;
- limitation–evidence.

#### Summary labels

- structured plan;
- short summary;
- detailed summary;
- source IDs cho từng claim.

### 21.3. Quy mô

#### Pilot

- 50–100 paper;
- đủ để xây schema, pipeline và baseline;
- chưa đủ để huấn luyện model lớn từ đầu.

#### Thesis-ready

- 300–500 paper được annotate có chọn lọc;
- pretraining/weak supervision trên corpus lớn;
- expert test set khoảng 50–100 paper.

#### Benchmark/paper lớn

- 1.000+ paper;
- temporal, venue và cross-domain splits;
- hidden expert test.

### 21.4. Weak supervision

Nguồn weak labels:

- section headings;
- “we propose”, “our contributions”;
- “where $x$ denotes”;
- “ablation”, “w/o”, “remove”;
- bold values;
- table caption;
- explicit Eq./Table references;
- LaTeX label/ref;
- abstract–body alignment.

Weak labels dùng để pretrain hoặc candidate generation, không dùng làm gold test.

### 21.5. Synthetic data

Tốt cho verifier và robustness:

- đổi number;
- đổi row/column;
- đổi metric direction;
- đổi symbol;
- xóa qualifier;
- đổi dataset;
- gán nhầm evidence.

Không dùng synthetic summaries như nguồn duy nhất cho generator vì dễ học phong cách và lỗi của teacher.

### 21.6. Data split và leakage

- split theo paper;
- gộp arXiv/camera-ready duplicates;
- temporal test;
- không để table/equation crop cùng paper rơi sang split khác;
- report riêng open và closed models;
- lưu paper version;
- kiểm tra near-duplicate bằng title/authors/text fingerprints.

---

## 22. Chiến lược negative sampling

Negative sampling là yếu tố quyết định cho evidence linking và verifier.

### 22.1. Easy negatives

Random object từ paper khác. Chỉ hữu ích lúc đầu.

### 22.2. In-paper negatives

- same section, khác claim;
- same table, khác row;
- same metric, khác dataset;
- same symbol, khác scope;
- same method name, khác variant.

### 22.3. Adversarial negatives

- đúng value nhưng sai provenance;
- đúng claim nhưng sai experimental setting;
- paraphrase có thêm một qualifier sai;
- equation giống nhưng đổi dấu;
- limitation hợp lý về mặt chung nhưng paper không nói.

### 22.4. Teacher-mined negatives

Dùng model hiện tại retrieve false positives khó nhất, human/rule filter rồi đưa lại training.

---

## 23. Evaluation theo module

### 23.1. M0/M1

- routing accuracy;
- block mAP/F1;
- reading-order accuracy;
- section-tree edit distance;
- bbox localization;
- cross-reference link F1.

### 23.2. M2

- equation detection mAP;
- normalized LaTeX exact/edit distance;
- compile success;
- render similarity;
- symbol grounding F1;
- equation-role macro-F1;
- salience Precision@k, Recall@k, nDCG;
- expert explanation faithfulness.

### 23.3. M3

- TEDS/GriTS;
- header-link F1;
- table-role macro-F1;
- result tuple F1;
- numeric accuracy;
- main-result Recall@k;
- control–variant accuracy;
- component/intervention F1;
- delta exact accuracy.

### 23.4. M4

- rhetorical macro-F1;
- NER span F1;
- relation F1;
- atomic claim precision/recall;
- contribution/limitation F1;
- source-type accuracy.

### 23.5. M6/M7

- edge prediction MRR/Hits@k;
- claim–evidence Recall@k;
- multimodal evidence correctness;
- salience nDCG;
- downstream task improvement so với no-graph.

### 23.6. M8

- content-unit coverage;
- type/section coverage;
- plan order agreement;
- redundancy;
- budget violation rate;
- unsupported item selection rate.

### 23.7. M9

- schema validity;
- citation validity;
- entity/number preservation;
- coverage;
- factuality;
- readability;
- ROUGE/BERTScore chỉ là metric phụ.

### 23.8. M10/M11

- factual error detection AUROC/F1;
- false acceptance rate;
- table/equation error accuracy;
- repair success;
- calibration ECE/Brier;
- risk–coverage curve.

### 23.9. End-to-end human evaluation

Các chuyên gia chấm blind:

1. correctness;
2. completeness;
3. equation usefulness;
4. result/ablation correctness;
5. contribution/limitation fairness;
6. traceability;
7. time saved;
8. overall preference.

---

## 24. Baselines

### B0 — Abstract

Dùng abstract gốc.

### B1 — Direct PDF-to-LLM

Raw extracted text/full PDF → một prompt → structured summary.

### B2 — Parser + flat chunk RAG + LLM

Không có section hierarchy hoặc graph.

### B3 — Hierarchical section summarization

Summary từng section rồi tổng hợp, text-only.

### B4 — Multimodal VLM direct

Page images → VLM → summary.

### B5 — Typed objects, no HGT

Equation/table/text extractors + rule-based planner.

### B6 — Homogeneous GAT/Graph Transformer

Không dùng node/edge-specific parameters.

### B7 — Proposed SciLens

Typed multimodal graph + HGT + planner + generator + verifier.

---

## 25. Ablation experiments của SciLens

Phải thực hiện:

1. bỏ HGT, dùng flat candidates;
2. HGT nhưng bỏ edge types;
3. bỏ equation image view;
4. bỏ symbol grounding;
5. linearize table thay scientific table encoder;
6. bỏ header-to-cell edges;
7. bỏ ablation-specific heads;
8. bỏ rhetorical roles;
9. bỏ evidence contrastive pretraining;
10. bỏ content planner;
11. bỏ deterministic numerical verifier;
12. bỏ typed verifier, dùng general NLI;
13. bỏ confidence/abstention;
14. single parser so với multi-path parsing;
15. generator frozen/API so với QLoRA fine-tune.

Mỗi ablation phải báo cáo metric liên quan. Ví dụ bỏ equation image view không nhất thiết ảnh hưởng ROUGE nhưng có thể ảnh hưởng LaTeX/explanation accuracy.

---

## 26. Ba cấu hình triển khai

### 26.1. Cấu hình A — MVP

```text
Docling/GROBID/Nougat
  → rule-based typed objects
  → prompts cho role/result/ablation
  → LLM API generation
  → deterministic number/source checks
```

Không có model mới đáng kể. Phù hợp demo 4–8 tuần.

### 26.2. Cấu hình B — Thesis-ready, khuyến nghị

```text
Pretrained parser/OCR
  → fine-tuned SciBERT/Longformer
  → scientific table/equation heads
  → HGT
  → multi-task salience/evidence/ablation
  → learned planner
  → LLM API hoặc QLoRA generator
  → typed verifier
```

Đóng góp chính:

- graph schema;
- HGT;
- ablation/evidence heads;
- SciLensBench-Pilot;
- typed verification.

### 26.3. Cấu hình C — Paper/benchmark đầy đủ

Thêm:

- dual-view equation encoder;
- custom table semantic encoder;
- graph self-supervised pretraining;
- QLoRA grounded generator;
- preference optimization;
- calibrated selective prediction;
- 1.000+ paper benchmark.

---

## 27. Khuyến nghị phạm vi mô hình

Nếu một người thực hiện, không nên tự xây đồng thời mọi branch. Chọn:

### Phương án 1 — Ablation-centered

- pretrained parser/table structure;
- tự xây scientific table encoder;
- result/ablation graph;
- evidence-aware ablation summarizer;
- typed numerical verifier.

Đây là hướng gọn và novelty rõ.

### Phương án 2 — Equation-centered

- pretrained equation OCR;
- tự xây dual-view equation encoder;
- symbol grounding;
- equation salience;
- faithful explanation verifier.

### Phương án 3 — Unified graph, khuyến nghị nếu có nhóm

- dùng pretrained branch models;
- tự xây HGT + planner + verifier;
- equation/table chỉ fine-tune heads.

---

## 28. Tài nguyên tính toán

Các con số dưới đây chỉ là planning estimate, phụ thuộc model và sequence size.

### MVP

- không cần huấn luyện model lớn;
- một GPU tầm trung cho local OCR/encoders hoặc dùng cloud/API;
- cache mọi parser output.

### Thesis-ready

- 1–4 GPU có VRAM phù hợp;
- mixed precision;
- gradient accumulation;
- LoRA/QLoRA cho generator;
- subgraph batching;
- freeze vision/OCR backbones;
- offline preprocessing cho page/table/equation embeddings.

### Full research

- multi-GPU cho long-context generator và graph pretraining;
- distributed data processing;
- object storage cho page images/crops;
- experiment tracking và dataset versioning.

Nguyên tắc tiết kiệm:

- không chạy VLM trên page không cần thiết;
- không reparse paper giữa experiments;
- cache normalized objects;
- crop-level vision inference;
- retrieve trước, generate sau;
- PEFT thay full fine-tuning generator.

---

## 29. Cấu trúc codebase đề xuất

```text
scilens/
├── configs/
├── schemas/
│   ├── document.py
│   ├── graph.py
│   └── summary.py
├── ingestion/
│   ├── source_router.py
│   └── version_resolver.py
├── parsing/
│   ├── grobid_adapter.py
│   ├── docling_adapter.py
│   ├── latex_parser.py
│   └── layout_quality.py
├── equations/
│   ├── detector.py
│   ├── transcriber.py
│   ├── normalizer.py
│   ├── dual_encoder.py
│   ├── symbol_grounder.py
│   └── salience.py
├── tables/
│   ├── detector.py
│   ├── structure.py
│   ├── canonicalizer.py
│   ├── semantic_encoder.py
│   ├── result_extractor.py
│   └── ablation_extractor.py
├── text/
│   ├── hierarchical_encoder.py
│   ├── rhetorical.py
│   ├── entities.py
│   ├── claims.py
│   ├── contributions.py
│   └── limitations.py
├── graph/
│   ├── builder.py
│   ├── constraints.py
│   ├── hgt.py
│   └── pretraining.py
├── planning/
│   ├── selector.py
│   ├── ordering.py
│   └── budget.py
├── generation/
│   ├── serializer.py
│   ├── generator.py
│   └── constrained_decode.py
├── verification/
│   ├── claim_decomposer.py
│   ├── text_nli.py
│   ├── numeric.py
│   ├── table.py
│   ├── equation.py
│   ├── repair.py
│   └── calibration.py
├── training/
│   ├── datasets.py
│   ├── losses.py
│   ├── curriculum.py
│   └── negative_mining.py
├── evaluation/
└── app/
```

---

## 30. Data flow example

Giả sử paper có:

- câu “We introduce a graph-aware fusion module”;
- Eq. (4) định nghĩa fusion;
- Table 2 báo kết quả chính;
- Table 4 bỏ fusion module làm F1 giảm từ 88.4 xuống 86.9.

### Parsing

```text
claim_12:
  "The paper introduces a graph-aware fusion module."

eq_4:
  role = model_definition

result_full:
  F1 = 88.4

result_wo_fusion:
  F1 = 86.9
```

### Graph

```text
claim_12 MENTIONS component_fusion
eq_4 DEFINES component_fusion
variant_wo_fusion REMOVES component_fusion
result_wo_fusion BELONGS_TO variant_wo_fusion
result_full COMPARES_WITH result_wo_fusion
```

### Ablation engine

$$
\Delta=88.4-86.9=1.5.
$$

### Content plan

```json
{
  "contribution": ["claim_12"],
  "key_equation": ["eq_4"],
  "main_result": ["result_full"],
  "ablation": ["variant_wo_fusion", "result_wo_fusion"]
}
```

### Generated output

```text
The method introduces a graph-aware fusion module [claim_12].
Equation (4) defines how the representations are combined [eq_4].
Removing the module is associated with a 1.5-point F1 decrease
on the reported setting [table_4:full_f1, table_4:wo_fusion_f1].
```

### Verification

- entity: correct;
- equation source: correct;
- values: exact;
- delta: correct;
- metric direction: correct;
- language “associated with” phù hợp controlled ablation;
- accepted.

---

## 31. Failure modes và fallback

| Failure | Detection | Fallback |
|---|---|---|
| Reading order sai | parser disagreement/low confidence | source-native hoặc page OCR |
| Equation OCR không compile | LaTeX validator | source LaTeX/manual warning |
| Symbol không có definition | grounding confidence thấp | không giải thích symbol |
| Table header mơ hồ | structure confidence thấp | crop-level VLM hoặc abstain |
| Không xác định metric direction | metric registry/context không đủ | báo raw scores, không kết luận |
| Multiple ablation changes | causal-strength classifier | dùng descriptive wording |
| Evidence retrieval conflict | verifier contradiction | bỏ claim/cảnh báo |
| LLM sinh source ID giả | schema validation | reject/regenerate |
| LLM bịa limitation | limitation verifier | xóa hoặc gắn hypothesis |
| Paper quá dài | section graph + subgraph retrieval | hierarchical processing |

---

## 32. Kiến trúc được khuyến nghị cuối cùng

Phiên bản cân bằng nhất:

```text
Pretrained:
  Docling/GROBID + TATR + Nougat

Fine-tuned:
  SciBERT/Longformer
  equation role/symbol heads
  table role/result heads

Built by project:
  Scientific Document Graph schema
  Heterogeneous Graph Transformer
  claim–evidence contrastive linker
  result/ablation relation heads
  budget-aware content planner
  typed verifier
  confidence/abstention

Generation:
  QLoRA fine-tuned open LLM hoặc API LLM
  nhưng chỉ nhận verified evidence packets
```

Đây không phải kiến trúc một neural network duy nhất. Nó là một **compound AI system** có nhiều neural modules, shared structured representation và deterministic safeguards. Đối với bài toán document intelligence phức tạp, kiến trúc module hóa này thực tế và dễ kiểm chứng hơn một model monolithic.

### Research claim nên kiểm nghiệm

> So với direct long-context LLM và hierarchical text-only summarization, typed heterogeneous graph reasoning kết hợp modality-specific verification sẽ tăng coverage của nội dung kỹ thuật và giảm unsupported claims, đặc biệt đối với equation explanation, numerical results và ablation conclusions.

---

## 33. Tài liệu tham khảo chính

### Document parsing và multimodal understanding

1. Huang et al. (2022), [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387).
2. Auer et al. (2024), [Docling Technical Report](https://arxiv.org/abs/2408.09869).
3. GROBID, [Official Documentation](https://grobid.readthedocs.io/en/latest/).
4. Blecher et al. (2023), [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418).

### Equation và table

5. Schmitt-Koopmann et al. (2024), [MathNet: A Data-Centric Approach for Printed Mathematical Expression Recognition](https://arxiv.org/abs/2404.13667).
6. Smock et al. (2021/2022), [PubTables-1M](https://arxiv.org/abs/2110.00061).
7. Microsoft, [Table Transformer official repository](https://github.com/microsoft/table-transformer).
8. Herzig et al. (2020), [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://research.google/pubs/tapas-weakly-supervised-table-parsing-via-pre-training/).
9. Kardas et al. (2020), [AxCell: Automatic Extraction of Results from Machine Learning Papers](https://aclanthology.org/2020.emnlp-main.692/).
10. Jain et al. (2020), [SciREX](https://aclanthology.org/2020.acl-main.670/).

### Scientific text, graph và summarization

11. Beltagy et al. (2019), [SciBERT](https://aclanthology.org/D19-1371/).
12. Beltagy et al. (2020), [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
13. Hu et al. (2020), [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332).
14. Cohan et al. (2018), [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/abs/1804.05685).
15. Cachola et al. (2020), [SciTLDR](https://aclanthology.org/2020.findings-emnlp.428/).
16. Hayashi et al. (2023), [What’s New? Summarizing Contributions in Scientific Literature](https://aclanthology.org/2023.eacl-main.72/).

### Grounding, factuality và efficient fine-tuning

17. Scirè et al. (2024), [FENICE](https://aclanthology.org/2024.findings-acl.841/).
18. Bishop et al. (2024), [LongDocFACTScore/LongSciVerify](https://aclanthology.org/2024.lrec-main.941/).
19. Chu et al. (2025), [TracSum](https://aclanthology.org/2025.emnlp-main.43/).
20. Hu et al. (2021), [LoRA](https://arxiv.org/abs/2106.09685).
21. Dettmers et al. (2023), [QLoRA](https://arxiv.org/abs/2305.14314).
22. Zhao et al. (2025), [AbGen](https://arxiv.org/abs/2507.13300).
