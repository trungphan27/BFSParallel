# SciLens: Kiến trúc mô hình theo luồng Input → Output

> Đây là bản kiến trúc rút gọn, tập trung vào **paper đi qua hệ thống như thế
> nào**, mỗi công đoạn giải quyết việc gì, dùng mô hình Deep Learning nào và tạo
> đầu ra gì.

---

## 1. Input và output

### Input

Hệ thống nhận:

- file PDF của scientific paper;
- source LaTeX, XML hoặc HTML nếu có;
- tùy chọn: ngôn ngữ và độ dài bản tóm tắt mong muốn.

### Output

Hệ thống trả về một báo cáo có cấu trúc:

- TL;DR;
- bài toán, động lực và research gap;
- đóng góp kỹ thuật;
- tóm tắt từng section;
- công thức quan trọng dưới dạng LaTeX và phần giải thích;
- experimental setup;
- bảng và kết quả chính;
- ablation study;
- ưu điểm và hạn chế;
- nguồn bằng chứng cho từng nhận định;
- confidence và cảnh báo khi dữ liệu không đủ tin cậy.

---

## 2. Toàn bộ luồng xử lý

```text
PDF / LaTeX / XML
        │
        ▼
[1] Kiểm tra và chọn nguồn tài liệu
        │
        ▼
[2] Parse layout và cấu trúc paper
        │
        ├──────────────┬────────────────┐
        ▼              ▼                ▼
[3A] Text         [3B] Equation    [3C] Table
understanding     understanding     & result extraction
        │              │                │
        └──────────────┴────────────────┘
                       │
                       ▼
[4] Dựng Scientific Document Graph
                       │
                       ▼
[5] Graph reasoning và các task heads
                       │
                       ▼
[6] Chọn nội dung và bằng chứng cần tóm tắt
                       │
                       ▼
[7] Sinh báo cáo có cấu trúc
                       │
                       ▼
[8] Kiểm chứng, sửa lỗi và gắn confidence
                       │
                       ▼
              STRUCTURED SUMMARY
```

Ba nhánh Text, Equation và Table chạy song song sau khi paper đã được parse.
Chúng không tự sinh summary riêng; kết quả được hợp nhất trước khi hệ thống lựa
chọn nội dung và sinh câu trả lời.

---

## 3. Công đoạn 1 — Kiểm tra và chọn nguồn tài liệu

### Vấn đề cần giải quyết

Một paper có thể:

- có source LaTeX/XML chất lượng cao;
- là PDF có text layer;
- là PDF scan phải OCR;
- có source và PDF không cùng version;
- bị lỗi font, reading order hoặc mất công thức.

Nếu chọn sai nguồn, mọi bước sau đều bị sai.

### Xử lý

1. Kiểm tra PDF có render được không.
2. Kiểm tra text layer, số trang và chất lượng ký tự.
3. Kiểm tra source LaTeX/XML có tồn tại và cùng version không.
4. Dự đoán chất lượng text, layout, equation và table.
5. Chọn một trong các đường:
   `source-native`, `PDF-layout`, `vision-OCR` hoặc `hybrid`.

### Deep Learning sử dụng

- **Mặc định:** rules và quality metrics.
- **Khi cần huấn luyện:** một CNN/ViT nhỏ mã hóa page thumbnails, kết hợp với
  metadata qua MLP để phân loại route và dự đoán quality score.

Đây không phải đóng góp nghiên cứu chính. Không cần dùng LLM.

### Output

- route được chọn;
- quality scores;
- PDF pages và source đã xác minh;
- cảnh báo document-level.

---

## 4. Công đoạn 2 — Parse layout và cấu trúc paper

### Vấn đề cần giải quyết

PDF chỉ lưu ký tự và tọa độ, không thực sự biết đâu là:

- title, heading, paragraph;
- equation;
- table và caption;
- figure;
- footnote;
- thứ tự đọc;
- section hierarchy.

### Xử lý

1. Render PDF thành page images.
2. Detect các layout blocks.
3. Gắn text spans/OCR vào từng block.
4. dự đoán reading order.
5. Ghép blocks thành paragraph và section tree.
6. Gắn caption với table/figure/equation.
7. Căn chỉnh với LaTeX/XML nếu có.

### Deep Learning sử dụng

- **Layout detector:** LayoutLMv3, DiT hoặc ViT/ResNet + DETR.
- **Reading-order model:** Transformer hoặc graph edge classifier.
- **OCR/page-to-markup fallback:** pretrained Nougat hoặc mô hình tương đương.
- **Table region detector:** Table Transformer.

Nên sử dụng pretrained model rồi fine-tune theo domain, không huấn luyện parser
từ đầu.

### Output

Một Document Object Model:

```text
Paper
 ├── Section
 │    ├── Paragraph
 │    ├── Equation
 │    ├── Table
 │    └── Figure
 └── Appendix / References
```

Mỗi object giữ page, bounding box, reading order và source ID.

---

## 5. Công đoạn 3A — Hiểu nội dung văn bản

### Vấn đề cần giải quyết

Hệ thống phải biết mỗi đoạn đang nói về:

- bài toán và động lực;
- research gap;
- phương pháp;
- experimental setup;
- kết quả;
- đóng góp;
- limitation.

Nó cũng phải trích xuất task, method, component, dataset, metric, baseline và
các claim quan trọng.

### Xử lý

1. Mã hóa sentence và paragraph.
2. Phân loại rhetorical role.
3. Trích xuất scientific entities.
4. Trích xuất relations giữa entities.
5. Tách câu thành atomic claims.
6. Phân loại contribution và limitation.

### Deep Learning sử dụng

- **Token encoder:** SciBERT, DeBERTa hoặc pretrained scientific encoder.
- **Long-document encoder:** Longformer, BigBird hoặc hierarchical
  Transformer.
- **Entity extraction:** span classifier hoặc biaffine span model.
- **Relation extraction:** pairwise/biaffine relation classifier.
- **Claim và rhetorical heads:** MLP multi-label classifiers.

### Output

- rhetorical labels;
- entities và relations;
- atomic claims;
- contribution candidates;
- limitation candidates;
- embeddings cho sentence/paragraph.

---

## 6. Công đoạn 3B — Nhận dạng và hiểu công thức

### Vấn đề cần giải quyết

Không phải mọi công thức đều quan trọng. Hệ thống phải:

- detect công thức;
- khôi phục đúng LaTeX;
- biết công thức có vai trò gì;
- tìm định nghĩa của từng ký hiệu;
- chọn công thức cần đưa vào summary;
- giải thích nhưng không bịa ý nghĩa.

### Xử lý

1. Detect và crop equation.
2. Chuyển ảnh công thức thành LaTeX.
3. Chuẩn hóa và kiểm tra LaTeX có render được không.
4. Mã hóa đồng thời LaTeX và context.
5. Liên kết symbol với definition spans.
6. Phân loại equation role và salience.
7. Tạo explanation có dẫn nguồn.

### Deep Learning sử dụng

- **Formula OCR:** Vision Transformer/Swin encoder +
  autoregressive Transformer decoder; dùng pretrained UniMERNet, MathNet hoặc
  im2latex-style model.
- **LaTeX encoder:** Transformer trên LaTeX tokens hoặc formula AST.
- **Context encoder:** scientific text Transformer.
- **Dual-view fusion:** cross-attention giữa formula và surrounding text.
- **Symbol grounding:** span-pair classifier hoặc cross-encoder.
- **Role/salience:** MLP heads trên fused representation.

Phần có giá trị nghiên cứu không nằm ở OCR, mà ở **symbol grounding,
role/salience và grounded explanation**.

### Output

Với mỗi công thức quan trọng:

- LaTeX đã xác minh;
- equation role;
- danh sách symbol và định nghĩa;
- surrounding evidence;
- salience score;
- explanation candidate;
- confidence.

---

## 7. Công đoạn 3C — Hiểu bảng, kết quả và ablation

### Vấn đề cần giải quyết

Một giá trị trong bảng chỉ có ý nghĩa khi biết:

- method/variant nào;
- dataset và split nào;
- metric nào;
- chiều metric cao hơn hay thấp hơn tốt;
- control và variant có cùng điều kiện không;
- ô nào là bằng chứng.

### Xử lý

1. Detect bảng.
2. Khôi phục rows, columns, headers và spanning cells.
3. Dựng canonical table.
4. Phân loại table role.
5. Trích xuất result tuples.
6. Chọn main results.
7. Với bảng ablation:
   - tìm control;
   - tìm variants;
   - xác định component/intervention;
   - kiểm tra comparability;
   - tính delta bằng code.

### Deep Learning sử dụng

- **Table detection/structure:** Table Transformer hoặc DETR.
- **Table encoder:** Transformer trên cells kết hợp row, column, header và
  visual embeddings.
- **Table role classifier:** MLP trên table embedding + caption + section.
- **Result extraction:** entity/relation classifier trên cells và text.
- **Ablation relation model:** cross-encoder hoặc graph relation classifier
  giữa control, variant và component.

Các phép tính score/delta dùng chương trình deterministic, không giao cho LLM.

### Output

- canonical tables;
- result tuples;
- main-result candidates;
- ablation relations;
- control–variant deltas;
- evidence cell IDs;
- confidence.

---

## 8. Công đoạn 4 — Dựng Scientific Document Graph

### Vấn đề cần giải quyết

Sau ba nhánh, thông tin vẫn rời rạc. Ví dụ:

- một claim trong Conclusion tham chiếu bảng ở Experiments;
- một equation được giải thích ở paragraph khác;
- một ablation row tương ứng với component trong Method.

Hệ thống cần một biểu diễn chung để nối chúng lại.

### Xử lý

Tạo graph gồm các node:

```text
section, paragraph, claim, entity, equation, symbol,
table, cell, result, ablation, citation
```

Các edge quan trọng:

```text
contains, defines, refers_to, supports, has_result,
has_cell, compares_with, ablates, contradicts
```

### Deep Learning sử dụng

- Bản thân graph builder chủ yếu dùng schema và rules.
- Edge chưa biết được dự đoán bằng relation classifiers từ các bước trước.

### Output

Scientific Document Graph có typed nodes, typed edges, source IDs và
provenance.

---

## 9. Công đoạn 5 — Graph reasoning và dự đoán thông tin quan trọng

### Vấn đề cần giải quyết

Một object chỉ có thể được đánh giá đúng khi xét quan hệ với toàn paper. Ví dụ,
một bảng có thể rất lớn nhưng không phải main result; một equation ngắn có thể
là objective quan trọng nhất.

### Xử lý

1. Khởi tạo embedding cho từng loại node.
2. Truyền thông tin qua typed edges.
3. Học quan hệ xuyên modality.
4. Dự đoán salience và evidence links.
5. Dự đoán contribution, limitation, result và ablation cuối cùng.

### Deep Learning sử dụng

- **Heterogeneous Graph Transformer — HGT** là mô hình cốt lõi.
- Mỗi node type và edge type có projection/attention parameters riêng.
- Các multi-task heads nhận contextualized node embeddings:
  - salience head;
  - evidence-linking head;
  - equation heads;
  - result/ablation heads;
  - contribution/limitation heads.

Đây là một trong các phần phù hợp nhất để tạo novelty cho project.

### Output

- ranked content candidates;
- verified graph relations;
- contribution/limitation predictions;
- important equations, results và ablations;
- evidence candidates.

---

## 10. Công đoạn 6 — Chọn nội dung và lập kế hoạch tóm tắt

### Vấn đề cần giải quyết

Đưa toàn bộ paper vào LLM khiến output:

- thiếu mục quan trọng;
- dùng quá nhiều token cho Related Work;
- bỏ equation/table;
- lặp ý;
- khó truy nguồn.

### Xử lý

Planner tạo kế hoạch cho từng output section:

```text
section cần viết
→ nội dung cần nói
→ evidence IDs
→ độ ưu tiên
→ token budget
→ trường hợp phải abstain
```

### Deep Learning sử dụng

- Transformer/cross-encoder mã hóa candidate và output intent.
- Budget-aware selector chọn nodes/evidence.
- Có thể dùng pointer network hoặc constrained autoregressive planner.
- HGT embeddings từ bước trước là input chính.

LLM có thể hỗ trợ tạo teacher plans, nhưng planner cuối nên được huấn luyện và
ràng buộc theo schema.

### Output

Một content plan có cấu trúc, đã chọn đủ text, equation, table cells và
claim–evidence.

---

## 11. Công đoạn 7 — Sinh bản tóm tắt có cấu trúc

### Vấn đề cần giải quyết

Generator phải viết dễ hiểu nhưng không được:

- thay đổi số;
- gắn kết quả cho sai method;
- giải thích sai symbol;
- biến limitation suy luận thành fact;
- tạo claim không có evidence.

### Xử lý

1. Nhận content plan và evidence bundle.
2. Sinh từng mục theo output schema.
3. Giữ source IDs trong intermediate output.
4. Sinh LaTeX từ object đã xác minh, không tự chép lại từ trí nhớ.
5. Tách output thành atomic claims để kiểm tra.

### Deep Learning sử dụng

Một trong hai phương án:

- **Open-weight encoder–decoder:** LongT5, LED hoặc model tương đương;
- **Decoder-only LLM:** fine-tune bằng LoRA/QLoRA với structured instruction.

Generator cần supervised fine-tuning trên grounded summaries. API LLM có thể
dùng cho baseline hoặc MVP, nhưng không nên là toàn bộ đóng góp của project.

### Output

Bản summary nháp có cấu trúc, source IDs và các atomic claims.

---

## 12. Công đoạn 8 — Kiểm chứng, sửa lỗi và gắn confidence

### Vấn đề cần giải quyết

Một textual verifier duy nhất không đủ:

- số phải đối chiếu đúng cell;
- delta phải tính lại;
- equation phải khớp LaTeX và symbol;
- contribution/limitation phải đúng scope.

### Xử lý

Mỗi atomic claim được route đến verifier phù hợp:

| Claim type | Cách kiểm tra |
|---|---|
| Textual claim | entailment với source spans |
| Numerical claim | đối chiếu tuple và tính lại |
| Table claim | kiểm tra cell, row và header path |
| Equation claim | kiểm tra LaTeX, symbol và context |
| Contribution | kiểm tra novelty statement và evidence |
| Limitation | kiểm tra tác giả nêu hay hệ thống suy luận |

Claim sai được sửa, hạ confidence, loại bỏ hoặc chuyển thành cảnh báo.

### Deep Learning sử dụng

- NLI/cross-encoder cho textual entailment.
- Cross-encoder cho claim–cell và claim–equation alignment.
- MLP/calibration model kết hợp parser confidence, generation confidence và
  verifier scores.
- Temperature scaling, isotonic regression hoặc conformal calibration cho
  confidence/abstention.
- Numerical recomputation và schema validation dùng code.

### Output cuối cùng

- structured summary đã kiểm chứng;
- claim–evidence map;
- confidence theo từng mục;
- parsing/verification warnings;
- abstention tại nơi không đủ bằng chứng.

---

## 13. Thành phần nào dùng pretrained, thành phần nào phải xây?

| Thành phần | Khuyến nghị |
|---|---|
| PDF OCR/layout | dùng pretrained, chỉ fine-tune nếu cần |
| Formula image-to-LaTeX | dùng pretrained |
| Table detection/structure | dùng pretrained |
| Scientific text encoder | dùng pretrained rồi fine-tune |
| Equation role/salience/symbol grounding | **tự xây và huấn luyện** |
| Result/ablation extraction | **tự xây và huấn luyện** |
| Scientific Document Graph | **tự thiết kế schema** |
| Heterogeneous Graph Transformer | **mô hình nghiên cứu chính** |
| Content Planner | **tự xây và huấn luyện** |
| Generator | fine-tune open model hoặc dùng LLM API cho baseline |
| Typed Verifier | **tự xây/fine-tune** kết hợp rules |
| Calibration/abstention | fit trên calibration set riêng |

Project vì vậy **không chỉ gọi API**. Các pretrained model xử lý perception cơ
bản; phần nghiên cứu nằm ở multimodal scientific understanding, graph
reasoning, content planning, ablation extraction và grounded verification.

---

## 14. Chiến lược huấn luyện rút gọn

1. **Không huấn luyện end-to-end ngay từ đầu.**
2. Benchmark và khóa parser.
3. Fine-tune riêng text, equation và table extractors.
4. Dựng graph từ output đã kiểm tra.
5. Pretrain HGT bằng edge prediction và cross-modal alignment.
6. Fine-tune HGT cùng các multi-task heads trên SciLensBench.
7. Huấn luyện planner từ gold/verified teacher plans.
8. Fine-tune generator trên evidence-grounded summaries.
9. Huấn luyện typed verifiers bằng gold data và controlled corruptions.
10. Fit calibrator trên calibration split; đánh giá một lần trên test.

Không cần backpropagate xuyên toàn bộ PDF parser → LLM. Huấn luyện module-wise
giúp xác định lỗi, giảm compute và phù hợp quy mô luận văn.

---

## 15. Cấu hình nên triển khai

### Nếu làm MVP

```text
Pretrained parser
→ pretrained formula/table models
→ rules + scientific encoder
→ structured evidence
→ LLM API
→ deterministic verifier
```

### Nếu làm luận văn có novelty

```text
Pretrained parser
→ fine-tuned Text/Equation/Table extractors
→ Scientific Document Graph
→ Heterogeneous Graph Transformer
→ trainable Content Planner
→ LoRA-fine-tuned Generator
→ Typed Verifier + Calibration
```

Phạm vi luận văn nên tập trung mạnh vào một trong hai hướng:

- **Equation-centered:** symbol grounding, equation salience và explanation;
- **Ablation-centered:** table understanding, control–variant extraction và
  evidence-grounded ablation summary.

HGT và planner đóng vai trò hợp nhất hệ thống, nhưng không nhất thiết phải
huấn luyện lại toàn bộ OCR/layout backbone.

---

## 16. Tóm tắt kiến trúc trong một câu

**SciLens chuyển PDF thành các object có cấu trúc, hiểu riêng text–equation–
table, liên kết chúng bằng Heterogeneous Scientific Document Graph, chọn bằng
chứng trước khi sinh, rồi kiểm chứng từng loại claim trước khi trả về bản tóm
tắt.**

