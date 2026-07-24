# SciLens: Evidence-Grounded Multimodal Scientific Paper Summarization

> Đề xuất project nghiên cứu và triển khai hệ thống tóm tắt bài báo khoa học theo cấu trúc, có khả năng xử lý công thức, bảng biểu, kết quả thực nghiệm, ablation study, đóng góp kỹ thuật, ưu điểm và hạn chế.
>
> Ngày rà soát tài liệu: **24/07/2026**

---

## 1. Tóm tắt đề xuất

### 1.1. Bài toán

Các hệ thống tóm tắt paper thông thường thường tạo ra một đoạn văn tương tự abstract hoặc TL;DR. Cách tiếp cận này chưa đáp ứng tốt nhu cầu đọc paper thực tế vì:

- không cho biết thông tin đến từ section, đoạn, bảng hoặc công thức nào;
- dễ bỏ qua chi tiết kỹ thuật nằm ngoài abstract và introduction;
- không giữ được cấu trúc toán học, quan hệ giữa công thức và định nghĩa ký hiệu;
- khó hiểu đúng các bảng có header nhiều tầng, metric khác chiều tối ưu hoặc nhiều experimental settings;
- thường trộn lẫn kết quả chính với ablation, baseline và phân tích phụ;
- dễ biến nhận xét của mô hình thành “kết luận của tác giả”;
- các metric như ROUGE không đủ để đo độ đúng của số liệu, công thức và claim.

### 1.2. Ý tưởng trung tâm

**SciLens** không nên được định nghĩa đơn thuần là “một LLM đọc PDF rồi tóm tắt”. Project nên được định nghĩa là:

> Một hệ thống hiểu tài liệu khoa học đa phương thức, tạo bản tóm tắt phân cấp và có cấu trúc; mỗi claim quan trọng được liên kết tới bằng chứng trong paper; công thức, bảng, kết quả và ablation được trích xuất thành các đối tượng có thể kiểm tra tự động.

Pipeline đề xuất:

```text
PDF / LaTeX / XML
        │
        ▼
Document parsing + layout recovery
        │
        ▼
Scientific Document Graph
(section, claim, equation, symbol, table, cell, experiment, result)
        │
        ├──► Section-aware summarization
        ├──► Equation selection + LaTeX + explanation
        ├──► Main-result extraction
        ├──► Ablation analysis
        └──► Contribution / strength / limitation analysis
        │
        ▼
Claim-level verification + confidence + citations
        │
        ▼
Structured summary (Markdown + JSON + source anchors)
```

### 1.3. Luận điểm novelty nên theo đuổi

Điểm mới mạnh nhất và có khả năng bảo vệ tốt không phải là ghép nhiều chức năng vào một giao diện. Đóng góp nghiên cứu nên tập trung vào ba thành phần liên quan chặt chẽ:

1. **Scientific Document Evidence Graph**: biểu diễn thống nhất text, section, công thức, ký hiệu, bảng, ô dữ liệu, experimental setting và claim.
2. **Evidence-grounded structured summarization**: mỗi nội dung sinh ra phải có source span/page/bounding box hoặc table-cell provenance; hệ thống được phép từ chối khi bằng chứng không đủ.
3. **SciLensBench**: benchmark đánh giá đồng thời coverage, factuality, equation fidelity, numerical/table faithfulness, ablation understanding và traceability.

Đây là hướng phù hợp cho đồ án nghiên cứu, luận văn hoặc paper. Nếu chỉ làm MVP sản phẩm, có thể dùng các parser và LLM sẵn có; nếu muốn có tính mới học thuật, cần xây dựng thêm representation, verification method và/hoặc benchmark.

### 1.4. Định vị so với nghiên cứu gần đây

Không nên tuyên bố novelty chung chung là “summary có citation” hoặc “summary theo aspect”. [TracSum](https://aclanthology.org/2025.emnlp-main.43/) đã nghiên cứu aspect-based summarization có sentence-level traceability trên 500 medical abstracts. [ARC](https://aclanthology.org/2026.eacl-long.167/) đã đánh giá mức độ bảo toàn các vai trò lập luận trong long-document summary, bao gồm cả scientific articles. Vì vậy, novelty cụ thể của SciLens nên nằm ở:

- full paper thay vì chỉ abstract;
- provenance đa phương thức tới text span, equation và table cell;
- representation thống nhất section–claim–symbol–cell–experiment;
- trích xuất và kiểm chứng main result/ablation;
- đánh giá equation fidelity, numerical faithfulness và selective abstention.

---

## 2. Phạm vi nên chọn

### 2.1. Phạm vi giai đoạn đầu

Nên giới hạn phiên bản nghiên cứu đầu tiên ở:

- paper tiếng Anh;
- lĩnh vực Computer Science, ưu tiên Machine Learning/NLP/Computer Vision;
- đầu vào là PDF born-digital hoặc LaTeX source từ arXiv;
- paper thực nghiệm có bảng kết quả và/hoặc ablation;
- đầu ra bằng tiếng Anh hoặc tiếng Việt, nhưng dữ liệu và evaluation chính nên dùng tiếng Anh để giảm nhiễu do dịch.

Lý do: cấu trúc `task–dataset–metric–method–score` trong paper ML đã có các tài nguyên như [SciREX](https://aclanthology.org/2020.acl-main.670/) và [AxCell](https://aclanthology.org/2020.emnlp-main.692/). Ablation cũng xuất hiện thường xuyên hơn và có cấu trúc tương đối rõ trong nhóm paper này.

### 2.2. Những phần chưa nên hứa trong phiên bản đầu

- hiểu đúng mọi công thức thuộc mọi ngành khoa học;
- suy ra chứng minh toán học không được viết trong paper;
- đọc hoàn hảo PDF scan chất lượng thấp;
- tự động đánh giá paper “tốt” hay “xấu”;
- kết luận rằng một thành phần “gây ra” mức tăng nếu bảng chỉ cho thấy tương quan;
- tạo limitations như một sự thật nếu paper không cung cấp đủ bằng chứng.

Các trường hợp này cần confidence, cảnh báo hoặc cơ chế abstention.

---

## 3. Các vấn đề kỹ thuật project bắt buộc phải giải quyết

### 3.1. Khôi phục cấu trúc tài liệu từ PDF

PDF lưu vị trí hiển thị tốt hơn là cấu trúc ngữ nghĩa. Hệ thống phải khôi phục:

- reading order của tài liệu nhiều cột;
- cây section/subsection;
- paragraph, list, footnote và appendix;
- caption và liên kết giữa caption với figure/table;
- inline equation và display equation;
- công thức được tham chiếu bởi “Eq. (x)”;
- bảng, header nhiều tầng, merged cell, chú thích và ký hiệu;
- vị trí trang và bounding box để truy vết.

Có thể thử nghiệm nhiều parser:

- [GROBID](https://grobid.readthedocs.io/en/latest/Introduction/) cho metadata, section, citation và TEI XML; full-text model của GROBID nhận diện paragraph, section title, figure, table, formula và các callout tương ứng.
- [Docling](https://arxiv.org/abs/2408.09869) cho layout analysis, table structure và unified document representation.
- [Nougat](https://arxiv.org/abs/2308.13418) cho hướng image-to-markup, đặc biệt hữu ích khi cần khôi phục biểu thức toán.
- LaTeX/XML source nếu có, vì đây thường là nguồn có cấu trúc tốt hơn PDF.

**Đề xuất:** dùng multi-path parsing. Ưu tiên LaTeX/XML; nếu chỉ có PDF thì chạy parser chính, sau đó gọi equation/table OCR chuyên biệt cho các vùng có độ tin cậy thấp. Không nên dùng một parser duy nhất cho mọi loại paper.

### 3.2. Tóm tắt phân cấp và nhận biết vai trò diễn ngôn

Hệ thống phải hiểu một câu đang trình bày:

- background/context;
- research gap;
- motivation;
- research question;
- method;
- theoretical statement;
- experimental setup;
- result;
- analysis;
- limitation;
- conclusion hoặc future work.

Việc chỉ chunk theo số token làm mất cây section và quan hệ diễn ngôn. [MuLMS-AZ](https://aclanthology.org/2023.codi-1.1/) cho thấy argumentative zoning có thể phân loại các vai trò như Motivation, Background và Result trong bài báo khoa học. [What’s New?](https://aclanthology.org/2023.eacl-main.72/) cũng chỉ ra lợi ích của việc tách riêng contribution summary và context summary.

Hệ thống nên tạo:

- một TL;DR;
- một executive summary;
- summary theo từng section;
- một technical deep dive;
- danh sách claim và bằng chứng tương ứng.

### 3.3. Chọn công thức quan trọng, không chỉ OCR tất cả công thức

Trích xuất toàn bộ equation sang LaTeX chỉ giải quyết bài toán nhận dạng. Project còn phải xác định **công thức nào đáng lưu ý**.

Ví dụ các tín hiệu salience:

- công thức định nghĩa objective/loss hoặc mô hình chính;
- được tham chiếu nhiều lần trong paper;
- xuất hiện ở Method/Theory thay vì appendix;
- có các cụm “we define”, “our objective”, “is computed as”;
- được sử dụng để suy ra thuật toán hoặc kết quả phía sau;
- chứa ký hiệu mới do paper giới thiệu;
- có liên hệ trực tiếp với contribution claim.

Mỗi công thức được chọn cần có:

1. LaTeX chuẩn hóa;
2. số equation nếu có;
3. source page/bounding box;
4. ý nghĩa ngắn gọn;
5. bảng ký hiệu;
6. input, output và điều kiện áp dụng;
7. vai trò của công thức trong phương pháp;
8. giải thích trực giác;
9. các giả định được nêu trong paper;
10. confidence và cảnh báo đối với ký hiệu không định nghĩa.

Ví dụ schema:

```json
{
  "equation_id": "eq_4",
  "latex": "\\mathcal{L}=\\mathcal{L}_{task}+\\lambda\\mathcal{L}_{reg}",
  "role": "training_objective",
  "source": {"page": 5, "section": "3.2", "bbox": [72, 310, 510, 355]},
  "symbols": [
    {"symbol": "\\lambda", "meaning": "regularization weight", "evidence": "span_231"}
  ],
  "explanation": "...",
  "confidence": 0.93
}
```

Kiểm chứng equation nên gồm:

- compile success;
- exact match hoặc normalized edit distance với LaTeX source;
- so sánh ảnh render của LaTeX dự đoán với crop gốc;
- symbol coverage;
- kiểm tra giải thích có dùng ký hiệu ngoài định nghĩa hay không.

[MathNet](https://arxiv.org/abs/2404.13667) nhấn mạnh khoảng cách giữa công thức sinh tổng hợp từ LaTeX và công thức lấy từ paper thực tế; vì vậy test set phải chứa công thức từ PDF thật, không chỉ dữ liệu render sạch.

### 3.4. Hiểu bảng thay vì chuyển bảng thành text phẳng

Bảng khoa học có các vấn đề:

- multi-row/multi-column header;
- cùng một metric nhưng nhiều dataset hoặc setting;
- ký hiệu `↑/↓` xác định chiều tốt hơn;
- bold/underline đánh dấu best/second-best;
- giá trị `mean ± std`;
- footnote thay đổi điều kiện so sánh;
- kết quả có thể nằm trong caption hoặc đoạn văn dẫn chiếu;
- một ô “Ours” chỉ có nghĩa khi giữ đúng row/column hierarchy.

Pipeline bảng nên có bốn tầng:

1. **Table detection và structure recognition**.
2. **Canonicalization** thành HTML/JSON với header path đầy đủ cho từng cell.
3. **Table-role classification**: main result, ablation, hyperparameter, data statistics, qualitative comparison, efficiency hoặc error analysis.
4. **Semantic extraction** thành result tuples.

Result tuple gợi ý:

```text
(method, variant, task, dataset, split, metric, score, direction,
 baseline, delta, experimental_setting, source_cell)
```

[PubTables-1M](https://arxiv.org/abs/2110.00061) cung cấp gần một triệu bảng từ bài báo khoa học và hỗ trợ table detection, structure recognition, functional analysis. [AxCell](https://aclanthology.org/2020.emnlp-main.692/) trực tiếp giải quyết trích xuất kết quả từ bảng trong paper ML. Các tài nguyên này phù hợp cho pretraining/baseline nhưng chưa thay thế annotation cần có cho summary và ablation của SciLens.

### 3.5. Nhận diện và tóm tắt kết quả chính

Không phải giá trị lớn nhất trong bảng luôn là kết quả chính. Cần xét:

- claim trong abstract/introduction/conclusion;
- table caption;
- câu dẫn chiếu tới bảng;
- tên method và variant;
- metric direction;
- statistical significance hoặc variance;
- điều kiện dữ liệu và test split;
- so sánh công bằng hay không;
- trade-off accuracy–speed–memory.

Đầu ra nên phân biệt:

- **reported result**: số liệu được paper báo cáo;
- **comparison**: chênh lệch do hệ thống tính từ các ô hợp lệ;
- **author interpretation**: giải thích do tác giả viết;
- **system inference**: suy luận của SciLens và phải được gắn nhãn.

### 3.6. Hiểu ablation study

Đây là một trong các phần có tiềm năng novelty cao nhất.

Hệ thống cần:

1. xác định table/paragraph nào là ablation;
2. nhận diện full model, control và từng ablated variant;
3. ánh xạ variant tới component bị thêm, bỏ hoặc thay đổi;
4. giữ cố định dataset, metric và experimental setting;
5. tính delta đúng chiều của metric;
6. mô tả ảnh hưởng theo từng dataset/metric;
7. phát hiện kết quả không nhất quán giữa các setting;
8. không kết luận quá mức khi không có significance hoặc nhiều yếu tố thay đổi cùng lúc.

Schema gợi ý:

```json
{
  "component": "cross-attention module",
  "intervention": "removed",
  "control": "full model",
  "outcomes": [
    {
      "dataset": "X",
      "metric": "F1",
      "control_score": 88.4,
      "variant_score": 86.9,
      "delta": -1.5,
      "source_cells": ["t4-r2-c5", "t4-r4-c5"]
    }
  ],
  "conclusion": "Removing the module is associated with a 1.5-point F1 decrease.",
  "causal_strength": "controlled_single_component",
  "confidence": 0.91
}
```

Nên dùng cách nói “associated with a decrease” nếu paper không đủ điều kiện cho kết luận nhân quả mạnh.

[AbGen](https://arxiv.org/abs/2507.13300) xây dựng benchmark về **thiết kế** ablation từ paper NLP và cho thấy còn khoảng cách giữa LLM với chuyên gia. SciLens khác ở nhiệm vụ **trích xuất, chuẩn hóa và giải thích ablation đã có**, nhưng AbGen là nguồn hữu ích để thiết kế taxonomy và expert evaluation.

### 3.7. Trích xuất đóng góp, ưu điểm và hạn chế

Ba loại thông tin này không nên được sinh bằng cùng một prompt chung.

#### Đóng góp kỹ thuật

Mỗi contribution nên có cấu trúc:

```text
problem → proposed artifact/method → technical difference → evidence → reported effect
```

Phải tách:

- novelty claim do tác giả tự tuyên bố;
- khác biệt kỹ thuật được paper mô tả;
- kết quả thực nghiệm hỗ trợ;
- mức độ mới so với related work, chỉ kết luận khi đã có ngữ cảnh so sánh phù hợp.

#### Ưu điểm

Ưu điểm cần có đối tượng so sánh và bằng chứng:

- tốt hơn baseline nào;
- trên dataset/metric nào;
- cải thiện accuracy, robustness, compute, memory hay interpretability;
- có nhất quán giữa các setting không.

#### Hạn chế

Tách ba nguồn:

1. `author_stated`: tác giả nêu trực tiếp trong Limitations/Discussion.
2. `evidence_supported`: suy ra trực tiếp từ thí nghiệm, ví dụ hiệu năng giảm ở low-resource setting.
3. `analyst_hypothesis`: giả thuyết của hệ thống; phải diễn đạt thận trọng, không được trình bày như fact.

Không nên cho phép hệ thống tự tạo “nhược điểm” chỉ để làm output có vẻ cân bằng.

### 3.8. Factuality, traceability và abstention

Mỗi câu trong final summary nên được tách thành atomic claims. Mỗi claim phải liên kết tới một hoặc nhiều:

- text span;
- equation;
- table cell;
- caption;
- figure;
- appendix item.

Verifier kiểm tra:

- entailment giữa claim và evidence;
- consistency của entity, method, dataset và metric;
- numerical consistency;
- phạm vi định lượng như “all”, “most”, “on average”;
- mâu thuẫn giữa abstract và body;
- source có thực sự hỗ trợ cách diễn giải hay chỉ có từ khóa giống nhau.

[LongDocFACTScore](https://aclanthology.org/2024.lrec-main.941/) cung cấp LongSciVerify và một framework đánh giá factuality cho long-document scientific summarization. [FENICE](https://aclanthology.org/2024.findings-acl.841/) dùng atomic claim extraction và NLI alignment. Tuy vậy, nghiên cứu mới về stress test factuality cho long document cho thấy các metric hiện hữu vẫn không ổn định trước những biến đổi tương đương về nghĩa và claim có mật độ thông tin cao; vì vậy không nên dùng một LLM-as-a-judge duy nhất làm ground truth ([Mujahid et al., 2026](https://aclanthology.org/2026.acl-long.1472/)).

[TabFaith](https://aclanthology.org/2026.surgellm-1.21/) còn chỉ ra các dạng lỗi đặc thù khi tóm tắt bảng như bịa số, gán sai hàng/cột, tạo ranking sai và trộn mốc thời gian. Do đó verifier của SciLens phải kiểm tra claim ở cấp cell/schema, không chỉ kiểm tra entailment với chuỗi text đã linearize.

Khi evidence yếu, hệ thống nên trả về:

```text
Không đủ bằng chứng để kết luận / ký hiệu chưa được định nghĩa rõ /
bảng không cho phép so sánh trực tiếp / parser có độ tin cậy thấp.
```

---

## 4. Đầu ra chuẩn của hệ thống

Một output đầy đủ nên có các mục:

```markdown
# Paper title

## 1. TL;DR
## 2. Bài toán và động lực
## 3. Khoảng trống nghiên cứu
## 4. Ý tưởng và đóng góp chính
## 5. Tóm tắt theo từng section
### 5.1 Introduction
### 5.2 Related Work
### 5.3 Method
### 5.4 Experiments
### 5.5 Conclusion
## 6. Công thức quan trọng
### Eq. 1 — LaTeX, ký hiệu, trực giác, vai trò, nguồn
## 7. Experimental setup
## 8. Bảng và kết quả chính
## 9. Ablation study
## 10. Ưu điểm có bằng chứng
## 11. Hạn chế
### Tác giả tự nêu
### Suy luận được hỗ trợ bởi bằng chứng
### Giả thuyết cần kiểm chứng thêm
## 12. Takeaways cho người triển khai
## 13. Claim–evidence map
## 14. Confidence và cảnh báo parsing
```

Ngoài Markdown dành cho người đọc, hệ thống nên xuất JSON theo schema cố định. JSON là điều kiện cần để đánh giá tự động và xây UI có thể click từ claim về đúng vị trí trong PDF.

---

## 5. Scientific Document Graph

### 5.1. Các node chính

| Node | Thuộc tính chính |
|---|---|
| `Section` | title, level, page range, rhetorical role |
| `Span` | text, page, bounding box, section |
| `Claim` | atomic proposition, claim type, confidence |
| `Equation` | LaTeX, number, role, image crop |
| `Symbol` | surface form, definition, scope, unit |
| `Table` | caption, role, header tree |
| `Cell` | row path, column path, value, style |
| `Experiment` | task, dataset, split, setting |
| `Method` | name, variant, components |
| `Metric` | name, unit, direction |
| `Result` | score, variance, source cell |
| `Contribution` | artifact, novelty claim, evidence |
| `Limitation` | source type, evidence, scope |

### 5.2. Các edge quan trọng

```text
Section CONTAINS Span
Span STATES Claim
Span DEFINES Symbol
Equation USES Symbol
Span REFERENCES Equation/Table/Figure
Table HAS_CELL Cell
Cell MEASURES Method ON Dataset WITH Metric
Result SUPPORTED_BY Cell
AblationVariant REMOVES/REPLACES Component
Claim SUPPORTED_BY Span/Equation/Cell
Contribution SUPPORTED_BY Claim/Result
Limitation QUALIFIES Claim
```

### 5.3. Vì sao graph có ý nghĩa nghiên cứu

- giữ quan hệ xuyên section mà chunking tuyến tính làm mất;
- cho phép retrieve evidence theo loại đối tượng;
- tạo negative samples có kiểm soát bằng cách đổi dataset, metric, cell hoặc symbol;
- kiểm chứng số liệu và công thức bằng rule trước khi gọi LLM;
- hỗ trợ nhiều output format mà không cần parse paper lại;
- tạo nền tảng cho benchmark claim-level traceability.

---

## 6. Kiến trúc hệ thống đề xuất

### 6.1. Stage A — Ingestion và source selection

Thứ tự ưu tiên:

1. publisher XML/JATS hoặc arXiv LaTeX;
2. born-digital PDF;
3. scanned PDF + OCR.

Lưu file gốc, checksum, license, DOI/arXiv ID và phiên bản paper. Không trộn camera-ready với preprint mà không ghi version.

### 6.2. Stage B — Multimodal parsing

- parse metadata và section tree;
- giữ page/bounding box cho mọi object;
- detect table/equation/figure;
- OCR hoặc image-to-markup cho vùng cần thiết;
- khôi phục cross-reference;
- đo confidence cho reading order, table structure và equation transcription.

### 6.3. Stage C — Normalization

- chuẩn hóa section title nhưng giữ title gốc;
- canonicalize citation/callout;
- chuẩn hóa Unicode/LaTeX;
- tạo symbol table theo scope;
- tạo table header path;
- parse số, phần trăm, `±`, range, scientific notation;
- chuẩn hóa metric direction.

### 6.4. Stage D — Scientific information extraction

- rhetorical role classification;
- task/dataset/method/metric extraction;
- claim extraction;
- contribution/gap/limitation extraction;
- equation-role classification;
- table-role classification;
- result và ablation tuple extraction.

### 6.5. Stage E — Salience planning

Thay vì sinh summary ngay, hệ thống tạo một **content plan**:

```json
{
  "problem": ["claim_12"],
  "gap": ["claim_19"],
  "method": ["claim_33", "eq_4", "eq_7"],
  "main_results": ["result_6", "result_9"],
  "ablations": ["ablation_2"],
  "limitations": ["claim_81"]
}
```

Salience score có thể học hoặc kết hợp rule:

```text
salience =
  section_prior
  + rhetorical_role_score
  + cross_reference_centrality
  + contribution_alignment
  + result_support
  + reader_profile_relevance
  - redundancy
  - parser_uncertainty
```

### 6.6. Stage F — Grounded generation

Sinh theo từng field/schema, không sinh một lần toàn bộ tài liệu. Generator chỉ nhận evidence packet cần thiết và bắt buộc trả source IDs.

Nên có hai chế độ:

- **faithful/extractive-first**: ưu tiên chính xác, phù hợp evaluation;
- **reader-friendly**: giải thích dễ hiểu hơn nhưng vẫn giữ citation và đánh dấu phần diễn giải.

### 6.7. Stage G — Verification và repair

1. tách output thành atomic claims;
2. retrieve source evidence;
3. rule-check number, unit, table coordinate và equation symbol;
4. NLI/LLM verification;
5. kiểm tra cross-field consistency;
6. sửa claim nếu có evidence rõ;
7. xóa hoặc abstain nếu không thể hỗ trợ;
8. calibration confidence.

---

## 7. Điểm mới: lựa chọn theo mức tham vọng

### 7.1. Mức A — MVP kỹ thuật

**Đóng góp:** hệ thống end-to-end hữu ích.

- PDF → structured Markdown;
- summary theo section;
- equation/table extraction;
- result/ablation summary;
- citations tới page.

Mức này phù hợp đồ án môn học hoặc portfolio. Novelty học thuật còn yếu vì chủ yếu tích hợp công cụ.

### 7.2. Mức B — Luận văn/paper khả thi, khuyến nghị

**Tên hướng:** *Evidence-Grounded Multimodal Structured Summarization of Machine Learning Papers*.

Đóng góp:

1. Scientific Document Evidence Graph;
2. content planner nhận biết section và object type;
3. verifier chuyên cho số liệu, bảng và equation;
4. tập annotation 300–500 paper với claim-evidence, important equations, main results và ablations;
5. evaluation protocol đa chiều.

Đây là lựa chọn cân bằng nhất giữa tính mới, khối lượng và khả năng có kết quả tốt.

### 7.3. Mức C — Hướng nghiên cứu tham vọng

**Tên hướng:** *SciLensBench: A Benchmark for Traceable Equation-, Table-, and Ablation-Aware Scientific Paper Summarization*.

Đóng góp:

- benchmark 1.000+ paper;
- annotation bởi người có chuyên môn;
- temporal split và cross-domain split;
- adversarial test cho number swap, metric swap, symbol swap, wrong-cell attribution;
- train/evaluate multimodal models;
- selective generation với calibrated confidence.

Mức này có tiềm năng paper mạnh hơn nhưng chi phí annotation và quality control cao.

---

## 8. Dữ liệu

### 8.1. Tài nguyên có thể tái sử dụng

| Tài nguyên | Công dụng | Giới hạn đối với SciLens |
|---|---|---|
| [arXiv/PubMed long summarization](https://arxiv.org/abs/1804.05685) | Full paper → abstract, huấn luyện long-document summarization | Abstract không bao phủ chi tiết equation/table/ablation |
| [SciTLDR](https://aclanthology.org/2020.findings-emnlp.428/) | 5,4K TLDR cho 3,2K paper | Extreme summary, thiếu structured evidence |
| [LongSumm](https://aclanthology.org/2020.sdp-1.39/) | Extended scientific summaries | Quy mô nhỏ, annotation không nhắm công thức/bảng |
| [S2ORC](https://aclanthology.org/2020.acl-main.447/) | Corpus lớn có structured full text, citation, table/figure mentions | Parse tự động, không phải gold multimodal summary |
| [What’s New?](https://aclanthology.org/2023.eacl-main.72/) | Contribution/context summaries | Không bao phủ equation/result/ablation |
| [MuLMS-AZ](https://aclanthology.org/2023.codi-1.1/) | Rhetorical/argumentative roles | 50 paper, domain materials science |
| [PubTables-1M](https://arxiv.org/abs/2110.00061) | Table detection/structure/function | Không có main-result/ablation summary |
| [AxCell](https://aclanthology.org/2020.emnlp-main.692/) | Result extraction từ paper ML | Tập trung leaderboard, chưa phải summary toàn paper |
| [SciREX](https://aclanthology.org/2020.acl-main.670/) | Document-level result tuples | Entity/relation scope khác SciLens output |
| [LongSciVerify](https://aclanthology.org/2024.lrec-main.941/) | Factuality annotation cho scientific long summary | Không chuyên equation/table/ablation |
| [AbGen](https://arxiv.org/abs/2507.13300) | Expert examples về ablation design | Thiết kế ablation, không phải trích xuất ablation hiện hữu |
| [TracSum](https://aclanthology.org/2025.emnlp-main.43/) | Aspect summary có sentence-level citations | Medical abstracts, không phải full-paper multimodal evidence |
| [ARC](https://aclanthology.org/2026.eacl-long.167/) | Structured argument coverage cho long summary | Không biểu diễn equation/table cell/ablation |
| [TabFaith](https://aclanthology.org/2026.surgellm-1.21/) | Lỗi và metric structural faithfulness của table summary | Không được thiết kế riêng cho end-to-end scientific paper summary |

#### Nhận xét

Không có một tài nguyên riêng lẻ nào cung cấp đầy đủ ground truth cho output mà SciLens cần. Đây chính là cơ sở hợp lý để xây một dataset/benchmark mới, nhưng cần tránh tuyên bố “đầu tiên” trước khi hoàn thành systematic literature review.

### 8.2. SciLensBench đề xuất

#### Pilot

- 100 paper ML/NLP/CV;
- 70 train, 10 validation, 20 test;
- ưu tiên paper open-access có LaTeX source;
- ít nhất 60 paper có ablation;
- ít nhất 70 paper có display equations;
- ít nhất 80 paper có result tables.

#### Bản nghiên cứu

- 500–1.000 paper;
- chia theo venue, subfield và năm;
- temporal test split để giảm nguy cơ model đã ghi nhớ paper;
- một test set PDF-only, không cho dùng LaTeX;
- một robustness set gồm layout phức tạp hoặc scan;
- một cross-domain set nhỏ từ bio/physics để đo generalization.

#### Đơn vị annotation

Mỗi paper cần:

- section tree;
- section-level key points;
- 5–15 atomic central claims;
- evidence spans/cells/equations;
- important equation labels và symbol definitions;
- table roles;
- main result tuples;
- ablation tuples;
- contributions;
- author-stated limitations;
- supported analyst limitations;
- one short summary và one structured summary.

#### Quy trình annotation

1. Parser pre-annotates.
2. Annotator A sửa cấu trúc và gắn nhãn.
3. Annotator B kiểm tra độc lập.
4. Expert adjudication cho disagreement.
5. Tính inter-annotator agreement theo từng task.
6. Lưu provenance của mọi chỉnh sửa.

Đối với equation và ablation, annotator cần hiểu ML hoặc toán ở mức đọc paper. Không nên crowdsource toàn bộ cho người không có chuyên môn.

#### Chống leakage

- split theo paper, không theo paragraph/table;
- loại near-duplicate giữa arXiv version và proceedings version;
- temporal split;
- báo cáo riêng closed-model vì không biết chính xác pretraining data;
- dùng paper mới hoặc paper ít phổ biến cho expert hidden test;
- không dùng abstract làm gold cho mọi trường vì abstract có thể không chứa ablation/limitation.

---

## 9. Evaluation

Không nên tối ưu hoặc công bố một điểm tổng hợp duy nhất. Cần báo cáo dashboard metric theo các trục độc lập.

### 9.1. Document parsing

| Thành phần | Metric |
|---|---|
| Section detection | Precision/Recall/F1, tree edit distance |
| Reading order | pairwise order accuracy |
| Paragraph/span alignment | boundary F1 |
| Cross-reference linking | link F1 |
| Page/bounding box | IoU hoặc localization accuracy |

### 9.2. Equation

| Năng lực | Metric |
|---|---|
| Equation detection | box mAP/F1 |
| LaTeX recognition | exact match, normalized edit distance, BLEU/Edit score |
| Render fidelity | image similarity hoặc human correctness |
| Important-equation selection | Precision@k, Recall@k, nDCG |
| Symbol grounding | entity/link F1 |
| Explanation faithfulness | expert score + claim-evidence accuracy |
| Validity | LaTeX compile success |

Exact match không đủ vì nhiều LaTeX string có thể render cùng một biểu thức. Cần dùng cả normalized representation và render-based checking.

### 9.3. Table và numerical results

| Năng lực | Metric |
|---|---|
| Table structure | TEDS/structure F1 |
| Header-to-cell linking | link F1 |
| Table-role classification | macro-F1 |
| Result tuple extraction | exact/micro-F1 |
| Numeric correctness | exact cell value accuracy |
| Best-model comparison | comparison accuracy |
| Table summary faithfulness | supported numerical claim rate |

### 9.4. Ablation

- ablation table detection F1;
- component/intervention extraction F1;
- control–variant pairing accuracy;
- delta exact accuracy;
- metric-direction accuracy;
- conclusion faithfulness;
- inconsistency detection accuracy;
- expert score về mức độ hữu ích và không overclaim.

### 9.5. Summary

- content coverage theo expert/pyramid units;
- section coverage;
- contribution/context disentanglement;
- redundancy;
- factual consistency;
- citation precision, citation recall và citation correctness;
- claim completeness;
- readability/coherence;
- ROUGE/BERTScore chỉ là metric phụ để so với baseline cũ.

Nghiên cứu về LongDocFACTScore cho thấy ROUGE không đánh giá được factual consistency của long summary. Vì vậy metric lexical không được dùng làm kết luận chính.

### 9.6. Contribution, strength và limitation

- span/claim extraction F1;
- evidence attribution accuracy;
- source-type accuracy: author-stated vs system-inferred;
- comparison completeness cho strength;
- overclaim rate;
- expert Likert score về usefulness và correctness.

### 9.7. Calibration và selective prediction

Đánh giá:

- Expected Calibration Error;
- Brier score;
- risk–coverage curve;
- accuracy khi chỉ giữ các output có confidence ≥ ngưỡng;
- abstention precision: hệ thống có từ chối đúng các trường hợp khó không.

Đây là phần quan trọng để biến hệ thống từ demo thành công cụ nghiên cứu đáng tin cậy.

### 9.8. Human evaluation

Nên có ít nhất ba nhóm tiêu chí:

1. **Correctness:** claim, số liệu, công thức có đúng không?
2. **Coverage:** có bỏ sót contribution/result/ablation quan trọng không?
3. **Usefulness:** một nghiên cứu sinh có tiết kiệm thời gian và hiểu paper tốt hơn không?

Blind evaluation giữa các hệ thống, random thứ tự output, ghi rõ chuyên môn annotator và đo agreement.

---

## 10. Baseline và thí nghiệm bắt buộc

### 10.1. Baseline

#### B0 — Abstract

Dùng abstract gốc như bản tóm tắt. Đây là baseline rất quan trọng: hệ thống chỉ có ý nghĩa nếu cung cấp thông tin chính xác và hữu ích hơn abstract.

#### B1 — Direct long-context LLM

PDF/text đầy đủ → một prompt → structured summary.

#### B2 — Parser + flat chunks + LLM

Parse PDF thành Markdown, chunk tuyến tính và sinh summary.

#### B3 — Hierarchical text-only

Summary theo section rồi tổng hợp, không có equation/table objects.

#### B4 — Multimodal VLM

Render page image → VLM → structured summary.

#### B5 — Proposed SciLens

Document Graph + typed retrieval + content plan + field-level generation + verifier.

### 10.2. Ablation của chính SciLens

Phải bỏ từng thành phần để chứng minh đóng góp:

- bỏ section hierarchy;
- bỏ graph, dùng flat chunks;
- bỏ equation branch;
- bỏ table structure, linearize table;
- bỏ rhetorical roles;
- bỏ content planner;
- bỏ claim-level verifier;
- bỏ rule-based numerical checking;
- bỏ confidence/abstention;
- single-parser so với multi-path parser.

Kết quả cần báo cáo theo từng năng lực, không chỉ summary score chung.

### 10.3. Robustness tests

Tạo perturbation có kiểm soát:

- đổi một chữ số;
- đổi tên dataset;
- đảo `↑` thành `↓`;
- lấy đúng value nhưng sai row/column;
- đổi symbol trong explanation;
- đưa table caption không khớp;
- xóa định nghĩa ký hiệu;
- thay thứ tự cột PDF;
- paraphrase claim nhưng giữ nghĩa;
- thêm câu gần nghĩa nhưng không entail claim.

Các test này đo đúng failure mode của scientific summarization tốt hơn ROUGE.

---

## 11. Research questions và hypothesis

### RQ1

**Biểu diễn typed Scientific Document Graph có cải thiện coverage và factuality so với flat chunking và hierarchical text-only không?**

Hypothesis: graph đặc biệt cải thiện các claim cần liên kết text với table/equation.

### RQ2

**Tách content planning khỏi generation có giúp giảm omission và redundancy không?**

Hypothesis: planner tăng section coverage và important-equation/result recall.

### RQ3

**Verifier chuyên biệt theo modality có tốt hơn một LLM judge chung không?**

Hypothesis: rule + cell provenance tốt hơn judge chung đối với số liệu; symbol grounding tốt hơn judge chung đối với equation explanation.

### RQ4

**Ablation-aware extraction có giúp người đọc hiểu vai trò component chính xác hơn direct summarization không?**

Hypothesis: control–variant pairing và delta calculation giảm overclaim.

### RQ5

**Selective generation dựa trên confidence có tạo trade-off risk–coverage tốt hơn luôn luôn sinh output không?**

Hypothesis: abstention làm giảm mạnh unsupported-claim rate với mức giảm coverage chấp nhận được.

---

## 12. Tiêu chí thành công

Một phiên bản nghiên cứu được xem là thành công khi:

- section summary bao phủ phần lớn expert key points;
- gần như mọi numerical claim đều truy được về đúng table cell hoặc text span;
- equation LaTeX có compile rate cao và explanation không bịa định nghĩa ký hiệu;
- ablation delta được tính đúng theo metric direction;
- tách rõ author claim và analyst inference;
- citation correctness cao hơn direct LLM baseline;
- hệ thống biết từ chối khi parsing/evidence không chắc chắn;
- human evaluators đánh giá structured output hữu ích hơn abstract và direct summary.

Không nên đặt mục tiêu “100% chính xác trên mọi paper”. Mục tiêu khoa học tốt hơn là đo được lỗi, calibration và điều kiện hệ thống thất bại.

---

## 13. Công nghệ triển khai gợi ý

### Parsing

- GROBID, Docling;
- Nougat hoặc formula OCR chuyên biệt;
- PyMuPDF để render/crop và quản lý coordinate;
- arXiv source/LaTeXML khi license cho phép.

### Representation và storage

- Pydantic/JSON Schema cho typed objects;
- PostgreSQL cho metadata;
- object storage cho PDF, page image và crops;
- graph database là tùy chọn; giai đoạn đầu có thể dùng adjacency lists trong JSON/SQL;
- vector index chỉ dùng cho retrieval, không thay thế graph/provenance.

### Modeling

- SciBERT hoặc scientific encoder cho rhetorical/entity classification;
- long-context LLM cho planning/generation;
- VLM chỉ cho page/table/equation crop cần thiết;
- constrained JSON decoding;
- NLI/claim verifier;
- deterministic arithmetic engine cho delta và ranking.

### Interface

- viewer hai panel: PDF bên trái, structured summary bên phải;
- click claim → highlight source;
- render LaTeX bằng KaTeX/MathJax;
- table viewer giữ header hierarchy;
- toggle “author-stated / system-inferred”;
- hiển thị confidence và parsing warnings.

### Nguyên tắc kiến trúc

LLM không nên làm các phép tính và phép nối cấu trúc mà code xác định có thể làm chính xác hơn. LLM dùng cho semantic interpretation; parser/rule/schema dùng cho structure, arithmetic và provenance.

---

## 14. Lộ trình 16 tuần

| Tuần | Công việc | Deliverable |
|---|---|---|
| 1–2 | Chốt scope, schema, literature matrix, 20 paper mẫu | Problem statement + annotation guide v0 |
| 3–4 | Pipeline PDF/LaTeX → document objects | Parser benchmark nhỏ |
| 5–6 | Section/rhetorical/claim extraction | Structured document JSON |
| 7–8 | Equation branch và symbol grounding | Equation report + evaluation |
| 9–10 | Table/result/ablation branch | Result/ablation tuples |
| 11 | Content planner và structured generator | End-to-end baseline |
| 12 | Claim/cell/equation verifier | Faithfulness report |
| 13 | Annotation pilot 100 paper | SciLensBench-Pilot |
| 14 | Baseline + system ablations | Experiment tables |
| 15 | Human evaluation + error analysis | Evaluation report |
| 16 | Demo, documentation và paper draft | Reproducible release |

Nếu nhân lực chỉ có một người, nên giảm pilot xuống 50–100 paper và tập trung vào một novelty chính: **ablation-aware evidence grounding** hoặc **equation-aware summary**, thay vì cố huấn luyện mọi module từ đầu.

---

## 15. Rủi ro và cách giảm thiểu

| Rủi ro | Hậu quả | Giảm thiểu |
|---|---|---|
| PDF parsing sai reading order | Summary sai ngữ cảnh | Multi-path parser, coordinate QA, confidence |
| Equation OCR đúng hình nhưng sai semantics | Giải thích sai | LaTeX source khi có, render check, symbol grounding |
| Table linearization mất header | Gán số sai method/dataset | Header tree và cell provenance |
| LLM bịa limitation | Đánh giá paper không công bằng | Source-type labels và abstention |
| Abstract không phản ánh toàn paper | Gold summary thiên lệch | Expert content units và structured annotation |
| Closed LLM đã thấy paper | Kết quả phóng đại | Temporal/hidden test, open-model baselines |
| Annotation quá tốn kém | Dataset nhỏ/chất lượng thấp | Pre-annotation + double review + expert adjudication |
| Một composite score che lỗi | Kết luận sai | Dashboard theo task và error category |
| Copyright/license | Không thể phát hành dữ liệu | Chỉ phát hành derived annotations/IDs theo license |
| Chi phí inference | Khó triển khai | Typed retrieval, crop-level VLM, cache parser output |

---

## 16. Deliverables cuối cùng nên có

1. `SciLensParser`: PDF/LaTeX → structured document JSON.
2. `Scientific Document Graph` schema và builder.
3. Equation salience, LaTeX và explanation module.
4. Table/result/ablation extraction module.
5. Hierarchical evidence-grounded summarizer.
6. Claim-level multimodal verifier.
7. `SciLensBench-Pilot` và annotation guideline.
8. Baseline scripts, ablation experiments và error taxonomy.
9. Web demo có click-to-source.
10. Model card/dataset card, giới hạn và ethical considerations.

---

## 17. Đề xuất tên đề tài

### Tên tiếng Anh khuyến nghị

**SciLens: Evidence-Grounded Multimodal Structured Summarization of Scientific Papers**

### Tên tiếng Việt

**SciLens: Tóm tắt bài báo khoa học đa phương thức có cấu trúc và truy vết bằng chứng**

### Nếu tập trung vào benchmark

**SciLensBench: Evaluating Equation-, Table-, and Ablation-Aware Scientific Paper Summarization**

### Nếu tập trung vào ablation

**Beyond Abstracts: Evidence-Grounded Extraction and Summarization of Experimental Results and Ablation Studies**

---

## 18. Kết luận

Ý tưởng ban đầu có tiềm năng tốt, nhưng để trở thành một project có tính mới và hiệu quả, cần chuyển trọng tâm:

```text
Từ: “LLM tóm tắt paper và liệt kê thêm công thức/bảng”

Sang: “Structured multimodal scientific understanding
      + typed evidence graph
      + grounded generation
      + modality-specific verification
      + benchmark đo đúng các failure mode”
```

Khuyến nghị thực tế nhất là:

1. bắt đầu với paper ML/NLP/CV;
2. dùng parser có sẵn, không tự huấn luyện OCR từ đầu;
3. coi equation/table/ablation là typed objects thay vì text chunks;
4. bắt buộc mọi claim quan trọng có provenance;
5. tập trung novelty vào Evidence Graph + verifier + SciLensBench-Pilot;
6. đánh giá theo factuality, numerical/equation correctness và expert usefulness, không chỉ ROUGE.

Với cách định nghĩa này, SciLens vừa có thể trở thành một sản phẩm hữu ích cho người đọc paper, vừa có research questions, dataset, metric, baseline và ablation đủ rõ để phát triển thành luận văn hoặc bài báo.

---

## 19. Tài liệu tham khảo chính

### Scientific summarization và scholarly document understanding

1. Cohan et al. (2018), [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/abs/1804.05685).
2. Cachola et al. (2020), [TLDR: Extreme Summarization of Scientific Documents](https://aclanthology.org/2020.findings-emnlp.428/).
3. Lo et al. (2020), [S2ORC: The Semantic Scholar Open Research Corpus](https://aclanthology.org/2020.acl-main.447/).
4. Hayashi et al. (2023), [What’s New? Summarizing Contributions in Scientific Literature](https://aclanthology.org/2023.eacl-main.72/).
5. Liu et al. (2024), [SumSurvey: An Abstractive Dataset of Scientific Survey Papers for Long Document Summarization](https://aclanthology.org/2024.findings-acl.574/).
6. Schrader et al. (2023), [MuLMS-AZ: An Argumentative Zoning Dataset for the Materials Science Domain](https://aclanthology.org/2023.codi-1.1/).

### PDF, equation và table understanding

7. Blecher et al. (2023), [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418).
8. Auer et al. (2024), [Docling Technical Report](https://arxiv.org/abs/2408.09869).
9. GROBID, [Official Documentation](https://grobid.readthedocs.io/en/latest/).
10. Smock et al. (2021), [PubTables-1M: Towards Comprehensive Table Extraction from Unstructured Documents](https://arxiv.org/abs/2110.00061).
11. Kardas et al. (2020), [AxCell: Automatic Extraction of Results from Machine Learning Papers](https://aclanthology.org/2020.emnlp-main.692/).
12. Jain et al. (2020), [SciREX: A Challenge Dataset for Document-Level Information Extraction](https://aclanthology.org/2020.acl-main.670/).
13. Schmitt-Koopmann et al. (2024), [MathNet: A Data-Centric Approach for Printed Mathematical Expression Recognition](https://arxiv.org/abs/2404.13667).
14. Borisova et al. (2025), [Table Understanding and (Multimodal) LLMs](https://arxiv.org/abs/2507.00152).

### Ablation và factuality

15. Zhao et al. (2025), [AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](https://arxiv.org/abs/2507.13300).
16. Bishop et al. (2024), [LongDocFACTScore: Evaluating the Factuality of Long Document Abstractive Summarisation](https://aclanthology.org/2024.lrec-main.941/).
17. Scirè et al. (2024), [FENICE: Factuality Evaluation Based on NLI and Claim Extraction](https://aclanthology.org/2024.findings-acl.841/).
18. Mujahid et al. (2026), [Stress Testing Factual Consistency Metrics for Long-Document Summarization](https://aclanthology.org/2026.acl-long.1472/).
19. Zhao et al. (2023), [QTSumm: Query-Focused Summarization over Tabular Data](https://aclanthology.org/2023.emnlp-main.74/).
20. Chu et al. (2025), [TracSum: A New Benchmark for Aspect-Based Summarization with Sentence-Level Traceability](https://aclanthology.org/2025.emnlp-main.43/).
21. Elaraby and Litman (2026), [ARC: Argument Representation and Coverage Analysis for Zero-Shot Long Document Summarization](https://aclanthology.org/2026.eacl-long.167/).
22. Bukkapatnam and Mehta (2026), [TabFaith: Benchmarking and Improving Structural Faithfulness in LLM Table Summarization](https://aclanthology.org/2026.surgellm-1.21/).
