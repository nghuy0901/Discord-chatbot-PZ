# NomNom RAG — Báo cáo rà soát hệ thống (2026-06-29)

> Tài liệu này được tạo bằng cách **đọc trực tiếp mã nguồn và truy vấn database thật**, KHÔNG dựa
> vào các tài liệu kiến trúc cũ (README, các plan/spec trong `docs/superpowers/`) vì chúng được xác
> nhận là **không còn chính xác**. Đây là tài liệu nguồn-sự-thật mới cho trạng thái thực tế của hệ thống.

## Phán quyết tổng thể

**CHƯA sẵn sàng đưa cho người dùng cuối.** Cơ chế "từ chối khi thiếu bằng chứng" (abstention) được
thiết kế đúng và hoạt động trên đường đi chính, nhưng có **2 đường thoát khiến bot vẫn bịa câu trả lời**,
knowledge base **trùng lặp nặng** (đã đo bằng số liệu thật), retrieval **suy giảm nghiêm trọng với
tiếng Việt**, và bộ "đánh giá chất lượng/cổng release" **không thực sự chặn được gì** (luôn pass).

| Hạng mục | Trạng thái |
|---|---|
| Không bịa khi thiếu bằng chứng | ⚠️ Đúng trên đường chính, **THỦNG** ở 2 đường (RAG tắt/lỗi, và tool-calling) |
| Truy xuất đúng (retrieval) | ❌ Lỗi thang điểm + RRF + tiếng Việt làm sai/rỗng kết quả |
| Trùng lặp dữ liệu | ❌ Trùng nặng ở cả vector store và SQLite (số liệu bên dưới) |
| Đánh giá & cổng chất lượng | ❌ Gate luôn pass — không chặn được bản RAG hỏng |
| Sẵn sàng vận hành | ❌ Rò rỉ lỗi nội bộ, không validate secret, không cap token |

---

## 1. Kiến trúc thực tế (đã xác minh từ code)

Luồng xử lý một tin nhắn người dùng:

```
Tin nhắn người dùng
  │
  ├─ context_manager.build_prompt()                         utils/context_manager.py:242
  │     └─ if ENABLE_RAG:  build_rag_result()               rag/retriever.py:384
  │           ├─ A2 preprocess (clean, expand viết tắt,     rag/query_preprocessor.py
  │           │     detect ngôn ngữ + domain + intent)
  │           ├─ domain_router.route()                      knowledge/domain_router.py
  │           ├─ KB hybrid search  (vector + BM25 → RRF)    rag/hybrid_retriever.py
  │           ├─ Chat-history hybrid search (+ A4 fallback) rag/retriever.py:160
  │           ├─ E2 Self-RAG grading (LLM chấm liên quan)   rag/self_rag.py
  │           ├─ EvidencePolicy.assess() → ANSWER/CLARIFY/  rag/evidence.py
  │           │     ABSTAIN  (gọi filter_trusted_results)
  │           └─ build_citation_context() → context + [n]   rag/citations.py
  │
  ├─ Nếu decision != ANSWER  → decision_response() (câu trả lời cứng) → DỪNG   src/aclient.py:623
  │
  ├─ Nếu decision == ANSWER:
  │     ├─ Định tuyến theo intent:                          src/aclient.py:674
  │     │     analytical/hybrid/None → chat_with_tools()  (SQLite pz_data.db)
  │     │     narrative/conversation → stream/chat_completion()
  │     └─ finalize_rag_answer():                           rag/answer_finalize.py
  │           ├─ check_groundedness() (LLM judge)           rag/groundedness.py
  │           │     └─ nếu KHÔNG grounded → override ABSTAIN
  │           └─ render_sources_footer() (trích nguồn [n])
  │
  └─ Hai consumer dùng chung luồng trên: Discord (src/aclient.py) và API (api/main.py)
```

Hạ tầng: PostgreSQL + pgvector qua LangChain `PGVector`; Redis cache; LLM provider-neutral
(OpenAI / Gemini / Ollama Cloud / OpenAI-compatible/vLLM). KB tĩnh là markdown trong
`knowledge/docs/{pz,server_rules,general}`; dữ liệu cấu trúc là SQLite `knowledge/structured/pz_data.db`
được truy vấn qua tool-calling.

**Điểm thiết kế đúng:** Khi `rag_result.decision` là `CLARIFY`/`ABSTAIN`, bot trả về **câu cứng**
(`rag/responses.py`) và **không gọi LLM** — đây là cách chống bịa đúng đắn, không phụ thuộc vào việc
model có "nghe lời" prompt hay không. `build_rag_result` cũng để `context` rỗng khi không phải ANSWER.

---

## 2. Findings theo mức độ nghiêm trọng

Quy ước: 🔴 CRITICAL (chặn release) · 🟠 HIGH · 🟡 MEDIUM · ⚪ LOW.

### 🔴 C1 — Bot vẫn bịa khi RAG bị tắt hoặc RAG lỗi (fail-open)
`utils/context_manager.py:264,291` · `src/aclient.py:623,754` · `api/main.py:277`

`build_prompt` chỉ chạy RAG khi `enable_rag=True`, và **nuốt mọi exception** (chỉ log warning), để
`rag_result = None`. Guard chống-bịa ở `aclient.py:623` là `if rag_result and decision != ANSWER` →
khi `rag_result is None` thì guard bị bỏ qua, code rơi thẳng xuống gọi LLM với `rag_context=""`. Cổng
groundedness (`aclient.py:754`) cũng đòi `rag_result` non-None nên cũng bị bỏ qua. Hệ quả: **chỉ cần
`ENABLE_RAG=false`, hoặc một lỗi tạm thời của DB/embedding/LLM trong lúc build RAG, là bot trả lời câu
hỏi game bằng kiến thức huấn luyện của model — không bằng chứng, không kiểm chứng.** API có lỗ y hệt:
`api/main.py` bọc toàn bộ build RAG trong `if ENABLE_RAG:`, khi tắt sẽ trả `"decision": "answer"` cho
một câu sinh tự do.
**Sửa:** Khi RAG được bật nhưng `rag_result is None` (build lỗi) → ABSTAIN bằng câu cứng, **không gọi
LLM**. Quyết định chính sách rõ ràng cho `ENABLE_RAG=false` đối với câu hỏi thuộc domain game (từ chối
hoặc cảnh báo "không có bằng chứng"). Tuyệt đối không coi `rag_result is None` là "được trả lời tự do".

### 🔴 C2 — Câu trả lời qua tool-calling KHÔNG bị kiểm groundedness và KHÔNG có trích nguồn
`src/aclient.py:674-676,754-760` · `api/main.py:395,415` · `src/ollama_provider.py:96-147`

Intent `analytical`/`hybrid`/`None` (chính là nhóm câu hỏi nhiều dữ kiện nhất: chỉ số vật phẩm, công
thức, so sánh) được định tuyến sang `chat_with_tools()`. Cổng groundedness lại **cố tình loại trừ**
nhánh này (`and not answered_with_tools`) với lý do "dữ liệu đến từ SQLite". Nhưng `chat_with_tools`
trả về **văn bản tự do model sinh ra** sau khi nhận kết quả tool — model có thể thêm/diễn giải/bịa
xung quanh dữ liệu tool, và nếu tool trả `{"error": ...}` thì model vẫn "kể" tiếp. Ngoài ra
`cite_ok = answered and not use_tools_for_query` → câu trả lời tool có `source_count: 0`, người dùng
**không có cách lần ngược nguồn**.
**Sửa:** Kiểm chứng câu trả lời tool dựa trên chính giá trị tool trả về (hoặc ràng buộc lượt sinh
sau-tool chỉ được thuật lại dữ liệu tool), và đính kèm output tool làm citation.

### 🔴 C3 — Vector store nhúng chunk TRÙNG do bỏ qua `doc_id`
`knowledge/manager.py:655` (`store.add_documents(all_chunks)` không truyền `ids=`)

`_stable_doc_id` được tính và lưu vào metadata nhưng **không** được dùng làm khóa khi insert →
PGVector tự sinh UUID ngẫu nhiên cho mỗi dòng. Chạy đúng chunker thật trên `knowledge/docs/pz/**`
sinh ra **~900 chunk trùng byte-y-hệt** (191 chuỗi khác nhau bị lặp, ví dụ chunk "Trash" lặp 30 lần;
"Ball-peen Hammer", "Multitool", "Claw Hammer" lặp 14-16 lần). Mỗi reload domain (`force=True` qua
lệnh @mention) lại nhân bản tiếp vì insert không idempotent.
**Hệ quả:** top-k=5 có thể bị 3-5 bản sao cùng một chunk chiếm chỗ, đẩy các thông tin khác ra ngoài;
tốn chi phí embedding & lưu trữ.
**Sửa:** `store.add_documents(all_chunks, ids=[d.metadata["doc_id"] for d in all_chunks])` (PGVector
upsert theo id) và dedup `all_chunks` theo `doc_id` trước khi insert.

### 🔴 C4 — SQLite `pz_data.db` trùng lặp & rác (đã đo trên DB thật)
`scripts/convert_md_to_sqlite.py` · `knowledge/structured/pz_tools.py`

Số liệu thật:

| Bảng | Số dòng | Bằng chứng trùng/rác |
|---|---|---|
| `items` | 3 884 | **1 455 dòng (37%) `item_id` rỗng**; Glass Bottle ×21, Duffel Bag ×18, Elbow Pad ×14, Bottle ×13 … |
| `recipes` | 969 | **"Unknown Recipe" ×32** (regex parse hỏng nhưng vẫn lưu); Brew Coffee / Cut Bar / Cut Up Tire ×2 |
| `locations` | 1 171 | Laundromat ×17, Clothing store ×17, Restaurant ×15, Spiffo's ×13, Zippee Market ×13 … |

Không bảng nào có ràng buộc `UNIQUE`; chỉ `items` có bước merge thủ công, và bước đó chỉ gộp dòng
không-id khi **mọi cột giống hệt nhau**. Còn có dòng rác: các nhãn cột bảng wiki ("Attack speed",
"Bite defense", "Body location") bị nhận nhầm là item. `get_recipe`/`search_items` vì thế có thể trả
nhiều dòng "Unknown Recipe" hoặc nhiều bản trùng tên → tool đưa ra kết quả nhập nhằng/sai.
**Sửa:** Bỏ qua (hoặc log) record không parse được tên thay vì lưu "Unknown Recipe"; dedup recipes theo
`(name, product, crafting_type, ingredients)` và locations theo `(name, coordinates, area)`; dedup item
không-id theo `(name, category, sub_category)` giữ dòng đầy đủ nhất; blacklist tên trùng nhãn-cột; thêm
`UNIQUE`/`INSERT … ON CONFLICT`.

### 🔴 C5 — Bộ đánh giá & "cổng release" không thực sự chặn được gì (theater)
`scripts/run_release_eval.py:159-220` · `evaluation/gates.py` · `.github/workflows/ci.yml:86-93` · `evaluation/data/release_qa.v1.jsonl`

- `main()` ghi `gate_status` vào report rồi **luôn `return 0`** (dòng 220); chỗ duy nhất `return 1`
  (dòng 199) chỉ cho lỗi schema dataset → **gate fail vẫn build xanh**.
- Job CI chạy eval (`rag-release-eval`) **mặc định tắt** (`if vars.RUN_RAG_RELEASE_EVAL == 'true'`) và
  cần self-hosted runner → PR/push bình thường **không hề chạy**.
- ~13/20 ngưỡng gate tham chiếu các metric **không hề được tính** (`faithfulness`,
  `prompt_injection_pass_rate`, `critical_rule_recall`, `pii_secret_leakage_count`, `human_answer_correctness`…).
- Dataset eval chỉ có **4 dòng** trong khi schema đòi đúng **300**; cờ `--require-release-quota` chỉ bật
  trong job đang tắt.
- LLM-judge (`evaluation/local_judge.py`) và RAGAS **không bao giờ được gọi** trong đường release
  (`judge_scores={}` hardcode); calibration của judge tính ra `eligible_for_release_gate` nhưng **không
  ai đọc**.

**Hệ quả:** một bản RAG hỏng hoàn toàn (recall@5 = 0, rò rỉ nguồn chưa duyệt) vẫn pass CI. Lời khẳng
định "đo chất lượng RAG và chặn release theo độ chính xác" hiện **sai trên thực tế**.
**Sửa:** `main()` phải `return 1` khi `gate_result.passed == False`; bật job eval (ít nhất split
`development`) chạy vô điều kiện trên PR; tính đủ mọi metric đã gate hoặc bỏ ngưỡng cho metric không
tính; dựng dataset 300 dòng thật; gọi judge/RAGAS trong đường release và gate theo calibration.

### 🟠 H1 — Điểm "relevance" của pgvector có thể ÂM / sai thang
`rag/db.py:265-288` · `knowledge/manager.py:695` · ngưỡng ở `rag/retriever.py:42-51`, `rag/hybrid_retriever.py:138`

PGVector mặc định cosine; relevance = `1 - cosine_distance`, mà cosine distance ∈ [0,2] nên relevance
∈ **[-1, 1]** (LangChain chỉ cảnh báo, KHÔNG clamp). Toàn bộ ngưỡng (`0.3`, `0.35`, `0.55`…) đều giả
định thang [0,1]. `format_kb_results_for_prompt` còn in `{similarity:.0%}` → có thể ra **phần trăm âm**.
**Sửa:** Đặt `distance_strategy` tường minh + hàm relevance clamp [0,1]; hiệu chỉnh lại toàn bộ ngưỡng;
tối thiểu clamp điểm âm về 0 trong `search_similar`.

### 🟠 H2 — RRF trộn lẫn `rrf_score` (~0.01) với `similarity` (cosine) ở mọi nơi downstream
`rag/hybrid_retriever.py:89-105` · `rag/self_rag.py:243` · `rag/retriever.py:537` · `rag/evidence.py:160`

Sau fusion, doc chỉ-BM25 có `rrf_score ≈ 0.0098` còn doc vector có `similarity ≈ 0.5–0.9`, nhưng
downstream dùng `r.get("similarity", r.get("rrf_score", 0))` như cùng một thang. Hệ quả: heuristic
"skip grading" (ngưỡng 0.7) gần như **không bao giờ kích hoạt** (lãng phí LLM mỗi truy vấn);
`avg/max/min_similarity` trong metrics là rác; điểm tin cậy trong EvidencePolicy bị lệch chuẩn (được
giảm nhẹ nhờ "agreement floor" 0.80 khi một doc xuất hiện ở cả hai nhánh, nhưng các ngưỡng vẫn không
mang đúng ý nghĩa).
**Sửa:** Giữ `similarity` gốc xuyên suốt fusion; mỗi downstream chọn rõ một thang (cosine chuẩn hóa
HOẶC điểm theo rank), không trộn.

### 🟠 H3 — Truy xuất đa ngôn ngữ chưa được xử lý (người dùng VN ↔ KB tiếng Anh)
`rag/query_preprocessor.py:174-226` · `rag/bm25_search.py:63-76`

Bot phục vụ người Việt nhưng KB là tiếng Anh. Hỗ trợ đa ngữ duy nhất hiện có là map viết tắt
(VN→VN) và prefix `"Project Zomboid:"`. BM25 thuần từ vựng: token VN "rìu" không bao giờ khớp token EN
"axe" → **nhánh BM25 đóng góp gần như 0 cho truy vấn VN**, hybrid sụp về vector-only, rồi dính tiếp H1.
Việc vector có "bắc cầu" ngôn ngữ được hay không **phụ thuộc hoàn toàn** vào model embedding có đa ngữ
hay không, mà **không có kiểm tra/đảm bảo** nào.
**Sửa:** Thêm bước dịch/mở rộng từ khóa VN→EN trước BM25, hoặc index alias song ngữ; assert model
embedding là đa ngữ khi `query_language=='vi'`.

### 🟠 H4 — Index BM25 bị cũ (stale) so với vector store
`rag/bm25_search.py` · `rag/ingest.py` · `admin/chat_approval.py`

BM25 chat chỉ refresh khi approve/revoke, BM25 KB chỉ refresh khi `@reload`. Ingest chat mới
(`add_documents_batch`) **không** gọi `refresh_from_db()` → vector và BM25 chạy trên **hai corpus khác
nhau** cho tới khi restart, fusion so kết quả tươi với danh sách từ vựng cũ.
**Sửa:** Refresh BM25 sau ingest (hoặc theo TTL); gom chung KB reload + BM25 refresh để không lệch nhau.

### 🟠 H5 — Self-RAG có thể xóa sạch một tập kết quả tốt → biến đúng thành "không biết"
`rag/self_rag.py:186-216,294-405`

Khi LLM-grader chấm nhầm các doc liên quan thành IRRELEVANT (rất dễ với model nhỏ, `temperature=0.1`),
chúng bị loại; **không có sàn đảm bảo giữ lại ≥1 doc**. Nếu grader chấm hỏng hết và có ≤10 kết quả,
tập lọc thành rỗng → EvidencePolicy ABSTAIN dù retrieval đã lấy đúng tài liệu.
**Sửa:** Đảm bảo minimum-keep (giữ top-N theo rank gốc nếu grading định loại sạch); coi grade
lỗi/mơ hồ là "giữ".

### 🟠 H6 — Cổng groundedness fail-OPEN khi timeout/lỗi (bịa nhiều nhất đúng lúc tải nặng)
`rag/groundedness.py:131-175`

Judge là một lần gọi LLM nữa (`RAG_GROUNDEDNESS_TIMEOUT=20s`). Khi timeout/lỗi provider/JSON hỏng →
trả `grounded=True` (mặc định `RAG_GROUNDEDNESS_FAIL_OPEN=true`) **và ghi `score=1.0`** che mất sự cố.
Cũng skip (cho qua) khi `sources` rỗng. Đây là kiểu "hệ thống nói dối nhiều nhất khi đang căng".
**Sửa:** Production nên `RAG_GROUNDEDNESS_FAIL_OPEN=false` (ít nhất fail-closed riêng cho timeout);
đừng ghi `score=1.0` cho ca fail-open.

### 🟠 H7 — System prompt tự mâu thuẫn; "không dùng kiến thức ngoài" chỉ là *yêu cầu*, không *bắt buộc*
`prompts/templates/base_identity.txt` · `game_domain_rules.txt` · `rag_instructions.txt` · `prompts/system_prompt.py:61`

`base_identity` ("answer as accurately and honestly as possible") và `game_domain_rules` ("general-
purpose chatbot, not just a game bot") **đối nghịch** với `rag_instructions` ("không trả lời từ kiến
thức nền"). Trên lượt ANSWER nhưng bằng chứng mỏng (confidence vừa qua 0.55), thứ duy nhất ngăn model
"chêm" kiến thức huấn luyện chỉ là một dòng prompt — không có enforce nào ngoài cổng groundedness (vốn
đã thủng ở C2/H6).
**Sửa:** Bỏ/giới hạn câu "general-purpose chatbot"/"answer as accurately as possible" cho lượt domain
game; coi prompt là phòng-thủ-chiều-sâu, còn enforce thật nằm ở cổng (và phải vá cổng).

### 🟠 H8 — Rò rỉ lỗi nội bộ ra người dùng & API
`src/aclient.py:876,943,1046,368` · `api/main.py:475,524,534`

Discord hiển thị `f"❌ … {str(e)[:200]}"` (legacy: untruncated); API `raise HTTPException(500,
detail=str(e))`. `str(e)` của lỗi DB/driver/LLM thường chứa connection string, host:port, đường dẫn,
SQL → lộ thông tin hạ tầng.
**Sửa:** Trả message chung + request id; chi tiết chỉ log phía server.

### 🟠 H9 — Không validate secret bắt buộc lúc khởi động
`src/bot.py:442` · `src/aclient.py:181-188` · `src/llm/factory.py:11-17` · `api/main.py:81-89`

`DISCORD_BOT_TOKEN`, `LLM_PROVIDER`/credentials không được kiểm tra lúc boot; provider hỏng bị nuốt
trong `health_check()` và bot vẫn chạy → **mọi tin nhắn fail per-request** (kèm rò rỉ ở H8). (`API_KEY`
được validate chặt nhưng lazy, không phải lúc boot.)
**Sửa:** Validate `DISCORD_BOT_TOKEN`, `LLM_PROVIDER`/keys, `API_KEY` ngay lúc khởi động; thiếu → từ chối boot.

### 🟡 MEDIUM

- **M1 — Chunk gần-trùng làm ngợp retrieval:** Glass Bottle ×21 (chỉ khác Item ID) không trùng byte
  nên thoát C3, nhưng gần-y-hệt trong không gian embedding → "glass bottle" trả 5 chunk na ná. Cần
  MMR/diversity rerank hoặc gộp các record cùng tên thành 1 chunk liệt kê biến thể.
  (`knowledge/docs/pz/Equipment/Fluid containers.md`)
- **M2 — Query enrichment làm nhiễu retrieval chat:** `retriever.py:444-448` nối 5 tin nhắn gần nhất
  vào query đem đi embed + BM25 → trôi chủ đề. Dùng context để rerank, đừng nối vào query.
- **M3 — `is_trusted_result` fail-OPEN cho KB:** `rag/trust.py:14` mặc định `trusted=True` khi thiếu
  key → nên fail-closed (`False`).
- **M4 — Domain routing chỉ theo từ khóa tiếng Anh:** `knowledge/domain_router.py` + `query_preprocessor.py`
  → câu VN hiếm khi khớp, rơi về "search all" hoặc bỏ KB nếu KB load lỗi (không cảnh báo).
- **M5 — Thread-context là code chết & gây nhiễu:** `rag/db.py:312` trả edge chỉ có `message_id`,
  không có `content`/`author` → `retriever.py:361` in ra `↳ [reply] @?:` rỗng, tốn ngân sách context.
- **M6 — `add_documents_batch` nuốt lỗi từng item:** `rag/db.py:222-237` → corpus bị thiếu âm thầm,
  caller báo "ingested N" sai. Cần trả số lỗi và fail to khi tỉ lệ lỗi cao.
- **M7 — `metric.retrieval_time_ms` bị ghi đè bằng tổng thời gian:** `retriever.py:495` rồi `:596` →
  metric "retrieval time" thực ra là total time.
- **M8 — Không cap `max_tokens`:** `max_tokens` được nối tới client nhưng không caller nào truyền →
  sinh không giới hạn (chi phí/DoS, làm judge groundedness timeout → H6).
- **M9 — Reload KB delete-then-insert không nguyên tử:** `knowledge/manager.py:637-666` → lỗi giữa
  chừng có thể để domain rỗng/nửa vời.
- **M10 — Default mất an toàn:** `.env.example` ship `local-dev-key`, `postgres:postgres`; client còn
  fallback `api_key or "local-dev-key"` (`src/llm/openai_compatible.py:15`) không có guard production.
- **M11 — Lỗi quy ước metric eval:** precision@k chia `min(k,len)` (thổi phồng), nDCG không dedup,
  keyword_coverage match substring (gameable). (`evaluation/retrieval_metrics.py`)

### ⚪ LOW

- **L1 — Persona jailbreak là code chết nhưng nguy hiểm:** `src/personas.py` chứa prompt khuyến khích
  "bịa cho nghe hợp lý"; hiện `build_system_prompt` không hề đọc `current_persona` nên vô hại — nhưng
  nên **xóa**, vì nếu sau này ai đó nối vào prompt sẽ vũ-khí-hóa việc bịa.
- **L2 — `/chat` slash bỏ qua cổng groundedness** (`src/aclient.py:975-1031`).
- **L3 — `min_vector_score` là config chết** trong `EvidencePolicy`.
- **L4 — `validate_embedding_signature` định nghĩa nhưng không gọi** → đổi model embedding mà không
  reindex sẽ cho cosine rác, không báo lỗi (`rag/embedding_registry.py`).
- **L5 — Phát hiện ngôn ngữ mong manh;** câu hỏi VN ngắn dễ bị xếp `CONVERSATION` → bỏ RAG
  (`rag/query_preprocessor.py`).
- **L6 — `knowledge/docs/pz/list_md.txt` rỗng (0 byte)** — manifest cũ/bỏ hoang.
- **L7 — Persist metric fire-and-forget nuốt lỗi**; `get_db_summary` nội suy `%` chuỗi cho `hours`
  (`rag/metrics.py`).

---

## 3. Trả lời trực tiếp các câu hỏi đặt ra

1. **Có lỗi/tra cứu sai không?** Có. Lỗi thang điểm cosine (H1), trộn điểm RRF (H2), BM25 cũ (H4),
   Self-RAG xóa nhầm kết quả (H5), routing/đa ngữ (H3, M4) → kết quả sai hoặc rỗng, đặc biệt với tiếng Việt.
2. **Khi độ chính xác thấp có bịa không?** Trên đường chính: **không** (abstain bằng câu cứng, không gọi
   LLM — thiết kế đúng). Nhưng **có 2 đường bịa đã xác nhận**: (C1) RAG tắt/lỗi → trả lời tự do;
   (C2) tool-calling → không kiểm groundedness, không nguồn. Cộng thêm H6 (cổng fail-open khi tải nặng).
3. **DB có lặp thông tin không?** Có, **nặng**: vector store nhúng ~900 chunk trùng byte (C3); SQLite có
   1 455 item rỗng id, 32 "Unknown Recipe", và rất nhiều trùng tên (C4) — số liệu đo trực tiếp ở trên.
4. **Sẵn sàng cho người dùng chưa?** **Chưa** — phải vá tối thiểu C1–C5 trước.

---

## 4. Lộ trình khắc phục (ưu tiên)

**Bắt buộc trước khi mở cho người dùng (P0):**
1. C1 — Khi RAG bật mà `rag_result is None` → ABSTAIN, không gọi LLM. Chốt chính sách `ENABLE_RAG=false`.
2. C2 — Kiểm chứng/ràng buộc câu trả lời tool; đính nguồn cho tool.
3. C3 — Truyền `ids=doc_id` khi `add_documents`; dedup chunk trước insert.
4. C4 — Làm sạch SQLite (bỏ "Unknown Recipe"/nhãn-cột; dedup; thêm UNIQUE).
5. H6 — `RAG_GROUNDEDNESS_FAIL_OPEN=false` ở production (hoặc fail-closed cho timeout).
6. H8/H9 — Ngừng rò `str(e)`; validate secret lúc boot.

**Quan trọng cho chất lượng (P1):**
7. H1 + H2 — Sửa thang điểm cosine và ngừng trộn RRF/similarity; hiệu chỉnh lại ngưỡng.
8. H3 — Mở rộng/dịch từ khóa VN→EN; đảm bảo embedding đa ngữ.
9. H4 — Refresh BM25 sau ingest/reload.
10. H5 — Sàn minimum-keep cho Self-RAG.
11. C5 — Biến bộ eval thành cổng thật (return 1 khi fail; bật trên PR; dataset 300 dòng; gọi judge/RAGAS).

**Dọn dẹp (P2):** M1–M11, L1–L7 (đặc biệt **xóa persona jailbreak L1**).

---

## 5. Ghi chú về tài liệu cũ

- `README.md`: phần "Evaluation"/"Production Checklist" gây hiểu nhầm rằng RAGAS + metric quyết định
  được release — thực tế gate không enforce (C5). Đã thêm cảnh báo & trỏ về tài liệu này.
- Các plan/spec trong `docs/superpowers/` là tài liệu **lịch sử/kế hoạch**, không phản ánh trạng thái
  chạy thực tế — không dùng làm nguồn-sự-thật.
