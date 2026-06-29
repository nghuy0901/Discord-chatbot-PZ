# NomNom RAG — Sổ tay lỗi & khắc phục (2026-06-29)

Tài liệu này tổng hợp **toàn bộ lỗi** đã phát hiện khi rà soát mã nguồn + database thật. Mỗi lỗi gồm 5 phần:

1. **Nguyên nhân** — gốc rễ kỹ thuật (kèm `file:dòng`).
2. **Cách khắc phục** — phương án sửa.
3. **Trước → Sau** — khác biệt sau khi sửa.
4. **Kết quả** — lợi ích đạt được.
5. **Ví dụ** — kịch bản cụ thể (❌ hành vi cũ → ✅ hành vi sau khi sửa).

Tài liệu kiến trúc & xếp hạng đầy đủ: [`rag-system-audit-2026-06-29.md`](rag-system-audit-2026-06-29.md).
Quy ước mức độ: 🔴 CRITICAL (chặn release) · 🟠 HIGH · 🟡 MEDIUM · ⚪ LOW.

---

## Bảng tổng hợp nhanh

| ID | Mức | Lỗi | Kết quả sau khắc phục |
|----|-----|-----|------------------------|
| C1 | 🔴 | RAG tắt/lỗi → bot trả lời tự do | Lỗi hạ tầng → từ chối an toàn, không bịa |
| C2 | 🔴 | Tool-calling không kiểm chứng, không nguồn | Câu trả lời nhiều-dữ-kiện được kiểm chứng + trích nguồn |
| C3 | 🔴 | Vector store nhúng ~900 chunk trùng | top-k đa dạng, reload không phình |
| C4 | 🔴 | SQLite trùng lặp & rác (1.455 item rỗng id, 32 "Unknown Recipe") | Mỗi thực thể 1 dòng sạch, tool không nhập nhằng |
| C5 | 🔴 | Cổng eval luôn pass | Cổng thật chặn được regression |
| H1 | 🟠 | Điểm relevance pgvector có thể âm/sai thang | Ngưỡng đúng nghĩa, % hợp lý |
| H2 | 🟠 | RRF trộn `rrf_score` với `similarity` | Điểm nhất quán, metrics & confidence đáng tin |
| H3 | 🟠 | Đa ngữ VN↔EN không xử lý | Recall cho người dùng VN tăng mạnh |
| H4 | 🟠 | Index BM25 cũ so với vector | Hybrid đồng bộ, không cần restart |
| H5 | 🟠 | Self-RAG xóa sạch kết quả tốt | Giảm từ chối oan |
| H6 | 🟠 | Groundedness fail-open khi timeout | Không "nói dối khi tải nặng" |
| H7 | 🟠 | System prompt tự mâu thuẫn | Giảm bịa khi bằng chứng mỏng |
| H8 | 🟠 | Rò lỗi nội bộ ra người dùng/API | Bịt rò rỉ hạ tầng |
| H9 | 🟠 | Không validate secret lúc boot | Phát hiện cấu hình sai ngay (fail-fast) |
| M1 | 🟡 | Chunk gần-trùng (Glass Bottle ×21) | Kết quả đa dạng hơn nhờ MMR |
| M2 | 🟡 | Query enrichment làm nhiễu retrieval | Truy vấn tập trung, đúng chủ đề |
| M3 | 🟡 | `is_trusted_result` fail-open cho KB | Mặc định an toàn (fail-closed) |
| M4 | 🟡 | Domain routing chỉ theo từ khóa EN | Câu VN định tuyến đúng domain |
| M5 | 🟡 | Thread-context là code chết & gây nhiễu | Bỏ rác khỏi prompt, tiết kiệm context |
| M6 | 🟡 | `add_documents_batch` nuốt lỗi từng item | Phát hiện ingest thiếu |
| M7 | 🟡 | `retrieval_time_ms` bị ghi đè bằng total time | Metric thời gian đúng |
| M8 | 🟡 | Không cap `max_tokens` | Chặn sinh vô hạn (chi phí/DoS) |
| M9 | 🟡 | Reload KB delete-then-insert không nguyên tử | Reload lỗi không để domain rỗng |
| M10 | 🟡 | Default mất an toàn (`local-dev-key`...) | Không lọt credential mặc định ra production |
| M11 | 🟡 | Lỗi quy ước metric eval | Số đo eval phản ánh đúng |
| L1 | ⚪ | Persona jailbreak (code chết) khuyến khích bịa | Xóa footgun |
| L2 | ⚪ | `/chat` bỏ qua cổng groundedness | Mọi đường vào đều được kiểm chứng |
| L3 | ⚪ | `min_vector_score` là config chết | Bỏ cấu hình gây hiểu nhầm |
| L4 | ⚪ | `validate_embedding_signature` không được gọi | Chặn đổi model embedding mà quên reindex |
| L5 | ⚪ | Phát hiện ngôn ngữ mong manh | Câu VN ngắn vẫn được RAG |
| L6 | ⚪ | `list_md.txt` rỗng | Manifest đúng/bỏ hẳn |
| L7 | ⚪ | Persist metric nuốt lỗi; SQL nội suy `%` | Không mất metric âm thầm |

---

# 🔴 CRITICAL

## C1 — RAG tắt hoặc RAG lỗi → bot trả lời tự do (bịa)
- **Nguyên nhân:** `build_prompt` chỉ chạy RAG khi `enable_rag=True` và **nuốt mọi exception** (`utils/context_manager.py:264,291`), để `rag_result=None`. Guard chống-bịa (`src/aclient.py:623`) và cổng groundedness (`src/aclient.py:754`) đều đòi `rag_result` non-None, nên khi `None` thì bị bỏ qua → code rơi xuống gọi LLM với context rỗng.
- **Cách khắc phục:** Khi RAG được bật mà `rag_result is None` (build lỗi) → trả câu ABSTAIN cứng, **không gọi LLM**. Với `ENABLE_RAG=false`: từ chối câu hỏi thuộc domain game (hoặc gắn nhãn "không có nguồn"). Phân biệt rõ "tắt có chủ đích" và "lỗi".
- **Trước → Sau:** Trước, một lỗi DB/embedding tạm thời biến bot thành mô hình trả lời tự do; sau, lỗi luôn dẫn tới từ chối an toàn.
- **Kết quả:** Xóa bỏ con đường bịa do sự cố hạ tầng — hành vi *fail-safe*.
- **Ví dụ:**
  ```
  Người dùng: "Fire Axe gây bao nhiêu sát thương?"   (Postgres timeout 2 giây)
  ❌ Trước: "Fire Axe gây 2.5 damage, tốc độ 1.1..."   (số bịa, không nguồn)
  ✅ Sau:  "Mình chưa có đủ thông tin đã được phê duyệt để trả lời chính xác câu này."
  ```

## C2 — Câu trả lời qua tool-calling không bị kiểm groundedness, không có nguồn
- **Nguyên nhân:** Intent `analytical/hybrid/None` định tuyến sang `chat_with_tools` (`src/aclient.py:674`), nhưng cổng groundedness **cố ý loại trừ** nhánh này (`and not answered_with_tools`, `src/aclient.py:758`). `chat_with_tools` trả **văn bản tự do** model sinh sau khi nhận kết quả tool, không đối chiếu lại; nếu tool trả `{"error":...}` model vẫn "kể" tiếp. Câu trả lời tool có `source_count: 0`.
- **Cách khắc phục:** Thêm bước kiểm chứng "tool-aware" (đối chiếu câu trả lời với chính giá trị tool trả về) hoặc ràng buộc lượt sinh sau-tool chỉ thuật lại dữ liệu tool; đính kèm output tool làm citation; tool lỗi/rỗng → ABSTAIN.
- **Trước → Sau:** Trước, model có thể bịa quanh dữ liệu tool và không để lại nguồn; sau, chỉ phát biểu điều tool trả về, kèm nguồn.
- **Kết quả:** Đường hỏi nhiều-dữ-kiện nhất (chỉ số, công thức, so sánh) được kiểm chứng + truy vết nguồn.
- **Ví dụ:**
  ```
  Người dùng: "Công thức làm Sterilized Bandage?"
  (tool get_recipe trả: Bandage + Disinfectant)
  ❌ Trước: "Cần Bandage, Disinfectant rồi đun sôi 5 phút..."   (thêm bước bịa)
  ✅ Sau:  "Sterilized Bandage = Bandage + Disinfectant.  [Nguồn: recipes/Medical]"
  ```

## C3 — Vector store nhúng chunk trùng (bỏ qua `doc_id`)
- **Nguyên nhân:** `knowledge/manager.py:655` gọi `store.add_documents(all_chunks)` **không truyền `ids=`**, nên PGVector sinh UUID ngẫu nhiên mỗi dòng và `_stable_doc_id` bị bỏ phí. Chunker thật sinh ~900 chunk **trùng byte** (chunk "Trash" lặp 30 lần); mỗi `@reload` nhân bản tiếp vì insert không idempotent.
- **Cách khắc phục:** `store.add_documents(all_chunks, ids=[d.metadata["doc_id"] for d in all_chunks])` (PGVector upsert theo id) + dedup `all_chunks` theo `doc_id` trước khi insert.
- **Trước → Sau:** Trước, một chunk tồn tại tới 30 bản và top-k bị bản sao chiếm chỗ; sau, mỗi chunk duy nhất 1 bản, reload idempotent.
- **Kết quả:** top-k đa dạng, tiết kiệm embedding/lưu trữ, reload không phình.
- **Ví dụ:**
  ```
  Người dùng: "Thứ gì dùng làm nhiên liệu (fuel)?"
  ❌ Trước: top-5 = [Trash, Trash, Trash, Trash, Trash]      → trả lời nghèo nàn
  ✅ Sau:  top-5 = [Trash, Notched Plank, Twigs, Magazine, Book] → trả lời đầy đủ
  ```

## C4 — SQLite `pz_data.db` trùng lặp & rác
- **Nguyên nhân:** `scripts/convert_md_to_sqlite.py` không có ràng buộc `UNIQUE`/upsert cho `recipes`/`locations`; chỉ `items` có merge thủ công và chỉ gộp khi **mọi cột giống hệt**; regex recipe hỏng vẫn lưu `name='Unknown Recipe'`; nhãn cột bảng wiki ("Attack speed"...) bị nhận nhầm là item. Số liệu thật: 1.455/3.884 item rỗng `item_id`, "Unknown Recipe" ×32, nhiều trùng tên.
- **Cách khắc phục:** Bỏ/log record không parse được tên; dedup `recipes` theo `(name,product,type,ingredients)`, `locations` theo `(name,coords,area)`, item-không-id theo `(name,category,sub_category)` giữ dòng đầy đủ nhất; blacklist tên trùng nhãn-cột; thêm `UNIQUE` + `INSERT … ON CONFLICT`.
- **Trước → Sau:** Trước, `get_recipe`/`search_items` trả nhiều "Unknown Recipe"/bản trùng; sau, mỗi thực thể một dòng sạch.
- **Kết quả:** Tool trả kết quả rõ ràng, không nhập nhằng.
- **Ví dụ:**
  ```
  Người dùng: "Cách nấu súp?"   → tool get_recipe
  ❌ Trước: [Unknown Recipe, Unknown Recipe, Unknown Recipe, ...]   → không rõ món gì
  ✅ Sau:  [Pot of Soup, Vegetable Stew]   → tên rõ ràng + nguyên liệu
  ```

## C5 — Bộ đánh giá & "cổng release" không chặn được gì (theater)
- **Nguyên nhân:** `scripts/run_release_eval.py:220` **luôn `return 0`** dù gate fail (chỉ `return 1` cho lỗi schema); job CI `rag-release-eval` **mặc định tắt** + cần self-hosted runner; ~13/20 ngưỡng gate trỏ tới metric **không hề được tính**; dataset eval chỉ **4 dòng** (schema đòi 300); LLM-judge & RAGAS **không bao giờ được gọi** trong đường release.
- **Cách khắc phục:** `main()` phải `return 1` khi `gate_result.passed == False`; bật job eval (ít nhất split `development`) chạy vô điều kiện trên PR; tính đủ mọi metric đã gate (hoặc bỏ ngưỡng cho metric không tính); dựng dataset 300 dòng thật; gọi judge/RAGAS và gate theo calibration.
- **Trước → Sau:** Trước, một bản RAG hỏng hoàn toàn vẫn "build xanh"; sau, build đỏ khi chất lượng tụt dưới ngưỡng.
- **Kết quả:** Cổng thực sự chặn regression trước khi release.
- **Ví dụ:**
  ```
  Lập trình viên đổi tham số làm recall@5 tụt 0.85 → 0.30
  ❌ Trước: CI xanh → merge → người dùng nhận RAG tệ
  ✅ Sau:  CI đỏ "gate failed: recall_at_5 0.30 < 0.70" → chặn merge
  ```

---

# 🟠 HIGH

## H1 — Điểm "relevance" của pgvector có thể ÂM / sai thang
- **Nguyên nhân:** PGVector cosine: relevance = `1 - distance`, mà distance ∈ [0,2] → relevance ∈ **[-1,1]** (LangChain chỉ cảnh báo, không clamp). Mọi ngưỡng (`0.3`, `0.35`, `0.55`) giả định thang [0,1]; `format_kb_results_for_prompt` còn in `{similarity:.0%}`.
- **Cách khắc phục:** Đặt `distance_strategy` tường minh + hàm relevance clamp [0,1]; hiệu chỉnh lại toàn bộ ngưỡng; tối thiểu clamp điểm âm về 0 trong `search_similar` (`rag/db.py:265-288`).
- **Trước → Sau:** Trước, ngưỡng không mang đúng ý nghĩa và có thể hiển thị phần trăm âm; sau, điểm chuẩn [0,1], ngưỡng đúng nghĩa.
- **Kết quả:** Lọc theo ngưỡng chính xác, hiển thị % hợp lý.
- **Ví dụ:**
  ```
  Câu VN khớp yếu với KB tiếng Anh
  ❌ Trước: hiển thị "(-12% match)" hoặc lọt qua ngưỡng do so sánh sai thang
  ✅ Sau:  hiển thị "(8% match)" và bị lọc đúng dưới ngưỡng 30%
  ```

## H2 — RRF trộn lẫn `rrf_score` (~0.01) với `similarity` (cosine)
- **Nguyên nhân:** Sau fusion, doc chỉ-BM25 có `rrf_score ≈ 0.0098` còn doc vector có `similarity ≈ 0.5–0.9`, nhưng downstream dùng `r.get("similarity", r.get("rrf_score", 0))` như cùng một thang (`rag/self_rag.py:243`, `rag/retriever.py:537`, `rag/evidence.py:160`).
- **Cách khắc phục:** Giữ `similarity` gốc xuyên suốt fusion; mỗi downstream chọn rõ **một** thang (cosine chuẩn hóa hoặc điểm theo rank), không trộn.
- **Trước → Sau:** Trước, heuristic "skip grading" gần như không kích hoạt (lãng phí LLM), metrics là rác, confidence lệch chuẩn; sau, điểm nhất quán nên các cơ chế hoạt động đúng.
- **Kết quả:** Tiết kiệm lời gọi LLM, metrics đáng tin, confidence chuẩn.
- **Ví dụ:**
  ```
  ❌ Trước: dashboard log avg_similarity = 0.01 dù kết quả rất khớp → hiểu nhầm "RAG kém"
  ✅ Sau:  avg_similarity = 0.72 phản ánh đúng chất lượng truy xuất
  ```

## H3 — Truy xuất đa ngôn ngữ chưa xử lý (người dùng VN ↔ KB tiếng Anh)
- **Nguyên nhân:** BM25 thuần từ vựng (`rag/bm25_search.py`): token "rìu" không bao giờ khớp token "axe"; hỗ trợ đa ngữ duy nhất là map viết tắt VN→VN + prefix EN (`rag/query_preprocessor.py`). Việc vector "bắc cầu" ngôn ngữ phụ thuộc hoàn toàn vào model embedding nhưng **không có kiểm tra**.
- **Cách khắc phục:** Thêm bước mở rộng/dịch từ khóa VN→EN trước BM25 (glossary thuật ngữ game), hoặc index alias song ngữ; assert model embedding đa ngữ khi `query_language=='vi'`.
- **Trước → Sau:** Trước, nhánh BM25 đóng góp ~0 cho câu VN → hybrid sụp về vector-only; sau, cả BM25 lẫn vector cùng bắt được câu VN.
- **Kết quả:** Recall cho người dùng VN (đối tượng chính) tăng mạnh.
- **Ví dụ:**
  ```
  Người dùng: "rìu nào chặt cây nhanh nhất?"
  ❌ Trước: BM25 tìm "rìu" trong KB tiếng Anh → 0 kết quả; chỉ còn vector
  ✅ Sau:  "rìu"→"axe" → BM25 khớp "Axe", "Wood Axe", "Hand Axe" → kết quả đầy đủ
  ```

## H4 — Index BM25 bị cũ (stale) so với vector store
- **Nguyên nhân:** BM25 chat chỉ refresh khi approve/revoke, BM25 KB chỉ khi `@reload`; ingest mới (`add_documents_batch`) **không** gọi `refresh_from_db()` → vector và BM25 chạy trên hai corpus khác nhau cho tới khi restart.
- **Cách khắc phục:** Refresh BM25 sau ingest/reload (hoặc theo TTL); gom chung KB reload + BM25 refresh để không lệch nhau.
- **Trước → Sau:** Trước, vector thấy tài liệu mới còn BM25 thì chưa → fusion lệch; sau, hai nhánh đồng bộ.
- **Kết quả:** Hybrid nhất quán, cập nhật không cần restart.
- **Ví dụ:**
  ```
  Admin @reload thêm tài liệu "Quy định PvP mới"
  ❌ Trước: vector tìm được, BM25 chưa có → xếp hạng kém ổn định
  ✅ Sau:  cả hai nhánh đều có tài liệu mới ngay lập tức
  ```

## H5 — Self-RAG có thể xóa sạch một tập kết quả tốt
- **Nguyên nhân:** Grader LLM nhỏ (`temperature=0.1`) dễ chấm nhầm relevant → irrelevant; **không có sàn minimum-keep** (`rag/self_rag.py:186-216`). Nếu ≤10 kết quả và grader chấm hỏng hết, tập lọc thành rỗng → EvidencePolicy ABSTAIN oan.
- **Cách khắc phục:** Luôn giữ top-N theo rank gốc nếu grading định loại sạch; coi grade lỗi/mơ hồ là "giữ".
- **Trước → Sau:** Trước, retrieval đúng nhưng grader chấm nhầm → từ chối oan; sau, luôn còn ≥1 doc tốt để trả lời.
- **Kết quả:** Giảm "false abstain" (từ chối oan), tận dụng retrieval đúng.
- **Ví dụ:**
  ```
  Người dùng: "Muldraugh có gì đặc biệt?"   (KB có trang Muldraugh rõ ràng)
  ❌ Trước: grader chấm nhầm "irrelevant" → "chưa đủ thông tin" (oan)
  ✅ Sau:  giữ lại top doc Muldraugh → trả lời đúng
  ```

## H6 — Cổng groundedness fail-OPEN khi timeout/lỗi
- **Nguyên nhân:** `rag/groundedness.py:167-175` khi timeout (`RAG_GROUNDEDNESS_TIMEOUT=20s`)/lỗi provider/JSON hỏng → trả `grounded=True` (mặc định `FAIL_OPEN=true`) và ghi `score=1.0` che mất sự cố.
- **Cách khắc phục:** Production set `RAG_GROUNDEDNESS_FAIL_OPEN=false` (ít nhất fail-closed riêng cho timeout); không ghi `score=1.0` cho ca fail-open.
- **Trước → Sau:** Trước, lúc tải nặng câu chưa kiểm vẫn được gửi; sau, timeout → ABSTAIN an toàn.
- **Kết quả:** Không "nói dối khi hệ thống đang căng"; metric phản ánh đúng tỉ lệ fail.
- **Ví dụ:**
  ```
  Giờ cao điểm, judge LLM timeout 20s
  ❌ Trước: câu trả lời (có thể bịa) vẫn gửi, log score=1.0
  ✅ Sau:  "Mình chưa đủ thông tin..." + log fail-closed
  ```

## H7 — System prompt tự mâu thuẫn; "không dùng kiến thức ngoài" chỉ là *yêu cầu*
- **Nguyên nhân:** `prompts/templates/base_identity.txt` ("answer as accurately and honestly as possible") và `game_domain_rules.txt` ("general-purpose chatbot") **đối nghịch** với `rag_instructions.txt` ("không trả lời từ kiến thức nền"). Trên lượt ANSWER bằng chứng mỏng, không có enforce nào ngoài cổng groundedness.
- **Cách khắc phục:** Bỏ/giới hạn câu "general-purpose chatbot"/"answer as accurately as possible" cho lượt domain game; làm RAG-only thành nguyên tắc chính.
- **Trước → Sau:** Trước, bằng chứng mỏng → model "chêm" kiến thức nền; sau, model bám nguồn, thiếu thì báo thiếu.
- **Kết quả:** Giảm bịa trên lượt ANSWER bằng chứng mỏng (phòng-thủ-chiều-sâu cho cổng groundedness).
- **Ví dụ:**
  ```
  Bằng chứng mỏng về một cơ chế game
  ❌ Trước: model bổ sung thông tin từ kiến thức ngoài game cho "đầy đủ"
  ✅ Sau:  chỉ nói điều có trong nguồn; phần thiếu → "nguồn chưa đề cập"
  ```

## H8 — Rò rỉ lỗi nội bộ ra người dùng & API
- **Nguyên nhân:** `src/aclient.py:876,943,1046` và `api/main.py:475,524,534` đưa `str(e)` ra người dùng/HTTP `detail` → lộ connection string, host:port, đường dẫn, SQL.
- **Cách khắc phục:** Trả message chung + request id; chi tiết chỉ log phía server.
- **Trước → Sau:** Trước, người dùng thấy DSN/đường dẫn nội bộ; sau, chỉ thấy thông báo chung + mã tra cứu.
- **Kết quả:** Bịt rò rỉ thông tin hạ tầng.
- **Ví dụ:**
  ```
  ❌ Trước: "❌ Error: connection to postgresql://user:pass@10.0.0.5:5432 failed"
  ✅ Sau:  "❌ Xin lỗi, có sự cố. Mã tham chiếu: a1b2c3"   (chi tiết ở log server)
  ```

## H9 — Không validate secret bắt buộc lúc khởi động
- **Nguyên nhân:** `DISCORD_BOT_TOKEN` (`src/bot.py:442`), `LLM_PROVIDER`/credentials (`src/llm/factory.py`) không kiểm lúc boot; provider hỏng bị nuốt trong `health_check()` và bot vẫn chạy.
- **Cách khắc phục:** Validate token/provider/key ngay lúc khởi động; thiếu → từ chối boot (fail-fast).
- **Trước → Sau:** Trước, bot "lên" bình thường rồi mọi tin nhắn fail; sau, cấu hình sai → không khởi động, báo rõ.
- **Kết quả:** Phát hiện lỗi cấu hình ngay, không để người dùng gặp lỗi per-request (kèm rò lỗi ở H8).
- **Ví dụ:**
  ```
  Quên set LLM_PROVIDER
  ❌ Trước: bot online; mỗi tin nhắn → "❌ ...LLM_PROVIDER is required"
  ✅ Sau:  khởi động dừng ngay: "FATAL: LLM_PROVIDER chưa cấu hình"
  ```

---

# 🟡 MEDIUM

## M1 — Chunk gần-trùng làm ngợp retrieval (Glass Bottle ×21)
- **Nguyên nhân:** Các record cùng tên chỉ khác `Item ID`/1 dòng stat tạo ra nhiều chunk gần-y-hệt trong không gian embedding (không trùng byte nên thoát C3). `knowledge/docs/pz/Equipment/Fluid containers.md` có 21 block `### Glass Bottle`.
- **Cách khắc phục:** MMR/diversity re-ranking khi truy xuất, hoặc gộp các record cùng tên thành 1 chunk liệt kê biến thể.
- **Trước → Sau:** Trước, "glass bottle" trả 5 chunk na ná; sau, kết quả đa dạng.
- **Kết quả:** Câu trả lời bao quát nhiều thực thể khác nhau.
- **Ví dụ:**
  ```
  Người dùng: "glass bottle dùng làm gì?"
  ❌ Trước: 5 chunk Glass Bottle gần giống nhau
  ✅ Sau:  1 Glass Bottle + Water Bottle + Empty Bottle + ...
  ```

## M2 — Query enrichment làm nhiễu retrieval chat
- **Nguyên nhân:** `rag/retriever.py:444-448` nối 5 tin nhắn gần nhất vào query rồi đem embed + BM25 → token lạ làm trôi chủ đề.
- **Cách khắc phục:** Dùng ngữ cảnh để re-rank, không nối vào query embed; nếu dùng thì giảm trọng số / chỉ lấy lượt ngay trước.
- **Trước → Sau:** Trước, câu hỏi rõ bị "pha loãng" bởi tán gẫu trước đó; sau, truy vấn tập trung.
- **Kết quả:** Tăng độ chính xác (precision) cho truy xuất chat.
- **Ví dụ:**
  ```
  Lịch sử: "ai online ko", "đói quá"  →  Câu hỏi: "cách sửa xe?"
  ❌ Trước: embed "cách sửa xe [context: ai online ko | đói quá]" → lệch
  ✅ Sau:  embed "cách sửa xe" → đúng trọng tâm
  ```

## M3 — `is_trusted_result` fail-open cho KB
- **Nguyên nhân:** `rag/trust.py:14` mặc định `trusted=True` khi metadata thiếu key `trusted`.
- **Cách khắc phục:** Mặc định `False` (fail-closed) khi thiếu key.
- **Trước → Sau:** Trước, chunk thiếu cờ trust vẫn được tin; sau, phải có cờ trust hợp lệ mới được tin.
- **Kết quả:** Ranh giới tin cậy an toàn hơn.
- **Ví dụ:**
  ```
  Chunk cũ insert thiếu field "trusted"
  ❌ Trước: được coi là đáng tin → có thể vào câu trả lời
  ✅ Sau:  bị loại cho tới khi gắn cờ trust đúng
  ```

## M4 — Domain routing chỉ theo từ khóa tiếng Anh
- **Nguyên nhân:** `knowledge/domain_router.py` + `_detect_domain` chỉ match keyword EN; câu VN hiếm khi khớp → rơi về "search all" hoặc bỏ KB nếu KB load lỗi (không cảnh báo).
- **Cách khắc phục:** Bổ sung keyword song ngữ; log/metric khi routing ra 0 domain do KB rỗng (khác với "không khớp").
- **Trước → Sau:** Trước, câu VN định tuyến kém ổn định; sau, định tuyến domain đúng.
- **Kết quả:** Truy xuất KB đúng domain, ít quét thừa.
- **Ví dụ:**
  ```
  Người dùng: "luật server về xây nhà?"
  ❌ Trước: không khớp keyword EN → quét tất cả domain
  ✅ Sau:  khớp "luật/quy định" → route đúng domain server_rules
  ```

## M5 — Thread-context là code chết & gây nhiễu
- **Nguyên nhân:** `rag/db.py:312` trả edge chỉ có `message_id`, không có `content`/`author`; `rag/retriever.py:361` vẫn in `↳ [reply] @?:` rỗng.
- **Cách khắc phục:** Hoặc join lấy nội dung+tác giả đã duyệt (và lọc trust), hoặc bỏ hẳn phần render này.
- **Trước → Sau:** Trước, prompt chứa dòng ngữ cảnh rỗng vô nghĩa; sau, không còn rác.
- **Kết quả:** Tiết kiệm ngân sách context, prompt sạch.
- **Ví dụ:**
  ```
  ❌ Trước: prompt có "↳ [reply] @?: " (rỗng) × nhiều dòng
  ✅ Sau:  không còn dòng rỗng, hoặc hiển thị ngữ cảnh thật có nội dung
  ```

## M6 — `add_documents_batch` nuốt lỗi từng item
- **Nguyên nhân:** `rag/db.py:222-237` bắt lỗi rồi tiếp tục; trả `count` có thể thấp hơn nhiều so với số message thật.
- **Cách khắc phục:** Trả số lỗi cho caller; fail to khi tỉ lệ lỗi vượt ngưỡng.
- **Trước → Sau:** Trước, ingest thiếu âm thầm, caller báo "ingested N" sai; sau, lỗi được phơi bày.
- **Kết quả:** Corpus đầy đủ, phát hiện sớm lỗi embedding/dimension.
- **Ví dụ:**
  ```
  Embedding sai dimension → 900/1000 doc fail
  ❌ Trước: log "ingested 1000" (sai), corpus chỉ có 100
  ✅ Sau:  "ingest failed: 900/1000 errors" → dừng & báo
  ```

## M7 — `metric.retrieval_time_ms` bị ghi đè bằng total time
- **Nguyên nhân:** `rag/retriever.py:495` set thời gian retrieval, rồi `:596` ghi đè bằng tổng thời gian.
- **Cách khắc phục:** Dùng hai field riêng (`retrieval_time_ms` và `total_time_ms`).
- **Trước → Sau:** Trước, "retrieval time" thực ra là total; sau, đo đúng từng giai đoạn.
- **Kết quả:** Quan sát hiệu năng chính xác.
- **Ví dụ:**
  ```
  ❌ Trước: dashboard báo retrieval 1200ms (gồm cả Self-RAG + format)
  ✅ Sau:  retrieval 300ms, total 1200ms — biết nút thắt nằm đâu
  ```

## M8 — Không cap `max_tokens` khi sinh
- **Nguyên nhân:** `max_tokens` được nối tới client nhưng không caller nào truyền → sinh không giới hạn.
- **Cách khắc phục:** Truyền `max_tokens` hợp lý cho provider.
- **Trước → Sau:** Trước, có thể sinh rất dài (chi phí/độ trễ, làm judge groundedness timeout → H6); sau, độ dài có trần.
- **Kết quả:** Kiểm soát chi phí/độ trễ, giảm DoS.
- **Ví dụ:**
  ```
  Prompt khiến model lặp vô hạn
  ❌ Trước: sinh 8000 token → chậm, tốn tiền, judge timeout
  ✅ Sau:  dừng ở max_tokens=1024
  ```

## M9 — Reload KB delete-then-insert không nguyên tử
- **Nguyên nhân:** `knowledge/manager.py:637-666` xóa cũ rồi insert mới trong hai kết nối khác nhau; lỗi giữa chừng để domain rỗng/nửa vời.
- **Cách khắc phục:** Dùng `ids=` upsert (C3) để insert idempotent, và/hoặc bọc delete+insert trong một transaction.
- **Trước → Sau:** Trước, reload lỗi có thể xóa sạch domain; sau, an toàn khi lỗi.
- **Kết quả:** Reload không làm mất dữ liệu.
- **Ví dụ:**
  ```
  @reload pz nhưng embedding lỗi giữa chừng
  ❌ Trước: domain "pz" còn 0 chunk → bot mất toàn bộ tri thức PZ
  ✅ Sau:  giữ nguyên dữ liệu cũ cho tới khi insert thành công
  ```

## M10 — Default mất an toàn (`local-dev-key`, `postgres:postgres`)
- **Nguyên nhân:** `.env.example` ship key/DB mặc định; `src/llm/openai_compatible.py:15` fallback `api_key or "local-dev-key"` không có guard production.
- **Cách khắc phục:** Không fallback `"local-dev-key"`; thêm guard từ chối DB cred mặc định ở production (như `API_KEY` đã làm).
- **Trước → Sau:** Trước, dễ deploy nhầm với cred mặc định; sau, bị chặn.
- **Kết quả:** Giảm rủi ro lộ/yếu credential.
- **Ví dụ:**
  ```
  Deploy quên đổi key
  ❌ Trước: chạy với "local-dev-key" / postgres:postgres
  ✅ Sau:  "FATAL: phát hiện credential mặc định, từ chối khởi động ở production"
  ```

## M11 — Lỗi quy ước trong metric eval
- **Nguyên nhân:** `evaluation/retrieval_metrics.py`: precision@k chia `min(k,len)` (thổi phồng), nDCG không dedup, keyword_coverage match substring (gameable bởi "axe" ⊂ "axed").
- **Cách khắc phục:** Chuẩn hóa: precision@k chia `k`; dedup `retrieved` trước nDCG; keyword match theo ranh giới từ + bỏ dấu cho tiếng Việt.
- **Trước → Sau:** Trước, số đo bị thổi phồng/sai; sau, phản ánh đúng.
- **Kết quả:** Eval đáng tin để gate.
- **Ví dụ:**
  ```
  Hệ thống trả 1 doc đúng, không trả gì thêm
  ❌ Trước: precision@5 = 1.0 (chia cho 1)
  ✅ Sau:  precision@5 = 0.2 (chia cho 5)
  ```

---

# ⚪ LOW

## L1 — Persona jailbreak (code chết) khuyến khích bịa
- **Nguyên nhân:** `src/personas.py` chứa prompt "make up answers... sound plausible"; hiện `build_system_prompt` **không** đọc `current_persona` nên vô hại, nhưng là footgun.
- **Cách khắc phục:** Xóa các persona jailbreak; nếu cần persona thì inject per-request và không cho ghi đè ràng buộc RAG-only.
- **Trước → Sau:** Trước, mã độc-hại nằm chờ; sau, không còn.
- **Kết quả:** Loại bỏ rủi ro nếu ai đó vô tình nối vào prompt.
- **Ví dụ:**
  ```
  ❌ Trước: tồn tại persona "SAM" chỉ thị model bịa cho hợp lý
  ✅ Sau:  xóa hẳn — không thể bị kích hoạt nhầm
  ```

## L2 — `/chat` slash bỏ qua cổng groundedness
- **Nguyên nhân:** `src/aclient.py:975-1031` (`handle_response`) tôn trọng ABSTAIN/CLARIFY nhưng trên ANSWER không gọi `finalize_rag_answer`.
- **Cách khắc phục:** Cho `/chat` đi qua cùng bước finalize.
- **Trước → Sau:** Trước, `/chat` không có kiểm chứng + không nguồn; sau, đồng nhất với mention/reply.
- **Kết quả:** Mọi đường vào đều được kiểm chứng.
- **Ví dụ:**
  ```
  Người dùng dùng "/chat câu hỏi game"
  ❌ Trước: trả lời không qua groundedness, không nguồn
  ✅ Sau:  qua groundedness + có footer nguồn
  ```

## L3 — `min_vector_score` là config chết
- **Nguyên nhân:** `EvidencePolicy.__init__` đặt `self.min_vector_score` nhưng không dùng ở đâu.
- **Cách khắc phục:** Dùng nó như sàn cho similarity, hoặc xóa để khỏi gây hiểu nhầm.
- **Trước → Sau:** Trước, cấu hình "ma" gây ngộ nhận; sau, sạch.
- **Kết quả:** Cấu hình phản ánh đúng hành vi.
- **Ví dụ:**
  ```
  ❌ Trước: chỉnh RAG_EVIDENCE_MIN_VECTOR_SCORE=0.8 → không có tác dụng gì
  ✅ Sau:  hoặc có tác dụng thật, hoặc biến mất khỏi config
  ```

## L4 — `validate_embedding_signature` định nghĩa nhưng không gọi
- **Nguyên nhân:** `rag/embedding_registry.py` có hàm so khớp chữ ký nhưng `get_vectorstore` chỉ log, không so với chữ ký đã lưu.
- **Cách khắc phục:** Gọi validate lúc khởi động; mismatch → từ chối phục vụ.
- **Trước → Sau:** Trước, đổi model embedding mà quên reindex → cosine rác, không báo; sau, bị chặn.
- **Kết quả:** Tránh "drift" model embedding âm thầm.
- **Ví dụ:**
  ```
  Đổi EMBEDDING_MODEL nhưng vector cũ vẫn của model cũ
  ❌ Trước: mọi truy vấn cho kết quả rác, không lỗi
  ✅ Sau:  "FATAL: embedding signature mismatch — cần reindex"
  ```

## L5 — Phát hiện ngôn ngữ mong manh; câu VN ngắn bị bỏ RAG
- **Nguyên nhân:** `rag/query_preprocessor.py` đếm ký tự có dấu (dễ nhầm); câu VN ngắn không pattern (<50 ký tự) bị xếp `CONVERSATION` → không RAG.
- **Cách khắc phục:** Dùng detector ngôn ngữ thực thụ; không mặc định câu ngắn có-domain thành CONVERSATION.
- **Trước → Sau:** Trước, "thuốc giảm đau?" bị coi là tán gẫu → bỏ RAG; sau, vẫn được RAG.
- **Kết quả:** Câu hỏi ngắn vẫn được trả lời có nguồn.
- **Ví dụ:**
  ```
  Người dùng: "thuốc giảm đau?"
  ❌ Trước: xếp CONVERSATION → trả lời chung chung, không tra KB
  ✅ Sau:  nhận diện câu hỏi → tra KB Medical → trả lời có nguồn
  ```

## L6 — `knowledge/docs/pz/list_md.txt` rỗng (0 byte)
- **Nguyên nhân:** Manifest liệt kê file KB bị bỏ trống (loader thực tế walk cây thư mục, không đọc file này).
- **Cách khắc phục:** Regenerate (`find ... -name '*.md' | sort`) trong build, hoặc xóa nếu không dùng.
- **Trước → Sau:** Trước, manifest gây hiểu nhầm về độ phủ; sau, đúng hoặc biến mất.
- **Kết quả:** Tín hiệu bảo trì/độ phủ đáng tin.
- **Ví dụ:**
  ```
  ❌ Trước: list_md.txt rỗng → reviewer tưởng KB trống
  ✅ Sau:  list_md.txt liệt kê đủ 160 file, hoặc bị xóa
  ```

## L7 — Persist metric fire-and-forget nuốt lỗi; SQL nội suy `%`
- **Nguyên nhân:** `rag/metrics.py` dùng `asyncio.create_task` không giữ tham chiếu + bắt lỗi ở mức `debug`; `get_db_summary` nội suy `%` chuỗi cho `hours`.
- **Cách khắc phục:** Giữ tham chiếu task / dùng hàng đợi giới hạn; log lỗi persist ở mức `warning`; tham số hóa SQL (`make_interval(hours => $1)`).
- **Trước → Sau:** Trước, metric mất âm thầm và có footgun SQL; sau, lỗi hiện rõ và SQL an toàn.
- **Kết quả:** Không mất metric âm thầm; loại footgun injection.
- **Ví dụ:**
  ```
  DB chập chờn
  ❌ Trước: metric rớt âm thầm (chỉ log debug) → /status thiếu số liệu
  ✅ Sau:  cảnh báo "metric persist failed" → biết để xử lý
  ```

---

## Thứ tự thực hiện đề xuất

- **P0 (bắt buộc trước khi mở cho người dùng):** C1, C2, C3, C4, H6, H8, H9
- **P1 (chất lượng):** H1, H2, H3, H4, H5, H7, C5
- **P2 (dọn dẹp):** M1–M11, L1–L7 (ưu tiên **xóa persona jailbreak L1**)
