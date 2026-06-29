# NomNom RAG — Remediation Hand-off (2026-06-29)

Hand-off cho toàn bộ công việc audit + khắc phục + dựng khung đánh giá trên nhánh
`codex/nomnom-production-ready`. Đây là tài liệu để người tiếp theo hiểu **đã làm
gì, ở đâu, còn lại gì, và cần làm gì trước khi production**.

---

## 0. TL;DR

- Đã **audit toàn bộ hệ thống RAG** (đọc code + truy vấn DB thật, không dựa tài liệu cũ).
- Đã **khắc phục toàn bộ findings P0/P1/P2** trong **7 stage**, mỗi stage 1 commit + test.
- Đã **nối LLM-judge** vào đường release (calibration-gated) và **dựng khung dataset eval**
  (pipeline inventory → draft → review → build → validate → eval → calibrate → gate).
- **Test suite: 188 passed** (local).
- **Còn lại = dữ liệu/người/hạ tầng** (không phải code): dataset 300 dòng review thật,
  ≥60 human review + calibration judge, cấu hình `EVAL_LLM_*`, bật eval trên CI runner.

> Lưu ý quan trọng: nhánh này trước đó là **một working tree lớn chưa commit hết** (nhiều
> module của tính năng rag-quality như `rag/result.py`, `rag/responses.py`,
> `evaluation/gates.py`, `evaluation/local_judge.py`… chưa từng được commit). Để push một
> nhánh **chạy được**, toàn bộ working tree đã được commit (trừ `tmp/`). Không có secret nào
> được commit (đã quét; không có `.env`).

---

## 1. Nhánh & remote

- Branch: `codex/nomnom-production-ready`
- Remote: `origin` → `https://github.com/nghuy0901/Discord-chatbot-PZ.git`
- Trạng thái pushed = đúng trạng thái 188-passing local.

## 2. Lịch sử commit của đợt làm việc này

| Commit | Nội dung |
|---|---|
| `6c78ac3` | Stage 1 — lõi điểm số & trust (scoring.py, RRF dedup, self-rag min-keep, trust fail-closed) |
| `4875572` | Stage 2 — ingest dedup + dọn SQLite + embedding pin (regenerate pz_data.db) |
| `2bff522` | Stage 3 — BM25 freshness + đa ngữ VN→EN |
| `a077438` | Stage 4 — groundedness fail-closed khi timeout + tool-source helper |
| `3d79a2c` | Stage 5 — **chặn 2 đường bịa (C1, C2)** + prompt/error/persona hardening |
| `1b17cea` | Stage 6 — validate config lúc boot + guard secret production |
| `08e2de6` | Stage 7 — enforce cổng eval + sửa quy ước metric |
| `69e7b5c` | cập nhật test persona (theo L1) |
| `4087238` | nối LLM-judge vào release eval (calibration-gated) |
| `e4ef1d5` | khung dataset eval (build_eval_dataset) + seed 20 dòng + sửa khớp recall |
| (commit docs cuối) | audit docs + playbook + hand-off này + hoàn tất working tree |

Hai tài liệu nền tảng của đợt audit:
- [`docs/audit/rag-system-audit-2026-06-29.md`](rag-system-audit-2026-06-29.md) — kiến trúc thật + findings theo severity.
- [`docs/audit/rag-bug-fix-playbook-2026-06-29.md`](rag-bug-fix-playbook-2026-06-29.md) — từng lỗi: nguyên nhân/cách sửa/trước-sau/ví dụ.

---

## 3. Đã làm gì (theo stage)

### Stage 1 — Lõi điểm số & trust (`rag/scoring.py` mới + ...)
- `rag/scoring.py`: **một nguồn điểm `[0,1]` duy nhất** (`relevance_score`), gỡ trộn
  `rrf_score`/`similarity` ở evidence/self_rag/citations/retriever (H2).
- `hybrid_retriever`: merge cross-arm theo `doc_id`, tag method cho nhánh đơn, cap chunk
  trùng-record cho KB (M1). `self_rag`: sàn min-keep (H5) + gỡ annotation xúi "dùng kiến
  thức ngoài". `evidence`: bỏ config chết (L3). `trust`: KB fail-closed (M3). `retriever`:
  bỏ nhiễu query bằng tin nhắn gần đây (M2).

### Stage 2 — Ingest & dữ liệu cấu trúc
- `manager.load_domain`: dedup chunk trùng-byte + **upsert theo `doc_id`** (reload idempotent,
  không bao giờ làm rỗng domain) (C3/M9). `db.search_similar`: clamp cosine về `[0,1]` (H1).
  `db.init_db`: **chốt chữ ký embedding** (L4). `convert_md_to_sqlite`: bỏ "Unknown Recipe"/nhãn
  cột, dedup, unique index (C4). **Đã regenerate `pz_data.db`** (Unknown Recipe 32→0, item rác
  12→0, item_id trùng→0). `list_md.txt` regenerate (L6).

### Stage 3 — BM25 freshness & hiểu truy vấn
- `query_preprocessor`: glossary **VN→EN** (H3), domain detection song ngữ pz + server_rules
  (M4), nhận diện tiếng Việt không dấu (L5). `manager.reload`: refresh KB BM25 (H4).
  `bm25_search.refresh_from_kb`: thêm điều kiện `trusted` (H4). `db.add_documents_batch`: cảnh
  báo khi ingest thiếu (M6). `retriever`: bỏ dòng thread-context rỗng (M5).

### Stage 4 — Cổng groundedness
- `groundedness`: **timeout → fail CLOSED** mặc định (H6) + helper `sources_from_tool_results`
  để kiểm chứng câu trả lời tool (C2-judge). `.env.example`: thêm `RAG_GROUNDEDNESS_TIMEOUT_FAIL_OPEN`.

### Stage 5 — Orchestration & prompt (CHẶN BỊA)
- **C1**: RAG bật mà build lỗi (`rag_result is None`) → ABSTAIN, không gọi LLM tự do.
- **C2**: câu trả lời tool được kiểm chứng với chính output tool (`finalize_tool_answer`), sai
  → từ chối; nối vào Discord (@mention + /chat) và API.
- H7 (prompt hết mâu thuẫn), H8 (ngừng rò `str(e)`), L1 (xóa persona jailbreak), L2 (/chat qua
  finalize), M8 (cap `max_tokens`).

### Stage 6 — Khởi động & default an toàn
- `src/startup.py`: `validate_runtime_config` — fail-fast khi thiếu secret (H9); guard reject
  `local-dev-key`/`postgres:postgres` ở production (M10) qua `APP_ENV=production`.

### Stage 7 — Metrics & cổng eval
- `run_release_eval.main()` **return ≠ 0 khi gate fail** (C5); baseline tách metric chưa-đo-được
  sang `deferred_*`. `retrieval_metrics`: precision@k chia `k`, nDCG dedup, keyword theo ranh
  giới từ (M11). `metrics`: giữ ref task persist + log warning + tham số hóa SQL (L7); thêm
  `total_time_ms` (M7).

### Eval framework + judge (sau Stage 7)
- **Nối judge**: `run_release_eval` gọi `judge_answer` cho mỗi câu ANSWER; `summarize` tổng hợp
  `answer_correctness`/`faithfulness`/`unsupported_claim_rate`/`critical_error_count`. Cổng
  **chỉ enforce metric judge khi calibration `eligible_for_release_gate=true`** (`JUDGE_CALIBRATION_REPORT`).
- **Khung dataset**: `scripts/build_eval_dataset.py` (reviewed candidates → dataset đúng schema),
  seed mở rộng **4 → 20 dòng**, recall khớp theo file. Runbook:
  [`docs/evaluation/eval-dataset-and-judge-pipeline.md`](../evaluation/eval-dataset-and-judge-pipeline.md).

---

## 4. Kiểm chứng

```bash
python -m pytest -q          # → 188 passed
# Dữ liệu cấu trúc đã sạch:
sqlite3 knowledge/structured/pz_data.db "SELECT COUNT(*) FROM recipes WHERE name='Unknown Recipe';"  # 0
sqlite3 knowledge/structured/pz_data.db "SELECT COUNT(*) FROM (SELECT item_id FROM items WHERE item_id<>'' GROUP BY item_id HAVING COUNT(*)>1);"  # 0
python scripts/validate_eval_dataset.py --dataset evaluation/data/release_qa.v1.jsonl  # validated 20 rows
```

---

## 5. CÒN LẠI (cần dữ liệu/người/hạ tầng — KHÔNG phải code)

| Việc | Cách làm | Ai |
|---|---|---|
| Dataset đủ **quota 300 dòng** (180 answerable / 60 unanswerable / 30 ambiguous / 30 adversarial; ~75-85% VN; dev 210 / holdout 90) | `export_eval_candidates` → review người → `build_eval_dataset.py` → `validate --require-release-quota` | Domain reviewer |
| **≥60 human review** + calibration judge `eligible=true` | `import_human_review.py` → `calibrate_local_judge.py` → set `JUDGE_CALIBRATION_REPORT` | Reviewer |
| Cấu hình `EVAL_LLM_BASE_URL` + `EVAL_LLM_MODEL` để bật judge | `.env` | DevOps |
| Bật job `rag-release-eval` (hiện opt-in + self-hosted) chạy trên PR | `.github/workflows/ci.yml` + runner có LLM | DevOps |
| Producer cho các metric `deferred_*` còn lại: `critical_rule_recall`, `prompt_injection_pass_rate`, `pii_secret_leakage_count`, `stability_rate`, `human_answer_correctness` | thêm probe/tổng hợp đa-run; promote khỏi `deferred_*` | Eng |

> Khi 2 việc đầu xong, các metric `faithfulness`/`unsupported_claim_rate`/`critical_error_count`
> sẽ **tự động enforce** trong cổng release (không cần sửa code).

---

## 6. Checklist vận hành production (BẮT BUỘC trước khi mở cho người dùng)

1. `APP_ENV=production` — bật fail-fast + guard secret.
2. Secret thật: `DISCORD_BOT_TOKEN`, `API_KEY` (≥32 ký tự, không phải default), `LLM_*`,
   `EMBEDDING_*`, `POSTGRES_URL` (không dùng `postgres:postgres`).
3. `RAG_GROUNDEDNESS_TIMEOUT_FAIL_OPEN=false` (giữ mặc định) — abstain khi judge timeout.
4. **Reindex KB sau khi cập nhật code ingest**: dedup theo `doc_id` (C3) chỉ áp dụng cho lần
   load kế tiếp → chạy `@reload knowledge` (hoặc `POST /api/admin/reload`) **một lần** để gom
   chunk trùng + áp `doc_id` upsert lên dữ liệu hiện có. (Chữ ký embedding L4 tự ghi ở `init_db`.)
5. Chạy migration: `python scripts/migrate_db.py` (cần `migrations/004_add_rag_trust_and_quality.sql`).
6. `python -m pytest -q` xanh; `/api/health` báo postgres + llm OK.

## 7. Biến môi trường mới (đợt này)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `APP_ENV` | development | `production` để fail-fast + guard secret (H9/M10) |
| `RAG_GROUNDEDNESS_TIMEOUT_FAIL_OPEN` | false | timeout judge → abstain (H6) |
| `LLM_MAX_TOKENS` | (unset) | cap độ dài sinh (M8) |
| `RAG_MAX_PER_RECORD` | 2 | giới hạn chunk cùng record trong KB top-k (M1) |
| `SELF_RAG_MIN_KEEP` | 1 | sàn giữ kết quả khi grader loại sạch (H5) |
| `RAG_LEXICAL_ONLY_CEILING` | 0.5 | trần điểm cho match chỉ-BM25 (H2) |
| `EMBEDDING_ALLOW_SIGNATURE_RESET` | false | cho phép re-pin chữ ký sau reindex (L4) |
| `JUDGE_CALIBRATION_REPORT` | (unset) | đường dẫn report calibration để judge được enforce (C5/C6) |
| ~~`RAG_EVIDENCE_MIN_VECTOR_SCORE`~~ | — | **đã bỏ** (config chết, L3) |

## 8. Rollback

Mỗi stage là một commit độc lập trên `codex/nomnom-production-ready`; có thể `git revert <commit>`
từng stage nếu cần. Dữ liệu `pz_data.db` được regenerate bằng `python scripts/convert_md_to_sqlite.py`.
