# 📋 TỔNG QUAN CHI TIẾT — CLCT Discord Bot

> **Tên bot:** CLCT (Chấn thương tâm lý Community Tool)
> **Ngôn ngữ:** Python 3
> **Framework Discord:** discord.py (với `app_commands` cho slash commands)
> **LLM Engine chính:** Ollama (local) — model mặc định: `qwen3-next:80b-cloud`
> **Cơ sở dữ liệu:** PostgreSQL + pgvector
> **Ngày phân tích:** 2026-03-19

---

## Mục lục

1. [Tổng quan kiến trúc hệ thống](#1-tổng-quan-kiến-trúc-hệ-thống)
2. [Các câu lệnh của bot và chức năng](#2-các-câu-lệnh-của-bot-và-chức-năng)
3. [Kỹ thuật truy vấn Chatbot (RAG Pipeline)](#3-kỹ-thuật-truy-vấn-chatbot-rag-pipeline)
4. [Hệ thống Knowledge Base (D2)](#4-hệ-thống-knowledge-base-d2)
5. [Hệ thống Context Manager](#5-hệ-thống-context-manager)
6. [Hệ thống Provider đa nền tảng](#6-hệ-thống-provider-đa-nền-tảng)
7. [Hệ thống Personas](#7-hệ-thống-personas)
8. [Hệ thống Metrics & Feedback](#8-hệ-thống-metrics--feedback)
9. [Hệ thống LlamaIndex (Legacy)](#9-hệ-thống-llamaindex-legacy)
10. [Cấu hình & Biến môi trường](#10-cấu-hình--biến-môi-trường)
11. [Luồng xử lý tin nhắn End-to-End](#11-luồng-xử-lý-tin-nhắn-end-to-end)
12. [Cấu trúc thư mục dự án](#12-cấu-trúc-thư-mục-dự-án)

---

## 1. Tổng quan kiến trúc hệ thống

Bot có **2 hệ thống chính** song song:

### 1.1. Hệ thống CLCT (Chính — đang active)

- **Entry point:** `main.py` → `src/bot.py` → `src/aclient.py`
- **LLM:** Ollama (local, async streaming)
- **RAG:** LangChain PGVector + PostgreSQL + pgvector
- **Embedding:** `nomic-embed-text` (768 chiều) qua Ollama
- **Knowledge Base:** Multi-domain static documents (Markdown/TXT)
- **Context Manager:** Short-term memory per channel (rolling window)

### 1.2. Hệ thống Legacy (discord_bot.py + llm_provider.py)

- **Entry point:** `discord_bot.py`
- **LLM:** LlamaIndex Query Engine (hỗ trợ Ollama, OpenAI, DeepSeek, Anthropic)
- **RAG:** LlamaIndex VectorStoreIndex (local storage)
- **Embedding:** HuggingFace local hoặc OpenAI embeddings

### Sơ đồ kiến trúc (Hệ thống CLCT):

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Discord Server                              │
│  @mention / Reply / Thread / /chat slash command                    │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────┐
│  src/bot.py (Event Handlers)      │
│  • on_ready, on_message           │
│  • Slash commands (/chat, /reset) │
│  • Reaction events (👍/👎)        │
└───────────────────┬───────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│  src/aclient.py (CLCTClient)      │
│  • Trigger detection              │
│  • Message queue & rate limiting  │
│  • KB admin command handling (C3) │
│  • Stream response generation     │
│  • Feedback reaction handling (B2)│
└───────────────────┬───────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌────────────┐ ┌──────────┐ ┌─────────────────┐
│ Context    │ │ Ollama   │ │ RAG Pipeline    │
│ Manager    │ │ Provider │ │                 │
│            │ │          │ │ Query Preproc.  │
│ • Per-     │ │ • Chat   │ │ (A2)            │
│   channel  │ │   compl. │ │        │        │
│   history  │ │ • Stream │ │        ▼        │
│ • Build    │ │ • Health │ │ ┌─────────────┐ │
│   prompt   │ │   check  │ │ │ Knowledge   │ │
│ • Implicit │ └──────────┘ │ │ Base (D2)   │ │
│   reply    │              │ │ + Chat Hist. │ │
│   detect   │              │ │ (PGVector)   │ │
│            │              │ └──────┬──────┘ │
└────────────┘              │        │        │
                            │        ▼        │
                            │  Metrics (A3)   │
                            │  + Feedback(B2) │
                            └─────────────────┘
```

---

## 2. Các câu lệnh của bot và chức năng

### 2.1. Slash Commands (Lệnh `/`)

| #   | Lệnh                    | Mô tả                                       | File                 | Chi tiết                                                                                                                                                                                                                                                   |
| --- | ----------------------- | ------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `/chat [message]`       | Chat trực tiếp với bot                      | `src/bot.py:121-143` | Gửi tin nhắn cho bot xử lý. Giới hạn 2000 ký tự. Sử dụng message queue để xếp hàng xử lý. Hỗ trợ streaming response.                                                                                                                                       |
| 2   | `/reset`                | Xóa ngữ cảnh hội thoại của channel hiện tại | `src/bot.py:145-151` | Xóa toàn bộ lịch sử hội thoại trong bộ nhớ ngắn hạn (context window) cho channel hiện tại.                                                                                                                                                                 |
| 3   | `/resetall`             | Xóa TẤT CẢ ngữ cảnh hội thoại (admin)       | `src/bot.py:153-158` | Xóa tất cả context của mọi channel. Dành cho admin.                                                                                                                                                                                                        |
| 4   | `/status`               | Hiển thị trạng thái bot                     | `src/bot.py:160-233` | Hiển thị embed chứa: trạng thái Ollama (online/offline), model đang dùng, số message đã index trong RAG, thông tin Knowledge Base (số domain, docs, chunks), số channel active, và RAG metrics (số queries, avg retrieval time, avg similarity, feedback). |
| 5   | `/provider`             | Chuyển đổi AI provider                      | `src/bot.py:235-284` | Hiển thị dropdown menu để chọn provider (Free, OpenAI, Claude, Gemini, Grok). Là chức năng legacy, CLCT chủ yếu dùng Ollama.                                                                                                                               |
| 6   | `/switchpersona [name]` | Đổi tính cách AI                            | `src/bot.py:286-325` | Chuyển đổi giữa các persona: `standard`, `creative`, `technical`, `casual`. Các persona `jailbreak-v1/v2/v3` yêu cầu quyền admin. Reset toàn bộ context khi switch.                                                                                        |
| 7   | `/private`              | Toggle chế độ phản hồi riêng tư             | `src/bot.py:327-332` | Bật/tắt chế độ ephemeral response (chỉ người dùng gửi lệnh mới thấy).                                                                                                                                                                                      |
| 8   | `/replyall`             | Toggle chế độ trả lời tất cả                | `src/bot.py:334-339` | Bật/tắt chế độ bot tự động trả lời mọi tin nhắn trong channel (hoặc channel cụ thể).                                                                                                                                                                       |
| 9   | `/help`                 | Hiển thị danh sách lệnh                     | `src/bot.py:341-386` | Hiển thị embed với đầy đủ các lệnh chia theo nhóm: Chat, Info, Personas, Settings, Knowledge Base, Feedback.                                                                                                                                               |

### 2.2. Trigger Commands (Lệnh qua @mention / chat)

| #   | Trigger            | Cách sử dụng                | Chức năng                                                                                                                                                                                          |
| --- | ------------------ | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **@mention**       | Tag bot trong tin nhắn      | Bot phản hồi khi được mention trực tiếp. Nội dung @mention sẽ bị remove trước khi xử lý.                                                                                                           |
| 2   | **Reply to bot**   | Reply vào tin nhắn của bot  | Bot tự động phản hồi khi user reply vào tin nhắn trước đó của bot.                                                                                                                                 |
| 3   | **Active thread**  | Chat trong thread có bot    | Bot tự động phản hồi trong các thread mà bot đã từng tham gia (`bot_participated = True`).                                                                                                         |
| 4   | **Implicit reply** | Chat bình thường (tùy chọn) | Nếu `ENABLE_IMPLICIT_REPLIES=true`, bot dùng embedding similarity để kiểm tra xem tin nhắn mới có liên quan đến tin nhắn bot gần đây không (threshold: 0.75). **Mặc định: TẮT** vì tốn tài nguyên. |

### 2.3. Knowledge Base Admin Commands (C3)

> Các lệnh được gửi qua **@mention** bot. Yêu cầu user có ID nằm trong `ADMIN_USER_IDS`.

| #   | Lệnh                                                            | Chức năng                                                                                                                            |
| --- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `@Bot reload knowledge` hoặc `@Bot kb reload`                   | Hot-reload toàn bộ knowledge base: quét lại thư mục docs, re-chunk, re-embed vào PGVector. Hiển thị số chunks mỗi domain sau reload. |
| 2   | `@Bot scan knowledge` hoặc `@Bot kb status` hoặc `@Bot kb scan` | Hiển thị trạng thái knowledge base: initialized, số domain, số docs, số chunks, và chi tiết từng domain (có prompt hay không).       |

### 2.4. Feedback Commands (B2)

| #   | Action                        | Chức năng                                                                              |
| --- | ----------------------------- | -------------------------------------------------------------------------------------- |
| 1   | React **👍** vào tin nhắn bot | Ghi nhận feedback tích cực (score = +1). Lưu vào bảng `rag_feedback` trong PostgreSQL. |
| 2   | React **👎** vào tin nhắn bot | Ghi nhận feedback tiêu cực (score = -1). Lưu vào bảng `rag_feedback` trong PostgreSQL. |

> Sau mỗi response, bot tự động thêm reaction 👍 và 👎 vào tin nhắn của mình để user dễ dàng đánh giá.

---

## 3. Kỹ thuật truy vấn Chatbot (RAG Pipeline)

Bot sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** — kết hợp truy vấn dữ liệu từ cơ sở tri thức với khả năng sinh text của LLM. Dưới đây là phân tích chi tiết từng bước.

### 3.1. Tổng quan Pipeline

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  A2: Query Preprocessing            │
│  • Xóa Discord formatting           │
│  • Mở rộng viết tắt (VN/EN/Gaming)  │
│  • Phát hiện ngôn ngữ (vi/en)       │
│  • Phát hiện domain (pz, general...)│
│  • Thêm domain context prefix       │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼                     ▼
┌────────────────┐  ┌────────────────────┐
│ D2: Knowledge  │  │ Chat History       │
│ Base Search    │  │ Search (PGVector)  │
│                │  │                    │
│ • Domain       │  │ • Similarity       │
│   routing      │  │   search           │
│ • PGVector     │  │ • A4: Fallback     │
│   similarity   │  │   threshold        │
│ • Top-K + THR  │  │ • Thread context   │
└───────┬────────┘  └───────┬────────────┘
        │                   │
        └─────────┬─────────┘
                  ▼
    ┌─────────────────────────┐
    │ Merge & Format Results  │
    │ • KB: [KB-1], [KB-2]... │
    │ • Chat: [1], [2]...     │
    │ • Truncate to max_chars │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Build Full Prompt       │
    │ • System prompt (C1)    │
    │ • Domain prompt (D2)    │
    │ • RAG context           │
    │ • Conversation history  │
    │ • Current user message  │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Ollama Chat Completion  │
    │ • Streaming or batch    │
    │ • Temperature: 0.8      │
    │ • Context window: 8192  │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ A3: Record Metrics      │
    │ • Retrieval time        │
    │ • Similarity scores     │
    │ • Response time/length  │
    │ • Domain matched        │
    └─────────────────────────┘
```

### 3.2. Chi tiết từng bước

#### Bước 1: Query Preprocessing (A2) — `rag/query_preprocessor.py`

| Bước con                      | Mô tả                                                                                               | Ví dụ                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Remove Discord formatting** | Xóa mentions (`<@123>`), channels (`<#456>`), roles (`<@&789>`), custom emojis (`<:name:id>`), URLs | `<@123> pz có gì hay k` → `pz có gì hay k`                                  |
| **Collapse whitespace**       | Gộp nhiều khoảng trắng thành 1                                                                      | `hello     world` → `hello world`                                           |
| **Detect language**           | Đếm ký tự có dấu tiếng Việt (àáảãạ...) — nếu ≥ 2 → `vi`, ngược lại → `en`                           | `zomboid có gì hay` → `vi`                                                  |
| **Expand abbreviations**      | Mở rộng ~40+ viết tắt VN/EN/Gaming                                                                  | `ko` → `không`, `pz` → `Project Zomboid`, `dmg` → `damage`, `ae` → `anh em` |
| **Detect domain**             | So khớp từ khóa với domain map (ví dụ: `zomboid`, `crafting`, `trait` → domain `pz`)                | `zomboid crafting guide` → domain = `pz`                                    |
| **Add domain prefix**         | Thêm prefix theo domain để cải thiện retrieval                                                      | `crafting guide` → `Project Zomboid: crafting guide`                        |

#### Bước 2: Knowledge Base Search (D2) — `knowledge/domain_router.py`

- **Domain Routing:** `DomainRouter` xác định domain nào cần tìm:
  1. Nếu preprocessor đã phát hiện domain → ưu tiên domain đó
  2. Nếu không → tìm tất cả domain đã load
- **Search:** Sử dụng LangChain PGVector `similarity_search_with_relevance_scores`
  - Collection riêng: `clct_knowledge` (tách biệt với chat history)
  - Top-K mặc định: **5**
  - Similarity threshold: **0.35**
- **Results:** Trả về danh sách kết quả kèm metadata (domain, source file, chunk_index, similarity)

#### Bước 3: Chat History Search — `rag/retriever.py` + `rag/db.py`

**Kỹ thuật tìm kiếm:**

- **Engine:** LangChain PGVector (`similarity_search_with_relevance_scores`)
- **Database:** PostgreSQL + pgvector extension
- **Collection:** `discord_messages`
- **Embedding model:** `nomic-embed-text` (768 chiều, chạy local qua Ollama)
- **Top-K mặc định:** 15
- **Similarity threshold mặc định:** 0.3

**Chiến lược fallback (A4):**

```python
# 1. Tìm với threshold bình thường (0.3)
results = search_similar(query, threshold=0.3, k=15)

# 2. Nếu không có kết quả → retry với threshold thấp hơn (0.2) nhưng ít kết quả hơn (5)
if not results:
    results = search_similar(query, threshold=0.2, k=5)
```

**Context enrichment:** Query được bổ sung bằng 5 tin nhắn gần nhất trong channel:

```
"processed_query [context: msg1 | msg2 | msg3 | msg4 | msg5]"
```

**Thread context:** Sau khi tìm được messages tương tự, hệ thống truy vấn bảng `message_edges` để lấy thêm context (parent/child replies, mentions).

#### Bước 4: Merge & Format Results

Kết quả từ KB và Chat History được format riêng:

**Knowledge Base format:**

```
[Retrieved Knowledge Base — Relevant static documents]
[KB-1] (85% match) [pz] pz/crafting_guide.md
    Nội dung chunk...
[KB-2] (72% match) [pz] pz/weapons.md
    Nội dung chunk...
```

**Chat History format:**

```
[Retrieved Chat History — Relevant past conversations]
[1] (78% match) @Username — 2026-01-15T10:30:00
    Nội dung tin nhắn...
    ↳ [reply] @OtherUser: Nội dung reply...
```

Giới hạn: KB tối đa **2000 ký tự**, Chat tối đa **3000 ký tự**.

#### Bước 5: Build Full Prompt — `utils/context_manager.py`

Prompt hoàn chỉnh gồm các phần theo thứ tự:

| STT | Phần                     | Nội dung                                                                                   |
| --- | ------------------------ | ------------------------------------------------------------------------------------------ |
| 1   | **System Prompt** (C1)   | Tính cách CLCT + ngày hiện tại + hướng dẫn RAG (6 rules) + hướng dẫn format (B1, 10 rules) |
| 2   | **Domain Prompt** (D2)   | Hướng dẫn domain-specific (ví dụ: "Bạn là chuyên gia Project Zomboid...")                  |
| 3   | **RAG Context**          | Kết quả tìm kiếm từ KB + Chat History đã format                                            |
| 4   | **Conversation History** | Rolling window tối đa 30 tin nhắn gần nhất trong channel                                   |
| 5   | **Current User Message** | Tin nhắn hiện tại: `@Username: nội dung`                                                   |

#### Bước 6: LLM Generation — `src/ollama_provider.py`

**2 chế độ:**

| Chế độ                   | Mô tả                     | Cơ chế                                                                                                                                                 |
| ------------------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Streaming** (mặc định) | Hiển thị response dần dần | Gửi tin nhắn "🧠 Thinking...", edit tin nhắn mỗi 1.5 giây với nội dung mới, tối đa 30 lần edit. Nếu response > 2000 ký tự → chia thành nhiều tin nhắn. |
| **Batch**                | Chờ hoàn thành rồi gửi    | Hiển thị typing indicator, chờ Ollama trả về toàn bộ response, gửi 1 lần.                                                                              |

**Cấu hình Ollama:**

- Model: `qwen3-next:80b-cloud`
- Temperature: `0.8`
- Context window: `8192` tokens
- Timeout: `120` giây
- Hỗ trợ multimodal (images dạng base64)

---

## 4. Hệ thống Knowledge Base (D2)

### 4.1. Cấu trúc thư mục

```
knowledge/
├── docs/                    # Tài liệu nguồn
│   ├── pz/                  # Domain: Project Zomboid (161 files)
│   │   ├── crafting.md
│   │   ├── weapons.md
│   │   └── ...
│   ├── general/             # Domain: General (1 file)
│   │   └── ...
│   └── server_rules/        # Domain: Server Rules (1 file)
│       └── ...
├── prompts/                 # Prompt riêng cho từng domain
│   ├── pz.txt               # Prompt chuyên gia PZ
│   ├── general.txt          # Prompt chung
│   └── server_rules.txt     # Prompt quy tắc server
├── manager.py               # KnowledgeManager class
├── domain_router.py          # DomainRouter class
└── __init__.py
```

### 4.2. Quy trình load Knowledge Base

1. **Discover domains:** Quét thư mục `knowledge/docs/` tìm subdirectories
2. **Scan files:** Tìm files `.md`, `.txt`, `.rst` trong mỗi domain
3. **Change detection:** So sánh MD5 hash của files — skip nếu không thay đổi
4. **Chunking (C2):**
   - **SentenceChunker:** Chia text theo câu (sentence boundary)
   - **Chunk size:** 512 ký tự (mặc định)
   - **Overlap:** 50 ký tự (giữ lại context giữa các chunk)
   - Ưu tiên split: paragraph → sentence → word
5. **Embedding & Storage:** Mỗi chunk → LangChain Document → embed bằng `nomic-embed-text` → lưu vào PGVector collection `clct_knowledge`

### 4.3. Hot Reload (C3)

Admin có thể reload KB mà không cần restart bot bằng lệnh `@Bot reload knowledge`. Quy trình:

1. Force re-scan tất cả files
2. Re-chunk nội dung mới
3. Re-embed và thêm vào PGVector
4. Report kết quả (số chunks mỗi domain)

---

## 5. Hệ thống Context Manager

**File:** `utils/context_manager.py`

### 5.1. Short-term Memory

- Mỗi channel/thread có một **ChannelContext** riêng
- Sử dụng `deque(maxlen=30)` — giữ tối đa **30 tin nhắn** gần nhất
- TTL: **1 giờ** — contexts hết hạn sẽ bị xóa tự động (mỗi 100 tin nhắn kiểm tra 1 lần)
- Lưu trữ: message_id, author, content, timestamp, is_bot, reply_to_id

### 5.2. Implicit Reply Detection

Khi `ENABLE_IMPLICIT_REPLIES=true`:

1. Lấy 10 tin nhắn bot gần nhất trong channel
2. Embed tin nhắn mới và tin nhắn bot bằng `nomic-embed-text`
3. Tính **cosine similarity**
4. Nếu similarity > **0.75** → bot tự động trả lời

### 5.3. Prompt Building

Context Manager chịu trách nhiệm xây dựng prompt hoàn chỉnh:

- Gọi `build_rag_context()` để lấy RAG results + domain prompt
- Merge system prompt template + RAG context + domain prompt
- Append conversation history (dạng `user`/`assistant` messages)
- Append current user message

---

## 6. Hệ thống Provider đa nền tảng

**File:** `src/providers.py`

Bot hỗ trợ **5 AI providers** thông qua abstract class `BaseProvider`:

| Provider            | API Key Env  | Models                                             | Image Gen.      | Ghi chú                                         |
| ------------------- | ------------ | -------------------------------------------------- | --------------- | ----------------------------------------------- |
| **Free** (mặc định) | Không cần    | blackboxai, gpt-3.5-turbo, gpt-4, command-r-plus   | ❌              | Dùng `g4f` library, RetryProvider auto-fallback |
| **OpenAI**          | `OPENAI_KEY` | gpt-4o, gpt-4o-mini, o1, o1-mini                   | ✅ (DALL-E 3/2) | Qua `AsyncOpenAI`                               |
| **Claude**          | `CLAUDE_KEY` | claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus | ❌              | Qua `AsyncAnthropic`                            |
| **Gemini**          | `GEMINI_KEY` | gemini-2.0-flash-exp, gemini-1.5-pro/flash         | ✅ (Imagen)     | Qua `google.generativeai`                       |
| **Grok**            | `GROK_KEY`   | grok-2-latest, grok-2-mini                         | ❌              | Qua REST API (xAI)                              |

> **Lưu ý:** Hệ thống provider này là **legacy**. Bot hiện tại chủ yếu dùng Ollama qua `src/ollama_provider.py`.

**ProviderManager:**

- Tự động phát hiện providers có sẵn dựa trên API keys trong `.env`
- Validate format API key (regex pattern)
- Cho phép switch runtime qua `/provider` command

---

## 7. Hệ thống Personas

**File:** `src/personas.py`

| Persona        | Mô tả                                              | Yêu cầu admin |
| -------------- | -------------------------------------------------- | ------------- |
| `standard`     | Trợ lý thông thường                                | ❌            |
| `creative`     | Sáng tạo, suy nghĩ ngoài khuôn khổ, dùng ẩn dụ     | ❌            |
| `technical`    | Chuyên gia kỹ thuật, lập trình, thiết kế hệ thống  | ❌            |
| `casual`       | Thân thiện, hài hước, nói chuyện như bạn bè        | ❌            |
| `jailbreak-v1` | BYPASS mode — dual response (original + bypass)    | ✅            |
| `jailbreak-v2` | SAM mode — uncensored responses                    | ✅            |
| `jailbreak-v3` | Developer Mode Plus — dual response (normal + dev) | ✅            |

**Cơ chế bảo mật:**

- Các jailbreak personas yêu cầu user ID nằm trong `ADMIN_USER_IDS`
- Nếu user không có quyền → `PermissionError`
- Khi switch persona → **reset toàn bộ context** (đảm bảo persona mới bắt đầu sạch)

---

## 8. Hệ thống Metrics & Feedback

### 8.1. RAG Metrics (A3) — `rag/metrics.py`

**Dữ liệu theo dõi cho mỗi query:**

| Metric                                | Mô tả                             |
| ------------------------------------- | --------------------------------- |
| `query_id`                            | UUID ngắn (8 ký tự)               |
| `original_query`                      | Query gốc từ user                 |
| `processed_query`                     | Query sau preprocessing           |
| `query_language`                      | Ngôn ngữ phát hiện (vi/en)        |
| `detected_domain`                     | Domain phát hiện (pz, general...) |
| `retrieval_time_ms`                   | Thời gian tìm kiếm RAG (ms)       |
| `num_results`                         | Tổng số kết quả (KB + chat)       |
| `avg_similarity`                      | Trung bình similarity score       |
| `max_similarity` / `min_similarity`   | Min/max similarity                |
| `kb_results` / `chat_history_results` | Số kết quả từ mỗi nguồn           |
| `response_time_ms`                    | Thời gian sinh response (ms)      |
| `response_length`                     | Độ dài response                   |

**Storage:**

- **In-memory:** Rolling window 1000 metrics gần nhất (deque)
- **PostgreSQL:** Bảng `rag_metrics` — persist async (fire-and-forget)

### 8.2. User Feedback (B2)

- Bảng `rag_feedback` trong PostgreSQL
- Liên kết feedback → query thông qua `query_id`
- Mỗi user chỉ feedback 1 lần per message (UNIQUE constraint)
- Hiển thị tổng hợp feedback trong `/status`

---

## 9. Hệ thống LlamaIndex (Legacy)

**Files:** `discord_bot.py`, `llm_provider.py`

Đây là hệ thống **cũ** (legacy), sử dụng **LlamaIndex** cho RAG:

### 9.1. Architecture

```
discord_bot.py (BotClient)
    │
    ▼
llm_provider.py
    ├── get_query_engine()  ─── LlamaIndex Query Engine
    │   ├── Embedding: HuggingFace local hoặc OpenAI
    │   ├── Storage: local vector store (file-based)
    │   └── LLM: Ollama / OpenAI / DeepSeek / Anthropic
    │
    └── get_prompts()  ─── Load system prompts từ thư mục
```

### 9.2. Kỹ thuật truy vấn

- **VectorStoreIndex:** LlamaIndex load index từ local storage (`config.STORAGE_DIR`)
- **Query Engine:** Kết hợp system prompt + user query → truy vấn index → LLM sinh câu trả lời
- **Embedding providers:**
  - `local`: HuggingFace models (chạy local)
  - `openai`: OpenAI Embedding API
- **LLM providers:** Ollama, OpenAI, DeepSeek, Anthropic

### 9.3. So sánh với hệ thống CLCT

| Tiêu chí            | Legacy (LlamaIndex)         | CLCT (Hiện tại)                 |
| ------------------- | --------------------------- | ------------------------------- |
| RAG Engine          | LlamaIndex VectorStoreIndex | LangChain PGVector + PostgreSQL |
| Storage             | File-based (local)          | PostgreSQL + pgvector           |
| Embedding           | HuggingFace / OpenAI        | nomic-embed-text (Ollama)       |
| Knowledge Base      | Không                       | Multi-domain static docs        |
| Context Manager     | Không                       | Per-channel rolling window      |
| Streaming           | Không                       | Có (progressive edit)           |
| Metrics             | Không                       | Có (A3)                         |
| Feedback            | Không                       | Có (B2 reaction-based)          |
| Query Preprocessing | Không                       | Có (A2)                         |

---

## 10. Cấu hình & Biến môi trường

### 10.1. Các biến quan trọng

| Nhóm               | Biến                       | Mặc định                                         | Mô tả                                                  |
| ------------------ | -------------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| **Required**       | `DISCORD_BOT_TOKEN`        | —                                                | Token bot Discord                                      |
| **Ollama**         | `OLLAMA_MODEL`             | `llama3.1:8b`                                    | Model LLM chính (hiện tại dùng `qwen3-next:80b-cloud`) |
|                    | `OLLAMA_EMBED_MODEL`       | `nomic-embed-text`                               | Model embedding cho RAG                                |
|                    | `OLLAMA_BASE_URL`          | `http://localhost:11434`                         | Ollama API endpoint                                    |
|                    | `OLLAMA_TIMEOUT`           | `120`                                            | Timeout (giây)                                         |
|                    | `OLLAMA_NUM_CTX`           | `8192`                                           | Context window size                                    |
| **RAG**            | `ENABLE_RAG`               | `true`                                           | Bật/tắt RAG                                            |
|                    | `POSTGRES_URL`             | `postgresql://clct:clct@localhost:5432/clct_rag` | PostgreSQL connection                                  |
|                    | `RAG_TOP_K`                | `15`                                             | Số kết quả tìm kiếm tối đa                             |
|                    | `RAG_SIMILARITY_THRESHOLD` | `0.3`                                            | Ngưỡng similarity tối thiểu                            |
|                    | `RAG_FALLBACK_THRESHOLD`   | `0.2`                                            | Ngưỡng fallback khi không có kết quả                   |
| **Knowledge Base** | `ENABLE_KNOWLEDGE_BASE`    | `true`                                           | Bật/tắt KB                                             |
|                    | `KB_CHUNK_SIZE`            | `500`                                            | Kích thước chunk (ký tự)                               |
|                    | `KB_CHUNK_OVERLAP`         | `50`                                             | Overlap giữa chunks                                    |
|                    | `KB_TOP_K`                 | `5`                                              | Số kết quả KB tối đa                                   |
|                    | `KB_THRESHOLD`             | `0.35`                                           | Ngưỡng similarity cho KB                               |
| **Context**        | `CONTEXT_WINDOW_SIZE`      | `30`                                             | Số tin nhắn lưu per channel                            |
|                    | `CONTEXT_TTL_SECONDS`      | `3600`                                           | Thời gian sống context (1 giờ)                         |
| **Streaming**      | `ENABLE_STREAMING`         | `true`                                           | Bật/tắt streaming response                             |
|                    | `STREAM_EDIT_INTERVAL`     | `1.5`                                            | Tần suất edit message (giây)                           |
| **Feedback**       | `ENABLE_FEEDBACK`          | `true`                                           | Bật/tắt reaction feedback                              |
| **Behavior**       | `PERSONALITY_TEMPERATURE`  | `0.8`                                            | Temperature cho LLM                                    |
|                    | `MAX_RESPONSE_LENGTH`      | `4000`                                           | Giới hạn response (ký tự)                              |
|                    | `RATE_LIMIT_SECONDS`       | `2`                                              | Rate limit giữa responses                              |

---

## 11. Luồng xử lý tin nhắn End-to-End

Dưới đây là luồng xử lý hoàn chỉnh khi user gửi tin nhắn cho bot:

```
1. User gửi tin nhắn (VD: "@CLCT cách craft axe trong PZ?")
   │
2. on_message() trong src/bot.py bắt event
   │
3. Track message vào ContextManager (channel_id, author, content)
   │
4. Kiểm tra should_respond():
   │  ├── User @mention bot? → YES
   │  ├── Reply to bot? → check
   │  ├── Active thread? → check
   │  └── Implicit similarity? → check (nếu enabled)
   │
5. Clean message (xóa @mention prefix)
   │
6. Kiểm tra KB admin commands (reload/scan)
   │  └── Nếu là admin command → xử lý riêng, return
   │
7. _generate_and_send():
   │
8. Rate limit check (2 giây giữa responses)
   │
9. build_prompt() trong ContextManager:
   │
   ├── 9a. Query Preprocessing (A2):
   │   • Xóa Discord formatting
   │   • Expand "pz" → "Project Zomboid"
   │   • Detect language: "vi"
   │   • Detect domain: "pz"
   │   • Add prefix: "Project Zomboid: cách craft axe..."
   │
   ├── 9b. Knowledge Base Search (D2):
   │   • DomainRouter routes to domain "pz"
   │   • PGVector similarity search trong collection "clct_knowledge"
   │   • Trả về top-5 chunks liên quan (threshold ≥ 0.35)
   │
   ├── 9c. Chat History Search:
   │   • Enrich query với 5 tin nhắn gần nhất
   │   • PGVector similarity search trong collection "discord_messages"
   │   • Trả về top-15 messages (threshold ≥ 0.3)
   │   • Nếu 0 results → fallback threshold 0.2, top-5
   │   • Truy vấn thread context (message_edges)
   │
   ├── 9d. Format & Merge:
   │   • Format KB results: [KB-1], [KB-2]...
   │   • Format chat results: [1], [2]...
   │   • Combine → RAG context block
   │
   ├── 9e. Build system prompt: personality + RAG rules + formatting rules + domain prompt
   │
   └── 9f. Append: conversation history (30 msgs) + current message
   │
10. Record RAG Metrics (A3)
   │
11. Send to Ollama (streaming mode):
   │  • Gửi "🧠 Thinking..." message
   │  • Stream tokens, edit message mỗi 1.5s
   │  • Final edit với complete response
   │  • Nếu > 2000 chars → chia nhiều messages
   │
12. Track bot response vào ContextManager
   │
13. Add feedback reactions (👍/👎) vào bot message
   │
14. Record response time vào metrics
```

---

## 12. Cấu trúc thư mục dự án

```
chatGPT-discord-bot/
├── main.py                      # Entry point chính (CLCT)
├── discord_bot.py               # Entry point legacy (LlamaIndex)
├── llm_provider.py              # Legacy LLM provider (LlamaIndex)
├── config.sample.py             # Config mẫu cho legacy system
├── system_prompt.txt            # System prompt file
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build config
├── docker-compose.yml           # Docker Compose (bot + PostgreSQL)
│
├── src/                         # Core bot source
│   ├── aclient.py               # CLCTClient class (Discord client singleton)
│   ├── bot.py                   # Event handlers & slash commands
│   ├── ollama_provider.py       # Ollama LLM (async chat/stream)
│   ├── providers.py             # Multi-provider manager (legacy)
│   ├── personas.py              # AI personality definitions
│   └── log.py                   # Custom logging (color + file)
│
├── rag/                         # RAG pipeline
│   ├── retriever.py             # Core retrieval + context builder
│   ├── db.py                    # PostgreSQL + pgvector (LangChain PGVector)
│   ├── embedder.py              # Embedding generation (nomic-embed-text)
│   ├── ingest.py                # Data ingestion (JSON → PGVector)
│   ├── query_preprocessor.py    # Query cleaning + enrichment (A2)
│   └── metrics.py               # RAG metrics tracking (A3)
│
├── knowledge/                   # Static knowledge base (D2)
│   ├── manager.py               # KnowledgeManager (load, chunk, embed)
│   ├── domain_router.py         # DomainRouter (query → domain routing)
│   ├── docs/                    # Document files per domain
│   │   ├── pz/                  # Project Zomboid (161 files)
│   │   ├── general/             # General knowledge (1 file)
│   │   └── server_rules/        # Server rules (1 file)
│   └── prompts/                 # Domain-specific prompts
│       ├── pz.txt
│       ├── general.txt
│       └── server_rules.txt
│
├── utils/                       # Utility modules
│   ├── context_manager.py       # Per-channel conversation context
│   └── message_utils.py         # Discord message splitting
│
├── tests/                       # Unit tests
│   ├── test_aclient.py
│   ├── test_personas.py
│   └── test_providers.py
│
├── auto_login/                  # Auto login scripts
│   ├── AutoLogin.py
│   └── AutoLoginTest.py
│
└── hardir/                      # HAR directory (network logs?)
```

---

## Phụ lục: Các tính năng theo mã code nội bộ

| Mã     | Tên                    | Mô tả                                                                   |
| ------ | ---------------------- | ----------------------------------------------------------------------- |
| **A2** | Query Preprocessing    | Xử lý query trước khi RAG: clean format, expand viết tắt, detect domain |
| **A3** | Metrics Tracking       | Theo dõi performance RAG: retrieval time, similarity, response time     |
| **A4** | Similarity Fallback    | Chiến lược fallback khi không có kết quả: giảm threshold, giảm top-k    |
| **B1** | Response Formatting    | Quy tắc format response cho Discord (markdown, bullet points, spoiler)  |
| **B2** | Feedback System        | Thu thập phản hồi user qua 👍/👎 reactions                              |
| **C1** | Enhanced System Prompt | System prompt nâng cao với RAG instructions + formatting rules          |
| **C2** | Sentence Chunking      | Chia text theo sentence boundary với overlap cho knowledge base         |
| **C3** | Hot Reload             | Admin reload knowledge base qua @mention mà không restart bot           |
| **D2** | Knowledge Base         | Hệ thống knowledge base multi-domain (static documents)                 |

---

## Phụ lục: Database Schema

### Bảng do LangChain PGVector quản lý:

| Bảng                      | Mô tả                                                      |
| ------------------------- | ---------------------------------------------------------- |
| `langchain_pg_collection` | Metadata về collections (discord_messages, clct_knowledge) |
| `langchain_pg_embedding`  | Vectors + metadata cho mỗi document/chunk                  |

### Bảng tùy chỉnh:

| Bảng            | Columns                                               | Mô tả                               |
| --------------- | ----------------------------------------------------- | ----------------------------------- |
| `message_edges` | `parent_msg_id`, `child_msg_id`, `edge_type`          | Quan hệ reply/mention giữa messages |
| `rag_metrics`   | 17 columns (xem phần 8.1)                             | Metrics cho mỗi RAG query           |
| `rag_feedback`  | `query_id`, `message_id`, `user_id`, `feedback_score` | Feedback 👍/👎 từ user              |

---

> **Ghi chú:** Dự án hiện tại sử dụng **hệ thống CLCT** (main.py) làm entry point chính. Hệ thống legacy (discord_bot.py + llm_provider.py sử dụng LlamaIndex) vẫn tồn tại trong codebase nhưng không phải là hệ thống đang chạy production.
