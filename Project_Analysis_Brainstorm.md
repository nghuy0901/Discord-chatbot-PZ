# Brainstorm & Phân tích Dự án: CLCT Discord Bot (Project Zomboid RAG Bot)

Dựa trên cấu trúc thư mục và mã nguồn của bạn, đây là một bản phân tích chi tiết và tổng hợp về dự án bot Discord hiện tại.

## 1. Nhiệm vụ chính của project (Quy mô & Mục đích)

Mặc dù dự án bắt nguồn từ bộ khung của một `chatGPT-discord-bot` đa năng, nhưng mã nguồn hiện tại đã được tùy chỉnh sâu và chuyển mình thành **"CLCT Discord Bot"** — một trợ lý ảo thông minh chuyên dụng cho một cộng đồng/server game **Project Zomboid (PZ)**.

**Nhiệm vụ cốt lõi:**
Đóng vai trò là một bách khoa toàn thư tương tác và người hướng dẫn cho người chơi trong server Discord. Bot giải đáp các thắc mắc về cơ chế game phức tạp của Project Zomboid (chế tạo, sinh tồn, kỹ năng, vật phẩm) và các quy định riêng (server rules) của server, sử dụng công nghệ kiểm chứng dữ liệu **RAG (Retrieval-Augmented Generation)** để đảm bảo câu trả lời chính xác, không bị "ảo giác" (hallucination).

---

## 2. Các chức năng hiện có của hệ thống

Dự án hiện đang sở hữu một hệ sinh thái tính năng rất đồ sộ, chia làm 3 mảng chính:

### A. Tích hợp AI & Xử lý Ngôn ngữ
*   **Hỗ trợ đa mô hình (Multi-Provider):** Vẫn giữ lại di sản hỗ trợ cổng API cho OpenAI, Claude, Gemini, Grok... 
*   **Tích hợp Local LLM (Ollama):** Hoạt động mạnh mẽ với Ollama (mặc định cấu hình dùng model `llama3.1:8b`). Tính năng này giúp server tiết kiệm chi phí API và xử lý thông tin nội bộ một cách bảo mật.
*   **Hệ thống Persona (Đóng vai):** Hỗ trợ nhiều tính cách khác nhau (standard, creative, technical, casual) và cả các chế độ "jailbreak" dành cho admin.

### B. Hệ thống RAG (Retrieval-Augmented Generation) & Knowledge Base
*   **Cơ sở dữ liệu tri thức đồ sộ (Knowledge Base):** Chứa hàng loạt tài liệu MarkDown (`knowledge/docs/pz/`) phân loại rõ ràng mọi khía cạnh của Project Zomboid (Appliances, Clothing, Crafting, Food, Weapons...). Ngoài ra còn có một mục riêng cho `server_rules`.
*   **Domain Router (Điều hướng tên miền):** Có khả năng tự động phân loại rẽ nhánh câu hỏi của người dùng xem họ đang hỏi về game (PZ), luật server (Server Rules) hay chỉ là trò chuyện thông thường (General).
*   **Vector Database & Retrieval:** Sử dụng PostgreSQL dưới dạng Vector DB để lưu trữ các bản nhúng (embeddings) tài liệu và truy xuất ngữ cảnh (retriever) cực kỳ nhanh chóng.

### C. Tương tác trên nền tảng Discord
*   **Trigger thông minh:** Bot phản hồi qua nhiều cách: khi được `@mention`, reply tin nhắn của bot, gọi lệnh `/chat`, hoặc chat trong các Thread / Kênh định sẵn.
*   **Automated Implicit Reply (Phản hồi tự động):** Có cơ chế nhận diện tự động trả lời khi nhận thấy độ tương đồng câu hỏi.
*   **Feedback Tracking:** Ghi nhận phản hồi của người dùng qua hệ thống thả cảm xúc (Reaction 👍/👎).

---

## 3. Cách thức các chức năng hoạt động (Luồng xử lý)

Để dễ hình dung, khi một người dùng nhắn tin hỏi bot: *"Làm sao để chế tạo áo giáp trong game?"*, hệ thống sẽ hoạt động theo luồng sau:

1.  **Tiếp nhận (Discord Bot - `src/bot.py`, `src/aclient.py`):** Bot bắt được sự kiện người dùng `@mention` hoặc dùng lệnh `/chat`. Chuyển tin nhắn thô vào hệ thống xử lý (utils & preprocessor).
2.  **Phân loại (Domain Router - `knowledge/domain_router.py`):** Phân tích câu hỏi và nhận diện từ khóa. Máy học/Logic router quyết định đây là câu hỏi thuộc mục **Project Zomboid (PZ)** chứ không phải nói chuyện phiếm hay hỏi luật server.
3.  **Tiền xử lý & Tìm kiếm véc-tơ (RAG Pipeline - `rag/`):**
    *   `rag/query_preprocessor.py` làm sạch và chuẩn hóa lại câu hỏi.
    *   `rag/embedder.py` chuyển đổi câu hỏi thành dạng vector số học.
    *   `rag/retriever.py` lấy vector đó so khớp trong PostgreSQL (`rag/db.py`) để tìm ra các đoạn tài liệu Markdown liên quan nhất (ví dụ: file `Armor(crafting).md` trong thư mục `Crafting`).
4.  **Tổng hợp Prompt & Sinh câu trả lời (LLM Provider):** 
    *   Thông tin lấy được từ database cùng câu hỏi của user được đưa vào một khuôn mẫu (prompt pattern trong `knowledge/prompts/pz.txt`).
    *   Dữ liệu được đẩy sang **Ollama** (hoặc OpenAI/Gemini tùy cấu hình) thông qua `src/ollama_provider.py`. LLM sẽ đọc ngữ cảnh game và soạn ra câu trả lời tự nhiên.
5.  **Phản hồi & Học tập:** Bot gửi câu trả lời về kênh chat Discord. Người dùng có thể thả 👍 hoặc 👎. Hệ thống sẽ ghi nhận tương tác này để đánh giá chất lượng phản hồi sau này.

---

## 4. Hướng phát triển & Tính năng tiềm năng (Brainstorm thêm)

Vì chức năng cốt lõi là một "Zomboid RAG Bot", dưới đây là những khía cạnh có thể mở rộng tiếp:
*   **Hỗ trợ đa ngôn ngữ tự động:** Dịch qua lại cho các server quốc tế.
*   **Tích hợp Webhook Báo cáo:** Nếu hệ thống RAG không tìm thấy câu trả lời (hallucination fallbacks), bot có thể tự động ping Admin hoặc mở ticket để admin giải đáp và tự động *Ingest* (nạp) file kiến thức mới.
*   **Lệnh trích xuất Wiki (Wiki-style Commands):** Bổ sung các lệnh tra cứu tĩnh không cần AI để phản hồi ngay lập tức, ví dụ `/item [tên]`, `/craft [tên]`. 
*   **Image Generation tích hợp map:** Cho phép người dùng gọi `/map` để lấy bản đồ các khu vực (nếu tích hợp được link map PZ hiện tại thay vì sinh ảnh DALL-E không sát với in-game).