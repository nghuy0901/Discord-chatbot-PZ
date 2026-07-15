"""Cross-lingual augmentation & bilingual domain routing (audit H3/M4/L5)."""

import pytest

from rag.query_preprocessor import QueryPreprocessor
from rag.query_preprocessor import QueryIntent
from rag.bm25_search import tokenize


CONVERSATION_QUERIES = [
    # Châm chọc / kháy đểu NomNom
    "nomnom gà quá",
    "gà quá sao không biết",
    "sao bot này cùi vậy",
    "nomnom trả lời chán thật",
    "bot bị lag não à",
    "hỏi dễ vậy mà cũng bí",
    "nomnom học lại đi",
    "bot nói nghe vô tri quá",
    "mày đang đoán mò đúng không",
    "trả lời như không trả lời ấy",
    "nomnom bị lú rồi hả",
    "bot tự tin nhưng sai bét",
    "Bot gì mà trả lời chậm như rùa, có biết gì không?",
    "Mày là bot ngu nhất tao từng gặp, mày còn trả lời được không?",
    "nomnom thông minh quá ha, chắc biết hết mọi thứ nhỉ",
    "ừ bot trả lời hay quá cơ, đúng là thiên tài",
    "bot giỏi quá ha, không biết có thật không nữa",
    # Banter / kháy đểu nhẹ
    "nói chuyện bớt nghiêm túc đi",
    "đừng làm mặt lạnh nữa nomnom",
    "cà khịa lại tôi xem nào",
    "roast nhẹ tôi một câu",
    "khịa tôi nhưng đừng ác quá",
    "nói gì cho đỡ nhạt đi",
    "pha trò đi nomnom",
    "bịa một câu xàm vui vui đi",
    # Tư vấn đời sống / cảm xúc
    "tư vấn cho tôi cách bớt trì hoãn khi làm việc với deadline dài",
    "hôm nay tôi hơi mệt, nói gì đó giúp tôi bình tĩnh lại",
    "tôi nên làm gì khi mất động lực học trong vài tuần liền",
    "làm sao để nói chuyện với bạn bè khi đang giận mà không làm quá lên",
    "cho tôi lời khuyên để ngủ sớm hơn mà không cầm điện thoại mãi",
    "tôi thấy áp lực vì nhiều việc cùng lúc, nên sắp xếp thế nào",
    "tư vấn giúp tôi cách xin lỗi cho tử tế sau khi lỡ nói nặng lời",
    "nếu hôm nay làm việc không hiệu quả thì nên xử lý tâm trạng ra sao",
    "tôi muốn tập thói quen đọc sách mỗi ngày thì bắt đầu thế nào",
    "nên phản hồi thế nào khi người khác góp ý hơi khó nghe",
    # Hỏi đáp cuộc sống thường ngày
    "tối nay ăn gì cho nhanh mà không quá dầu mỡ",
    "nên uống cà phê lúc nào để tối còn ngủ được",
    "làm sao để dọn bàn làm việc cho đỡ rối",
    "cách ghi chú cuộc họp sao cho dễ nhớ lại",
    "nên chia thời gian học và nghỉ như thế nào cho hợp lý",
    "có mẹo nào để đỡ quên việc nhỏ trong ngày không",
    "giải thích đơn giản vì sao ngủ đủ lại quan trọng",
    "cho tôi một checklist ngắn trước khi đi ngủ",
    "nói ngắn gọn cách giữ tập trung trong 25 phút",
    "làm sao để bắt đầu một cuộc trò chuyện bớt gượng",
    # Đố mẹo / đố vui
    "đố mẹo tôi một câu đi",
    "ra câu đố vui khó vừa thôi",
    "cái gì càng lấy đi càng lớn",
    "đố vui: có cổ mà không có đầu là gì",
    "cho tôi câu đố logic ngắn",
    "hỏi tôi một câu hack não nhẹ",
    "đố tôi câu nào vui vui để giải trí",
    "có câu đố mẹo nào khiến người ta dễ trả lời sai không",
    # Reply / follow-up to prior bot answer
    "ý tôi không phải vậy",
    "nói lại ngắn hơn đi",
    "giải thích dễ hiểu hơn",
    "ví dụ khác đi",
    "đoạn trên sai chỗ nào",
    "tóm tắt câu vừa rồi thành một câu",
    "trả lời kiểu bớt máy móc hơn được không",
    "sao lúc nãy bạn nói ngược lại",
    "vậy cuối cùng nên chọn cái nào",
    "đọc lại câu trước rồi trả lời cho đúng trọng tâm",
]


def test_vietnamese_query_augmented_with_english_terms():
    p = QueryPreprocessor()
    out, meta = p.preprocess("rìu nào chặt cây nhanh nhất?")
    assert meta["language"] == "vi"
    assert "rìu" in out.lower()
    assert "axe" not in out.lower()
    lexical = meta["lexical_query"].lower()
    assert "axe" in lexical
    assert "tree" in lexical or "wood" in lexical


def test_vietnamese_spawn_query_adds_normal_starting_location_terms():
    p = QueryPreprocessor()

    _, meta = p.preprocess("Muldraugh có phải điểm spawn bình thường không?")

    lexical = meta["lexical_query"].lower()
    assert "starting location" in lexical
    assert "normal" in lexical


def test_vietnamese_pz_domain_detected():
    p = QueryPreprocessor()
    _, meta = p.preprocess("công thức làm băng gạc vô trùng")
    assert meta["domain"] == "pz"


def test_server_rules_domain_detected_for_vietnamese():
    p = QueryPreprocessor()
    _, meta = p.preprocess("nội quy server về xây nhà là gì")
    assert meta["domain"] in ("server_rules", "pz")


def test_english_query_not_augmented():
    p = QueryPreprocessor()
    out, meta = p.preprocess("what is the strongest axe")
    assert meta["language"] == "en"
    # No Vietnamese glossary expansion duplicated onto an English query
    assert out.lower().count("axe") == 1


def test_ascii_vietnamese_detected():
    p = QueryPreprocessor()
    # typed without diacritics — should still be detected as Vietnamese (L5)
    assert p._detect_language("minh khong biet lam sao") == "vi"


def test_vietnamese_weapon_damage_superlative_is_analytical():
    p = QueryPreprocessor()
    _, meta = p.preprocess("vũ khí nào có sát thương cao nhất nomnom ?")
    assert meta["query_intent"] == QueryIntent.ANALYTICAL


def test_vietnamese_joke_request_is_conversation():
    p = QueryPreprocessor()
    _, meta = p.preprocess("kể chuyện cười đi nomnom")
    assert meta["query_intent"] == QueryIntent.CONVERSATION


@pytest.mark.parametrize("query", CONVERSATION_QUERIES)
def test_general_conversation_queries_are_not_routed_to_rag(query):
    p = QueryPreprocessor()
    _, meta = p.preprocess(query)
    assert meta["query_intent"] == QueryIntent.CONVERSATION


def test_vietnamese_gun_damage_query_detects_pz_domain():
    p = QueryPreprocessor()
    out, meta = p.preprocess("súng nào có sát thương cao nhất nomnom ?")
    assert meta["domain"] == "pz"
    assert meta["query_intent"] == QueryIntent.ANALYTICAL
    assert "gun" in meta["lexical_query"].lower()
    assert "damage" in meta["lexical_query"].lower()


def test_pz_explanation_with_simple_wording_still_uses_rag():
    p = QueryPreprocessor()
    _, meta = p.preprocess("giải thích đơn giản máy phát điện hoạt động như thế nào trong pz")
    assert meta["domain"] == "pz"
    assert meta["query_intent"] != QueryIntent.CONVERSATION


def test_insult_with_game_question_still_uses_rag():
    p = QueryPreprocessor()
    _, meta = p.preprocess("bot ngu vậy chứ rìu gây bao nhiêu sát thương?")

    assert meta["query_intent"] != QueryIntent.CONVERSATION


def test_pz_domain_is_metadata_not_retrieval_prefix():
    p = QueryPreprocessor()
    out, meta = p.preprocess("rìu gây bao nhiêu sát thương?")

    assert meta["domain"] == "pz"
    assert not out.lower().startswith("project zomboid:")
    assert not meta["lexical_query"].lower().startswith("project zomboid:")


def test_vietnamese_ra_sao_is_not_rewritten_as_tai_sao():
    p = QueryPreprocessor()
    out, _ = p.preprocess("rìu dùng ra sao?")

    assert "ra sao" in out.lower()
    assert "ra tại sao" not in out.lower()


def test_longest_glossary_phrase_does_not_add_shorter_overlap():
    p = QueryPreprocessor()
    _, meta = p.preprocess("đèn pin dùng thế nào?")
    lexical = meta["lexical_query"].lower()

    assert "flashlight" in lexical
    assert "battery" not in lexical


def test_bm25_tokenizer_keeps_accented_and_ascii_vietnamese_forms():
    tokens = tokenize("nhiễm trùng và đèn pin")

    assert "nhiễm" in tokens
    assert "nhiem" in tokens
    assert "đèn" in tokens
    assert "den" in tokens
