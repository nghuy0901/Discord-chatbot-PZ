"""
Query Preprocessor (A2) — cleans and enriches user queries
before they are sent to the RAG retrieval pipeline.

Features:
- Remove Discord formatting (@mentions, emojis, URLs)
- Expand common abbreviations (Vietnamese + English + gaming)
- Detect language (vi/en)
- Add domain context if a knowledge domain is detected
- 🆕 Phase 4: Query intent classification (analytical / narrative / hybrid)
"""

import re
import logging
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query Intent Classification (Phase 4)
# ---------------------------------------------------------------------------
class QueryIntent(str, Enum):
    """Classifies what kind of response pipeline a query needs."""
    ANALYTICAL = "analytical"    # → SQLite Tool Calling (data/comparison/ranking)
    NARRATIVE = "narrative"      # → Vector Search streaming (lore/prose/explanation)
    HYBRID = "hybrid"           # → Tool Calling + Vector Search (factual + context)
    CONVERSATION = "conversation"  # → No tools, no RAG needed (casual chat)


def requires_structured_tools(intent: object) -> bool:
    value = intent.value if isinstance(intent, QueryIntent) else str(intent or "")
    return value == QueryIntent.ANALYTICAL.value

# ---------------------------------------------------------------------------
# Abbreviation/slang expansion maps
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    # Vietnamese common
    "ko": "không",
    "k": "không",
    "dc": "được",
    "đc": "được",
    "ng": "người",
    "ntn": "như thế nào",
    "j": "gì",
    "gì v": "gì vậy",
    "đi": "đi",
    "r": "rồi",
    "oy": "rồi",
    "bt": "biết",
    "bik": "biết",
    "lm": "làm",
    "lm sao": "làm sao",
    "tl": "trả lời",
    "đang": "đang",
    "nma": "nhưng mà",
    "vs": "với",
    "mk": "mình",
    "ik": "đi",
    "cx": "cũng",
    "mn": "mọi người",
    "ae": "anh em",
    # Gaming common
    "gg": "good game",
    "wp": "well played",
    "afk": "away from keyboard",
    "npc": "non-player character",
    "hp": "health points",
    "xp": "experience points",
    "dmg": "damage",
    "dps": "damage per second",
    "op": "overpowered",
    "nerf": "nerf (giảm sức mạnh)",
    "buff": "buff (tăng sức mạnh)",
    "pz": "Project Zomboid",
    "mc": "Minecraft",
}

# Vietnamese → English game-term glossary (audit H3). Vietnamese users query an
# English knowledge base, so the BM25 (lexical) arm never matches and hybrid
# search collapses to vector-only. Appending the English equivalent gives the
# lexical arm something to hit and anchors the embedding cross-lingually. The
# original Vietnamese text is always preserved.
GAME_GLOSSARY_VI_EN = {
    "rìu": "axe", "búa": "hammer", "dao": "knife", "kiếm": "sword",
    "súng": "gun", "đạn": "ammo bullet", "vũ khí": "weapon",
    "sát thương": "damage",
    "giáp": "armor", "áo giáp": "armor", "quần áo": "clothing", "mũ": "hat helmet",
    "băng gạc": "bandage", "băng": "bandage", "thuốc giảm đau": "painkillers",
    "thuốc": "medicine pills", "vết thương": "wound injury",
    "sơ cứu": "first aid", "nhiễm trùng": "infection", "nhiễm": "infection",
    "thức ăn": "food", "đồ ăn": "food", "nước": "water",
    "xăng": "gasoline fuel petrol", "nhiên liệu": "fuel",
    "máy phát điện": "generator", "máy phát": "generator",
    "xe cộ": "vehicle car", "xe": "vehicle car",
    "gỗ": "wood plank", "cây": "tree wood", "đinh": "nails",
    "lửa": "fire", "nấu ăn": "cooking", "câu cá": "fishing",
    "trồng trọt": "farming", "nông nghiệp": "farming",
    "mộc": "carpentry", "rèn": "metalworking blacksmith",
    "may vá": "tailoring", "kỹ năng": "skill",
    "công thức": "recipe", "chế tạo": "crafting",
    "điểm spawn": "starting location spawn", "bình thường": "normal",
    "địa điểm": "location", "bản đồ": "map",
    "thây ma": "zombie", "đói": "hunger", "khát": "thirst",
    "mệt": "fatigue", "ngủ": "sleep",
    "trạm xăng": "gas station", "bệnh viện": "hospital",
    "đồn cảnh sát": "police station", "siêu thị": "supermarket grocery store",
    "kho": "storage warehouse", "ba lô": "backpack bag", "túi": "bag",
    "cửa": "door", "tường": "wall", "khóa": "lock",
    "pin": "battery", "đèn pin": "flashlight",
    "cắn": "bite", "kết nối": "connect", "điều kiện": "requirements",
}

# Discord formatting patterns
DISCORD_MENTION_PATTERN = re.compile(r"<@!?\d+>")
DISCORD_CHANNEL_PATTERN = re.compile(r"<#\d+>")
DISCORD_ROLE_PATTERN = re.compile(r"<@&\d+>")
DISCORD_EMOJI_PATTERN = re.compile(r"<a?:\w+:\d+>")
URL_PATTERN = re.compile(r"https?://\S+")


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------
class QueryPreprocessor:
    """
    Preprocesses user queries before RAG retrieval.
    Cleans Discord formatting, expands abbreviations,
    and enriches queries with domain context.
    """

    def __init__(self):
        self._abbreviations = ABBREVIATIONS.copy()

    def preprocess(
        self,
        raw_query: str,
        channel_name: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Full preprocessing pipeline.

        Args:
            raw_query: Raw user message text.
            channel_name: Discord channel name for domain hinting.

        Returns:
            Tuple of (cleaned_query, metadata_dict)
            metadata_dict contains: language, domain, original_length, etc.
        """
        metadata = {
            "original_length": len(raw_query),
            "language": "unknown",
            "domain": None,
            "was_cleaned": False,
        }

        # Step 1: Remove Discord formatting
        cleaned = self._remove_discord_formatting(raw_query)
        if cleaned != raw_query:
            metadata["was_cleaned"] = True

        # Step 2: Basic cleanup
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)  # collapse whitespace

        # Step 3: Detect language
        metadata["language"] = self._detect_language(cleaned)

        # Step 4: Expand abbreviations
        cleaned = self._expand_abbreviations(cleaned)

        # Step 5: Detect domain from keywords
        metadata["domain"] = self._detect_domain(cleaned, channel_name)

        # Keep semantic and lexical retrieval inputs separate. Domain is already
        # carried as metadata, so adding it to every query only creates noise.
        metadata["lexical_query"] = self._augment_cross_lingual(
            cleaned, metadata["language"]
        )

        # Step 7 (Phase 4): Classify query intent for routing
        metadata["query_intent"] = self.classify_query(
            cleaned, detected_domain=metadata["domain"]
        )

        logger.debug(
            f"Query preprocessed: '{raw_query[:60]}' → '{cleaned[:60]}' "
            f"(lang={metadata['language']}, domain={metadata['domain']}, "
            f"intent={metadata['query_intent'].value})"
        )

        return cleaned, metadata

    def _remove_discord_formatting(self, text: str) -> str:
        """Remove Discord-specific formatting from text."""
        text = DISCORD_MENTION_PATTERN.sub("", text)
        text = DISCORD_CHANNEL_PATTERN.sub("", text)
        text = DISCORD_ROLE_PATTERN.sub("", text)
        text = DISCORD_EMOJI_PATTERN.sub("", text)
        text = URL_PATTERN.sub("[link]", text)
        return text

    # Distinctive Vietnamese function words for ASCII-typed (no-diacritic) text.
    # Kept narrow to avoid misflagging English.
    _ASCII_VI_MARKERS = {
        "khong", "duoc", "nhieu", "minh", "biet", "muon", "vay",
        "roi", "chua", "bao nhieu", "nhung", "lam sao", "the nao",
    }

    def _detect_language(self, text: str) -> str:
        """Detect Vietnamese via diacritics, with an ASCII-Vietnamese fallback."""
        viet_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
        low = text.lower()
        viet_count = sum(1 for c in low if c in viet_chars)
        if viet_count >= 2 or (viet_count >= 1 and len(text) < 20):
            return "vi"
        # ASCII Vietnamese (typed without diacritics) is common in chat (L5).
        hits = sum(1 for m in self._ASCII_VI_MARKERS if m in low)
        if hits >= 2:
            return "vi"
        return "en"

    def _augment_cross_lingual(self, text: str, language: str) -> str:
        """Append English game-term equivalents for Vietnamese queries (H3).

        The original text is preserved; English synonyms are appended only when
        not already present, so the lexical (BM25) arm can match the English KB.
        """
        if language != "vi":
            return text
        low = text.lower()
        extra: list = []
        occupied: list[tuple[int, int]] = []
        for vi_term, en_term in sorted(
            GAME_GLOSSARY_VI_EN.items(), key=lambda item: len(item[0]), reverse=True
        ):
            for match in re.finditer(re.escape(vi_term), low):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in occupied):
                    continue
                occupied.append(span)
                for en in en_term.split():
                    if en not in low and en not in extra:
                        extra.append(en)
                break
        return f"{text} {' '.join(extra)}" if extra else text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations/slang."""
        words = text.split()
        expanded = []
        for word in words:
            lower = word.lower().strip(".,!?")
            if lower in self._abbreviations:
                expanded.append(self._abbreviations[lower])
            else:
                expanded.append(word)
        return " ".join(expanded)

    def _detect_domain(
        self, text: str, channel_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Detect the knowledge domain from keywords in text or channel name.
        Returns domain key or None.
        """
        text_lower = text.lower()
        channel = (channel_name or "").lower()

        # Domain keyword maps (bilingual — Vietnamese users rarely type the
        # English wiki terms, audit M4)
        domain_keywords = {
            "pz": [
                "project zomboid", "zomboid", "pz", "zombie",
                "carpentry", "foraging", "trapping",
                "moodle", "trait", "occupation", "knox",
                "crafting", "metalwork", "tailoring",
                "first aid", "farming", "fishing",
                "barricade", "generator", "vehicle",
                # Vietnamese
                "thây ma", "sinh tồn", "chế tạo", "nấu ăn", "câu cá",
                "trồng trọt", "mộc", "rèn", "may vá", "sơ cứu",
                "vết thương", "máy phát", "xe cộ", "vũ khí", "rìu",
                "súng", "sát thương", "công thức", "kỹ năng", "đói",
                "khát", "nhiễm", "damage", "gun", "firearm", "shotgun",
                "rifle", "pistol",
            ],
            "server_rules": [
                "server rule", "server rules", "rules of", "regulation",
                "quy định", "quy tắc", "nội quy", "luật server", "luật chơi",
                "vi phạm", "hình phạt",
            ],
        }

        for domain, keywords in domain_keywords.items():
            # Check channel name first
            if any(kw in channel for kw in keywords):
                return domain
            # Check message content
            if any(kw in text_lower for kw in keywords):
                return domain

        return None

    def _add_domain_context(self, text: str, domain: str) -> str:
        """Add domain-specific context prefix to improve retrieval."""
        prefixes = {
            "pz": "Project Zomboid:",
        }
        prefix = prefixes.get(domain, "")
        if prefix and not text.lower().startswith(prefix.lower()):
            return f"{prefix} {text}"
        return text

    def add_abbreviation(self, abbr: str, expansion: str) -> None:
        """Add a custom abbreviation mapping."""
        self._abbreviations[abbr.lower()] = expansion

    # ------------------------------------------------------------------
    # Phase 4: Query Intent Classification
    # ------------------------------------------------------------------

    # Patterns that strongly indicate ANALYTICAL intent (need SQL/tool calling)
    _ANALYTICAL_PATTERNS: list = [
        # Ranking / superlative (Vietnamese)
        r"nào\s+(mạnh|tốt|nhanh|nhẹ|nặng|cao|thấp|nhiều|ít|giỏi|dày)\s*nhất",
        r"(mạnh|tốt|nhanh|nhẹ|nặng|cao|thấp|nhiều|ít)\s*nhất",
        r"top\s*\d+",
        r"xếp\s*hạng",
        r"bảng\s*(xếp|so)\s*sánh",
        # Ranking / superlative (English)
        r"(best|worst|strongest|weakest|fastest|slowest|lightest|heaviest|highest|lowest)\b",
        r"\btop\s*\d+\b",
        r"\branking\b",
        # Comparison
        r"so\s*sánh",
        r"\bvs\.?\b",
        r"\bcompare\b",
        r"khác\s*(nhau|gì)",
        r"hơn\s*(gì|không|ko)",
        # Crafting / recipe lookup
        r"(cần|cần\s+gì|nguyên\s*liệu).*?(craft|chế|làm|tạo)",
        r"(craft|chế|làm|tạo).*?(cần|cần\s+gì|nguyên\s*liệu)",
        r"\brecipe\b",
        r"cách\s*(craft|chế\s*tạo|làm)\b",
        r"công\s*thức",
        # Stats / data lookup
        r"\b(damage|dmg|defense|def|encumbrance|weight|hunger|thirst|capacity)\b.*?\d",
        r"\d+.*?\b(damage|dmg|defense|def|encumbrance|weight)\b",
        r"(chỉ\s*số|stats?|thông\s*số)\s+(của|cho)",
        # Location queries
        r"(ở\s*đâu|chỗ\s*nào|tìm\s*(được|thấy)?\s*ở)",
        r"\b(find|where)\b.*?(location|station|shop|store|hospital)",
        r"(gas\s*station|police|hospital|gun\s*store|restaurant)",
        r"tọa\s*độ",
        r"coordinates",
        # Explicit item detail queries
        r"\bitem[\s_]*id\b",
        r"Base\.[\w.]+",
        # Filter queries
        r"(damage|defense|hunger)\s*(>|<|>=|<=|lớn\s*hơn|nhỏ\s*hơn|trên|dưới)\s*\d",
    ]

    # Patterns that strongly indicate NARRATIVE intent (prose/lore/explanation)
    _NARRATIVE_PATTERNS: list = [
        # "What is" / explanation
        r"(là\s+gì|là\s+cái\s+gì)",
        r"\b(what\s+is|what\s+are|how\s+does|how\s+do|how\s+to)\b",
        r"(tác\s*dụng|ảnh\s*hưởng|công\s*dụng|hiệu\s*quả)\s+(của|gì|như\s*thế)",
        r"(giải\s*thích|explain|describe|mô\s*tả)",
        r"(hướng\s*dẫn|guide|tutorial|tips?\b)",
        # Lore / narrative
        r"(lore|story|câu\s*chuyện|lịch\s*sử|background)",
        r"(kể|tell\s*me\s*about)",
        # Server rules / community
        r"(server\s*rules?|quy\s*(tắc|định)|luật|rules?\s+of)",
        r"(nội\s*quy|regulation)",
        # Mechanics explanation
        r"(hoạt\s*động|works?|mechanism|cơ\s*chế)\s+(như\s*thế\s*nào|how|ra\s*sao)",
        r"(skill|kỹ\s*năng)\s+(làm\s*gì|tác\s*dụng|level\s*up)",
    ]

    # Patterns that indicate CONVERSATION (casual chat, no game data needed)
    _CONVERSATION_PATTERNS: list = [
        r"^(hi|hello|hey|xin\s*chào|chào|ê|oi|yo|sup)\b",
        r"^(cảm\s*ơn|thanks?|thank\s*you|tks)\b",
        r"^(bye|tạm\s*biệt|goodbye|bb)\b",
        r"(bạn\s*(là\s*ai|tên\s*gì|là\s*gì)|who\s*are\s*you|your\s*name)",
        r"(đùa|joke|funny|haha|lol|kk|:v|xD)",
        r"(chuyện\s*cười|câu\s*chuyện\s*cười|truyện\s*cười)",
        r"(kể|nói|bịa|chế).*(vui|hài|cười|joke|troll|roast)",
        r"(làm|viết).*(thơ|rap|vè).*(vui|troll|chọc|trêu|hài)?",
        r"(meme|troll|roast|chọc|trêu)\b",
        r"(khỏe\s*(không|ko)|how\s*are\s*you)",
        r"(tâm\s*sự|vent|rant)",
        r"(gà|cùi|lú|lag\s*não|vô\s*tri|đoán\s*mò|sai\s*bét|nhạt|máy\s*móc)",
        r"(nomnom|bot).*?(chậm\s+như|ngu\s+(nhất|quá)|dở\s+(quá|thật)|tệ\s+(quá|thật))",
        r"(nomnom|bot).*(thông\s*minh|giỏi|thiên\s*tài|hay\s*quá)",
        r"(tư\s*vấn|lời\s*khuyên|xin\s*lỗi|động\s*lực|áp\s*lực|tâm\s*trạng)",
        r"(trì\s*hoãn|deadline|bình\s*tĩnh|ngủ\s*sớm|ngủ\s*đủ|thói\s*quen|góp\s*ý)",
        r"(bạn\s*bè|đang\s*giận|nói\s*chuyện\s+với)",
        r"(dọn\s*bàn|ghi\s*chú|cuộc\s*họp|cà\s*phê|điện\s*thoại)",
        r"(học\s+và\s+nghỉ|giữ\s+tập\s*trung|cuộc\s+trò\s*chuyện)",
        r"(tối\s*nay\s*ăn\s*gì|uống\s*cà\s*phê|dọn\s*bàn|ghi\s*chú\s*cuộc\s*họp)",
        r"(mẹo.*quên|checklist.*ngủ|bắt\s*đầu\s+một\s+cuộc\s+trò\s*chuyện)",
        r"\b(tôi|mình|em|anh|chị|bạn)\s+nên\b",
        r"(đố\s*mẹo|đố\s*vui|câu\s*đố|hack\s*não)",
        r"(ý\s*tôi\s*không\s*phải|nói\s*lại|ví\s*dụ\s*khác)",
        r"(đoạn\s*trên|câu\s*vừa\s*rồi|lúc\s*nãy|cuối\s*cùng|đúng\s*trọng\s*tâm)",
        r"(giải\s*thích|tóm\s*tắt).*(dễ\s*hiểu\s*hơn|ngắn\s*hơn|một\s*câu)",
    ]

    # A short sentence is not necessarily small talk. These cues cover common
    # item/mechanic questions that carry no domain keyword. Sending them
    # through hybrid retrieval is safer than answering from model memory.
    _FACTUAL_QUERY_CUES: list = [
        r"\?$",
        r"(dùng\s*để|làm\s*gì|cần\s*gì|ở\s*đâu|bao\s*nhiêu|tìm\s*(được|thấy)?)",
        r"(cách\s*(chế|làm|tạo)|tác\s*dụng|công\s*dụng|hoạt\s*động)",
        r"\b(what|where|when|which|why|how|does|do|can)\b",
    ]

    def classify_query(
        self,
        text: str,
        detected_domain: Optional[str] = None,
    ) -> QueryIntent:
        """
        Classify query intent to determine the optimal response pipeline.

        Priority:
        1. CONVERSATION — casual chat, no game data needed
        2. ANALYTICAL — data/comparative/ranking → needs SQLite tools
        3. NARRATIVE — lore/prose/explanation → needs vector search
        4. HYBRID — unclear, might benefit from both

        Args:
            text: Preprocessed query text.
            detected_domain: Domain detected by preprocess (e.g., "pz").

        Returns:
            QueryIntent enum value.
        """
        text_lower = text.lower().strip()

        # Greetings, banter, jokes, and creative prompts do not need game data.
        for pattern in self._CONVERSATION_PATTERNS:
            if re.search(pattern, text_lower):
                return QueryIntent.CONVERSATION

        # Check analytical patterns first (higher priority for PZ domain)
        analytical_score = 0
        for pattern in self._ANALYTICAL_PATTERNS:
            if re.search(pattern, text_lower):
                analytical_score += 1

        # Check narrative patterns
        narrative_score = 0
        for pattern in self._NARRATIVE_PATTERNS:
            if re.search(pattern, text_lower):
                narrative_score += 1

        # Decision logic
        if analytical_score > 0 and narrative_score == 0:
            return QueryIntent.ANALYTICAL

        if narrative_score > 0 and analytical_score == 0:
            return QueryIntent.NARRATIVE

        if analytical_score > 0 and narrative_score > 0:
            # Both matched — prefer analytical if score is higher
            if analytical_score >= narrative_score:
                return QueryIntent.ANALYTICAL
            return QueryIntent.HYBRID

        # No game-related patterns matched
        if detected_domain:
            # Domain detected but no specific intent → hybrid
            return QueryIntent.HYBRID

        # A concise factual question without a keyword still needs evidence.
        if any(re.search(pattern, text_lower) for pattern in self._FACTUAL_QUERY_CUES):
            return QueryIntent.HYBRID

        # No domain, no factual cue, short text → conversation.
        if len(text_lower) < 50:
            return QueryIntent.CONVERSATION

        # Longer messages without clear intent → hybrid (let RAG try)
        return QueryIntent.HYBRID


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_preprocessor: Optional[QueryPreprocessor] = None


def get_preprocessor() -> QueryPreprocessor:
    """Return the singleton QueryPreprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = QueryPreprocessor()
    return _preprocessor


def classify_query(
    text: str,
    detected_domain: Optional[str] = None,
) -> QueryIntent:
    """
    Standalone helper: classify query intent for routing.

    Args:
        text: Raw or preprocessed query text.
        detected_domain: Domain hint.

    Returns:
        QueryIntent enum value.
    """
    return get_preprocessor().classify_query(text, detected_domain)

