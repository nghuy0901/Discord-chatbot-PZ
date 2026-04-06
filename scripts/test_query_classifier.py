"""
Test suite for Phase 4: Query Intent Classifier.

Validates that the classifier correctly routes queries to the optimal pipeline:
  - ANALYTICAL → SQLite Tool Calling (data/comparison/ranking)
  - NARRATIVE → Vector Search streaming (lore/prose/explanation)
  - HYBRID → Both (factual + context)
  - CONVERSATION → No tools, no RAG (casual chat)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query_preprocessor import (
    QueryPreprocessor,
    QueryIntent,
    classify_query,
)


def test_classifier():
    pp = QueryPreprocessor()

    # ============================================================
    # TEST CASES: (query, expected_intent, description)
    # ============================================================
    test_cases = [
        # --- ANALYTICAL: Ranking / Superlative ---
        ("Axe nào mạnh nhất?", QueryIntent.ANALYTICAL, "Vietnamese superlative"),
        ("food giảm hunger nhiều nhất?", QueryIntent.ANALYTICAL, "Vietnamese superlative + stat"),
        ("weapon damage cao nhất", QueryIntent.ANALYTICAL, "Vietnamese ranking"),
        ("best armor in the game", QueryIntent.ANALYTICAL, "English superlative"),
        ("top 5 weapons by damage", QueryIntent.ANALYTICAL, "English ranking"),
        ("strongest melee weapon", QueryIntent.ANALYTICAL, "English superlative adj"),
        ("lightest axe", QueryIntent.ANALYTICAL, "English superlative adj"),

        # --- ANALYTICAL: Comparison ---
        ("so sánh Axe vs Hatchet", QueryIntent.ANALYTICAL, "Vietnamese comparison"),
        ("compare Firefighter Axe and Wood Axe", QueryIntent.ANALYTICAL, "English comparison"),
        ("Axe vs Hatchet vs Hand Axe", QueryIntent.ANALYTICAL, "Multi-item comparison"),
        ("Axe khác gì Hatchet?", QueryIntent.ANALYTICAL, "Vietnamese difference"),

        # --- ANALYTICAL: Recipe / Crafting ---
        ("craft Baguette cần gì?", QueryIntent.ANALYTICAL, "Vietnamese recipe lookup"),
        ("recipe for axe", QueryIntent.ANALYTICAL, "English recipe lookup"),
        ("cách craft bandage", QueryIntent.ANALYTICAL, "Vietnamese craft how"),
        ("công thức nấu ăn", QueryIntent.ANALYTICAL, "Vietnamese recipe word"),
        ("nguyên liệu để làm bread", QueryIntent.ANALYTICAL, "Vietnamese ingredient"),

        # --- ANALYTICAL: Location ---
        ("gas station ở đâu?", QueryIntent.ANALYTICAL, "Vietnamese location"),
        ("where to find police station", QueryIntent.ANALYTICAL, "English location"),
        ("tọa độ gas station Louisville", QueryIntent.ANALYTICAL, "Vietnamese coordinates"),
        ("find hospital in Louisville", QueryIntent.ANALYTICAL, "English find location"),

        # --- ANALYTICAL: Stats / Data ---
        ("stats của Firefighter Axe", QueryIntent.ANALYTICAL, "Vietnamese stats lookup"),
        ("damage > 2 weapons", QueryIntent.ANALYTICAL, "Filter query"),
        ("item_id Base.Axe_Old", QueryIntent.ANALYTICAL, "Item ID lookup"),

        # --- NARRATIVE: Explanation ---
        ("Sneaking skill là gì?", QueryIntent.NARRATIVE, "Vietnamese what-is"),
        ("what is carpentry in Project Zomboid?", QueryIntent.NARRATIVE, "English what-is"),
        ("how does fishing work?", QueryIntent.NARRATIVE, "English how-does"),
        ("tác dụng của trait Athletic", QueryIntent.NARRATIVE, "Vietnamese effect"),
        ("giải thích moodle anxious", QueryIntent.NARRATIVE, "Vietnamese explain"),
        ("hướng dẫn tăng skill carpentry", QueryIntent.NARRATIVE, "Vietnamese guide"),

        # --- NARRATIVE: Lore ---
        ("lore của Knox country", QueryIntent.NARRATIVE, "Lore query"),
        ("kể về câu chuyện Project Zomboid", QueryIntent.NARRATIVE, "Vietnamese tell-about"),

        # --- NARRATIVE: Rules ---
        ("server rules là gì?", QueryIntent.NARRATIVE, "Server rules query"),
        ("nội quy server", QueryIntent.NARRATIVE, "Vietnamese rules"),
        ("rules of the server", QueryIntent.NARRATIVE, "English rules"),

        # --- NARRATIVE: Mechanics ---
        ("skill Carpentry hoạt động như thế nào?", QueryIntent.NARRATIVE, "Vietnamese mechanics"),
        ("how does the hunger system work?", QueryIntent.NARRATIVE, "English mechanics"),

        # --- CONVERSATION: Casual ---
        ("hi", QueryIntent.CONVERSATION, "Simple greeting"),
        ("hello bot", QueryIntent.CONVERSATION, "Greeting"),
        ("xin chào", QueryIntent.CONVERSATION, "Vietnamese greeting"),
        ("cảm ơn nha", QueryIntent.CONVERSATION, "Vietnamese thanks"),
        ("bạn là ai?", QueryIntent.CONVERSATION, "Who are you"),
        ("haha", QueryIntent.CONVERSATION, "Laughing"),
        ("bye", QueryIntent.CONVERSATION, "Farewell"),

        # --- HYBRID: Ambiguous ---
        ("Axe", QueryIntent.HYBRID, "Single word PZ-related"),
        ("tell me everything about weapons", QueryIntent.HYBRID, "Broad PZ query"),
    ]

    # ============================================================
    # Run tests
    # ============================================================
    passed = 0
    failed = 0
    failures = []

    print("\n🧪 Phase 4: Query Intent Classifier Test Suite")
    print("=" * 70)

    for query, expected, desc in test_cases:
        result = pp.classify_query(query, detected_domain="pz" if "pz" not in query.lower() else None)

        # Also test with preprocess pipeline
        _, meta = pp.preprocess(query)
        result_from_preprocess = meta.get("query_intent")

        status = "✅" if result == expected else "❌"
        if result != expected:
            failed += 1
            failures.append((query, expected, result, desc))
        else:
            passed += 1

        print(f"  {status} [{result.value:13s}] {desc:40s} | \"{query}\"")

    print()
    print("=" * 70)

    # Test standalone helper
    standalone_result = classify_query("Axe nào mạnh nhất?")
    assert standalone_result == QueryIntent.ANALYTICAL, \
        f"Standalone helper failed: {standalone_result}"
    print("  ✅ Standalone classify_query() helper works")

    # Test preprocess integration
    _, meta = pp.preprocess("so sánh Axe vs Hatchet")
    assert "query_intent" in meta, "preprocess() missing query_intent"
    assert meta["query_intent"] == QueryIntent.ANALYTICAL, \
        f"preprocess() intent wrong: {meta['query_intent']}"
    print("  ✅ preprocess() includes query_intent correctly")

    print()
    print("=" * 70)

    if failures:
        print(f"\n⚠️  {failed} FAILED ({passed} passed)")
        for q, exp, got, desc in failures:
            print(f"  ❌ [{desc}] \"{q}\"")
            print(f"     Expected: {exp.value}, Got: {got.value}")
        print()
    else:
        print(f"ALL {passed} TESTS PASSED ✅")

    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = test_classifier()
    sys.exit(0 if success else 1)
