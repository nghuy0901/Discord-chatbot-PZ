"""
Test PZ tools — validate all 5 tool functions return correct data.
Run standalone: python scripts/test_pz_tools.py
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge.structured.pz_tools import (
    search_items,
    compare_items,
    get_recipe,
    get_item_details,
    find_location,
    PZ_TOOLS,
    PZ_TOOL_FUNCTIONS,
)


def test_search_items():
    print("=" * 70)
    print("TEST 1: search_items — Top 5 weapons by max_damage")
    print("=" * 70)
    result = json.loads(search_items(
        category="Weapons",
        sort_by="max_damage",
        order="DESC",
        limit=5
    ))
    assert result["count"] > 0, "No weapons found!"
    for i, item in enumerate(result["items"], 1):
        print(f"  {i}. {item['name']:<35} | dmg: {item.get('max_damage', 'N/A')}")
    print()


def test_search_items_filter():
    print("=" * 70)
    print("TEST 2: search_items — Axes with damage > 1.5, sorted by encumbrance")
    print("=" * 70)
    result = json.loads(search_items(
        category="Weapons",
        sub_category="Axes",
        sort_by="encumbrance",
        order="ASC",
        filter_field="max_damage",
        min_value=1.5,
        limit=10
    ))
    assert result["count"] > 0, "No filtered axes found!"
    for i, item in enumerate(result["items"], 1):
        print(
            f"  {i}. {item['name']:<30} "
            f"| dmg: {item.get('max_damage', 'N/A'):>4} "
            f"| enc: {item.get('encumbrance', 'N/A')}"
        )
    print()


def test_compare_items():
    print("=" * 70)
    print("TEST 3: compare_items — Axe vs Hatchet vs Hand Axe")
    print("=" * 70)
    result = json.loads(compare_items(
        names=["Axe", "Hatchet", "Hand Axe"],
        fields=["max_damage", "min_damage", "attack_speed", "encumbrance", "knockback"]
    ))
    assert result["count"] > 0, "No items found for comparison!"
    for item in result["comparison"]:
        print(f"  {item['name']:<20} | {item}")
    print()


def test_get_recipe():
    print("=" * 70)
    print("TEST 4: get_recipe — Baguette recipe")
    print("=" * 70)
    result = json.loads(get_recipe(product_name="Baguette"))
    assert result["count"] > 0, "No Baguette recipe found!"
    for r in result["recipes"]:
        print(f"  Recipe: {r['name']}")
        print(f"    Product: {r.get('product', 'N/A')}")
        print(f"    Ingredients: {r.get('ingredients', 'N/A')[:80]}...")
        print(f"    Tools: {r.get('tools', 'N/A')[:80]}...")
        print(f"    Workstation: {r.get('workstation', 'N/A')}")
    print()


def test_get_recipe_by_ingredient():
    print("=" * 70)
    print("TEST 5: get_recipe — Recipes that need 'flour'")
    print("=" * 70)
    result = json.loads(get_recipe(ingredient="flour", limit=5))
    assert result["count"] > 0, "No flour recipes found!"
    for r in result["recipes"]:
        print(f"  {r['name']:<40} → {r.get('product', 'N/A')}")
    print()


def test_get_item_details():
    print("=" * 70)
    print("TEST 6: get_item_details — Firefighter Axe (full stats)")
    print("=" * 70)
    result = json.loads(get_item_details(name="Firefighter Axe"))
    assert result["count"] > 0, "Firefighter Axe not found!"
    item = result["items"][0]
    for k, v in item.items():
        if k not in ("id", "created_at", "source_file"):
            print(f"  {k:<25}: {v}")
    print()


def test_find_location():
    print("=" * 70)
    print("TEST 7: find_location — Gas stations in Louisville")
    print("=" * 70)
    result = json.loads(find_location(
        business_type="Gas station",
        area="Louisville"
    ))
    assert result["count"] > 0, "No gas stations found!"
    for loc in result["locations"]:
        print(
            f"  {loc['name']:<35} "
            f"| {loc.get('business_type', 'N/A'):<20} "
            f"| {loc.get('coordinates', '')}"
        )
    print()


def test_registry():
    print("=" * 70)
    print("TEST 8: Registry — PZ_TOOLS and PZ_TOOL_FUNCTIONS")
    print("=" * 70)
    assert len(PZ_TOOLS) == 5, f"Expected 5 tools, got {len(PZ_TOOLS)}"
    assert len(PZ_TOOL_FUNCTIONS) == 5, f"Expected 5 functions, got {len(PZ_TOOL_FUNCTIONS)}"
    print(f"  PZ_TOOLS: {len(PZ_TOOLS)} tools")
    for func in PZ_TOOLS:
        print(f"    - {func.__name__}")
    print(f"  PZ_TOOL_FUNCTIONS: {list(PZ_TOOL_FUNCTIONS.keys())}")
    print()


if __name__ == "__main__":
    print("\n🔧 PZ Tools Validation Suite\n")
    test_search_items()
    test_search_items_filter()
    test_compare_items()
    test_get_recipe()
    test_get_recipe_by_ingredient()
    test_get_item_details()
    test_find_location()
    test_registry()
    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
