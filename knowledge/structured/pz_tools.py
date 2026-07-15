"""
PZ Tools — SQLite-backed tool functions for LLM tool calling.

Provides 5 tools that query the Project Zomboid SQLite database:
  1. search_items     — Find and rank items (weapons, food, clothing, etc.)
  2. compare_items    — Side-by-side comparison of specific items
  3. get_recipe       — Look up crafting recipes
  4. get_item_details — Full details for a single item
  5. find_location    — Find map locations / businesses

Each tool is a bare callable with a Google-style docstring. For OpenAI /
OpenAI-compatible / Gemini providers they are converted to JSON tool schemas by
``src.llm.tool_schema`` before being sent (passing raw functions is not
JSON-serializable); native Ollama can also accept the generated schemas.

Usage:
    from knowledge.structured.pz_tools import PZ_TOOLS, PZ_TOOL_FUNCTIONS
    # PZ_TOOLS          — list of tool callables (see src.llm.tool_schema)
    # PZ_TOOL_FUNCTIONS — dict mapping name→callable for executing tool calls
"""

import os
import json
import sqlite3
import logging
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(
    os.path.dirname(__file__), "pz_data.db"
)
PZ_DOCS_PATH = Path(__file__).resolve().parents[1] / "docs" / "pz"


def _get_connection() -> sqlite3.Connection:
    """Get a read-only SQLite connection with row factory."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"PZ database not found at {DB_PATH}")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows: list) -> List[dict]:
    """Convert sqlite3.Row objects to plain dicts, stripping None values."""
    results = []
    for row in rows:
        d = {k: row[k] for k in row.keys() if row[k] is not None}
        # Parse extra_fields JSON if present
        if "extra_fields" in d and isinstance(d["extra_fields"], str):
            try:
                d["extra_fields"] = json.loads(d["extra_fields"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


def _disambiguate_names(results: List[dict]) -> List[dict]:
    """Disambiguate results that share a display name.

    Different items can share a name (e.g. ``Base.WoodAxe`` and
    ``Base.WoodAxeForged`` are both "Wood Axe"). When that happens within one
    result set, append a short item_id suffix so the user can tell them apart.
    """
    from collections import Counter

    counts = Counter(r.get("name") for r in results if r.get("name"))
    for r in results:
        name = r.get("name")
        if name and counts[name] > 1 and r.get("item_id"):
            suffix = str(r["item_id"]).split(".")[-1]
            r["name"] = f"{name} ({suffix})"
    return results


# ---------------------------------------------------------------------------
# Allowed columns for safe SQL (prevent injection)
# ---------------------------------------------------------------------------
SORTABLE_COLUMNS = {
    "name", "category", "sub_category", "encumbrance",
    "max_damage", "min_damage", "door_damage", "tree_damage",
    "attack_speed", "crit_chance", "crit_multiplier", "knockback",
    "min_range", "max_range", "max_condition",
    "hunger", "thirst", "unhappiness", "boredom",
    "fresh_days", "rotten_days", "cooked_mins", "burned_mins",
    "bite_defense", "scratch_defense", "bullet_defense",
    "insulation", "wind_resistance",
    "capacity", "points",
}

FILTER_COLUMNS = SORTABLE_COLUMNS | {
    "item_id", "equipped", "body_location", "movement_speed",
    "body_parts_protected", "source_file", "spice",
}


def _validate_column(col: str, allowed: set) -> str:
    """Validate column name to prevent SQL injection."""
    col = col.strip().lower()
    if col not in allowed:
        raise ValueError(f"Invalid column: '{col}'. Allowed: {sorted(allowed)}")
    return col


def _normalize_alias_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


ITEM_FILTER_ALIASES = [
    (
        {
            "gun", "guns", "firearm", "firearms", "sung", "súng",
            "rifle", "rifles", "shotgun", "shotguns", "pistol", "pistols",
        },
        "Weapons",
        "Firearms",
    ),
    (
        {"ammo", "ammunition", "bullet", "bullets", "dan", "đạn", "bang dan", "băng đạn"},
        "Weapons",
        "Ammo",
    ),
    (
        {"tool", "tools", "cong cu", "công cụ", "dung cu", "dụng cụ"},
        "Equipment",
        "Tools",
    ),
    (
        {"medical", "medicine", "meds", "first aid", "thuoc", "thuốc", "y te", "y tế"},
        "Equipment",
        "Medical",
    ),
    (
        {"food", "foods", "do an", "đồ ăn", "thuc an", "thức ăn"},
        "Food",
        None,
    ),
    (
        {"armor", "armour", "giap", "giáp", "ao giap", "áo giáp", "protection"},
        "Clothing",
        None,
    ),
    (
        {"trait", "traits", "perk", "perks", "dac tinh", "đặc tính"},
        "Player",
        "Trait",
    ),
    (
        {"storage", "container", "containers", "crate", "crates", "tu do", "tủ đồ"},
        "Storage",
        None,
    ),
    (
        {"fishing", "fish", "cau ca", "câu cá"},
        "Equipment",
        "Fishing",
    ),
    (
        {"camping", "cam trai", "cắm trại"},
        "Equipment",
        "Camping",
    ),
    (
        {"trap", "traps", "bay", "bẫy"},
        "Equipment",
        "Traps",
    ),
]


CRAFTING_TYPE_ALIASES = {
    "armor": "Armor",
    "armour": "Armor",
    "giap": "Armor",
    "giáp": "Armor",
    "assembly": "Assembly",
    "assemble": "Assembly",
    "lap rap": "Assembly",
    "lắp ráp": "Assembly",
    "blacksmith": "Blacksmithing",
    "blacksmithing": "Blacksmithing",
    "ren": "Blacksmithing",
    "rèn": "Blacksmithing",
    "blade": "Blade",
    "carpentry": "Carpentry",
    "moc": "Carpentry",
    "mộc": "Carpentry",
    "woodwork": "Carpentry",
    "carving": "Carving",
    "khac": "Carving",
    "khắc": "Carving",
    "cooking": "Cooking",
    "cook": "Cooking",
    "nau an": "Cooking",
    "nấu ăn": "Cooking",
    "cong thuc nau an": "Cooking",
    "công thức nấu ăn": "Cooking",
    "electrical": "Electrical",
    "electricity": "Electrical",
    "dien": "Electrical",
    "điện": "Electrical",
    "farming": "Farming",
    "farm": "Farming",
    "nong nghiep": "Farming",
    "nông nghiệp": "Farming",
    "fishing": "Fishing",
    "cau ca": "Fishing",
    "câu cá": "Fishing",
    "medical": "Medical",
    "first aid": "Medical",
    "y te": "Medical",
    "y tế": "Medical",
    "metalworking": "Metalworking",
    "metal": "Metalworking",
    "co khi": "Metalworking",
    "cơ khí": "Metalworking",
    "repair": "Repair",
    "sua chua": "Repair",
    "sửa chữa": "Repair",
    "tailoring": "Tailoring",
    "may va": "Tailoring",
    "may vá": "Tailoring",
    "tools": "Tools",
    "tool": "Tools",
    "dung cu": "Tools",
    "dụng cụ": "Tools",
    "cong cu": "Tools",
    "công cụ": "Tools",
    "weapon": "Weaponry",
    "weapons": "Weaponry",
    "weaponry": "Weaponry",
    "vu khi": "Weaponry",
    "vũ khí": "Weaponry",
}

LITERATURE_MARKERS = {
    "book",
    "magazine",
    "manual",
    "notebook",
    "journal",
    "newspaper",
    "flier",
    "flyer",
}

LITERATURE_GENERIC_TERMS = {
    "a",
    "an",
    "and",
    "for",
    "how",
    "of",
    "the",
    "to",
    "use",
    "book",
    "books",
    "sach",
    "skill",
    "skills",
    "magazine",
    "magazines",
    "manual",
    "manuals",
    "literature",
    "nomnom",
    "dung",
    "dùng",
    "lam",
    "làm",
    "gi",
    "gì",
}


def _normalize_item_filters(
    category: Optional[str],
    sub_category: Optional[str],
    name_contains: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Normalize common user/LLM aliases to database categories.

    Tool-calling models often pass natural labels such as "Guns" or Vietnamese
    "súng" even though the shipped DB stores guns as Weapons / Firearms.
    """
    category_text = _normalize_alias_text(category)
    sub_category_text = _normalize_alias_text(sub_category)
    name_text = _normalize_alias_text(name_contains)

    for aliases, normalized_category, normalized_sub_category in ITEM_FILTER_ALIASES:
        if (
            category_text in aliases
            or sub_category_text in aliases
            or name_text in aliases
        ):
            category = normalized_category
            sub_category = normalized_sub_category
            if name_text in aliases:
                name_contains = None
            break

    return category, sub_category, name_contains


def _normalize_crafting_type(crafting_type: Optional[str]) -> Optional[str]:
    key = _normalize_alias_text(crafting_type)
    return CRAFTING_TYPE_ALIASES.get(key, crafting_type)


def _normalize_literature_query(query: Optional[str]) -> str:
    text = _normalize_alias_text(query)
    replacements = {
        "sách": "book",
        "sach": "book",
        "tạp chí": "magazine",
        "tap chi": "magazine",
        "cẩm nang": "manual",
        "cam nang": "manual",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _extract_markdown_field(body: str, field_name: str) -> Optional[str]:
    match = re.search(
        rf"^\s*-\s+\*\*{re.escape(field_name)}:\*\*\s*(.+)$",
        body,
        re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def _iter_markdown_sections():
    if not PZ_DOCS_PATH.exists():
        return
    heading_re = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)
    for path in PZ_DOCS_PATH.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        matches = list(heading_re.finditer(text))
        for idx, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            yield path, heading, body


def _is_literature_record(heading: str, body: str, item_id: str) -> bool:
    if heading.lower().startswith("recipe:"):
        return False
    name = _extract_markdown_field(body, "Name") or ""
    haystack = f"{heading} {name} {item_id}".lower()
    return any(marker in haystack for marker in LITERATURE_MARKERS)


def _score_literature_record(query: str, record: dict) -> int:
    if not query:
        return 1
    searchable = " ".join(
        str(record.get(key, ""))
        for key in ("name", "item_id", "summary", "source")
    ).lower()
    name = str(record.get("name", "")).lower()
    item_id = str(record.get("item_id", "")).lower()

    score = 0
    if query in name:
        score += 100
    elif query in searchable:
        score += 50

    tokens = [
        token
        for token in re.findall(r"[\w']+", query, flags=re.UNICODE)
        if token not in LITERATURE_GENERIC_TERMS and len(token) > 1
    ]
    for token in tokens:
        if token in name:
            score += 10
        if token in item_id:
            score += 5
        if token in searchable:
            score += 2
    return score


# ---------------------------------------------------------------------------
# Tool 1: search_items
# ---------------------------------------------------------------------------
def search_items(
    category: str = None,
    sub_category: str = None,
    name_contains: str = None,
    sort_by: str = None,
    order: str = "DESC",
    limit: int = 10,
    min_value: float = None,
    max_value: float = None,
    filter_field: str = None,
) -> str:
    """
    Search and rank items from the Project Zomboid database.

    Use this for item/stat/ranking questions like "axe nào mạnh nhất?",
    "food giảm hunger nhiều nhất?", "armor nào có bite defense cao nhất?",
    "tool nào nhẹ nhất?", "ammo nào?", "trait nào tốn nhiều points?".

    Args:
        category: Filter by category. Options: "Weapons", "Food", "Clothing",
            "Equipment", "Storage", "Comfort", "Appliances", "Plumbing",
            "Miscellaneous", "Player". Common aliases like "guns"/"súng",
            "ammo"/"đạn", "tools"/"dụng cụ", "medical"/"thuốc",
            "armor"/"giáp", and "traits"/"đặc tính" are normalized.
        sub_category: Filter by sub-category. Examples: "Axes", "Canned food",
            "Armor", "Cooking", "Electricity", "Positive", "Negative"
        name_contains: Search items whose name contains this text (partial match)
        sort_by: Column to sort results by. Common options:
            "max_damage" (weapons), "hunger" (food), "bite_defense" (armor),
            "encumbrance" (weight), "points" (traits), "capacity" (storage)
        order: Sort direction. "DESC" for highest first, "ASC" for lowest first
        limit: Maximum number of results to return (default: 10, max: 25)
        min_value: Minimum value for filter_field (e.g. damage > 1.5)
        max_value: Maximum value for filter_field (e.g. encumbrance < 5)
        filter_field: Column to apply min_value/max_value filter on

    Returns:
        str: JSON string with matching items and their stats
    """
    try:
        category, sub_category, name_contains = _normalize_item_filters(
            category, sub_category, name_contains
        )
        conn = _get_connection()
        conditions = []
        params = []

        if category:
            conditions.append("category = ?")
            params.append(category)
        if sub_category:
            conditions.append("sub_category = ?")
            params.append(sub_category)
        if name_contains:
            conditions.append("name LIKE ?")
            params.append(f"%{name_contains}%")
        if filter_field and min_value is not None:
            col = _validate_column(filter_field, FILTER_COLUMNS)
            conditions.append(f"{col} >= ?")
            params.append(min_value)
        if filter_field and max_value is not None:
            col = _validate_column(filter_field, FILTER_COLUMNS)
            conditions.append(f"{col} <= ?")
            params.append(max_value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Build ORDER BY
        order_clause = ""
        if sort_by:
            sort_col = _validate_column(sort_by, SORTABLE_COLUMNS)
            direction = "ASC" if order.upper() == "ASC" else "DESC"
            order_clause = f"ORDER BY {sort_col} {direction} NULLS LAST"

        # Clamp limit
        limit = min(max(1, limit), 25)

        query = f"""
            SELECT name, item_id, category, sub_category,
                   encumbrance, max_damage, min_damage, attack_speed,
                   door_damage, tree_damage, knockback,
                   crit_chance, max_condition, equipped,
                   hunger, thirst, unhappiness, boredom,
                   fresh_days, rotten_days,
                   bite_defense, scratch_defense, bullet_defense,
                   body_location, movement_speed,
                   capacity, points, description, effects
            FROM items
            WHERE {where_clause}
            {order_clause}
            LIMIT ?
        """
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        results = _disambiguate_names(_rows_to_dicts(rows))
        return json.dumps({
            "count": len(results),
            "items": results,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"search_items failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 2: compare_items
# ---------------------------------------------------------------------------
def compare_items(
    names: list,
    fields: list = None,
) -> str:
    """
    Compare multiple items side-by-side from the Project Zomboid database.
    
    Use this for questions like "so sánh Axe vs Hatchet", "Firefighter Axe vs Wood Axe",
    "compare Baseball Bat and Crowbar".

    Args:
        names: List of item names to compare. Example: ["Axe", "Hatchet", "Hand Axe"]
        fields: Optional list of specific fields to compare.
            Examples: ["max_damage", "attack_speed", "encumbrance", "knockback"]
            If not provided, all available stats are returned.

    Returns:
        str: JSON string with comparison table of all items and their stats
    """
    try:
        conn = _get_connection()

        placeholders = ",".join(["?"] * len(names))
        query = f"""
            SELECT name, item_id, category, sub_category,
                   encumbrance, max_damage, min_damage, attack_speed,
                   door_damage, tree_damage, knockback,
                   crit_chance, crit_multiplier, max_condition,
                   min_range, max_range, equipped,
                   hunger, thirst, unhappiness, boredom,
                   fresh_days, rotten_days, cooked_mins, burned_mins,
                   bite_defense, scratch_defense, bullet_defense,
                   body_location, body_parts_protected, movement_speed,
                   insulation, wind_resistance,
                   capacity, points, description, effects
            FROM items
            WHERE name IN ({placeholders})
            ORDER BY name
        """
        rows = conn.execute(query, names).fetchall()
        conn.close()

        all_results = _rows_to_dicts(rows)

        # Deduplicate: keep the row with most data per name+item_id
        # Prefer Weapons category (has damage stats) over Equipment
        seen = {}
        for item in all_results:
            key = (item.get("name"), item.get("item_id"))
            existing = seen.get(key)
            if existing is None:
                seen[key] = item
            else:
                # Prefer the row with more non-null fields
                if len(item) > len(existing):
                    seen[key] = item

        results = list(seen.values())

        # Filter to requested fields only
        if fields and results:
            # Always keep name + item_id for identification
            keep = {"name", "item_id", "category"} | set(fields)
            results = [
                {k: v for k, v in item.items() if k in keep}
                for item in results
            ]

        results = _disambiguate_names(results)
        return json.dumps({
            "count": len(results),
            "comparison": results,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"compare_items failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 3: get_recipe
# ---------------------------------------------------------------------------
def get_recipe(
    product_name: str = None,
    crafting_type: str = None,
    ingredient: str = None,
    limit: int = 10,
) -> str:
    """
    Find crafting recipes from the Project Zomboid database.

    Use this for crafting/recipe questions like "craft Baguette cần gì?",
    "recipe Axe", "những recipe nào cần flour?", "công thức nấu ăn nào có?",
    "dụng cụ craft được gì?".

    Args:
        product_name: Search by product name (partial match).
            Example: "Baguette", "Axe", "Bandage"
        crafting_type: Filter by crafting category.
            Options: "Cooking", "Carpentry", "Tailoring", "Metalworking",
            "Electrical", "Blacksmithing", "Medical", "Repair", "Tools",
            "Weaponry". Common aliases like "Blacksmith", "rèn", "nấu ăn",
            "sửa chữa", "may vá", and "dụng cụ" are normalized.
        ingredient: Search for recipes that use this ingredient (partial match).
            Example: "flour", "nails", "Sheet"
        limit: Maximum results to return (default: 10, max: 25)

    Returns:
        str: JSON string with matching recipes including ingredients, tools, and workstation
    """
    try:
        crafting_type = _normalize_crafting_type(crafting_type)
        conn = _get_connection()
        conditions = []
        params = []

        if product_name:
            conditions.append("(name LIKE ? OR product LIKE ?)")
            params.extend([f"%{product_name}%", f"%{product_name}%"])
        if crafting_type:
            conditions.append("crafting_type = ?")
            params.append(crafting_type)
        if ingredient:
            conditions.append("ingredients LIKE ?")
            params.append(f"%{ingredient}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit = min(max(1, limit), 25)

        query = f"""
            SELECT name, product, crafting_type, ingredients,
                   tools, skill_required, workstation, xp_gained
            FROM recipes
            WHERE {where_clause}
            LIMIT ?
        """
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        results = _rows_to_dicts(rows)
        return json.dumps({
            "count": len(results),
            "recipes": results,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"get_recipe failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 4: get_item_details
# ---------------------------------------------------------------------------
def get_item_details(
    name: str = None,
    item_id: str = None,
) -> str:
    """
    Get full details for a specific item from the Project Zomboid database.

    Use this for item detail/use questions like "thông tin chi tiết về
    Firefighter Axe", "stats của Wood Axe", "Base.Axe_Old là gì?",
    "Hammer dùng làm gì?", "tác dụng của Needle là gì?".

    Args:
        name: The item name to look up. Example: "Firefighter Axe", "Turkey (Whole)"
        item_id: The game's internal item ID. Example: "Base.Axe_Old", "Base.PizzaWhole"

    Returns:
        str: JSON string with all available stats for the item, including
            any extra_fields that don't fit the main schema
    """
    try:
        conn = _get_connection()

        if item_id:
            query = "SELECT * FROM items WHERE item_id = ?"
            rows = conn.execute(query, [item_id]).fetchall()
        elif name:
            query = "SELECT * FROM items WHERE name = ?"
            rows = conn.execute(query, [name]).fetchall()
            # Fallback to LIKE if exact match fails
            if not rows:
                query = "SELECT * FROM items WHERE name LIKE ? LIMIT 5"
                rows = conn.execute(query, [f"%{name}%"]).fetchall()
        else:
            conn.close()
            return json.dumps({"error": "Please provide either 'name' or 'item_id'"})

        conn.close()
        results = _disambiguate_names(_rows_to_dicts(rows))

        # Also look up associated recipes
        if results:
            item_name = results[0].get("name", "")
            try:
                conn2 = _get_connection()
                recipe_q = """
                    SELECT name, product, ingredients, tools, workstation
                    FROM recipes
                    WHERE product LIKE ? OR name LIKE ?
                    LIMIT 5
                """
                recipe_rows = conn2.execute(
                    recipe_q, [f"%{item_name}%", f"%{item_name}%"]
                ).fetchall()
                conn2.close()

                if recipe_rows:
                    for r in results:
                        r["related_recipes"] = _rows_to_dicts(recipe_rows)
            except Exception:
                pass  # Non-fatal

        return json.dumps({
            "count": len(results),
            "items": results,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"get_item_details failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 5: find_location
# ---------------------------------------------------------------------------
def find_location(
    name: str = None,
    business_type: str = None,
    area: str = None,
    limit: int = 15,
) -> str:
    """
    Find locations and businesses on the Project Zomboid map.

    Use this for questions like "gas station ở đâu?", "tìm police station Louisville",
    "quán ăn nào gần Louisville?", "map locations".

    Args:
        name: Search by location/business name (partial match).
            Example: "Fossoil", "Gas 2 Go", "Spiffo"
        business_type: Filter by business type (partial match).
            Examples: "Gas station", "Restaurant", "Police station",
            "Gun store", "Hospital", "School"
        area: Filter by map area.
            Options: "Louisville", "Muldraugh", "Riverside", "Rosewood",
            "West Point", "March Ridge"
        limit: Maximum results (default: 15, max: 30)

    Returns:
        str: JSON string with matching locations including coordinates
    """
    try:
        conn = _get_connection()
        conditions = []
        params = []

        if name:
            conditions.append("name LIKE ?")
            params.append(f"%{name}%")
        if business_type:
            conditions.append("business_type LIKE ?")
            params.append(f"%{business_type}%")
        if area:
            conditions.append("area LIKE ?")
            params.append(f"%{area}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit = min(max(1, limit), 30)

        query = f"""
            SELECT name, business_type, coordinates, area, source_file
            FROM locations
            WHERE {where_clause}
            LIMIT ?
        """
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        results = _rows_to_dicts(rows)
        return json.dumps({
            "count": len(results),
            "locations": results,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"find_location failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool 6: search_literature
# ---------------------------------------------------------------------------
def search_literature(
    query: str = None,
    limit: int = 10,
) -> str:
    """
    Search Project Zomboid literature records from the markdown knowledge docs.

    Use this for books, skill books, recipe magazines, magazines, manuals, and
    questions like "sách skill Carpentry là gì?", "How to Use Generators
    magazine dùng làm gì?", "manual nào cho Mechanics?", "tạp chí nào mở recipe?".

    Args:
        query: Search text for a book/magazine/manual name, skill, or purpose.
            Vietnamese aliases like "sách" and "tạp chí" are normalized.
        limit: Maximum results to return (default: 10, max: 25)

    Returns:
        str: JSON string with matching literature records and source snippets.
    """
    try:
        normalized_query = _normalize_literature_query(query)
        limit = min(max(1, limit), 25)

        records = []
        for path, heading, body in _iter_markdown_sections() or []:
            item_id = _extract_markdown_field(body, "Item ID") or ""
            if not _is_literature_record(heading, body, item_id):
                continue

            name = _extract_markdown_field(body, "Name") or heading
            record = {
                "name": name,
                "item_id": item_id,
                "encumbrance": _extract_markdown_field(body, "Encumbrance"),
                "burn_time": _extract_markdown_field(body, "Burn time"),
                "source": str(path.relative_to(PZ_DOCS_PATH)).replace(os.sep, "/"),
                "summary": " ".join(body.split())[:700],
            }
            score = _score_literature_record(normalized_query, record)
            if normalized_query and score <= 0:
                continue
            record["_score"] = score
            records.append(record)

        records.sort(key=lambda item: (-item.pop("_score", 0), item["name"]))
        return json.dumps(
            {
                "count": min(len(records), limit),
                "records": records[:limit],
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"search_literature failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Registry — exported for use in LLM integration
# ---------------------------------------------------------------------------

# List of tool functions — pass to ollama.chat(tools=PZ_TOOLS)
PZ_TOOLS = [
    search_items,
    compare_items,
    get_recipe,
    get_item_details,
    find_location,
    search_literature,
]

# Map of function name → callable — for executing tool calls
PZ_TOOL_FUNCTIONS = {
    "search_items": search_items,
    "compare_items": compare_items,
    "get_recipe": get_recipe,
    "get_item_details": get_item_details,
    "find_location": find_location,
    "search_literature": search_literature,
}
