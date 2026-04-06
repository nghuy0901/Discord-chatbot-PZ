"""
Phase 2: Markdown → SQLite Converter
=====================================
Parses PZ Wiki Markdown files and inserts structured records into SQLite.

Supports 3 record types:
  - items:     Weapons, Food, Clothing, Equipment, Storage, Comfort, Appliances, etc.
  - recipes:   Crafting recipes from Crafting/*.md
  - locations: Business listings from Locations/*.md

Usage:
    python scripts/convert_md_to_sqlite.py [--db PATH] [--docs PATH] [--verbose]
"""

import os
import re
import sys
import json
import yaml
import sqlite3
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge", "docs", "pz",
)
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge", "structured", "pz_data.db",
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
-- Items table: Weapons, Food, Clothing, Equipment, Storage, Comfort, etc.
CREATE TABLE IF NOT EXISTS items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         TEXT,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    sub_category    TEXT,
    source_file     TEXT,

    -- Shared numeric
    encumbrance     REAL,

    -- Weapons
    min_damage      REAL,
    max_damage      REAL,
    door_damage     REAL,
    tree_damage     REAL,
    min_range       REAL,
    max_range       REAL,
    attack_speed    REAL,
    crit_chance     REAL,
    crit_multiplier REAL,
    knockback       REAL,
    max_condition   INTEGER,
    condition_lower_chance REAL,
    average_condition REAL,

    -- Food
    hunger          REAL,
    thirst          REAL,
    unhappiness     REAL,
    boredom         REAL,
    fresh_days      REAL,
    rotten_days     REAL,
    cooked_mins     REAL,
    burned_mins     REAL,

    -- Clothing / Armor
    body_location   TEXT,
    movement_speed  TEXT,
    attack_speed_mod TEXT,
    bite_defense    REAL,
    scratch_defense REAL,
    bullet_defense  REAL,
    insulation      REAL,
    wind_resistance REAL,
    neck_protection TEXT,
    body_parts_protected TEXT,

    -- Storage / Comfort / Furniture
    size_tiles      REAL,
    capacity        REAL,
    crafting_surface TEXT,
    bed_type        TEXT,
    skill_required  TEXT,
    tools_required  TEXT,

    -- Equipment
    types           TEXT,
    equip_category  TEXT,

    -- Traits
    points          REAL,
    description     TEXT,
    effects         TEXT,

    -- General text fields
    equipped        TEXT,
    spice           TEXT,

    -- Catch-all for extra fields
    extra_fields    TEXT,

    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
CREATE INDEX IF NOT EXISTS idx_items_sub_category ON items(sub_category);
CREATE INDEX IF NOT EXISTS idx_items_name ON items(name);
CREATE INDEX IF NOT EXISTS idx_items_item_id ON items(item_id);

-- Recipes table
CREATE TABLE IF NOT EXISTS recipes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    product         TEXT,
    crafting_type   TEXT NOT NULL,
    source_file     TEXT,
    ingredients     TEXT,
    tools           TEXT,
    skill_required  TEXT,
    workstation     TEXT,
    xp_gained       TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_recipes_crafting_type ON recipes(crafting_type);
CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(name);

-- Locations table
CREATE TABLE IF NOT EXISTS locations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    business_type   TEXT,
    coordinates     TEXT,
    area            TEXT,
    district        TEXT,
    source_file     TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_locations_area ON locations(area);
CREATE INDEX IF NOT EXISTS idx_locations_business ON locations(business_type);
"""


# ---------------------------------------------------------------------------
# Utility: Parse YAML frontmatter
# ---------------------------------------------------------------------------
def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Extract YAML frontmatter and return (metadata_dict, remaining_text)."""
    fm_match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", text, re.DOTALL)
    if fm_match:
        try:
            meta = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        body = text[fm_match.end():]
        return meta, body
    return {}, text


# ---------------------------------------------------------------------------
# Utility: Parse numeric value from string
# ---------------------------------------------------------------------------
def parse_number(val: str) -> Optional[float]:
    """Parse a numeric value, handling %, ×, ∞, '-', 'N/A'."""
    if not val or val.strip() in ("-", "N/A", "∞", "—", ""):
        return None
    # Remove %, ×, commas
    cleaned = val.strip().replace("%", "").replace("×", "").replace(",", "")
    # Handle negative percentages like "-15%"
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Generic record parser: ### Heading + - **Key:** Value lines
# ---------------------------------------------------------------------------
def parse_records_from_md(body: str) -> List[Dict[str, str]]:
    """
    Parse blocks of:
        ### Record Name
        - **Key:** Value
        - **Key:** Value
    Returns list of dicts with raw string values.
    """
    records = []
    current_record = None
    current_heading = None
    current_section_h2 = None  # Track ## headings for context

    for line in body.splitlines():
        line = line.rstrip()

        # Track ## section headings
        h2_match = re.match(r"^##\s+(.+)$", line)
        if h2_match:
            current_section_h2 = h2_match.group(1).strip()
            continue

        # Detect ### heading (new record)
        h3_match = re.match(r"^###\s+(.+)$", line)
        if h3_match:
            heading = h3_match.group(1).strip()
            # Save previous record
            if current_record and len(current_record) > 1:
                records.append(current_record)
            # Start new record
            current_record = {"_heading": heading}
            if current_section_h2:
                current_record["_section"] = current_section_h2
            current_heading = heading
            continue

        # Detect - **Key:** Value
        kv_match = re.match(r"^-\s+\*\*(.+?):\*\*\s*(.*)", line)
        if kv_match and current_record is not None:
            key = kv_match.group(1).strip()
            val = kv_match.group(2).strip()
            current_record[key] = val
            continue

    # Don't forget the last record
    if current_record and len(current_record) > 1:
        records.append(current_record)

    return records


# ---------------------------------------------------------------------------
# Category: Items (Weapons, Food, Clothing, Equipment, Storage, Comfort, etc.)
# ---------------------------------------------------------------------------

# Categories that contain record-type item data
ITEM_CATEGORIES = {
    "Weapons", "Food", "Clothing", "Equipment", "Storage",
    "Comfort", "Appliances", "Plumbing", "Miscellaneous",
}

# Field mapping: markdown key → (db_column, is_numeric)
FIELD_MAP = {
    # Shared — Name/Object handled separately in record_to_item_row
    "Encumbrance": ("encumbrance", True),
    "Item ID": ("item_id", False),

    # Weapons
    "Minimum damage": ("min_damage", True),
    "Maximum damage": ("max_damage", True),
    "Door damage": ("door_damage", True),
    "Tree damage": ("tree_damage", True),
    "Minimum range": ("min_range", True),
    "Maximum range": ("max_range", True),
    # NOTE: "Attack speed" is handled dynamically in record_to_item_row
    # because Weapons use it as numeric and Clothing as text
    "Critical hit chance": ("crit_chance", True),
    "Crit multiplier": ("crit_multiplier", True),
    "Knockback": ("knockback", True),
    "Max condition": ("max_condition", True),
    "Condition lower chance": ("condition_lower_chance", True),
    "Average condition": ("average_condition", True),
    "Equipped": ("equipped", False),
    "Endurance modifier": (None, False),  # Skip, rarely useful

    # Food
    "Hunger": ("hunger", True),
    "Hunger / Encumbrance": (None, False),  # Derived, skip
    "Thirst": ("thirst", True),
    "Unhappiness": ("unhappiness", True),
    "Boredom": ("boredom", True),
    "Fresh (days)": ("fresh_days", True),
    "Rotten (days)": ("rotten_days", True),
    "Cooked (mins)": ("cooked_mins", True),
    "Burned (mins)": ("burned_mins", True),
    "Spice": ("spice", False),

    # Clothing / Armor
    "Body location": ("body_location", False),
    "Movement speed": ("movement_speed", False),
    "Body parts protected": ("body_parts_protected", False),
    "Bite defense": ("bite_defense", True),
    "Scratch defense": ("scratch_defense", True),
    "Bullet defense": ("bullet_defense", True),
    "Neck protection": ("neck_protection", False),
    "Insulation": ("insulation", True),
    "Wind resistance": ("wind_resistance", True),
    "Water resistance": (None, False),  # Rarely has values
    "Condition loss": (None, False),  # Skip

    # Storage / Comfort
    "Size  (tiles)": ("size_tiles", True),
    "Capacity": ("capacity", True),
    "Crafting surface?": ("crafting_surface", False),
    "Bed type": ("bed_type", False),
    "Skill": ("skill_required", False),
    "Tool(s)": ("tools_required", False),

    # Equipment
    "Types": ("types", False),
    "Category": ("equip_category", False),

    # Traits
    "Points": ("points", True),
    "Description": ("description", False),
    "Effects": ("effects", False),
}


def record_to_item_row(
    record: Dict[str, str],
    category: str,
    sub_category: str,
    source_file: str,
) -> Dict[str, Any]:
    """Convert a parsed record dict to a database row dict."""
    row = {
        "category": category,
        "sub_category": sub_category,
        "source_file": source_file,
    }
    extra = {}

    # Get name from record — try multiple keys
    name = (
        record.get("Name")
        or record.get("Object")
        or record.get("Business")
        or record.get("_heading", "Unknown")
    )
    row["name"] = name

    for key, val in record.items():
        if key.startswith("_"):
            continue

        # Skip Name/Object — already handled above to avoid duplication
        if key in ("Name", "Object", "Business"):
            continue

        # Handle "Attack speed" dynamically based on category
        if key == "Attack speed":
            if category == "Clothing":
                # Clothing uses text like "-3%"
                row["attack_speed_mod"] = val if val and val not in ("-", "N/A") else None
            else:
                # Weapons use numeric attack speed
                row["attack_speed"] = parse_number(val)
            continue

        if key in FIELD_MAP:
            col, is_numeric = FIELD_MAP[key]
            if col is None:
                continue  # Explicitly skipped field

            if is_numeric:
                row[col] = parse_number(val)
            else:
                # For fields that may appear multiple times (Skill, Tool(s))
                if col in row and row[col]:
                    row[col] = f"{row[col]} | {val}"
                else:
                    row[col] = val if val and val not in ("-", "N/A") else None
        else:
            # Unknown field → put in extra_fields
            if val and val not in ("-", "N/A"):
                extra[key] = val

    if extra:
        row["extra_fields"] = json.dumps(extra, ensure_ascii=False)

    return row


# ---------------------------------------------------------------------------
# Category: Recipes (Crafting/*.md)
# ---------------------------------------------------------------------------
def parse_recipes_from_md(body: str, crafting_type: str, source_file: str) -> List[Dict[str, Any]]:
    """Parse recipe blocks from Crafting markdown files."""
    recipes = []
    # Split by ### Recipe: headings
    recipe_blocks = re.split(r"(?=^### Recipe:)", body, flags=re.MULTILINE)

    for block in recipe_blocks:
        block = block.strip()
        if not block.startswith("### Recipe:"):
            continue

        # Parse heading line: ### Recipe: Name, Product info
        heading_match = re.match(r"^### Recipe:\s*(.+?)(?:\r?\n|$)", block)
        if not heading_match:
            continue

        heading_text = heading_match.group(1).strip()

        # Split heading: name may be followed by product info after comma
        # Example: "Prepare Baguette, Baguette ×1"
        parts = heading_text.split(",", 1)
        recipe_name = parts[0].strip()
        product = parts[1].strip() if len(parts) > 1 else None

        recipe = {
            "name": recipe_name,
            "product": product,
            "crafting_type": crafting_type,
            "source_file": source_file,
        }

        # Parse - **Key:** Value lines
        for line in block.splitlines()[1:]:
            kv_match = re.match(r"^-\s+\*\*(.+?):\*\*\s*(.*)", line)
            if kv_match:
                key = kv_match.group(1).strip()
                val = kv_match.group(2).strip()

                if key == "Ingredients":
                    recipe["ingredients"] = val
                elif key == "Tools":
                    recipe["tools"] = val
                elif key in ("Recipes", "Skills"):
                    recipe["skill_required"] = val
                elif key == "Workstation":
                    recipe["workstation"] = val
                elif key == "XP":
                    recipe["xp_gained"] = val

        recipes.append(recipe)

    return recipes


# ---------------------------------------------------------------------------
# Category: Locations (Locations/*.md business listings)
# ---------------------------------------------------------------------------
def parse_locations_from_md(
    body: str, area: str, source_file: str,
) -> List[Dict[str, Any]]:
    """Parse business listing records from Location markdown files."""
    locations = []
    current_district = None

    records = parse_records_from_md(body)
    # Also track district from ## headings
    for line in body.splitlines():
        h2_match = re.match(r"^##\s+(.+)$", line)
        if h2_match:
            section = h2_match.group(1).strip()
            if section not in ("Overview", "Businesses", "See also",
                               "Gallery", "References", "Navigation"):
                current_district = section

    # Re-parse with district tracking
    current_district = None
    for line in body.splitlines():
        h2_match = re.match(r"^##\s+(.+)$", line)
        if h2_match:
            current_district = h2_match.group(1).strip()

    # Use parsed records
    for rec in records:
        if "Business" in rec or "Type" in rec or "Coordinates" in rec:
            loc = {
                "name": rec.get("Business", rec.get("_heading", "Unknown")),
                "business_type": rec.get("Type"),
                "coordinates": rec.get("Coordinates"),
                "area": area,
                "district": rec.get("_section"),
                "source_file": source_file,
            }
            locations.append(loc)

    return locations


# ---------------------------------------------------------------------------
# Trait-specific parsing (Player/Trait.md)
# ---------------------------------------------------------------------------

# Trait category markers
TRAIT_SECTIONS = {"Positives", "Negatives", "Occupation exclusive",
                  "Removed traits", "Future traits"}


def parse_traits_from_md(body: str, source_file: str) -> List[Dict[str, Any]]:
    """Parse trait records, tagging each with positive/negative."""
    records = parse_records_from_md(body)
    items = []
    for rec in records:
        # Skip headings that are just section names
        heading = rec.get("_heading", "")
        if heading in TRAIT_SECTIONS or heading in ("Positives", "Negatives",
                                                     "List of traits", "Overview"):
            continue
        if "Name" not in rec and "Points" not in rec and "Description" not in rec:
            continue

        row = record_to_item_row(rec, "Player", "Trait", source_file)

        # Determine trait_type from section context
        section = rec.get("_section", "")
        if "Positive" in section:
            row["sub_category"] = "Trait_Positive"
        elif "Negative" in section:
            row["sub_category"] = "Trait_Negative"
        elif "Occupation" in section:
            row["sub_category"] = "Trait_Occupation"
        else:
            row["sub_category"] = "Trait"

        items.append(row)
    return items


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------
class MDToSQLiteConverter:
    """Orchestrates parsing MD files and inserting into SQLite."""

    def __init__(self, docs_dir: str, db_path: str, verbose: bool = False):
        self.docs_dir = Path(docs_dir)
        self.db_path = db_path
        self.verbose = verbose
        self.stats = {
            "files_processed": 0,
            "items_inserted": 0,
            "recipes_inserted": 0,
            "locations_inserted": 0,
            "errors": [],
        }

    def run(self):
        """Main entry point."""
        logger.info(f"📂 Source: {self.docs_dir}")
        logger.info(f"💾 Database: {self.db_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Delete existing DB to rebuild fresh
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            logger.info("🗑️  Removed old database")

        # Create DB and schema
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA_SQL)
        conn.commit()

        # Process all categories
        self._process_items(conn)
        self._process_recipes(conn)
        self._process_locations(conn)
        self._process_traits(conn)

        conn.commit()
        conn.close()

        self._print_report()

    def _read_file(self, filepath: Path) -> str:
        """Read file with encoding fallback."""
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return filepath.read_text(encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return filepath.read_text(encoding="utf-8", errors="replace")

    # ---- Items (Weapons, Food, Clothing, Equipment, Storage, Comfort, etc.)
    def _process_items(self, conn: sqlite3.Connection):
        """Process all item-type categories."""
        for category_dir in sorted(self.docs_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            cat_name = category_dir.name
            # Skip categories handled separately
            if cat_name in ("Crafting", "Locations", "Lore", "Other",
                            "Environment"):
                continue
            # Player handled separately for traits
            if cat_name == "Player":
                continue

            for md_file in sorted(category_dir.glob("*.md")):
                try:
                    self._process_item_file(conn, md_file, cat_name)
                except Exception as e:
                    self.stats["errors"].append(f"{md_file.name}: {e}")
                    logger.error(f"❌ Error processing {md_file.name}: {e}")

    def _process_item_file(self, conn: sqlite3.Connection, filepath: Path, cat_name: str):
        """Parse a single item-type MD file."""
        text = self._read_file(filepath)
        meta, body = parse_frontmatter(text)

        sub_category = meta.get("type", filepath.stem)
        source_file = filepath.name

        records = parse_records_from_md(body)
        count = 0

        for rec in records:
            # Skip records that are just section headers (no key-value data)
            data_keys = [k for k in rec if not k.startswith("_")]
            if len(data_keys) < 2:
                continue

            row = record_to_item_row(rec, cat_name, sub_category, source_file)
            self._insert_item(conn, row)
            count += 1

        self.stats["files_processed"] += 1
        self.stats["items_inserted"] += count
        if self.verbose:
            logger.info(f"  📄 {filepath.name}: {count} items")

    def _insert_item(self, conn: sqlite3.Connection, row: Dict[str, Any]):
        """Insert a single item row into the database."""
        columns = [
            "item_id", "name", "category", "sub_category", "source_file",
            "encumbrance", "min_damage", "max_damage", "door_damage", "tree_damage",
            "min_range", "max_range", "attack_speed", "crit_chance", "crit_multiplier",
            "knockback", "max_condition", "condition_lower_chance", "average_condition",
            "hunger", "thirst", "unhappiness", "boredom",
            "fresh_days", "rotten_days", "cooked_mins", "burned_mins",
            "body_location", "movement_speed", "attack_speed_mod",
            "bite_defense", "scratch_defense", "bullet_defense",
            "insulation", "wind_resistance", "neck_protection", "body_parts_protected",
            "size_tiles", "capacity", "crafting_surface", "bed_type",
            "skill_required", "tools_required",
            "types", "equip_category",
            "points", "description", "effects",
            "equipped", "spice", "extra_fields",
        ]
        values = [row.get(col) for col in columns]
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)
        conn.execute(f"INSERT INTO items ({col_str}) VALUES ({placeholders})", values)

    # ---- Recipes
    def _process_recipes(self, conn: sqlite3.Connection):
        """Process Crafting/*.md files."""
        crafting_dir = self.docs_dir / "Crafting"
        if not crafting_dir.exists():
            return

        for md_file in sorted(crafting_dir.glob("*.md")):
            try:
                text = self._read_file(md_file)
                meta, body = parse_frontmatter(text)
                crafting_type = meta.get("type", md_file.stem.replace("(crafting)", ""))
                source_file = md_file.name

                recipes = parse_recipes_from_md(body, crafting_type, source_file)
                for recipe in recipes:
                    conn.execute(
                        """INSERT INTO recipes (name, product, crafting_type, source_file,
                           ingredients, tools, skill_required, workstation, xp_gained)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            recipe.get("name"),
                            recipe.get("product"),
                            recipe.get("crafting_type"),
                            recipe.get("source_file"),
                            recipe.get("ingredients"),
                            recipe.get("tools"),
                            recipe.get("skill_required"),
                            recipe.get("workstation"),
                            recipe.get("xp_gained"),
                        ),
                    )

                self.stats["files_processed"] += 1
                self.stats["recipes_inserted"] += len(recipes)
                if self.verbose:
                    logger.info(f"  📄 {md_file.name}: {len(recipes)} recipes")

            except Exception as e:
                self.stats["errors"].append(f"{md_file.name}: {e}")
                logger.error(f"❌ Error processing {md_file.name}: {e}")

    # ---- Locations
    def _process_locations(self, conn: sqlite3.Connection):
        """Process Locations/*.md files."""
        loc_dir = self.docs_dir / "Locations"
        if not loc_dir.exists():
            return

        for md_file in sorted(loc_dir.glob("*.md")):
            try:
                text = self._read_file(md_file)
                meta, body = parse_frontmatter(text)
                area = meta.get("title", md_file.stem)
                source_file = md_file.name

                locations = parse_locations_from_md(body, area, source_file)
                for loc in locations:
                    conn.execute(
                        """INSERT INTO locations (name, business_type, coordinates,
                           area, district, source_file)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            loc.get("name"),
                            loc.get("business_type"),
                            loc.get("coordinates"),
                            loc.get("area"),
                            loc.get("district"),
                            loc.get("source_file"),
                        ),
                    )

                self.stats["files_processed"] += 1
                self.stats["locations_inserted"] += len(locations)
                if self.verbose:
                    logger.info(f"  📄 {md_file.name}: {len(locations)} locations")

            except Exception as e:
                self.stats["errors"].append(f"{md_file.name}: {e}")
                logger.error(f"❌ Error processing {md_file.name}: {e}")

    # ---- Traits (Player/Trait.md)
    def _process_traits(self, conn: sqlite3.Connection):
        """Process Player/Trait.md specifically."""
        trait_file = self.docs_dir / "Player" / "Trait.md"
        if not trait_file.exists():
            return

        try:
            text = self._read_file(trait_file)
            _, body = parse_frontmatter(text)
            traits = parse_traits_from_md(body, "Trait.md")

            for row in traits:
                self._insert_item(conn, row)

            self.stats["files_processed"] += 1
            self.stats["items_inserted"] += len(traits)
            if self.verbose:
                logger.info(f"  📄 Trait.md: {len(traits)} traits")

        except Exception as e:
            self.stats["errors"].append(f"Trait.md: {e}")
            logger.error(f"❌ Error processing Trait.md: {e}")

    # ---- Report
    def _print_report(self):
        """Print conversion statistics."""
        s = self.stats
        print("\n" + "=" * 60)
        print("📊 CONVERSION REPORT")
        print("=" * 60)
        print(f"  Files processed:    {s['files_processed']}")
        print(f"  Items inserted:     {s['items_inserted']}")
        print(f"  Recipes inserted:   {s['recipes_inserted']}")
        print(f"  Locations inserted: {s['locations_inserted']}")
        print(f"  Total records:      {s['items_inserted'] + s['recipes_inserted'] + s['locations_inserted']}")
        print(f"  Errors:             {len(s['errors'])}")
        if s["errors"]:
            print("\n⚠️  Errors:")
            for err in s["errors"]:
                print(f"    - {err}")
        print("=" * 60)
        print(f"💾 Database saved to: {self.db_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert PZ Wiki Markdown files to SQLite database"
    )
    parser.add_argument(
        "--docs", default=DEFAULT_DOCS_DIR,
        help=f"Path to PZ docs directory (default: {DEFAULT_DOCS_DIR})"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB_PATH,
        help=f"Output SQLite database path (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print details for each file processed"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    converter = MDToSQLiteConverter(args.docs, args.db, args.verbose)
    converter.run()


if __name__ == "__main__":
    main()
