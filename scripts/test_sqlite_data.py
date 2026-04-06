"""Test SQLite data quality with real-world queries."""
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")

def fmt(val, width=5, decimals=1, suffix=""):
    """Safe format for nullable numeric fields."""
    if val is None:
        return f"{'N/A':>{width+decimals+1}}{suffix}"
    return f"{val:>{width}.{decimals}f}{suffix}"

db = sqlite3.connect(r"knowledge\structured\pz_data.db")
db.row_factory = sqlite3.Row

print("=" * 70)
print("TEST 1: Axe nao manh nhat? (ORDER BY max_damage DESC)")
print("=" * 70)
rows = db.execute("""
    SELECT name, max_damage, min_damage, attack_speed, encumbrance, item_id
    FROM items
    WHERE sub_category = 'Axes' AND max_damage IS NOT NULL
    ORDER BY max_damage DESC
    LIMIT 10
""").fetchall()
for i, r in enumerate(rows, 1):
    print(f"  {i:2d}. {r['name']:35s} | max_dmg: {r['max_damage']:5.1f} | spd: {str(r['attack_speed']):>5s} | enc: {str(r['encumbrance']):>4s} | {r['item_id']}")

print(f"\n{'=' * 70}")
print("TEST 2: So sanh Axe vs Hatchet (WHERE name IN ...)")
print("=" * 70)
rows = db.execute("""
    SELECT name, max_damage, min_damage, attack_speed, knockback, encumbrance, item_id
    FROM items
    WHERE sub_category = 'Axes'
    AND (name = 'Axe' OR name LIKE 'Hatchet%' OR name LIKE 'Hand Axe%')
    AND max_damage IS NOT NULL
    ORDER BY max_damage DESC
""").fetchall()
for r in rows:
    print(f"  {r['name']:35s} | max: {r['max_damage']:5.1f} | min: {str(r['min_damage']):>5s} | spd: {str(r['attack_speed']):>5s} | kb: {str(r['knockback']):>5s}")

print(f"\n{'=' * 70}")
print("TEST 3: Food giam hunger nhieu nhat (ORDER BY hunger ASC)")
print("=" * 70)
rows = db.execute("""
    SELECT name, hunger, thirst, encumbrance, item_id
    FROM items
    WHERE category = 'Food' AND hunger IS NOT NULL
    ORDER BY hunger ASC
    LIMIT 10
""").fetchall()
for i, r in enumerate(rows, 1):
    print(f"  {i:2d}. {r['name']:35s} | hunger: {r['hunger']:6.1f} | thirst: {str(r['thirst']):>6s} | {r['item_id']}")

print(f"\n{'=' * 70}")
print("TEST 4: Recipe cho Baguette (text search)")
print("=" * 70)
rows = db.execute("""
    SELECT name, product, ingredients, tools, workstation, xp_gained
    FROM recipes
    WHERE name LIKE '%Baguette%' OR product LIKE '%Baguette%'
""").fetchall()
for r in rows:
    print(f"  Recipe: {r['name']}")
    print(f"    Product:     {r['product']}")
    print(f"    Ingredients: {r['ingredients']}")
    print(f"    Tools:       {r['tools']}")
    print(f"    Workstation: {r['workstation']}")
    print(f"    XP:          {r['xp_gained']}")

print(f"\n{'=' * 70}")
print("TEST 5: Armor tot nhat (bite defense, ORDER BY DESC)")
print("=" * 70)
rows = db.execute("""
    SELECT name, bite_defense, scratch_defense, bullet_defense, body_location, item_id
    FROM items
    WHERE sub_category = 'Armor' AND bite_defense IS NOT NULL
    ORDER BY bite_defense DESC
    LIMIT 10
""").fetchall()
for i, r in enumerate(rows, 1):
    bd = fmt(r['bite_defense'], 3, 0, '%')
    sd = fmt(r['scratch_defense'], 3, 0, '%')
    bud = fmt(r['bullet_defense'], 3, 0, '%')
    print(f"  {i:2d}. {r['name']:45s} | bite: {bd} | scratch: {sd} | bullet: {bud}")

print(f"\n{'=' * 70}")
print("TEST 6: Trait positive tot nhat (points)")
print("=" * 70)
rows = db.execute("""
    SELECT name, points, description, effects
    FROM items
    WHERE category = 'Player' AND sub_category LIKE 'Trait%' AND points IS NOT NULL
    ORDER BY points ASC
    LIMIT 10
""").fetchall()
for i, r in enumerate(rows, 1):
    desc = (r['description'] or '')[:60]
    print(f"  {i:2d}. {r['name']:25s} | pts: {r['points']:3.0f} | {desc}")

print(f"\n{'=' * 70}")
print("TEST 7: Location search - Gas Stations in Louisville")
print("=" * 70)
rows = db.execute("""
    SELECT name, business_type, coordinates, district
    FROM locations
    WHERE area = 'Louisville' AND business_type LIKE '%Gas%'
""").fetchall()
for r in rows:
    print(f"  {r['name']:30s} | {str(r['business_type']):>15s} | coords: {r['coordinates']} | {r['district']}")

print(f"\n{'=' * 70}")
print("TEST 8: Data quality - NULL check critical fields")
print("=" * 70)
# Items with no name
null_names = db.execute("SELECT COUNT(*) FROM items WHERE name IS NULL OR name = ''").fetchone()[0]
# Items with no item_id (expected for some categories)
null_ids = db.execute("SELECT COUNT(*) FROM items WHERE item_id IS NULL OR item_id = ''").fetchone()[0]
# Weapons with no damage
null_dmg = db.execute("SELECT COUNT(*) FROM items WHERE category = 'Weapons' AND max_damage IS NULL").fetchone()[0]
total_weapons = db.execute("SELECT COUNT(*) FROM items WHERE category = 'Weapons'").fetchone()[0]
# Food with no hunger
null_hunger = db.execute("SELECT COUNT(*) FROM items WHERE category = 'Food' AND hunger IS NULL").fetchone()[0]
total_food = db.execute("SELECT COUNT(*) FROM items WHERE category = 'Food'").fetchone()[0]
# Recipes with no ingredients
null_ingr = db.execute("SELECT COUNT(*) FROM recipes WHERE ingredients IS NULL").fetchone()[0]
total_recipes = db.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]

print(f"  Items with NULL name:        {null_names}")
print(f"  Items with NULL item_id:     {null_ids} / 5980  (expected: Storage/Comfort have no item_id)")
print(f"  Weapons with NULL max_dmg:   {null_dmg} / {total_weapons}")
print(f"  Food with NULL hunger:       {null_hunger} / {total_food}  (expected: sealed cans have no hunger)")
print(f"  Recipes with NULL ingredients: {null_ingr} / {total_recipes}")

print(f"\n{'=' * 70}")
print("ALL TESTS COMPLETE")
print("=" * 70)

db.close()
