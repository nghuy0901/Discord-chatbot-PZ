"""Item de-duplication: ingestion merge + tool-layer name disambiguation."""

import json
import os
import sqlite3

import pytest

from knowledge.structured.pz_tools import _disambiguate_names, search_items
from scripts.convert_md_to_sqlite import MDToSQLiteConverter, SCHEMA_SQL

DB_PATH = os.path.join("knowledge", "structured", "pz_data.db")


# --------------------------------------------------------------------------- #
#  Ingestion merge (unit — in-memory DB)
# --------------------------------------------------------------------------- #
def _mem_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    return conn


def _rows(conn):
    cur = conn.execute("SELECT * FROM items")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def test_merge_collapses_exact_repeats_and_multifacet():
    conn = _mem_db()
    # exact repeat of the same (item_id, source_file) x3 (the Tools.md facet)
    for _ in range(3):
        conn.execute(
            "INSERT INTO items (item_id,name,category,sub_category,source_file,encumbrance)"
            " VALUES ('Base.X','X','Equipment','Tools','Tools.md',2.0)"
        )
    # a richer facet for the same item_id from a different file (has damage)
    conn.execute(
        "INSERT INTO items (item_id,name,category,sub_category,source_file,max_damage,min_damage)"
        " VALUES ('Base.X','X','Weapons','Axes','Axes.md',3.0,1.0)"
    )
    # two identical trait rows with no item_id
    for _ in range(2):
        conn.execute(
            "INSERT INTO items (item_id,name,category,sub_category,points,description)"
            " VALUES (NULL,'Brave','Player','Trait',4,'no fear')"
        )
    conn.commit()

    conv = MDToSQLiteConverter(docs_dir=".", db_path=":memory:")
    conv._dedup_and_merge_items(conn)

    rows = _rows(conn)
    xs = [r for r in rows if r["item_id"] == "Base.X"]
    assert len(xs) == 1                       # all 4 X facets collapsed to one
    x = xs[0]
    assert x["max_damage"] == 3.0             # weapon stat preserved
    assert x["encumbrance"] == 2.0            # tool stat coalesced in
    assert x["sub_category"] == "Axes"        # richest facet wins

    traits = [r for r in rows if r["name"] == "Brave"]
    assert len(traits) == 1                   # exact-duplicate trait deduped
    assert len(rows) == 2


# --------------------------------------------------------------------------- #
#  Tool-layer name disambiguation (unit)
# --------------------------------------------------------------------------- #
def test_disambiguate_appends_item_id_on_name_clash():
    out = _disambiguate_names([
        {"name": "Wood Axe", "item_id": "Base.WoodAxe"},
        {"name": "Wood Axe", "item_id": "Base.WoodAxeForged"},
        {"name": "Pickaxe", "item_id": "Base.PickAxe"},
    ])
    names = [r["name"] for r in out]
    assert "Wood Axe (WoodAxe)" in names
    assert "Wood Axe (WoodAxeForged)" in names
    assert "Pickaxe" in names                 # unique name left untouched


# --------------------------------------------------------------------------- #
#  Real shipped DB integrity (regression guard)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="pz_data.db not present")
def test_shipped_db_has_no_duplicate_item_ids():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    dupes = con.execute(
        "SELECT COUNT(*) FROM (SELECT item_id FROM items "
        "WHERE item_id IS NOT NULL GROUP BY item_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    con.close()
    assert dupes == 0


@pytest.mark.skipif(not os.path.exists(DB_PATH), reason="pz_data.db not present")
def test_search_items_axes_are_distinct_and_disambiguated():
    data = json.loads(search_items(sub_category="Axes", sort_by="max_damage", limit=8))
    items = data["items"]
    ids = [it.get("item_id") for it in items]
    assert len(ids) == len(set(ids))          # no duplicate item_id in output
    # the top weapon-facet Wood Axe must carry real damage, not the NULL tools facet
    wood = [it for it in items if "Wood Axe" in it["name"]]
    assert wood and all(w.get("max_damage") for w in wood)
    if len(wood) > 1:                         # name clash → disambiguated
        assert all("(" in w["name"] for w in wood)
