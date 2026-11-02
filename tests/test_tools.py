"""Function tests for tools.py. No LLM involved, so they run offline in a couple of seconds.

Run from the project root:  python test_tools.py
"""
import time

import pandas as pd

import tools


# ---------- normalize


def test_normalize_accepts_non_strings():
    assert tools.normalize(123) == "123"


# ---------- load_csv / load_tables

def test_load_csv_keeps_accents():
    categories = tools.load_csv("categories")
    assert "Violões" in categories["name"].tolist()


def test_load_tables_returns_all_six_tables():
    t = tools.load_tables()
    assert {k: v.shape for k, v in t.items()} == {
        "categories": (9, 3), "customers": (50, 6), "orders": (20, 9),
        "order_items": (22, 3), "products": (65, 11), "promotions": (25, 5),
    }


def test_customers_contacts_are_normalized():
    c = tools.load_tables()["customers"].iloc[0]
    assert c["phone"] == "(67) 99812-3456"
    assert c["phone_digits"] == "67998123456"
    assert c["email"] == "lucas.mendes@jmail.com"


def test_orders_dates_are_parsed():
    orders = tools.load_tables()["orders"]
    assert orders.loc[0, "order_date"] == pd.Timestamp("2025-10-15")
    assert orders["estimated_delivery"].isna().sum() == 12  # not shipped yet


def test_products_get_category_specs_and_search_text():
    p = tools.load_tables()["products"].iloc[0]
    assert p["name"] == "Yamaha C40 Nylon Natural"
    assert p["category"] == "Violões"  # original text keeps the accent
    assert p["specs"]["top"] == "Spruce"  # JSON string became a dict
    assert p["search_text"].startswith("yamaha c40 nylon natural violao")  # search copy has none


def test_four_promotions_are_active():
    assert tools.load_tables()["promotions"]["is_active"].sum() == 4


# ---------- load_policy

def test_policy_has_ten_sections_in_order():
    sections = tools.load_policy()
    assert list(sections) == list(tools.POLICY_HEADINGS)


def test_policy_text_is_clean():
    text = " ".join(tools.load_policy().values())
    assert "Página" not in text
    assert "Manual de Políticas e Procedimentos" not in text  # repeated page header removed


def test_policy_content_lands_in_the_right_section():
    sections = tools.load_policy()
    assert "Rua 14 de Maio, 3200" in sections["about"]  # cover page joined to "about"
    assert "09:00 às 18:00" in sections["hours"]
    assert "7 (sete) dias" in sections["returns"]
    assert "Garantia" not in sections["returns"]  # returns stops before the warranty section


# ---------- tables() / policy() reload

def test_tables_are_cached_while_files_dont_change():
    assert tools.tables() is tools.tables()
    assert tools.policy() is tools.policy()


def test_tables_reload_when_a_csv_changes():
    orders_csv = tools.CSV_FILES[2]
    original = orders_csv.read_text(encoding="utf-8")
    try:
        assert tools.tables()["orders"].loc[0, "status"] == "delivered"
        time.sleep(1)
        orders_csv.write_text(original.replace("1,3,2025-10-15,delivered", "1,3,2025-10-15,cancelled"), encoding="utf-8")
        assert tools.tables()["orders"].loc[0, "status"] == "cancelled"
    finally:
        orders_csv.write_text(original, encoding="utf-8")  # always put the real data back
    assert tools.tables()["orders"].loc[0, "status"] == "delivered"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}  {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")