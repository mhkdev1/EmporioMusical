"""Function tests for tools.py. No LLM involved, so they run offline in a couple of seconds.

Run from the project root:  pytest -v
"""
import shutil
import time

import pandas as pd
import pytest

from src import tools


# ---------- normalize

@pytest.mark.parametrize("text, expected", [
    ("Violões Elétricos", "violoes eletricos"),
    ("AÇÃO", "acao"),
    ("violao", "violao"),
    (123, "123"),
])
def test_normalize(text, expected):
    assert tools.normalize(text) == expected


# ---------- load_csv / load_tables

def test_load_csv_keeps_accents():
    assert "Violões" in tools.load_csv("categories")["name"].tolist()


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
    assert list(tools.load_policy()) == list(tools.POLICY_HEADINGS)


def test_policy_text_is_clean():
    text = " ".join(tools.load_policy().values())
    assert "Página" not in text
    assert "Manual de Políticas e Procedimentos" not in text  # repeated page header removed


@pytest.mark.parametrize("section, snippet", [
    ("about", "Rua 14 de Maio, 3200"),  # cover page joined to "about"
    ("hours", "09:00 às 18:00"),
    ("returns", "7 (sete) dias"),
])
def test_policy_content_lands_in_the_right_section(section, snippet):
    assert snippet in tools.load_policy()[section]


def test_returns_section_stops_before_warranty():
    assert "Garantia" not in tools.load_policy()["returns"]


# ---------- tables() / policy() reload

def test_tables_are_cached_while_files_dont_change():
    assert tools.tables() is tools.tables()
    assert tools.policy() is tools.policy()


def test_tables_reload_when_a_csv_changes(tmp_path, monkeypatch):
    # Work on a copy of the data so the real CSVs are never touched.
    for f in tools.CSV_FILES:
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.setattr(tools, "DATA", tmp_path)
    monkeypatch.setattr(tools, "CSV_FILES", [tmp_path / f.name for f in tools.CSV_FILES])

    assert tools.tables()["orders"].loc[0, "status"] == "delivered"

    orders_csv = tools.CSV_FILES[2]
    time.sleep(1)
    orders_csv.write_text(
        orders_csv.read_text(encoding="utf-8").replace("1,3,2025-10-15,delivered", "1,3,2025-10-15,cancelled"),
        encoding="utf-8",
    )
    assert tools.tables()["orders"].loc[0, "status"] == "cancelled"