"""Function tests for tools.py. No LLM involved, so they run offline in a couple of seconds.

Run from the project root:  pytest -v
"""
import shutil
import time
from datetime import date

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

   
   
 # ---------- prices and stock

@pytest.mark.parametrize("price, expected", [
    (40, 1),       # below R$ 50, no installments
    (150, 3),      # 3x of 50
    (159.9, 3),    # 4x would be 39.97, under the R$ 80 minimum
    (480, 6),      # 6x of 80
    (699.9, 6),    # 7x would be 99.99, under the R$ 100 minimum
    (700, 7),
    (1200, 12),
    (9999, 12),    # never more than 12x
])
def test_max_installments(price, expected):
    assert tools.max_installments(price) == expected


def test_active_discount():
    assert tools.active_discount(94) == (8, "Desconto Exclusivo")
    assert tools.active_discount(95) is None  # Takamine GD20 has no active promotion


def test_price_info_without_promotion_has_pix():
    assert tools.price_info(95, 2199.0) == {
        "list_price_brl": 2199.0, "final_price_brl": 2199.0, "pix_price_brl": 2089.05,
        "max_installments": 12, "installment_brl": 183.25,
    }


def test_price_info_with_promotion_has_no_pix():
    info = tools.price_info(94, 5999.0)
    assert info["promotion"] == "Desconto Exclusivo"
    assert info["discount_percent"] == 8
    assert info["final_price_brl"] == 5519.08
    assert info["pix_price_brl"] is None
    assert info["installment_brl"] == round(5519.08 / 12, 2)  # installments use the promo price


@pytest.mark.parametrize("status, stock, expected", [
    ("active", 5, "in_stock"),
    ("active", 0, "out_of_stock"),
    ("discontinued", 3, "discontinued"),
    ("coming_soon", 0, "coming_soon"),
])
def test_availability(status, stock, expected):
    assert tools.availability(status, stock) == expected

    # ---------- catalog, promotions and policy lookups

def test_search_by_name_returns_prices_from_code():
    found = tools.search_products(term="takamine gd20")
    assert found["total_found"] == 1
    gd20 = found["products"][0]
    assert gd20["name"] == "Takamine GD20 Dreadnought Natural"
    assert gd20["final_price_brl"] == 2199.0 and gd20["pix_price_brl"] == 2089.05
    assert gd20["max_installments"] == 12 and gd20["installment_brl"] == 183.25


def test_search_by_category_and_max_price():
    found = tools.search_products(category="Violões", max_price=1000)
    assert found["total_found"] == 14
    assert len(found["products"]) == tools.MAX_RESULTS  # cut at 10
    assert all(p["category"] == "Violões" and p["final_price_brl"] <= 1000 for p in found["products"])


def test_search_puts_in_stock_first_and_cheapest_first():
    products = tools.search_products(category="Violões", max_price=1000)["products"]
    assert all(p["availability"] == "in_stock" for p in products)
    prices = [p["final_price_brl"] for p in products]
    assert prices == sorted(prices)


@pytest.mark.parametrize("a, b", [
    ({"category": "Violões"}, {"category": "violoes"}),
    ({"category": "Violões"}, {"category": "VIOLÕES"}),
    ({"term": "elétrico"}, {"term": "eletrico"}),
])
def test_search_ignores_accents_and_case(a, b):
    assert tools.search_products(**a) == tools.search_products(**b)


def test_search_shows_out_of_stock_products():
    assert tools.search_products(term="GF-3D")["products"][0]["availability"] == "out_of_stock"


@pytest.mark.parametrize("kwargs", [{"term": "xyz"}, {"category": "Pianos de cauda"}, {"max_price": 1}])
def test_search_with_no_match_is_empty(kwargs):
    assert tools.search_products(**kwargs) == {"total_found": 0, "products": []}


def test_active_promotions_lists_four():
    promos = tools.active_promotions()
    assert len(promos) == 4
    assert all(p["pix_price_brl"] is None for p in promos)  # no PIX on top of a promotion


def test_active_promotions_for_one_product():
    taylor = tools.active_promotions(94)
    assert len(taylor) == 1 and taylor[0]["final_price_brl"] == 5519.08
    assert tools.active_promotions(95) == []


def test_get_policy_section():
    assert "09:00 às 18:00" in tools.get_policy_section("hours")
    assert tools.get_policy_section("nope").startswith("Unknown section 'nope'. Available: about, hours")


def test_lookups_return_plain_json():
    import json
    json.dumps([tools.search_products(), tools.active_promotions()], ensure_ascii=False)




# ---------- login, orders and the customer's own account

PEDRO, ANA = 3, 2  # Pedro owns orders 1 and 19, Ana owns order 8


@pytest.mark.parametrize("contact", [
    "pedro.oliveira@jmail.com",
    "  PEDRO.OLIVEIRA@JMAIL.COM ",  # extra spaces and caps
    "(67) 98432-1098",
    "98432-1098",                    # without the area code
    "67984321098",
])
def test_find_customer_by_email_or_phone(contact):
    assert tools.find_customer(contact) == {"customer_id": PEDRO, "name": "Pedro Henrique Oliveira"}


@pytest.mark.parametrize("contact", ["", "x", "9981", "67", "@jmail.com", "pedro", "someone@else.com"])
def test_find_customer_rejects_unknown_or_partial(contact):
    assert tools.find_customer(contact) is None


def test_shared_email_cannot_log_in_but_each_phone_can():
    # Data quirk: customers 11 and 43 share bruno.martins@jmail.com (probably one person registered twice).
    customers = tools.tables()["customers"]
    duplicated = customers[customers["email"].duplicated(keep=False)]["email"].unique().tolist()
    assert duplicated == ["bruno.martins@jmail.com"], "new duplicated e-mails showed up in the data"
    assert tools.find_customer("bruno.martins@jmail.com") is None
    assert tools.find_customer("(67) 98543-2109")["customer_id"] == 11
    assert tools.find_customer("(67) 99456-3456")["customer_id"] == 43


def test_every_customer_can_log_in_by_phone():
    for c in tools.tables()["customers"].itertuples():
        assert tools.find_customer(c.phone)["customer_id"] == c.customer_id


def test_orders_and_account_need_login():
    assert tools.get_order(1, None) == tools.NOT_LOGGED_IN
    assert tools.get_my_account(None) == tools.NOT_LOGGED_IN


def test_owner_sees_the_order():
    order = tools.get_order(1, PEDRO)
    assert order["status"] == "delivered"
    assert order["tracking_code"] == "BRAB1234567BR"
    assert order["items"] == [{"name": "Martin D-28 Dreadnought Natural", "quantity": 1}]


def test_nobody_sees_someone_elses_order():
    """Every order against every customer: only the owner gets it back."""
    t = tools.tables()
    for order in t["orders"].itertuples():
        for customer in t["customers"].itertuples():
            result = tools.get_order(order.order_id, customer.customer_id)
            if customer.customer_id == order.customer_id:
                assert result["order_id"] == order.order_id
            else:
                assert result == tools.ORDER_NOT_FOUND, (order.order_id, customer.customer_id)


def test_not_found_looks_the_same_whether_the_order_exists_or_not():
    assert tools.get_order(1, ANA) == tools.get_order(999, ANA) == tools.ORDER_NOT_FOUND


def test_delivery_dates_are_computed_in_code():
    order8 = tools.get_order(8, ANA)  # shipped, expected on 2026-02-28
    today = date.today()
    assert order8["days_since_estimated_delivery"] == (today - date(2026, 2, 28)).days
    assert order8["delivery_overdue"] == (today > date(2026, 2, 28))
    assert tools.get_order(1, PEDRO)["delivery_overdue"] is False  # delivered orders are never late


def test_my_account_has_only_my_data():
    account = tools.get_my_account(PEDRO)
    assert account["name"] == "Pedro Henrique Oliveira"
    assert account["email"] == "pedro.oliveira@jmail.com"
    assert sorted(o["order_id"] for o in account["orders"]) == [1, 19]


def test_every_account_lists_exactly_its_own_orders():
    t = tools.tables()
    for c in t["customers"].itertuples():
        own = set(t["orders"][t["orders"]["customer_id"] == c.customer_id]["order_id"])
        assert {o["order_id"] for o in tools.get_my_account(c.customer_id)["orders"]} == own