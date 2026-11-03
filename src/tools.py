#Store data/ CSVs, policy manual and the tools the agent can call.

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from datetime import date



PIX_DISCOUNT = 0.05  #5% off the list price
MAX_RESULTS = 10  # keeps the tool output short for the model

# Words people use to talk about price or to mean "any product". They aren't product names, so I drop
# them from the search term instead of letting "barato" or "menor custo" make the search come back empty.
FILLER_WORDS = {
    "barato", "barata", "baratos", "baratas", "caro", "cara", "caros", "caras", "mais", "menos",
    "menor", "maior", "custo", "preco", "precos", "valor", "produto", "produtos", "equipamento",
    "equipamentos", "item", "itens", "qual", "quais", "o", "a", "os", "as", "de", "do", "da", "um", "uma",
}
ANY_CATEGORY = {"todos", "todas", "todo", "toda", "qualquer", "geral", "all"}

DATA = Path(__file__).parent / "data"
CSV_PREFIX = "desafio_tecnico_ai_eng - "
CSV_FILES = [DATA / f"{CSV_PREFIX}{name}.csv" for name in
             ("categories", "customers", "orders", "order_items", "products", "promotions")]
POLICY_FILE = DATA / "políticas_da_loja.pdf"

# Top-level headings of the manual, in the order they appear.
# I list them by hand because section 7.2 has its own numbered steps

POLICY_HEADINGS = {
    "about": "1. Sobre a Empório da Música",
    "hours": "2. Horário de Funcionamento",
    "payment": "3. Formas de Pagamento",
    "returns": "4. Política de Trocas e Devoluções",
    "shipping": "5. Política de Frete e Entregas",
    "promotions": "6. Promoções e Descontos",
    "service": "7. Atendimento via WhatsApp",
    "warranty": "8. Garantia",
    "privacy": "9. Privacidade e Proteção de Dados",
    "final": "10. Disposições Finais",
}


#Lowercase and drop accents used to compare text.
def normalize(text: str) -> str:

    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{CSV_PREFIX}{name}.csv", encoding="utf-8")



#Read the six CSVs and clean the columns the tools filter on.
def load_tables() -> dict[str, pd.DataFrame]:
    
    categories = load_csv("categories")
    customers = load_csv("customers")
    orders = load_csv("orders")
    order_items = load_csv("order_items")
    products = load_csv("products")
    promotions = load_csv("promotions")

    # Login compares these, so people can type the phone with or without (67), spaces, dashes.
    customers["email"] = customers["email"].str.strip().str.lower()
    customers["phone_digits"] = customers["phone"].str.replace(r"\D", "", regex=True)

    # Orders that haven't shipped yet have no estimated_delivery, those stay empty (NaT).
    for col in ("order_date", "estimated_delivery"):
        orders[col] = pd.to_datetime(orders[col])

    # specs comes as a JSON string in the CSV.
    products["specs"] = products["specs"].map(json.loads)
    products = products.merge(
        categories[["category_id", "name"]].rename(columns={"name": "category"}), on="category_id"
    )
    # Accent-free copy of name + description + category, just for searching.
    products["search_text"] = (products["name"] + " " + products["description"] + " " + products["category"]).map(normalize)

    promotions["is_active"] = promotions["is_active"].astype(bool)

    return {
        "categories": categories,
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "promotions": promotions,
    }


#Read the policy PDF and split it into {section_key: text}
def load_policy() -> dict[str, str]:
    
    text = " ".join(page.extract_text() for page in PdfReader(POLICY_FILE).pages)
    # pypdf returns this file with almost one word per line, so I flatten the whitespace first.
    text = re.sub(r"\s+", " ", text)
    text = text.replace("Empório da Música Manual de Políticas e Procedimentos", "")  # page header
    text = re.sub(r"Página \d+", "", text)  # page footer
    text = re.sub(r"(\d) - (\d)", r"\1-\2", text)  # phone numbers split in the tables, "3341 - 4444"
    text = re.sub(r"\s+", " ", text).strip()

    starts = [text.index(heading) for heading in POLICY_HEADINGS.values()] + [len(text)]
    sections = {key: text[starts[i]:starts[i + 1]].strip() for i, key in enumerate(POLICY_HEADINGS)}
    # The cover page (address, phone, e-mail) comes before section 1, so I keep it with "about".
    sections["about"] = text[: starts[0]].strip() + "\n" + sections["about"]
    return sections   

# The CSVs and the manual get updated while the agent is running, so loading them once at startup
# isn't enough. Reading everything on every call is wasteful too (the PDF alone takes ~0.4 s).
# lru_cache keys on the files' modification time: same time, cached data; any file changed, reload.
@lru_cache(maxsize=1)
def _tables_at(mtime: float) -> dict[str, pd.DataFrame]:
    return load_tables()


@lru_cache(maxsize=1)
def _policy_at(mtime: float) -> dict[str, str]:
    return load_policy()


def tables() -> dict[str, pd.DataFrame]:
    """Current store data. Reloads only when one of the CSVs changed on disk."""
    return _tables_at(max(f.stat().st_mtime for f in CSV_FILES))


def policy() -> dict[str, str]:
    """Current policy sections. Reloads only when the PDF changed on disk."""
    return _policy_at(POLICY_FILE.stat().st_mtime)


# ---------- prices and stock (the model never does this math)



    # Best active promotion for a product. promotions don't stack so if a product has more than one I take the biggest.
    
def active_discount(product_id: int) -> tuple[int, str] | None:
    
    promos = tables()["promotions"]
    rows = promos[(promos["product_id"] == product_id) & promos["is_active"]]
    if rows.empty:
        return None
    best = rows.loc[rows["discount_percent"].idxmax()]
    return int(best["discount_percent"]), best["description"]


#  Most interest-free installments allowed for this price (manual 3.1). Minimum installment: R$ 50 up to 3x, R$ 80 from 4x to 6x, R$ 100 from 7x to 12x.   
def max_installments(price: float) -> int:
   
    for n in range(12, 1, -1):
        minimum = 100 if n >= 7 else 80 if n >= 4 else 50
        if price / n >= minimum:
            return n
    return 1


def price_info(product_id: int, list_price: float) -> dict:
    """Every price figure the agent is allowed to quote for a product."""
    promo = active_discount(product_id)
    if promo:
        percent, campaign = promo
        final = round(list_price * (1 - percent / 100), 2)
        info = {"list_price_brl": list_price, "promotion": campaign, "discount_percent": percent,
                "final_price_brl": final, "pix_price_brl": None}  # PIX doesn't apply on top of a promo
    else:
        final = list_price
        info = {"list_price_brl": list_price, "final_price_brl": final,
                "pix_price_brl": round(list_price * (1 - PIX_DISCOUNT), 2)}
    n = max_installments(final)
    return {**info, "max_installments": n, "installment_brl": round(final / n, 2)}


def availability(status: str, stock: int) -> str:
    """in_stock, out_of_stock, discontinued or coming_soon."""
    if status != "active":
        return status
    return "in_stock" if stock > 0 else "out_of_stock"

# ---------- catalog, promotions and policy lookups


def search_products(category: str | None = None, max_price: float | None = None, term: str | None = None,
                    sort: str = "cheapest") -> dict:
    """Products filtered by category, max final price and words in name/description. All optional.

    sort is "cheapest" or "most_expensive".
    """
    products = tables()["products"]
    if category and normalize(category) not in ANY_CATEGORY:
        products = products[products["category"].map(normalize) == normalize(category)]
    if term:
        for word in normalize(term).split():  # every word has to show up somewhere
            if word in FILLER_WORDS:
                continue
            products = products[products["search_text"].str.contains(word, regex=False)]

    results = []
    for p in products.itertuples():
        item = {
            "product_id": int(p.product_id),
            "name": p.name,
            "category": p.category,
            "availability": availability(p.status, p.stock_quantity),
            "stock_quantity": int(p.stock_quantity),
            **price_info(int(p.product_id), float(p.price_brl)),
            "specs": p.specs,
        }
        # max_price is checked on the final price, so a product on sale can fit under the limit
        if max_price is None or item["final_price_brl"] <= max_price:
            results.append(item)


    # In-stock first, then by price. Otherwise the 10-item cut could hide products people can buy.
    sign = -1 if sort == "most_expensive" else 1
    results.sort(key=lambda r: (r["availability"] != "in_stock", sign * r["final_price_brl"]))
    found = {"total_found": len(results), "products": results[:MAX_RESULTS]}
    # Nothing found can be bought: hand over real alternatives, or a small model makes some up.
    if results and all(r["availability"] != "in_stock" for r in results):
        found["in_stock_alternatives"] = similar_in_stock(results[0])
    return found

def similar_in_stock(product: dict, limit: int = 3) -> list[dict]:
    """In-stock products from the same category, closest in price to the given one."""
    products = tables()["products"]
    in_stock = products[(products["category"] == product["category"])
                        & (products["status"] == "active") & (products["stock_quantity"] > 0)]
    candidates = [
        {"product_id": int(p.product_id), "name": p.name,
         "final_price_brl": price_info(int(p.product_id), float(p.price_brl))["final_price_brl"]}
        for p in in_stock.itertuples()
    ]
    candidates.sort(key=lambda c: abs(c["final_price_brl"] - product["final_price_brl"]))
    return candidates[:limit]


def active_promotions(product_id: int | None = None) -> list[dict]:
    """Active promotions with the discounted price, for every product or just one."""
    promos = tables()["promotions"]
    promos = promos[promos["is_active"]]
    if product_id is not None:
        promos = promos[promos["product_id"] == int(product_id)]
    products = tables()["products"].set_index("product_id")

    result = []
    for promo in promos.itertuples():
        p = products.loc[promo.product_id]
        result.append({
            "product_id": int(promo.product_id),
            "name": p["name"],
            "availability": availability(p["status"], p["stock_quantity"]),
            **price_info(int(promo.product_id), float(p["price_brl"])),
        })
    return result


def get_policy_section(section: str) -> str:
    """Text of one section of the manual. Unknown keys get the list of valid ones back."""
    sections = policy()
    if section not in sections:
        return f"Unknown section '{section}'. Available: {', '.join(sections)}."
    return sections[section]



# ---------- login, orders and the customer's own account

NOT_LOGGED_IN = {"error": "Customer is not logged in. Ask them to type /login followed by their e-mail or phone."}
# Same answer for "this order doesn't exist" and "this order belongs to someone else",
# so nobody can find out which order numbers exist by trying them.
ORDER_NOT_FOUND = {"error": "No order with this number for the logged-in customer. Ask them to check the number."}

#find duplicated contacts, so the CLI can ask for the other one when a login is ambiguous
def duplicate_contacts() -> dict[str, list[str]]:
    """E-mails and phones that show up in more than one customer record."""
    c = tables()["customers"]
    return {
        "emails": sorted(c.loc[c["email"].duplicated(keep=False), "email"].unique()),
        "phones": sorted(c.loc[c["phone_digits"].duplicated(keep=False), "phone_digits"].unique()),
    }


def _matches(customers: pd.DataFrame, email_or_phone: str) -> pd.Series:
    """Rows whose e-mail is exactly this, or whose phone ends with these digits (8 or more)."""
    given = email_or_phone.strip().lower()
    digits = "".join(ch for ch in given if ch.isdigit())
    by_email = customers["email"] == given
    by_phone = customers["phone_digits"].str.endswith(digits) if len(digits) >= 8 else False
    return by_email | by_phone


def find_customer(email_or_phone: str, confirm: str | None = None) -> dict:
    """Login lookup, used by /login. Never exposed to the model.

    Returns {"status": "ok", "customer_id", "name"} when exactly one record matches.
    When more than one does (customers 11 and 43 share an e-mail), it returns
    {"status": "need_more", "ask_for": "phone" or "email"} so the CLI can ask for the other contact,
    and the second call passes it in confirm. Anything else is {"status": "not_found"}.
    """
    customers = tables()["customers"]
    found = customers[_matches(customers, email_or_phone)]
    if confirm is not None:
        found = found[_matches(found, confirm)]  # the second contact must belong to one of the first hits
    if len(found) == 1:
        c = found.iloc[0]
        return {"status": "ok", "customer_id": int(c["customer_id"]), "name": c["name"]}
    if len(found) > 1 and confirm is None:
        return {"status": "need_more", "ask_for": "phone" if "@" in email_or_phone else "email"}
    return {"status": "not_found"}

def get_order(order_id: int, customer_id: int | None) -> dict:
    """One order of the logged-in customer: status, tracking, dates and items."""
    if customer_id is None:
        return NOT_LOGGED_IN
    t = tables()
    orders = t["orders"]
    # The owner check is part of the filter: someone else's order just isn't found.
    match = orders[(orders["order_id"] == int(order_id)) & (orders["customer_id"] == customer_id)]
    if match.empty:
        return ORDER_NOT_FOUND
    order = match.iloc[0]

    items = t["order_items"].merge(t["products"], on="product_id")
    items = items[items["order_id"] == order["order_id"]]

    # Date math happens here. Small models are bad at it and called an 8-month-late delivery "recent".
    today = date.today()
    ordered = order["order_date"].date()
    eta = None if pd.isna(order["estimated_delivery"]) else order["estimated_delivery"].date()
    return {
        "order_id": int(order["order_id"]),
        "status": order["status"],  # pending, confirmed, shipped, delivered or cancelled
        "order_date": ordered.isoformat(),
        "days_since_order": (today - ordered).days,
        "estimated_delivery": eta.isoformat() if eta else None,
        "days_since_estimated_delivery": (today - eta).days if eta else None,
        "delivery_overdue": bool(order["status"] == "shipped" and eta is not None and eta < today),
        "tracking_code": None if pd.isna(order["tracking_code"]) else order["tracking_code"],
        "payment_method": order["payment_method"],
        "total_brl": float(order["total_brl"]),
        "items": [{"name": i.name, "quantity": int(i.quantity)} for i in items.itertuples()],
        "notes": None if pd.isna(order["notes"]) else order["notes"],
    }


def get_my_account(customer_id: int | None) -> dict:
    """Personal data and order list of the logged-in customer."""
    if customer_id is None:
        return NOT_LOGGED_IN
    t = tables()
    c = t["customers"][t["customers"]["customer_id"] == customer_id].iloc[0]
    orders = t["orders"][t["orders"]["customer_id"] == customer_id].sort_values("order_date", ascending=False)
    return {
        "name": c["name"],
        "email": c["email"],
        "phone": c["phone"],
        "city": c["city"],
        "orders": [
            {"order_id": int(o.order_id), "order_date": o.order_date.date().isoformat(),
             "status": o.status, "total_brl": float(o.total_brl)}
            for o in orders.itertuples()
        ],
    }

