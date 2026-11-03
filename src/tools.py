"""Store data and business rules.

Loads the six CSV files and the policy manual (PDF) from `src/data`, applies the store's rules (prices, PIX
discount, installments, availability, delivery dates, login) and answers the lookups the agent's tools use.

Everything here is plain Python and pandas. Nothing in this module depends on LangChain or on a model.
"""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from datetime import date



PIX_DISCOUNT = 0.05  #5% off the list price
"""Discount for paying with PIX (manual 3). Not applied on top of a promotional price (manual 6.2)."""

MAX_RESULTS = 10  # keeps the tool output short for the model
"""Most products a search returns, so the model gets a short list to choose from."""

# Words people use to talk about price or to mean "any product". They aren't product names, so I drop
# them from the search term instead of letting "barato" or "menor custo" make the search come back empty.
FILLER_WORDS = {
    "barato", "barata", "baratos", "baratas", "caro", "cara", "caros", "caras", "mais", "menos",
    "menor", "maior", "custo", "preco", "precos", "valor", "produto", "produtos", "equipamento",
    "equipamentos", "item", "itens", "qual", "quais", "o", "a", "os", "as", "de", "do", "da", "um", "uma",
}
"""Words ignored in search terms: price words and generic words like "produto"."""

ANY_CATEGORY = {"todos", "todas", "todo", "toda", "qualquer", "geral", "all"}
"""Category values that mean "no category filter"."""

DATA = Path(__file__).parent / "data"
"""Folder with the CSV files and the policy PDF."""

CSV_PREFIX = "desafio_tecnico_ai_eng - "
"""Common start of the CSV file names."""

CSV_FILES = [DATA / f"{CSV_PREFIX}{name}.csv" for name in
             ("categories", "customers", "orders", "order_items", "products", "promotions")]
"""The six CSV files, watched for changes by `tables()`."""

POLICY_FILE = DATA / "políticas_da_loja.pdf"
"""The store's policy manual, watched for changes by `policy()`."""

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
"""Section key used by the agent -> heading exactly as it appears in the manual, in document order."""


def normalize(text: str) -> str:
    """Lowercase a text and drop its accents, for comparisons only.

    Customers' text and catalog text are compared through this function, so "Violões", "violoes" and
    "VIOLÕES" match. The original text, with accents, is what the customer sees.

    Args:
        text: Any value; it is converted to `str` first.

    Returns:
        The text in lower case, without accents. Example: "Violões Elétricos" -> "violoes eletricos".
    """
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def load_csv(name: str) -> pd.DataFrame:
    """Read one of the store's CSV files.

    Args:
        name: Short table name, e.g. "products" or "orders".

    Returns:
        The table as a DataFrame, read as UTF-8 so accents are kept.
    """
    return pd.read_csv(DATA / f"{CSV_PREFIX}{name}.csv", encoding="utf-8")


def load_tables() -> dict[str, pd.DataFrame]:
    """Read the six CSVs and prepare the columns the lookups filter on.

    Cleaning done here:

    - customers: e-mail in lower case and a `phone_digits` column (digits only), so login accepts any format.
    - orders: `order_date` and `estimated_delivery` parsed as dates (empty for orders not shipped yet).
    - products: `specs` parsed from JSON, the category name joined in, and an accent-free `search_text`.
    - promotions: `is_active` as a boolean.

    Use `tables()` instead of calling this directly; it adds caching.

    Returns:
        A dict with the keys categories, customers, orders, order_items, products and promotions.
    """
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


def load_policy() -> dict[str, str]:
    """Read the policy manual and split it into its 10 sections.

    pypdf returns this PDF with almost one word per line, so whitespace is flattened first. Repeated page
    headers, page numbers and phone numbers broken across lines are cleaned up. The cover page, which holds
    the address and phone numbers, comes before section 1 and is joined to the "about" section.

    Use `policy()` instead of calling this directly; it adds caching.

    Returns:
        A dict mapping each key of `POLICY_HEADINGS` to the text of that section.

    Raises:
        ValueError: If a heading in `POLICY_HEADINGS` is not found in the PDF.
    """
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
    """Cached `load_tables()`; a new `mtime` means a file changed, so it loads again."""
    return load_tables()


@lru_cache(maxsize=1)
def _policy_at(mtime: float) -> dict[str, str]:
    """Cached `load_policy()`; a new `mtime` means the PDF changed, so it loads again."""
    return load_policy()


def tables() -> dict[str, pd.DataFrame]:
    """Current store data, reloaded only when one of the CSVs changed on disk.

    The cache key is the newest modification time among the six files. If any of them is saved again, the
    key changes and all tables are reloaded (products depend on categories). Otherwise the data in memory
    is returned right away.

    Returns:
        The same dict as `load_tables()`.
    """
    return _tables_at(max(f.stat().st_mtime for f in CSV_FILES))


def policy() -> dict[str, str]:
    """Current policy sections, reloaded only when the PDF changed on disk.

    Returns:
        The same dict as `load_policy()`.
    """
    return _policy_at(POLICY_FILE.stat().st_mtime)


# ---------- prices and stock (the model never does this math)


def active_discount(product_id: int) -> tuple[int, str] | None:
    """Best active promotion for a product.

    Promotions don't stack (manual 6.2), so when a product has more than one, the biggest discount wins.

    Args:
        product_id: Product to check.

    Returns:
        `(discount_percent, campaign_name)`, or `None` if the product has no active promotion.
    """
    promos = tables()["promotions"]
    rows = promos[(promos["product_id"] == product_id) & promos["is_active"]]
    if rows.empty:
        return None
    best = rows.loc[rows["discount_percent"].idxmax()]
    return int(best["discount_percent"]), best["description"]


def max_installments(price: float) -> int:
    """Most interest-free installments allowed for a price (manual 3.1).

    The minimum installment is R$ 50 up to 3x, R$ 80 from 4x to 6x and R$ 100 from 7x to 12x. The function
    tries 12x first and goes down until the installment value meets the minimum for that range.

    Args:
        price: Final price in BRL, after any promotion.

    Returns:
        Number of installments, from 1 (no installments) to 12. Examples: 159.90 -> 3, 2199.00 -> 12.
    """
    for n in range(12, 1, -1):
        minimum = 100 if n >= 7 else 80 if n >= 4 else 50
        if price / n >= minimum:
            return n
    return 1


def price_info(product_id: int, list_price: float) -> dict:
    """Every price figure the agent is allowed to quote for a product.

    The model copies these numbers instead of calculating them. The installment table lists every allowed
    option, so a question like "em 9x?" is a lookup; a number missing from the table is not allowed.

    Args:
        product_id: Product, used to find an active promotion.
        list_price: Catalog price in BRL.

    Returns:
        A dict with `list_price_brl`, `final_price_brl`, `pix_price_brl` (5% off, or `None` when there is
        a promotion), `installments` (e.g. `{"1x": 2199.0, ..., "12x": 183.25}`) and, when a promotion
        applies, `promotion` and `discount_percent`.
    """
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
        # Every installment option the customer can pick, so "em 6x?" or "em 9x?" is a lookup, never a division.
    options = {f"{n}x": round(final / n, 2) for n in range(1, max_installments(final) + 1)}
    return {**info, "installments": options}


def availability(status: str, stock: int) -> str:
    """Turn the catalog status and the stock into one availability value.

    Args:
        status: Catalog status: "active", "discontinued" or "coming_soon".
        stock: Units in stock.

    Returns:
        "in_stock", "out_of_stock", "discontinued" or "coming_soon".
    """
    if status != "active":
        return status
    return "in_stock" if stock > 0 else "out_of_stock"

# ---------- catalog, promotions and policy lookups


def search_products(category: str | None = None, max_price: float | None = None, term: str | None = None,
                    sort: str = "cheapest") -> dict:
    """Search the catalog. All filters are optional and can be combined.

    Every word of `term` must appear in the product's name, description or category, ignoring accents and
    case; generic words in `FILLER_WORDS` are skipped. `max_price` is checked against the final price, so a
    product on sale can fit under the limit. Results put in-stock products first, then sort by price.

    Args:
        category: Category name ("Violões", "violoes"...). Values in `ANY_CATEGORY` mean no filter.
        max_price: Highest final price in BRL.
        term: Brand, model or feature words, e.g. "takamine gd20" or "nylon".
        sort: "cheapest" or "most_expensive".

    Returns:
        `{"total_found": int, "products": [...]}` with at most `MAX_RESULTS` short rows (`product_id`,
        `name`, `category`, `availability`, `final_price_brl` and, on sale, `discount_percent`). When
        nothing found is in stock, also `in_stock_alternatives` from `similar_in_stock()`.
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
        info = price_info(int(p.product_id), float(p.price_brl))
        item = {
            "product_id": int(p.product_id),
            "name": p.name,
            "category": p.category,
            "availability": availability(p.status, p.stock_quantity),
            "final_price_brl": info["final_price_brl"],
        }
        if "promotion" in info:
            item["discount_percent"] = info["discount_percent"]
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
    """In-stock products from the same category, closest in price to the given one.

    Gives the model real alternatives to suggest when a product is unavailable, so it doesn't make
    one up.

    Args:
        product: A search row with at least `category` and `final_price_brl`.
        limit: How many alternatives to return.

    Returns:
        Up to `limit` dicts with `product_id`, `name` and `final_price_brl`, closest price first.
    """
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

def get_product(product_id: int) -> dict:
    """Everything about one product.

    Search returns short rows; this returns the full detail of the product the customer is talking
    about, so follow-up questions reuse the same `product_id`.

    Args:
        product_id: Product to look up.

    Returns:
        `product_id`, `name`, `category`, `availability`, `stock_quantity`, every field of `price_info()`,
        `description` and `specs`. If the id doesn't exist, `{"error": ...}`.
    """
    products = tables()["products"]
    row = products[products["product_id"] == int(product_id)]
    if row.empty:
        return {"error": f"No product with id {product_id}."}
    p = row.iloc[0]
    return {
        "product_id": int(p["product_id"]),
        "name": p["name"],
        "category": p["category"],
        "availability": availability(p["status"], p["stock_quantity"]),
        "stock_quantity": int(p["stock_quantity"]),
        **price_info(int(p["product_id"]), float(p["price_brl"])),
        "description": p["description"],
        "specs": p["specs"],
    }


def active_promotions(product_id: int | None = None) -> list[dict]:
    """Active promotions with their prices.

    Args:
        product_id: Only this product's promotions. `None` returns all of them.

    Returns:
        A list of dicts with `product_id`, `name`, `availability` and every field of `price_info()`.
    """
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
    """Text of one section of the policy manual.

    Args:
        section: A key of `POLICY_HEADINGS`, e.g. "hours" or "returns".

    Returns:
        The section text. For an unknown key, a message listing the valid keys, so the model can retry.
    """
    sections = policy()
    if section not in sections:
        return f"Unknown section '{section}'. Available: {', '.join(sections)}."
    return sections[section]



# ---------- login, orders and the customer's own account

NOT_LOGGED_IN = {"error": "Customer is not logged in. Ask them to type /login followed by their e-mail or phone."}
"""Returned by order and account lookups when nobody is logged in."""

# Same answer for "this order doesn't exist" and "this order belongs to someone else",
# so nobody can find out which order numbers exist by trying them.
ORDER_NOT_FOUND = {"error": "No order with this number for the logged-in customer. Ask them to check the number."}
"""Returned for a missing order and for someone else's order alike, so order numbers can't be probed."""


def duplicate_contacts() -> dict[str, list[str]]:
    """E-mails and phones that show up in more than one customer record.

    A data-quality check: in the provided data, customers 11 and 43 share one e-mail.

    Returns:
        `{"emails": [...], "phones": [...]}`, each list sorted.
    """
    c = tables()["customers"]
    return {
        "emails": sorted(c.loc[c["email"].duplicated(keep=False), "email"].unique()),
        "phones": sorted(c.loc[c["phone_digits"].duplicated(keep=False), "phone_digits"].unique()),
    }


def _matches(customers: pd.DataFrame, email_or_phone: str) -> pd.Series:
    """Which customers match an e-mail or a phone.

    The e-mail must match exactly (ignoring case and spaces). A phone matches when the stored number ends
    with the digits typed, so it works with or without the area code; at least 8 digits are required.

    Args:
        customers: Customers table, or a subset of it.
        email_or_phone: What the customer typed.

    Returns:
        A boolean Series, True for each matching row.
    """
    given = email_or_phone.strip().lower()
    digits = "".join(ch for ch in given if ch.isdigit())
    by_email = customers["email"] == given
    by_phone = customers["phone_digits"].str.endswith(digits) if len(digits) >= 8 else False
    return by_email | by_phone


def find_customer(email_or_phone: str, confirm: str | None = None) -> dict:
    """Find the customer behind a login attempt. Used by `/login` only, never exposed to the model.

    When the contact matches more than one record (customers 11 and 43 share an e-mail), the CLI asks for
    the other contact and calls this again with it in `confirm`. The confirmation is only looked for among
    the records the first contact matched, so another person's phone can't be used.

    Args:
        email_or_phone: E-mail or phone typed after `/login`.
        confirm: The second contact, when the first one was ambiguous.

    Returns:
        `{"status": "ok", "customer_id": int, "name": str}` for exactly one match,
        `{"status": "need_more", "ask_for": "phone" | "email"}` for an ambiguous first contact, or
        `{"status": "not_found"}` otherwise.
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
    """One order of the logged-in customer.

    The order number and the owner are filtered together, so another customer's order is simply not
    found. Day counts are computed here because the model is unreliable at date math.

    Args:
        order_id: Order number.
        customer_id: Logged-in customer, or `None` when nobody is logged in.

    Returns:
        `order_id`, `status`, `order_date`, `days_since_order`, `estimated_delivery`,
        `days_since_estimated_delivery`, `delivery_overdue` (shipped and past the estimate),
        `tracking_code`, `payment_method`, `total_brl`, `items` and `notes`. Otherwise `NOT_LOGGED_IN`
        or `ORDER_NOT_FOUND`.
    """
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
    """Personal data and order list of the logged-in customer.

    Args:
        customer_id: Logged-in customer, or `None` when nobody is logged in.

    Returns:
        `name`, `email`, `phone`, `city` and `orders` (newest first, each with `order_id`, `order_date`,
        `status` and `total_brl`), or `NOT_LOGGED_IN`.
    """
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

