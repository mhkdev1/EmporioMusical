#Store data/ CSVs, policy manual and the tools the agent can call.

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

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

