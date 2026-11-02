#Data layer: preprocess the raw CSVs and the policy PDF once into a SQLite database that the tools query.

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parent.parent / "data" #path of the data folder
CSV_PREFIX = "desafio_tecnico_ai_eng - "
POLICY_PDF = DATA_DIR / "políticas_da_loja.pdf"
DB_PATH = DATA_DIR / "emporio.db"  # generated file, rebuilt automatically when a source file changes
TABLES = ["categories", "customers", "orders", "order_items", "products", "promotions"]
PRIMARY_KEYS = {"categories": "category_id", "customers": "customer_id", "orders": "order_id",
                "products": "product_id", "promotions": "promotion_id"}

# Top-level headings of the policy manual, in document order -> section key used by the get_policy tool.
# ponytail: hardcoded because the manual is a single fixed document; a generic heading detector would also
# match the numbered steps inside section 7.2 ("1. Saudação", "2. Entendimento"...).
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


#Lower-case and strip accents, used only for matching, never for display.
def normalize(text: str) -> str:
    
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()