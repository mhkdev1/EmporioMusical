# Architecture

Architecture documentation for the Empório da Música customer service agent. The structure follows the
[arc42](https://arc42.org) template, trimmed to what a project of this size needs. Setup instructions live in
the [README](../README.md).

## Contents

1. [Introduction and goals](#1-introduction-and-goals)
2. [Constraints](#2-constraints)
3. [Context and scope](#3-context-and-scope)
4. [Solution strategy](#4-solution-strategy)
5. [Building blocks](#5-building-blocks)
6. [Runtime scenarios](#6-runtime-scenarios)
7. [Deployment](#7-deployment)
8. [Crosscutting concepts](#8-crosscutting-concepts)
9. [Architecture decisions](#9-architecture-decisions)
10. [Quality and testing](#10-quality-and-testing)
11. [Risks and technical debt](#11-risks-and-technical-debt)
12. [Glossary](#12-glossary)
13. [Module reference](#13-module-reference)

---

## 1. Introduction and goals

### 1.1 Purpose

Empório da Música is a fictional musical instrument store in Campo Grande/MS. Its staff spends much of the day
on repeated questions: opening hours, order status, prices and stock. This project is a text chat agent,
"Melodia", that answers those questions in Brazilian Portuguese using the store's own data and policy manual.

### 1.2 Main features

| Feature | Description |
|---|---|
| Product questions | Price, promotions, PIX price, installments, stock and specs, from the catalog CSVs. |
| Store information | Address, hours, payment, returns, shipping, warranty and privacy, from the policy manual. |
| Orders | Status, tracking code and delivery dates for the logged-in customer only. |
| Account | The logged-in customer's own data and order list. |
| Scope control | Declines questions unrelated to the store and requests for accessories the store doesn't sell. |

### 1.3 Quality goals

| Priority | Goal | What it means here |
|---|---|---|
| 1 | Correctness | Every price, installment, stock level and date the agent states matches the data. |
| 2 | Privacy | A customer can only see their own orders and personal data. |
| 3 | Freshness | Changes to the data files show up without restarting the agent. |
| 4 | Portability | The model provider can be changed through configuration alone. |
| 5 | Testability | Business rules and agent wiring can be tested without a model or network. |

### 1.4 Stakeholders

| Role | Interest |
|---|---|
| Store customers | Fast, correct answers in a friendly tone. |
| Store staff | Fewer repeated questions; complaints and edge cases still reach them. |
| Developers | Clear separation between rules, agent wiring and interface. |
| Evaluators | Reasoning behind the technical choices. |

## 2. Constraints

| Constraint | Origin |
|---|---|
| Python as the main language | Challenge requirement. |
| Data comes as six CSV files and one PDF manual | Provided by the store. |
| Free model options must work (local Ollama or a free API tier) | Project choice. |
| Python 3.11 to 3.13 | LangChain does not install on Python 3.15 yet. |
| No passwords in the customer data | Provided data; login uses e-mail or phone. |

## 3. Context and scope

```
                 ┌──────────────────────────────────────┐
  Customer ─────►│           Melodia agent              │─────► LLM provider
  (terminal)     │  cli.py ─► agent.py ─► tools.py      │       (Ollama local or
             ◄───│                          │           │        Gemini API)
                 └──────────────────────────┼───────────┘
                                            ▼
                               src/data: 6 CSVs + policy PDF
```

| External element | Interface |
|---|---|
| Customer | Text lines typed in the terminal; commands `/login`, `/logout`, `sair`. |
| LLM provider | Chat API with tool calling, reached through LangChain. Chosen by `LLM_MODEL`. |
| Data files | Read-only CSV and PDF files in `src/data`. |

## 4. Solution strategy

| Goal | Approach |
|---|---|
| Correctness | Tool calling with fixed Python functions. The model decides which lookup to make and writes the reply; the code does every calculation and returns ready-made numbers. |
| Privacy | Login handled in code. The logged-in customer reaches the tools through the runtime context, outside the model's view. |
| Freshness | Data cached in memory, keyed by each file's modification time. |
| Portability | LangChain's `init_chat_model` with a `provider:model` string from `.env`. |
| Testability | Rules in a framework-free module; the agent accepts any model, so tests pass a scripted fake one. |

## 5. Building blocks

### 5.1 Level 1: modules

```
src/
├── cli.py      Interface: terminal loop, commands, session state
├── agent.py    Agent: LangChain tools, system prompt, model, agent loop
├── tools.py    Domain: data loading, business rules, lookups
└── data/       Store data (read-only)
tests/
├── test_tools.py   Unit tests for tools.py
└── test_agent.py   Agent tests (fake model offline, real model with -m live)
```

Dependencies point one way: `cli.py` → `agent.py` → `tools.py`. `tools.py` imports nothing from the other two
and nothing from LangChain.

### 5.2 Level 2: inside `tools.py`

| Group | Responsibility | Main functions |
|---|---|---|
| Data loading | Read and clean CSVs, split the PDF, cache by modification time | `load_tables`, `load_policy`, `tables`, `policy` |
| Pricing | Promotions, PIX, installment limits and table, availability | `active_discount`, `max_installments`, `price_info`, `availability` |
| Catalog | Product search, product detail, promotions, policy sections | `search_products`, `get_product`, `similar_in_stock`, `active_promotions`, `get_policy_section` |
| Customers and orders | Login lookup, order detail, account data | `find_customer`, `get_order`, `get_my_account` |

### 5.3 Level 2: inside `agent.py`

| Element | Responsibility |
|---|---|
| `Session` | Holds the logged-in `customer_id` for one call. |
| Tool wrappers | Six `@tool` functions that expose the lookups to the model. |
| `SYSTEM_PROMPT` | Persona, tone and rules (facts only from tools, login done by the system, scope). |
| `make_model` / `build_agent` / `ask` | Model creation, agent assembly and the single entry point for a message. |

### 5.4 Tools exposed to the model

| Tool | Input from the model | Returns |
|---|---|---|
| `search_products` | category, max_price, term, sort | Short rows (id, name, category, availability, final price), up to 10, in stock first and sorted by price. Adds `in_stock_alternatives` when nothing found can be bought. |
| `get_product` | product_id | Prices, PIX price, installment table, stock, description and specs of one product. |
| `active_promotions` | product_id (optional) | Running promotions with original price, discount and final price. |
| `get_policy` | section (one of 10 keys) | Text of that section of the manual. |
| `get_order` | order_id | Status, tracking, dates and items of an order of the logged-in customer. |
| `get_my_account` | nothing | Name, e-mail, phone, city and orders of the logged-in customer. |

## 6. Runtime scenarios

### 6.1 Product question

```
Customer: "Quanto custa o Takamine GD20?"
cli.Chat.handle ─► agent.ask ─► model
model ─► search_products(term="takamine gd20") ─► one short row, product_id 95
model ─► get_product(95) ─► final price 2199.0, PIX 2089.05, installments {"1x": ... "12x": 183.25}
model ─► reply quoting those numbers
```

### 6.2 Order question with login

```
Customer: "Qual o status do pedido 1?"     (not logged in)
model ─► get_order(1) ─► NOT_LOGGED_IN ─► reply: "digite /login ..."
Customer: "/login pedro.oliveira@jmail.com"
cli ─► tools.find_customer ─► ok, customer 3   (no model call)
Customer: "Qual o status do pedido 1?"
model ─► get_order(1), session customer 3 ─► order found ─► reply with status and tracking
```

### 6.3 Another customer's order

```
Session: customer 2 (Ana)
model ─► get_order(1) ─► filter: order 1 AND customer 2 ─► ORDER_NOT_FOUND
reply: asks to check the number, without saying the order exists
```

### 6.4 Ambiguous login

```
Customer: "/login bruno.martins@jmail.com"   (e-mail shared by customers 11 and 43)
find_customer ─► need_more, ask_for phone ─► "digite o telefone do seu cadastro"
Customer: "(67) 98543-2109"
find_customer(e-mail, confirm=phone) ─► only customer 11 has both ─► logged in
```

### 6.5 Model or network failure

```
agent.ask ─► provider error (503, 429, timeout)
exception raised before ask() returns ─► Chat keeps its previous history
cli.main prints "[Não consegui falar com o modelo agora: ...]" and waits for the next line
```

## 7. Deployment

The prototype runs as a single local process:

```
Developer machine
├── Python 3.13 virtual environment (.venv)
├── python -m src.cli
├── src/data (CSV and PDF files)
└── Ollama service on localhost:11434   or   HTTPS calls to the Gemini API
```

A production setup would put the agent behind a WhatsApp webhook, read from the store's database instead of
files, and keep history per phone number in Redis or Postgres. See section 11.

## 8. Crosscutting concepts

### 8.1 Numbers come from code

Every value the agent may quote is computed in `tools.py`: final price with promotion, PIX price, the
installment table, availability, days since order and `delivery_overdue`. The system prompt tells the model to
copy numbers from tool results and never calculate. This rule came from testing: whenever the model did the
math, it got it wrong.

### 8.2 Data loading and freshness

- CSVs are read with pandas and cleaned: phones as digits, e-mails in lower case, parsed dates, specs JSON as
  dicts, category name joined into products, and an accent-free `search_text` column.
- The policy PDF is cleaned of repeated headers, page numbers and broken phone numbers, then split into 10
  sections by their headings. The cover page is joined to `about`.
- `tables()` and `policy()` call an `lru_cache` keyed by the files' modification time, so edited files are
  reloaded on the next question and unchanged ones come from memory.

### 8.3 Security and privacy

- Login is a command handled by `cli.py`; the model is never involved.
- The `customer_id` reaches `get_order` and `get_my_account` through LangChain's `ToolRuntime`. It is not in
  the schema the model sees, so the model cannot request someone else's data, even under prompt injection.
- Order lookups filter by order number and owner together. A missing order and someone else's order return
  the same message.
- Switching customers or logging out clears the history.
- Text from customers is matched literally (`regex=False`) when searching.

### 8.4 Text handling

Comparisons use `normalize`, which drops accents and case ("Violões" and "violoes" match). The original text,
with accents, is what the customer sees. Generic words such as "barato" or "produto" are ignored in search
terms.

### 8.5 Error handling

- Tools return `{"error": ...}` messages the model can act on: unknown category (with the valid list),
  unknown policy section, unknown product, not logged in, order not found.
- Provider errors are caught in `cli.py`; the session stays open and the history is unchanged.
- The agent loop is capped at 12 steps per message.

### 8.6 Configuration

| Variable | Example | Notes |
|---|---|---|
| `LLM_MODEL` | `ollama:qwen3:8b`, `google_genai:gemini-flash-lite-latest` | `provider:model`. Defaults to `ollama:qwen3:8b`. |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | | Only for Gemini. Kept in `.env`, which git ignores. |

For Ollama, `make_model` turns off Qwen3's thinking mode and sets a low temperature. Cloud models keep their
provider defaults.

## 9. Architecture decisions

| # | Decision | Alternatives considered | Reason |
|---|---|---|---|
| 1 | Tool calling with fixed functions | Vector RAG, SQL agent | Prices and stock need exact matches; a SQL agent could read every customer under prompt injection. |
| 2 | LangChain `create_agent` | Hand-written tool loop | Provider switching via `.env` and the hidden runtime context for the logged-in customer. |
| 3 | CSV files with pandas | SQLite | The data arrives as files; SQLite would need the same freshness check plus a build step. |
| 4 | Policy split by section, looked up by key | Embeddings | Ten fixed sections; a key lookup is exact and needs no extra infrastructure. |
| 5 | Short search rows plus `get_product` | Full detail in every search row | Long result lists made a small model pick the wrong product. |
| 6 | Installment table in the price data | Only the maximum, or a separate calculator tool | Any "em Nx?" becomes a lookup; a missing entry means not allowed. |
| 7 | Qwen3 8B on Ollama as default | Gemini, OpenAI | Free, local, no quota. Gemini remains one setting away. |
| 8 | CLI with in-memory history | API, web UI, persisted history | The challenge focuses on the agent; history may hold personal data. |

## 10. Quality and testing

| Suite | Scope | How it runs |
|---|---|---|
| `tests/test_tools.py` | Data cleaning, PDF sections, cache reload, installment limits, PIX and promotion prices, search, login formats, order privacy matrix (every order against every customer) | `pytest`, offline, a few seconds |
| `tests/test_agent.py` (offline) | Tool schemas hide `customer_id`, session injection, history, made-up category handling | `pytest`, with a scripted `GenericFakeChatModel` |
| `tests/test_agent.py` (live) | Price, login prompt, tracking for the owner, nothing for others, prompt injection, real alternatives | `pytest -m live`, needs the configured model |

Example conversations recorded with Gemini are in `examples/`.

## 11. Risks and technical debt

| Item | Impact | Mitigation or next step |
|---|---|---|
| Small models sometimes skip instructions (e.g. not passing a category) | Wrong product in larger catalogs | Larger model, or a labeled question set to compare models |
| Live tests match strings in model replies | A valid rephrasing can fail them | Check tool calls instead of wording, or use a grader |
| Gemini free tier quotas and 503 errors | Interrupted sessions | Paid key or retry with backoff |
| Reading a CSV while it is being written | Half-loaded data | Write to a temp file and rename; in production read from a database |
| Login without a second factor | Anyone with a customer's e-mail or phone can log in | One-time code by e-mail or WhatsApp |
| History in memory, one session per process | No multi-user support | Webhook service with history per phone number |

## 12. Glossary

| Term | Meaning |
|---|---|
| Tool | A Python function the model can ask the agent to run. |
| Tool calling | The model replying with "call this function with these arguments" instead of text. |
| Agent loop | Model, tool, model, repeated until the model answers without a tool. |
| Runtime context | Data passed to tools by the code (the logged-in customer), not visible to the model. |
| PIX | Brazilian instant payment; 5% off the list price at this store. |
| Installments (parcelas) | Interest-free credit card payments, limited by a minimum installment value. |

## 13. Module reference

### `src/tools.py`

| Function | Description |
|---|---|
| `normalize(text)` | Lower case without accents, used only for comparisons. |
| `load_csv(name)` | Reads one CSV from `src/data` by short name. |
| `load_tables()` | Reads and cleans the six CSVs; returns a dict of DataFrames. |
| `load_policy()` | Reads the policy PDF and returns `{section_key: text}`. |
| `tables()` | Current data, reloaded when any CSV changes. |
| `policy()` | Current policy sections, reloaded when the PDF changes. |
| `active_discount(product_id)` | Best active promotion as `(percent, name)`, or `None`. |
| `max_installments(price)` | Most interest-free installments allowed for a price (manual 3.1). |
| `price_info(product_id, list_price)` | Final price, promotion, PIX price and the installment table. |
| `availability(status, stock)` | `in_stock`, `out_of_stock`, `discontinued` or `coming_soon`. |
| `search_products(category, max_price, term, sort)` | Short product rows with `total_found`; may add in-stock alternatives. |
| `similar_in_stock(product, limit)` | In-stock products of the same category, closest in price. |
| `get_product(product_id)` | Full detail of one product. |
| `active_promotions(product_id)` | Running promotions with prices. |
| `get_policy_section(section)` | Text of one section, or the list of valid keys. |
| `duplicate_contacts()` | E-mails and phones shared by more than one customer record. |
| `find_customer(email_or_phone, confirm)` | Login lookup: `ok`, `need_more` or `not_found`. |
| `get_order(order_id, customer_id)` | One order of that customer, with computed day counts. |
| `get_my_account(customer_id)` | That customer's data and order list. |

### `src/agent.py`

| Element | Description |
|---|---|
| `Session` | Dataclass with the logged-in `customer_id`. |
| `search_products`, `get_product`, `active_promotions`, `get_policy`, `get_order`, `get_my_account` | Tools exposed to the model (see 5.4). |
| `SYSTEM_PROMPT` | Persona, tone and rules for the model. |
| `make_model(name)` | Builds the chat model from `provider:model`. |
| `build_agent(model)` | Creates the LangChain agent; accepts a custom model for tests. |
| `ask(agent, history, text, customer_id)` | Sends one message; returns the reply and the updated history. |

### `src/cli.py`

| Element | Description |
|---|---|
| `Chat` | State of one conversation: agent, logged-in customer, history, pending login confirmation. |
| `Chat.handle(text)` | Routes a line to a command, a login confirmation or the agent. |
| `Chat.login(contact, confirm)` | Logs in, asks for a second contact when needed, clears history on customer change. |
| `Chat.logout()` | Ends the session and clears the history. |
| `main()` | Terminal loop. |
