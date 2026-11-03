"""Empório da Música customer service agent.

A text chat agent ("Melodia") for a musical instrument store. It answers in Brazilian Portuguese using the
store's CSV data and policy manual.

Modules:

- `src.tools`: data loading, business rules and lookups (no LangChain).
- `src.agent`: LangChain tools, system prompt, model selection and the agent.
- `src.cli`: terminal chat with `/login` and `/logout`.
"""
