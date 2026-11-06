"""Prompt text for rag."""

PROMPTS = {}

PROMPTS["rag_context_chunk"] = r"""$text"""

PROMPTS["rag_context_intro"] = ""

PROMPTS["rag_query"] = r"""$speaker. Этап $phase. $intent
Текущая тема переговоров: $recent"""

