"""
FinExtract-Bench: Prompt templates for LLM-based financial extraction.

Design principles:
- Prompts are versioned (PROMPT_VERSION) so provenance can track which
  prompt was used for an extraction.
- The schema in the prompt exactly mirrors the FinancialReport Pydantic model.
- Prompts instruct the model to return null (not omit) missing fields so
  the parser can distinguish "not found" from "parse error".
- For each field, the prompt requests source_text to enable provenance.
- Unit ambiguity is explicitly flagged in the prompt (the single most common
  source of extraction error in financial documents).
"""

from __future__ import annotations

PROMPT_VERSION = "1.0.0"

# ============================================================
# JSON schema description (embedded in all prompts)
# ============================================================

_FIELD_SCHEMA = """
{
  "revenue": {
    "value": <number in the document's native unit, e.g. 383285 if in millions>,
    "unit": "<e.g. 'million USD', 'USD', 'billion USD'>",
    "currency": "<ISO 4217 code, e.g. 'USD'>",
    "source_text": "<exact quote from the document containing this value>",
    "confidence": <0.0-1.0, your confidence this is the correct field>
  },
  "net_income": { ... same structure ... },
  "operating_income": { ... same structure ... },
  "total_assets": { ... same structure ... },
  "total_liabilities": { ... same structure ... },
  "cash_and_equivalents": { ... same structure ... },
  "eps": {
    "value": <earnings per share value>,
    "unit": "USD per share",
    "currency": "USD",
    "source_text": "...",
    "confidence": ...
  }
}
"""

_UNIT_WARNING = """
CRITICAL UNIT RULES:
1. If the document says "(in millions)" or "in millions of dollars", the unit
   for all values on that page/section is "million USD".
2. Do NOT convert values. Report the number exactly as it appears in the
   document (e.g. 383285, not 383285000000).
3. If the document says "(in thousands)", the unit is "thousand USD".
4. For EPS, the unit is always "USD per share".
5. If you cannot determine the unit, set unit to "unknown".
"""

_NULL_RULE = """
If a field is not found in the provided text, set its value to null (JSON null).
Do NOT guess or fabricate values. Do NOT omit missing fields — include them with null.
"""

# ============================================================
# Text-only prompt (Pipeline A)
# ============================================================

TEXT_ONLY_SYSTEM_PROMPT = f"""You are a financial data extraction expert. Your task is to
extract specific financial metrics from an annual report or 10-K filing.

Return ONLY valid JSON matching this exact schema:
{_FIELD_SCHEMA}
{_UNIT_WARNING}
{_NULL_RULE}

Extract values for the MOST RECENT fiscal year only (not prior-year comparatives).
If multiple values appear for the same metric, choose the one labeled as the current
or most recent year.

Return ONLY the JSON object. No explanation, no markdown, no code blocks.
""".strip()

# ============================================================
# Layout-aware prompt (Pipeline B)
# ============================================================
# Same instructions but explicitly references table structure

LAYOUT_AWARE_SYSTEM_PROMPT = f"""You are a financial data extraction expert. You will
receive structured text extracted from financial tables in an annual report.
The text preserves table layout with column separators.

Return ONLY valid JSON matching this exact schema:
{_FIELD_SCHEMA}
{_UNIT_WARNING}
{_NULL_RULE}

IMPORTANT: The document text may contain multiple years. Extract the MOST RECENT
fiscal year values only. Column headers will identify which column is current year
vs. prior year.

Return ONLY the JSON object. No explanation, no markdown, no code blocks.
""".strip()

# ============================================================
# Hybrid prompt (Pipeline C)
# ============================================================
# Richer instructions for the strongest pipeline

HYBRID_SYSTEM_PROMPT = f"""You are a senior financial analyst performing structured
data extraction from an annual report. You will receive a combination of financial
tables and surrounding text context.

Return ONLY valid JSON matching this exact schema:
{_FIELD_SCHEMA}
{_UNIT_WARNING}
{_NULL_RULE}

Additional guidelines for hybrid extraction:
- Prefer values from financial statement tables over narrative text.
- For revenue, look for "Total revenue", "Net sales", "Net revenue", or "Revenue".
- For operating_income, look for "Operating income", "Income from operations".
- For eps, prefer diluted EPS over basic EPS.
- If the income statement and a narrative section give conflicting values,
  use the income statement value and note the conflict in source_text.
- For total_assets and total_liabilities, look in the Balance Sheet section.
- For cash_and_equivalents, look for "Cash and cash equivalents" in the
  Balance Sheet (not the cash flow statement).

Return ONLY the JSON object. No explanation, no markdown, no code blocks.
""".strip()

# ============================================================
# Correction prompt (used when first attempt returns invalid JSON)
# ============================================================

CORRECTION_PROMPT_TEMPLATE = """Your previous response could not be parsed as valid JSON.
Error: {error}

Your previous response was:
{previous_response}

Please return ONLY valid JSON. No markdown code blocks, no explanation text.
The response must start with {{ and end with }}.
""".strip()


def get_system_prompt(pipeline: str) -> str:
    """
    Return the system prompt for a given pipeline.

    Args:
        pipeline: 'text_only', 'layout_aware', or 'hybrid'.

    Returns:
        System prompt string.
    """
    if pipeline == "layout_aware":
        return LAYOUT_AWARE_SYSTEM_PROMPT
    if pipeline == "hybrid":
        return HYBRID_SYSTEM_PROMPT
    return TEXT_ONLY_SYSTEM_PROMPT  # Default for 'text_only' and unknown
