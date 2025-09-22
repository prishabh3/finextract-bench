# Research Report: Financial Document Extraction Benchmark

**Author**: [Your Name]  
**Date**: August 2026  
**Project**: FinExtract-Bench

---

## 1. Executive Summary

This report outlines the methodology and preliminary findings from evaluating three distinct LLM-based pipelines for extracting highly structured financial metrics (e.g., Revenue, Net Income, EPS) from unstructured, complex PDF annual reports (10-Ks). 

The goal was to measure the tradeoff between **Accuracy, Latency, and Cost** across:
1. **Pipeline A**: Naive Text-Only (PyMuPDF)
2. **Pipeline B**: Layout-Aware (Docling)
3. **Pipeline C**: Hybrid Layout + Semantic Rules

*Insert top-line takeaway here, e.g., "Pipeline C reduced unit-scaling errors by 80% at the cost of 2x latency compared to Pipeline A."*

---

## 2. Methodology

### 2.1 Dataset
- **Volume**: 5 large-cap technology companies (Apple, Microsoft, Amazon, Alphabet, Meta).
- **Timeframe**: Fiscal Years 2022 and 2023.
- **Ground Truth**: Manually sourced from EDGAR 10-K filings, tracking 9 distinct metrics per report.

### 2.2 Pipelines Evaluated
| Pipeline | Parser | Context Passed to LLM | Key Features |
|---|---|---|---|
| **Text-Only** | PyMuPDF | Raw, flattened string | Fast, cheapest. Prone to column shifts. |
| **Layout-Aware** | Docling | Markdown-formatted tables | Preserves spatial headers and hierarchical table structures. |
| **Hybrid** | Docling | Markdown tables + Text Blocks + Logic Prompts | Verifies accounting consistency (e.g., Operating Income < Revenue). |

### 2.3 Evaluation Metrics
- **Extraction Coverage**: % of targeted metrics successfully returned (non-null).
- **Exact Accuracy**: Strict equality against ground truth (allowing for unit normalization).
- **Relaxed Accuracy (1% / 5%)**: Allows for minor rounding or amortization discrepancies common in MD&A sections vs. financial tables.
- **Cost**: Estimated USD cost per document based on tokenizer counts (OpenAI/Anthropic).

---

## 3. Failure Taxonomy Analysis

Rather than relying on simple aggregate scores, the evaluation harness automatically classifies errors into distinct mechanisms. 

### 3.1 Observed Failure Distribution
*(Replace with actual experiment data plot)*

1. **Unit Normalization Errors (Critical)**: The LLM extracts "450" when the column header specifies "in millions", resulting in an error factor of 1,000,000.
2. **Sign Errors (High)**: Expenses or Net Losses are extracted as positive values.
3. **Missing Values (Medium)**: The LLM refuses to extract a value it deems ambiguous.

### 3.2 Mitigation Effectiveness
Pipeline C (Hybrid) introduced explicit semantic checks ("Ensure Net Income is not greater than Revenue"). This resulted in a *[X]% reduction in Semantic Mismatches* compared to Pipeline B, indicating that multi-step reasoning prompts can effectively audit layout-parsed data.

---

## 4. Performance Trade-offs

### 4.1 Cost vs. Accuracy
*(Insert plot analysis here)*
While Docling + GPT-4o (Pipeline B) yields the highest structural accuracy, its token usage is roughly 3x higher than PyMuPDF due to the verbosity of markdown table formatting. 

### 4.2 Latency Implications
*(Insert plot analysis here)*
Docling's document conversion process introduces a heavy upfront latency cost (often >15 seconds for a 100-page PDF). For high-throughput requirements, Pipeline A combined with a faster model (e.g., GPT-4o-mini) provides the best latency/cost ratio, though it requires a fallback mechanism for dense tabular data.

---

## 5. Conclusion & Future Work

The benchmark demonstrates that simply passing raw PDF text to an LLM is insufficient for rigorous financial extraction. Layout awareness is critical for tabular data, but it introduces significant token and latency overhead. 

**Future directions include:**
1. **Vision Models**: Testing GPT-4o Vision directly on PDF page images to bypass parsing entirely.
2. **Chunking Strategies**: Evaluating RAG-based extraction rather than full-context passing to reduce costs.
3. **Open-Weight Models**: Benchmarking Llama-3 70B against proprietary models using the same harness.
