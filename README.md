# Materials Science Jargon Simplifier

A fine-tuned BART model that turns jargon-heavy materials science abstracts into plain-language summaries, paired with a data-driven jargon detector that highlights technical terms in the original text.

## Overview

Materials science papers are dense with domain-specific vocabulary that makes them inaccessible to non-specialists. This project fine-tunes a BART-base sequence-to-sequence model to simplify jargon-heavy abstracts, and separately derives a jargon vocabulary directly from the training data — no external dictionary or hand-labeling required.

**Live demo:** _[add your URL here once deployed]_

## How it works

### 1. Data collection
~1,000 materials science abstracts (`cond-mat.mtrl-sci`) pulled from the arXiv API, cleaned of LaTeX markup and formatting artifacts.

### 2. Synthetic data generation
Since no paired complex→simple materials-science dataset exists, each abstract was rewritten in plain language using a locally-run Llama 3.1 8B model (via Ollama) — avoiding reliance on paid APIs while producing high-quality training pairs.

### 3. Data cleaning & filtering
Pairs were filtered on length ratio, word count bounds, and duplicate/degenerate generations, then split into train/validation/test sets (834 / 98 / 50).

### 4. Jargon vocabulary extraction
Rather than using a hand-built dictionary, the jargon vocabulary was derived empirically: words present in the complex abstract but absent from its simplified counterpart were tallied across the whole corpus, then filtered using real-world English word frequency (via `wordfreq`) to isolate terms that are genuinely rare in everyday English — separating true domain jargon (e.g. *phonon*, *anisotropy*, *antiferromagnetic*) from generic academic vocabulary (e.g. *demonstrate*, *results*).

### 5. Fine-tuning
`facebook/bart-base` (139M parameters) was fine-tuned as a sequence-to-sequence simplifier using HuggingFace `Trainer`, with mixed-precision training on a consumer GPU (RTX 4060).

### 6. Evaluation
The fine-tuned model was evaluated on the held-out test set across three independent axes:

| Metric | Result |
|---|---|
| Average Flesch Reading Ease (original) | −1.03 |
| Average Flesch Reading Ease (simplified) | 44.83 |
| Average Readability improvement | +45.86 points |
| Faithfulness (LLM-as-judge, 1–5 scale) | Median 3/5 (45/50 examples scored 3, 5/50 scored 2) |

The readability gain is large and consistent. Faithfulness is more modest and highly consistent across examples — the model reliably preserves the gist and some key facts while dropping some specific details, a known tradeoff for small-parameter simplification models trained on a limited dataset. This tradeoff is discussed further in the writeup rather than treated as a flaw to be eliminated.

### 7. Backend & frontend
A FastAPI backend serves the fine-tuned model and jargon dictionary through a single `/simplify` endpoint. A lightweight HTML/CSS/JS frontend (light/dark mode, mobile-responsive) lets users paste text and see the original (with jargon highlighted in red) alongside the simplified version, plus readability scores for both.

## Tech stack

- **Model:** BART-base, fine-tuned with HuggingFace `transformers` + `datasets`
- **Data generation:** Llama 3.1 8B via Ollama (local inference)
- **Backend:** FastAPI, PyTorch
- **Frontend:** Vanilla HTML/CSS/JS (no build step)
- **Evaluation:** ROUGE, Flesch Reading Ease (`textstat`), LLM-as-judge faithfulness scoring

## Deployment status

An attempt was made to deploy the backend on Render's free tier. Even after converting the model to float16 (halving its size to ~279MB) and pushing it via Git LFS, the combined memory footprint of PyTorch, `transformers`, and the model weights exceeded the free tier's 512MB RAM limit, causing the service to be killed on startup. Free-tier CPU-only hosts with such tight memory caps aren't well suited to serving even a relatively small (139M parameter) transformer model alongside its runtime dependencies. The project currently runs locally (backend + frontend); deploying to a host with a larger memory allowance (or serving a further-compressed/quantized model) would resolve this.

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend
uvicorn main:app --reload --port 8080

# Open index.html in a browser (make sure API_URL in index.html points to your local backend)
```

## Notes on design choices

- **BART over T5:** BART's denoising-autoencoder pretraining objective is a closer match to simplification (corrupted→clean text) than T5's general text-to-text framing.
- **Local generation over paid APIs:** synthetic training pairs were generated entirely with a locally-run open model, keeping the pipeline reproducible without API costs.
- **Frequency-based jargon detection over a classifier:** avoids the need for labeled jargon data by reusing the complex/simple pairs already produced during training data generation.
