import json
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizerFast
import torch
import textstat

# --- Config ---
MODEL_PATH = "./bart-simplifier-final"
JARGON_PATH = "jargon_terms.json"

# --- Load model once, at startup (not per-request — this is the key performance point) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

tokenizer = BartTokenizerFast.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

with open(JARGON_PATH, "r") as f:
    jargon_terms = set(json.load(f).keys())

print(f"Loaded model and {len(jargon_terms)} jargon terms.")

# --- FastAPI app setup ---
app = FastAPI(title="Materials Science Jargon Simplifier")

# Allow requests from any frontend origin (fine for a portfolio project; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/response schemas ---

# If input is not a string, this will ensure that an error is propogated. 
# This basically ensures input is of type string and is not empty
class SimplifyRequest(BaseModel):
    text: str

# This basically ensures input is of a specific type and is not empty
class SimplifyResponse(BaseModel):
    original: str
    simplified: str
    jargon_terms: list[str]
    original_readability: float
    simplified_readability: float


# --- Core functions ---
def simplify_text(text: str, max_length: int = 256) -> str:
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def find_jargon(text: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", text.lower())
    found = sorted(set(w for w in words if w in jargon_terms))
    return found


# --- Endpoint ---
@app.post("/simplify", response_model=SimplifyResponse)
def simplify(request: SimplifyRequest):
    text = request.text

    simplified = simplify_text(text)
    jargon_found = find_jargon(text)

    original_score = textstat.flesch_reading_ease(text)
    simplified_score = textstat.flesch_reading_ease(simplified)

    return SimplifyResponse(
        original=text,
        simplified=simplified,
        jargon_terms=jargon_found,
        original_readability=original_score,
        simplified_readability=simplified_score,
    )


# --- Health check endpoint (useful for deployment platforms) ---
@app.get("/health")
def health():
    return {"status": "ok", "device": device}