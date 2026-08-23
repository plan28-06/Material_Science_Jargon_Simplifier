import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

SIMPLIFY_PROMPT = """Rewrite the following materials science abstract in plain, simple English for someone with no Material Science background — like explaining it to a curious high schooler.

Rules:
- Keep all key facts, findings, and numbers accurate — do not invent anything
- Avoid jargon, technical terms, and chemical/mathematical notation where possible
- Write in 3-5 clear sentences
- Do not add commentary like "Here is the simplified version" — just output the simplified text directly

Abstract: {text}

Simplified version:"""


def simplify_text(text, model=MODEL, retries=3):
    prompt = SIMPLIFY_PROMPT.format(text=text)

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }, timeout=120)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5)

    return None  # failed after retries


def load_abstracts(path="raw_abstracts.jsonl"):
    abstracts = []
    with open(path, "r") as f:
        for line in f:
            abstracts.append(json.loads(line))
    return abstracts


def main():
    abstracts = load_abstracts()
    print(f"Loaded {len(abstracts)} abstracts.")

    out_path = "pairs.jsonl"
    done = 0

    with open(out_path, "w") as f:
        for i, item in enumerate(abstracts):
            complex_text = item["abstract"]

            print(f"[{i+1}/{len(abstracts)}] Simplifying...")
            simple_text = simplify_text(complex_text)

            if simple_text is None:
                print(f"  Skipped (failed after retries)")
                continue

            f.write(json.dumps({
                "id": item["id"],
                "complex": complex_text,
                "simple": simple_text
            }) + "\n")
            f.flush()  # save progress immediately, in case script crashes

            done += 1

    print(f"Done. Successfully generated {done}/{len(abstracts)} pairs.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()