import json
import requests
import re
import statistics


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
INPUT_PATH = "test_predictions.jsonl"
OUTPUT_PATH = "faithfulness_scores.jsonl"

JUDGE_PROMPT = """You are evaluating whether a simplified summary faithfully preserves the key facts of a technical abstract.

Original abstract: {complex_text}

Simplified version: {simple_text}

Rate the simplified version on a scale of 1-5 for FAITHFULNESS:
5 = All key facts/findings preserved accurately, no invented information
3 = Some facts preserved, but notable omissions or minor inaccuracies
1 = Major facts missing, or the summary contains invented/incorrect information

Respond in EXACTLY this format, nothing else:
SCORE: <number>
REASON: <one sentence>"""


def judge(complex_text, simple_text):
    prompt = JUDGE_PROMPT.format(complex_text=complex_text, simple_text=simple_text)
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}  # deterministic for evaluation consistency
    }, timeout=120)
    return response.json()["response"].strip()


def parse_judge_output(text):
    score_match = re.search(r"SCORE:\s*(\d)", text)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    score = int(score_match.group(1)) if score_match else None
    reason = reason_match.group(1).strip() if reason_match else text
    return score, reason


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    data = load_jsonl(INPUT_PATH)
    results = []

    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] Judging...")
        judge_output = judge(item["complex"], item["model_generated"])
        score, reason = parse_judge_output(judge_output)
        results.append({
            "id": item["id"],
            "score": score,
            "reason": reason,
        })

    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    valid_scores = [r["score"] for r in results if r["score"] is not None]
    avg_score = sum(valid_scores) / len(valid_scores)
    print(f"\nAverage faithfulness score: {avg_score:.2f} / 5")
    print(f"Median faithfulness score: {statistics.median(valid_scores):.2f} / 5")
    print(f"Score distribution: {sorted(valid_scores)}")
    print(f"Scored {len(valid_scores)}/{len(results)} examples successfully.")


if __name__ == "__main__":
    main()