import json
from transformers import BartForConditionalGeneration, BartTokenizerFast
import torch

MODEL_PATH = "./bart-simplifier-final"
TEST_PATH = "test.jsonl"
OUTPUT_PATH = "test_predictions.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = BartTokenizerFast.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()  # switch to inference mode (disables dropout etc.)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def simplify(text, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():  # no need to track gradients, we're not training
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,          # beam search: explores multiple possible outputs, picks the best
            length_penalty=1.0,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    test_data = load_jsonl(TEST_PATH)
    print(f"Loaded {len(test_data)} test examples.")

    results = []
    for i, item in enumerate(test_data):
        print(f"[{i+1}/{len(test_data)}] Generating...")
        generated = simplify(item["complex"])
        results.append({
            "id": item.get("id", i),
            "complex": item["complex"],
            "reference_simple": item["simple"],       # Llama's version (our training target)
            "model_generated": generated,              # your fine-tuned BART's actual output
        })

    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()