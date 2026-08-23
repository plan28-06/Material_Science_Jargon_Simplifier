import json
import pandas as pd
import random

INPUT_PATH = "pairs.jsonl"
TRAIN_PATH = "train.jsonl"
VAL_PATH = "val.jsonl"
TEST_PATH = "test.jsonl"

# --- Filtering thresholds (tweakable) ---
MIN_SIMPLE_WORDS = 15       # too short = likely a bad/truncated generation
MAX_SIMPLE_WORDS = 150      # too long = model rambled
MIN_LENGTH_RATIO = 0.15     # simple shouldn't be less than 15% of complex length
MAX_LENGTH_RATIO = 0.90     # simple shouldn't be more than 90% of complex length (barely simplified)


def load_pairs(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def word_count(text):
    return len(text.split())


def filter_pairs(pairs):
    kept = []
    dropped = []

    for p in pairs:
        complex_text = p.get("complex", "").strip()
        simple_text = p.get("simple", "").strip()

        # Drop empty or missing fields
        if not complex_text or not simple_text:
            dropped.append((p, "empty field"))
            continue

        simple_len = word_count(simple_text)
        complex_len = word_count(complex_text)

        # Drop too short / too long simplified text
        if simple_len < MIN_SIMPLE_WORDS:
            dropped.append((p, "simple too short"))
            continue
        if simple_len > MAX_SIMPLE_WORDS:
            dropped.append((p, "simple too long"))
            continue

        # Drop bad length ratios
        ratio = simple_len / complex_len if complex_len > 0 else 0
        if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
            dropped.append((p, f"bad ratio ({ratio:.2f})"))
            continue

        # Drop near-duplicates (simple text too similar to complex text)
        if simple_text.strip().lower() == complex_text.strip().lower():
            dropped.append((p, "identical to original"))
            continue

        kept.append(p)

    return kept, dropped


def split_dataset(pairs, train_frac=0.85, val_frac=0.10, seed=42):
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    pairs = load_pairs(INPUT_PATH)
    print(f"Loaded {len(pairs)} raw pairs.")

    kept, dropped = filter_pairs(pairs)
    print(f"Kept {len(kept)} pairs, dropped {len(dropped)} pairs.")

    # Show a breakdown of why pairs were dropped
    if dropped:
        reasons = pd.Series([reason for _, reason in dropped]).value_counts()
        print("\nDrop reasons:")
        print(reasons)

    train, val, test = split_dataset(kept)
    print(f"\nSplit: {len(train)} train / {len(val)} val / {len(test)} test")

    save_jsonl(train, TRAIN_PATH)
    save_jsonl(val, VAL_PATH)
    save_jsonl(test, TEST_PATH)

    print(f"\nSaved: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")


if __name__ == "__main__":
    main()