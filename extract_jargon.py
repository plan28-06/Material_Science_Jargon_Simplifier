import json
import re
from collections import Counter
from wordfreq import word_frequency

INPUT_PATH = "pairs.jsonl"
OUTPUT_PATH = "jargon_terms.json"

MIN_WORD_LEN = 4
MIN_OCCURRENCES = 2          # must appear in at least 2 pairs
MAX_GENERAL_FREQ = 3e-6     # words more common than this in everyday English are NOT jargon


def tokenize(text):
    words = re.findall(r'\b[a-zA-Z][a-zA-Z-]+\b', text.lower())
    return words


def load_pairs(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def extract_jargon(pairs):
    jargon_counter = Counter()

    for p in pairs:
        complex_words = set(tokenize(p["complex"]))
        simple_words = set(tokenize(p["simple"]))
        candidates = complex_words - simple_words

        for word in candidates:
            if len(word) < MIN_WORD_LEN:
                continue
            jargon_counter[word] += 1

    jargon_terms = {}
    for word, count in jargon_counter.items():
        if count < MIN_OCCURRENCES:
            continue

        # This is the key filter: how common is this word in everyday English?
        freq = word_frequency(word, 'en')

        # Rare in general English -> likely a real technical term
        if freq < MAX_GENERAL_FREQ:
            jargon_terms[word] = {"corpus_count": count, "general_freq": freq}

    return jargon_terms


def main():
    pairs = load_pairs(INPUT_PATH)
    print(f"Loaded {len(pairs)} pairs.")

    jargon_terms = extract_jargon(pairs)
    print(f"Extracted {len(jargon_terms)} candidate jargon terms.")

    # Sort by corpus frequency (most common technical terms first)
    sorted_terms = dict(
        sorted(jargon_terms.items(), key=lambda x: -x[1]["corpus_count"])
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(sorted_terms, f, indent=2)

    print(f"Saved to {OUTPUT_PATH}")

    print("\nTop 20 jargon terms:")
    for word, info in list(sorted_terms.items())[:20]:
        print(f"  {word}: corpus={info['corpus_count']}, general_freq={info['general_freq']:.2e}")


if __name__ == "__main__":
    main()