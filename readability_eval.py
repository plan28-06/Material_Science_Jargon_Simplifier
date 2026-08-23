import json
import textstat

INPUT_PATH = "test_predictions.jsonl"


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    data = load_jsonl(INPUT_PATH)

    complex_scores = []
    generated_scores = []

    for item in data:
        complex_scores.append(textstat.flesch_reading_ease(item["complex"]))
        generated_scores.append(textstat.flesch_reading_ease(item["model_generated"]))

    avg_complex = sum(complex_scores) / len(complex_scores)
    avg_generated = sum(generated_scores) / len(generated_scores)

    print(f"Average Flesch-Kincaid Reading Ease — Original (complex): {avg_complex:.2f}")
    print(f"Average Flesch-Kincaid Reading Ease — Model output (simplified): {avg_generated:.2f}")
    print(f"Improvement: +{avg_generated - avg_complex:.2f} points")


if __name__ == "__main__":
    main()