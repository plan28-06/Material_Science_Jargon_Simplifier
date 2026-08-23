import feedparser
import urllib.request
import json
import time
import re


def clean_latex(text):
    # Convert common LaTeX Greek letters to actual unicode characters
    greek_map = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ', r'\mu': 'μ',
        r'\pi': 'π', r'\sigma': 'σ', r'\omega': 'ω'
    }
    for latex, unicode_char in greek_map.items():
        text = text.replace(latex, unicode_char)

    # Remove $ math delimiters but keep the content
    text = text.replace('$', '')

    # Convert LaTeX subscripts/superscripts like _2 -> 2, ^2 -> 2
    text = re.sub(r'_(\d+)', r'\1', text)   # MnO_2 -> MnO2
    text = re.sub(r'\^(\d+)', r'\1', text)  # x^2 -> x2

    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def fetch_arxiv_abstracts(category="cond-mat.mtrl-sci", total=1000, batch_size=100):
    all_abstracts = []
    start = 0

    while start < total:
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=cat:{category}"
            f"&start={start}&max_results={batch_size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        print(f"Fetching results {start} to {start + batch_size}...")

        data = urllib.request.urlopen(url).read()
        feed = feedparser.parse(data)

        if not feed.entries:
            print("No more entries found, stopping early.")
            break

        for entry in feed.entries:
            all_abstracts.append({
                "id": entry.id,
                "title": clean_latex(entry.title.replace("\n", " ").strip()),
                "abstract": clean_latex(entry.summary.replace("\n", " ").strip())
            })

        start += batch_size
        time.sleep(3)  # be polite to arXiv's servers

    return all_abstracts


if __name__ == "__main__":
    abstracts = fetch_arxiv_abstracts(total=1000)
    print(f"Fetched {len(abstracts)} abstracts.")

    with open("raw_abstracts.jsonl", "w") as f:
        for item in abstracts:
            f.write(json.dumps(item) + "\n")

    print("Saved to raw_abstracts.jsonl")