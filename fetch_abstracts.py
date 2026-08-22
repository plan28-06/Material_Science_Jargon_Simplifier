import feedparser
import urllib.request
import json
import time

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
                "title": entry.title.replace("\n", " ").strip(),
                "abstract": entry.summary.replace("\n", " ").strip()
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