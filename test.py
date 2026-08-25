import json
terms = json.load(open("jargon_terms.json"))
print(f"Total jargon terms: {len(terms)}")