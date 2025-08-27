import argparse
import re
import math
import sys
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?]) +")
def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENTENCE_SPLIT_REGEX.split(text)
def textrank_summarize(text: str, ratio: float = 0.2, min_sentences: int = 3) -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)
    scores = nx.pagerank(nx.from_numpy_array(sim))
    n = max(min_sentences, int(math.ceil(len(sentences) * ratio)))
    ranked = sorted(scores, key=scores.get, reverse=True)[:n]
    ranked.sort()
    return " ".join(sentences[i] for i in ranked)
def main():
    parser = argparse.ArgumentParser(description="Simple Text Summarizer")
    parser.add_argument("--text", required=True, help="Input text")
    parser.add_argument("--method", choices=["textrank"], default="textrank", help="Summarization method")
    parser.add_argument("--ratio", type=float, default=0.2, help="Fraction of sentences to keep")
    parser.add_argument("--min-sentences", type=int, default=3, help="Minimum sentences in result")
    args = parser.parse_args()
    if args.method == "textrank":
        summary = textrank_summarize(args.text, ratio=args.ratio, min_sentences=args.min_sentences)
    else:
        summary = "Unsupported method."
    print("\nSummary:\n", summary)
if __name__ == "__main__":
    main()
