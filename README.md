# candidate-job-matcher

Ranks candidate profiles against a job query and measures how good the ranking is.

- **BM25 retrieval** over profile text (stopword-filtered, length-normalized).
- **Skill-overlap reranking**: the text score is blended with the Jaccard overlap of required and held skills. `alpha=0` is the text-only baseline, so the two can be compared on the same labeled queries.
- **Ranking metrics**: `recall_at_k` and `ndcg_at_k` over graded relevance labels, plus an `evaluate()` harness that averages them across queries.

## Why the reranking matters

A profile that repeats "python" six times beats a real backend engineer on text alone. The tests build exactly that case: the baseline ranks the keyword-stuffed profile first, and the blended score puts the engineer with the right skills first. NDCG@k improves on the labeled query.

## Limits

- The data in the tests is synthetic and tiny. This shows the method and the evaluation, not results on real hiring data.
- Retrieval is lexical. There are no learned embeddings or recommendation model, so synonyms ("backend" vs "server-side") are not matched.
- No third-party packages, Python 3.10 or newer.

## Run

```bash
python3 -m unittest
```

Files: `matcher.py`, `test_matcher.py`
