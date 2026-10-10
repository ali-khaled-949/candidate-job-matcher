"""Candidate-to-job matching: BM25 text retrieval, skill-overlap reranking, and ranking metrics.

The ranking quality is measured (recall@k and NDCG@k), not assumed, so a change to the scoring can be compared
against the baseline on the same labeled queries.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field

WORD = re.compile(r"[a-z0-9+#.]+")
STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "for", "with", "on", "is", "are", "we", "you", "our"}


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


@dataclass
class Doc:
    id: str
    text: str
    skills: set[str] = field(default_factory=set)


class BM25:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokens(d.text)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


def skill_overlap(wanted: set[str], have: set[str]) -> float:
    """Jaccard overlap of skill sets, 0..1."""
    if not wanted or not have:
        return 0.0
    return len(wanted & have) / len(wanted | have)


def rank(query: str, wanted_skills: set[str], docs: list[Doc], alpha: float = 0.5, k: int | None = None) -> list[str]:
    """Rank docs by BM25 (scaled 0..1) blended with skill overlap. alpha=0 is the text-only baseline."""
    bm = BM25(docs).score(query)
    top = max(bm) or 1.0
    scored = [((1 - alpha) * (s / top) + alpha * skill_overlap(wanted_skills, d.skills), d.id) for s, d in zip(bm, docs)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    ids = [i for _, i in scored]
    return ids[:k] if k else ids


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: list[str], gains: dict[str, int], k: int) -> float:
    dcg = sum(gains.get(i, 0) / math.log2(pos + 2) for pos, i in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(pos + 2) for pos, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(cases: list[dict], docs: list[Doc], alpha: float, k: int = 3) -> dict:
    """cases: {query, skills, gains{doc_id: 0..3}}. Returns mean recall@k and NDCG@k."""
    rec = ndcg = 0.0
    for c in cases:
        ranked = rank(c["query"], set(c["skills"]), docs, alpha)
        rec += recall_at_k(ranked, {i for i, g in c["gains"].items() if g > 0}, k)
        ndcg += ndcg_at_k(ranked, c["gains"], k)
    n = len(cases)
    return {"recall@k": rec / n, "ndcg@k": ndcg / n, "k": k, "queries": n}
