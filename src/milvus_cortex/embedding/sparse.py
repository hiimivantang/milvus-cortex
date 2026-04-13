"""BM25-style sparse vectorizer — no external dependencies.

Converts text into sparse vectors using term frequency with IDF-like weighting.
Produces dict[int, float] format compatible with Milvus SPARSE_FLOAT_VECTOR.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


# Simple tokenizer: lowercase, split on non-alphanumeric, remove short tokens
_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Common English stop words to skip
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could and but or nor not no so if then else for "
    "to of in on at by with from as into through during before after above below "
    "between out off over under again further once here there when where why how "
    "all each every both few more most other some such that this these those i me "
    "my we our you your he him his she her it its they them their what which who".split()
)

# Hash space size — maps tokens to sparse vector dimensions via hashing.
# 2^16 = 65536 dimensions gives low collision rate for typical vocabulary.
HASH_SPACE = 65536


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase terms, removing stop words."""
    tokens = _SPLIT_RE.split(text.lower())
    return [t for t in tokens if t and len(t) > 1 and t not in _STOP_WORDS]


def _token_to_dim(token: str) -> int:
    """Hash a token to a dimension index in the sparse vector space."""
    h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    return h % HASH_SPACE


def text_to_sparse(text: str) -> dict[int, float]:
    """Convert text to a sparse vector using BM25-like term frequency weighting.

    Returns a dict mapping dimension indices to weights, compatible with
    Milvus SPARSE_FLOAT_VECTOR field type.
    """
    tokens = tokenize(text)
    if not tokens:
        return {}

    tf = Counter(tokens)
    doc_len = len(tokens)

    # BM25-like TF saturation: tf / (tf + k1)
    k1 = 1.2
    sparse: dict[int, float] = {}
    for token, count in tf.items():
        dim = _token_to_dim(token)
        score = (count / (count + k1)) * (1.0 + math.log(doc_len + 1))
        # Accumulate in case of hash collision
        sparse[dim] = sparse.get(dim, 0.0) + score

    return sparse


def query_to_sparse(query: str) -> dict[int, float]:
    """Convert a search query to a sparse vector.

    Uses simpler weighting than document vectors — just binary presence
    with slight boost for repeated terms.
    """
    tokens = tokenize(query)
    if not tokens:
        return {}

    tf = Counter(tokens)
    sparse: dict[int, float] = {}
    for token, count in tf.items():
        dim = _token_to_dim(token)
        score = 1.0 + math.log(count) if count > 1 else 1.0
        sparse[dim] = sparse.get(dim, 0.0) + score

    return sparse
