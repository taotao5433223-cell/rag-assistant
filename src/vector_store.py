import json
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum([x * x for x in a]))
    nb = math.sqrt(sum([x * x for x in b]))
    return dot / (na * nb) if na and nb else 0

def search(query_vector, records, top_k=3):
    scored = [(cosine(query_vector, r["vector"]), r) for r in records]
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]