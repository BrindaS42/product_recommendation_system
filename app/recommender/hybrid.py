import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def zscore(s: pd.Series):
    v = s.fillna(0.0).values
    if v.std() == 0:
        return pd.Series(np.zeros_like(v), index=s.index)
    return pd.Series((v - v.mean()) / (v.std() + 1e-12), index=s.index)

def blend(content_s: pd.Series, cf_s: pd.Series, compat_s: pd.Series, weights=(0.45,0.35,0.20)):
    c = zscore(content_s)
    f = zscore(cf_s.reindex(content_s.index).fillna(0.0))
    d = zscore(compat_s.reindex(content_s.index).fillna(0.0))
    w = np.array(weights, dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    return w[0]*c + w[1]*f + w[2]*d

def mmr(candidates: list, relevance: pd.Series, item_vecs: pd.DataFrame, k=10, lam=0.7):
    selected = []
    pool = list(candidates)
    while pool and len(selected) < k:
        best = None
        best_val = -1e9
        for pid in pool:
            rel = float(relevance.get(pid, 0.0))
            if not selected:
                div = 0.0
            else:
                sims = cosine_similarity(item_vecs.loc[[pid]].values, item_vecs.loc[selected].values).ravel()
                div = sims.max() if len(sims) else 0.0
            val = lam * rel - (1-lam) * div
            if val > best_val:
                best_val = val
                best = pid
        selected.append(best)
        pool.remove(best)
    return selected
