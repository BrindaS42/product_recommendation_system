import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math

# NOTE: This file contains two approaches:
# 1) Simple explicit matrix MF (SVD-init) - light baseline using sklearn TruncatedSVD on user-item rating matrix.
# 2) PMI co-occurrence fallback if data is sparse.

def build_user_item_matrix(reviews: pd.DataFrame, min_user_ratings=1):
    """
    Build explicit user-item matrix (sparse) for users who rated items.
    returns: user_index, item_index, R (DataFrame: users x items, with NaN for missing)
    """
    df = reviews[['user_id','product_id','rating']].copy()
    pivot = df.pivot_table(index='user_id', columns='product_id', values='rating', aggfunc='mean')
    # Optionally filter users/items
    user_counts = pivot.notna().sum(axis=1)
    item_counts = pivot.notna().sum(axis=0)
    pivot = pivot.loc[user_counts >= min_user_ratings, item_counts[item_counts >= 1].index]
    return pivot

def simple_svd_predict(pivot_df, n_components=64):
    """
    Very simple baseline: fill NaN with zeros, run TruncatedSVD, reconstruct scores.
    (Be careful: zeros imply negative signal; this is a baseline only.)
    """
    from sklearn.decomposition import TruncatedSVD
    X = pivot_df.fillna(0.0).values
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(X)   # users x k
    Vt = svd.components_       # k x items
    X_hat = U.dot(Vt)          # approx matrix
    preds = pd.DataFrame(X_hat, index=pivot_df.index, columns=pivot_df.columns)
    return preds, svd

# PMI co-occurrence across users (items co-rated by same users)
def build_item_pmi(reviews: pd.DataFrame):
    user_items = reviews.groupby('user_id')['product_id'].apply(lambda s: set(s.tolist()))
    item_counts = Counter()
    co_counts = defaultdict(int)
    n_users = len(user_items)
    for items in user_items:
        items = sorted(items)
        for i in items:
            item_counts[i] += 1
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                a, b = items[i], items[j]
                co_counts[(a,b)] += 1
                co_counts[(b,a)] += 1
    pmi = {}
    for (a,b), c in co_counts.items():
        p_a = item_counts[a] / n_users
        p_b = item_counts[b] / n_users
        p_ab = c / n_users
        val = math.log((p_ab / (p_a * p_b)) + 1e-12)
        if val > 0:
            pmi.setdefault(a, {})[b] = val
    return pmi, item_counts
