# app/api/routes.py
from fastapi import APIRouter, HTTPException
from ..models import BuildRequest, RecommendRequest, RecommendResponse, RecommendItem
from ..recommender import preprocessing, content, collaborative, demographic, hybrid, persistence
from ..config import ARTIFACTS_DIR, DATA_DIR
import pandas as pd
import numpy as np
from pathlib import Path

router = APIRouter()
_cache = {}

@router.post("/build")
def build_artifacts(req: BuildRequest):
    
    # Load & split
    products, reviews = preprocessing.load_and_prepare()

    persistence.save(products, "products_meta.pkl")
    persistence.save(reviews, "reviews.pkl")

    # product genome (content vectors)
    prod_vecs, tfidf, svd, ohe, scaler, prod_index = content.build_product_genome(products.reset_index(), reviews.reset_index(), force=req.force)

    pivot = collaborative.build_user_item_matrix(reviews, min_user_ratings=1)
    cf_preds = None
    try:
        cf_preds, svd_cf = collaborative.simple_svd_predict(pivot, n_components=64)
        persistence.save(cf_preds, "cf_preds.pkl")
    except Exception:
        cf_preds = None

    pmi_graph, item_counts = collaborative.build_item_pmi(reviews)
    persistence.save({'pmi': pmi_graph, 'counts': item_counts}, "pmi_graph.pkl")
    persistence.save(prod_vecs, "product_vectors.pkl")

    _cache.update({
        'products': products,
        'reviews': reviews,
        'product_vectors': prod_vecs,
        'cf_preds': cf_preds,
        'pmi_graph': pmi_graph
    })
    return {"status": "ok", "message": "Artifacts built and saved."}

@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):

    try:
        if 'product_vectors' not in _cache:
            _cache['products'] = persistence.load("products_meta.pkl")
            _cache['reviews'] = persistence.load("reviews.pkl")
            _cache['product_vectors'] = persistence.load("product_vectors.pkl")
            pmi_blob = persistence.load("pmi_graph.pkl")
            _cache['pmi_graph'] = pmi_blob['pmi']
            _cache['cf_preds'] = persistence.load("cf_preds.pkl") if (ARTIFACTS_DIR / "cf_preds.pkl").exists() else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Artifacts missing. Run /build first. Err: {e}")

    products = _cache['products']
    prod_vecs = _cache['product_vectors']
    reviews = _cache['reviews']
    pmi_graph = _cache['pmi_graph']
    cf_preds = _cache.get('cf_preds', None)

    q = req.questionnaire.dict()
    # intent vector: choose products in favorite categories or explicit_favorites
    mask = pd.Series(False, index=products.index)
    if q.get('favorite_categories'):
        fav = [c.lower() for c in q['favorite_categories']]
        mask = products['category'].astype(str).str.lower().apply(lambda s: any(f in s for f in fav))
    if q.get('explicit_favorites'):
        exp = set([e.lower() for e in q['explicit_favorites']])
        mask |= products['product_name'].astype(str).str.lower().apply(lambda s: any(e in s for e in exp))

    if mask.sum() == 0:
        intent_vec = prod_vecs.values.mean(axis=0)
    else:
        selected_ids = products.loc[mask].index.tolist()
        # prod_vecs index: product_id strings
        # ensure selected_ids exist in prod_vecs index; if not, fallback to mean
        selected_ids = [sid for sid in selected_ids if sid in prod_vecs.index]
        if not selected_ids:
            intent_vec = prod_vecs.values.mean(axis=0)
        else:
            intent_vec = prod_vecs.loc[selected_ids].values.mean(axis=0)

    # content scores
    content_scores = content.content_score_for_user(intent_vec, prod_vecs)

    # collaborative / cf scores
    if cf_preds is not None:
        # cf_preds: DataFrame users x items; aggregate to item score by mean predicted rating
        cf_scores = cf_preds.mean(axis=0)
        cf_scores = cf_scores.reindex(content_scores.index).fillna(0.0)
    else:
        # fallback: use PMI co-occurrence using explicit favorites (match by product name substring)
        seeds = [s.lower() for s in q.get('explicit_favorites', []) if s.strip()]
        ex_scores = {}
        for seed_name in seeds:
            matched = products[products['product_name'].str.lower().str.contains(seed_name, na=False)]
            for pid in matched.index:
                for nbr, w in pmi_graph.get(pid, {}).items():
                    ex_scores[nbr] = max(ex_scores.get(nbr, 0.0), w)
        cf_scores = pd.Series(ex_scores)
        cf_scores = cf_scores.reindex(content_scores.index).fillna(0.0)

    # demographic/compatibility
    compat = demographic.compatibility_score(q, products)

    # blend
    blended = hybrid.blend(content_scores, cf_scores, compat, weights=tuple(req.weights))
    candidates = list(blended.sort_values(ascending=False).index)

    # rerank using MMR with content as relevance
    mmr_selected = hybrid.mmr(candidates, content_scores, prod_vecs, k=req.top_k, lam=req.mmr_lambda)

    results = []
    for pid in mmr_selected:
        meta = products.loc[pid] if pid in products.index else {}
        results.append(RecommendItem(
            product_id=str(pid),
            product_name=str(meta.get('product_name', '')),
            score=float(blended.get(pid, 0.0)),
            content=float(content_scores.get(pid, 0.0)),
            cf=float(cf_scores.get(pid, 0.0)),
            compatibility=float(compat.get(pid, 0.0))
        ))

    return RecommendResponse(recommendations=results)
