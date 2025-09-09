# app/recommender/demographic.py
import pandas as pd
import numpy as np

def user_price_bucket(price, breaks=(0,20,100,1000)):
    # returns 'low','mid','high'
    if price <= breaks[1]: return 'low'
    if price <= breaks[2]: return 'mid'
    return 'high'

def compatibility_score(questionnaire: dict, products_meta: pd.DataFrame) -> pd.Series:
    """
    Compute compatibility score between user questionnaire and each product.
    Uses favorite categories, price sensitivity and preferred brands.
    """
    q = questionnaire
    fav_cats = [c.lower() for c in q.get('favorite_categories', [])]
    pref_brands = [b.lower() for b in q.get('preferred_brands', [])]
    price_sensitivity = float(q.get('price_sensitivity', 1.0))

    # category match
    cats = products_meta['category'].fillna('').astype(str).str.lower()
    cat_score = cats.apply(lambda c: 1.0 if any(fc in c for fc in fav_cats) else (0.6 if fav_cats else 0.8))

    # price match: if user price level is given, compute similarity
    if q.get('avg_price_level'):
        user_bucket = q['avg_price_level'].lower()  
        price_bucket = products_meta['price'].fillna(products_meta['price'].median()).apply(lambda p: user_price_bucket(p))
        price_score = price_bucket.apply(lambda pb: 1.0 if pb == user_bucket else (0.7 if (pb=='mid' or user_bucket=='mid') else 0.4))
    else:
        price_score = pd.Series(0.8, index=products_meta.index)

    # price sensitivity: if user is price sensitive, boost low-priced items
    ps = float(price_sensitivity)
    #normalized rank
    price_norm = (products_meta['price'].fillna(products_meta['price'].median()) - products_meta['price'].min()) / (products_meta['price'].max() - products_meta['price'].min() + 1e-9)
    # lower price -> higher match if price sensitivity >1
    price_sens_score = 1.0 - price_norm * (ps - 1) if ps >= 1 else 1.0 - price_norm * (ps*0.5)

    final = 0.5*cat_score + 0.3*price_score + 0.2*price_sens_score
    return final.clip(0.0, 1.0)
