import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
from ..config import ARTIFACTS_DIR, TFIDF_MAX_FEATURES, SVD_DIM
from pathlib import Path

ART_TFIDF = ARTIFACTS_DIR / "tfidf_joblib.pkl"
ART_SVD = ARTIFACTS_DIR / "svd_joblib.pkl"
ART_OHE = ARTIFACTS_DIR / "ohe_categories.pkl"
ART_PROD_VEC = ARTIFACTS_DIR / "product_vectors.pkl"
ART_PRODUCT_INDEX = ARTIFACTS_DIR / "product_index.pkl"
ART_SCALER = ARTIFACTS_DIR / "scaler_num.pkl"

def build_product_genome(products: pd.DataFrame, reviews: pd.DataFrame, force=False):
    """
    Build product genome: text TF-IDF + SVD, category OHE, numeric features (price, discount, rating, sentiment)
    Returns: product_vectors (DataFrame indexed by product_id)
    """
    if not force and ART_PROD_VEC.exists():
        prod_vecs = joblib.load(ART_PROD_VEC)
        tfidf = joblib.load(ART_TFIDF)
        svd = joblib.load(ART_SVD)
        ohe = joblib.load(ART_OHE)
        scaler = joblib.load(ART_SCALER)
        index = joblib.load(ART_PRODUCT_INDEX)
        return prod_vecs, tfidf, svd, ohe, scaler, index

    # 1) Text: product descriptions + aggregated reviews (optionally include review summaries)
    products = products.copy()
    reviews = reviews.copy()
    # aggregate reviews text & sentiment per product
    rev_agg = reviews.groupby('product_id').agg({
        'review_text': lambda xs: " ".join(xs.astype(str).tolist()[:30]),
        'sentiment': 'mean'
    }).rename(columns={'review_text': 'reviews_concat', 'sentiment': 'review_sentiment'})
    products = products.set_index('product_id').join(rev_agg, how='left')
    products['reviews_concat'] = products['reviews_concat'].fillna('')
    products['combined_text'] = (products['description'].fillna('') + ' ' + products['reviews_concat']).astype(str)

    corpus = products['combined_text'].fillna('').tolist()
    tfidf = TfidfVectorizer(min_df=3, ngram_range=(1,2), max_features=TFIDF_MAX_FEATURES)
    X_text = tfidf.fit_transform(corpus)

    # 2) Numeric features
    num_df = pd.DataFrame({
        'price': products['price'].fillna(products['price'].median()),
        'discount': products['discount'].fillna(0.0),
        'rating': products['rating'].fillna(0.0),
        'rating_count': np.log1p(products['rating_count'].fillna(0).astype(float)),
        'review_sentiment': products.get('review_sentiment', 0.0).fillna(0.0)
    }, index=products.index)
    scaler = StandardScaler(with_mean=False)
    X_num = scaler.fit_transform(num_df.values)

    # 3) Category one-hot
    cats = products['category'].fillna('unknown').astype(str)
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_cat = ohe.fit_transform(cats.values.reshape(-1,1))

    # 4) Concatenate sparse & dense features
    X_num_sparse = csr_matrix(X_num)
    X = hstack([X_text, X_cat, X_num_sparse], format='csr')

    # 5) TruncatedSVD to compress to SVD_DIM
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42, n_iter=7)
    X_latent = svd.fit_transform(X)  # shape = (n_products, SVD_DIM)

    #product vectors DataFrame
    prod_index = list(products.index)
    prod_vecs = pd.DataFrame(X_latent, index=prod_index)

    joblib.dump(tfidf, ART_TFIDF)
    joblib.dump(svd, ART_SVD)
    joblib.dump(ohe, ART_OHE)
    joblib.dump(prod_vecs, ART_PROD_VEC)
    joblib.dump(prod_index, ART_PRODUCT_INDEX)
    joblib.dump(scaler, ART_SCALER)

    return prod_vecs, tfidf, svd, ohe, scaler, prod_index

def content_score_for_user(intent_vec: np.ndarray, product_vectors: pd.DataFrame):
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(product_vectors.values, intent_vec.reshape(1,-1)).ravel()
    return pd.Series(sims, index=product_vectors.index)
