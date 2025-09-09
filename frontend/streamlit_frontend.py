import streamlit as st
import requests
import json

API_BASE = st.text_input("API base URL", value="http://localhost:8000/api")

st.title("Product Genome Hybrid Recommender — Single CSV")

if st.button("Build artifacts (run once)"):
    r = requests.post(f"{API_BASE}/build", json={"force": False})
    st.write(r.json())

st.header("Tell us about you / preferences")
fav_cats = st.text_input("Favorite categories (comma separated)", "electronics, books")
pref_brands = st.text_input("Preferred brands (comma separated)", "")
price_level = st.selectbox("Avg price level", ["", "low", "mid", "high"])
price_sens = st.slider("Price sensitivity", 0.0, 2.0, 1.0)
explicit_fav = st.text_input("Explicit favorite product name substring", "")

top_k = st.slider("Top K", 1, 30, 10)
w_content = st.slider("Content weight", 0.0, 1.0, 0.45)
w_cf = st.slider("CF weight", 0.0, 1.0, 0.35)
w_demo = st.slider("Demographic weight", 0.0, 1.0, 0.20)
mmr_lambda = st.slider("MMR lambda", 0.0, 1.0, 0.7)

if st.button("Get recommendations"):
    payload = {
        "questionnaire": {
            "avg_price_level": price_level,
            "favorite_categories": [c.strip() for c in fav_cats.split(",") if c.strip()],
            "preferred_brands": [b.strip() for b in pref_brands.split(",") if b.strip()],
            "price_sensitivity": float(price_sens),
            "explicit_favorites": [s.strip() for s in explicit_fav.split(",") if s.strip()]
        },
        "top_k": top_k,
        "weights": [w_content, w_cf, w_demo],
        "mmr_lambda": mmr_lambda
    }
    r = requests.post(f"{API_BASE}/recommend", json=payload)
    if r.status_code != 200:
        st.error(f"API error: {r.status_code}\n{r.text}")
    else:
        resp = r.json()
        for rec in resp['recommendations']:
            st.markdown(f"### {rec['product_name']}  (id: {rec['product_id']})")
            st.write(f"score: {rec['score']:.3f} | content: {rec['content']:.3f} | cf: {rec['cf']:.3f} | compat: {rec['compatibility']:.3f}")
            st.markdown("---")
