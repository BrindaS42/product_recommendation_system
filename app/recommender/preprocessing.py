import pandas as pd
from typing import Tuple
from pathlib import Path
from ..utils import clean_text, safe_float, simple_sentiment
from ..config import DATA_DIR

XLSX_PATH = DATA_DIR / "amazon.xlsx"

def load_and_split(xlsx_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    path = Path(xlsx_path) if xlsx_path else XLSX_PATH
    df = pd.read_excel(path, engine="openpyxl")  

    df.columns = [c.strip() for c in df.columns]

    prod_cols = [
        'product_id', 'product_name', 'category',
        'discounted_price', 'actual_price', 'discount_percentage',
        'rating', 'rating_count', 'about_product', 'img_link', 'product_link'
    ]
    present_prod_cols = [c for c in prod_cols if c in df.columns]
    products = df[present_prod_cols].drop_duplicates(subset=['product_id']).set_index('product_id')

    # clean + add derived fields
    products['product_name'] = products['product_name'].fillna('').astype(str)
    products['about_product'] = products.get('about_product', '').fillna('').astype(str)
    products['description'] = (products['product_name'] + ' ' + products['about_product']).map(clean_text)

    def choose_price(row):
        if 'actual_price' in row and not pd.isna(row['actual_price']):
            return safe_float(row['actual_price'])
        if 'discounted_price' in row and not pd.isna(row['discounted_price']):
            return safe_float(row['discounted_price'])
        return 0.0

    products['price'] = products.apply(choose_price, axis=1)
    products['discount'] = products.get('discount_percentage', 0).fillna(0).apply(safe_float)
    products['rating'] = products.get('rating', 0).fillna(0).apply(safe_float)
    products['rating_count'] = products.get('rating_count', 0).fillna(0).apply(lambda x: int(safe_float(x, 0)))

    # Reviews dataframe
    review_cols = ['review_id', 'user_id', 'user_name', 'review_title', 'review_content', 'rating', 'product_id']
    present_review_cols = [c for c in review_cols if c in df.columns]
    reviews = df[present_review_cols].copy()
    reviews = reviews.rename(columns={'review_content': 'review_text'})
    if 'review_text' not in reviews.columns:
        reviews['review_text'] = ''

    reviews['product_id'] = reviews['product_id'].astype(str)
    reviews['user_id'] = reviews['user_id'].astype(str).fillna('anon')
    reviews['rating'] = reviews.get('rating', 0).apply(safe_float)
    reviews['review_text'] = reviews['review_text'].fillna('').astype(str)
    reviews['sentiment'] = reviews['review_text'].apply(simple_sentiment)

    products = products.reset_index().set_index('product_id')
    return products, reviews

def load_and_prepare(xlsx_path: str = None):
    products, reviews = load_and_split(xlsx_path)
    return products, reviews
