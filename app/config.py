from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
ARTIFACTS_DIR = BASE / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS_CSV = DATA_DIR / "amazon_products.csv"
REVIEWS_CSV = DATA_DIR / "amazon_reviews.csv"

# SVD / embedding sizes
TFIDF_MAX_FEATURES = 8000
SVD_DIM = 128
USER_ITEM_DIM = 64
