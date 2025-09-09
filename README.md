# Amazon Product Recommendation System  

A modular, extensible **recommendation engine** built on the **Amazon product dataset** (~1500 entries).  
This project demonstrates **multiple recommendation strategies** — content-based (CB), collaborative filtering (CF), demographic filtering (DF), hybrid approaches and experimental **recommendation genome** structure approach.  

---

## Dataset  

The dataset contains Amazon products and reviews with the following fields:  

- `product_id`, `product_name`, `category`  
- `discounted_price`, `actual_price`, `discount_percentage`  
- `rating`, `rating_count`  
- `about_product` (text description)  
- `review_id`, `user_id`, `user_name`, `review_title`, `review_content`  

Dataset link : https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset

--> **Limitation**: Dataset contains only ~1500 entries which is very small scale, so accuracy is limited.

---

## Recommendation Genome  

The **genome** is a structured representation of items and user preferences using a set of interpretable features (signals) that capture different aspects of relevance.  
Mathematically, it is a **high-dimensional vector space** where each dimension corresponds to a feature (e.g., topic, sentiment, price behavior, category match), and items/users are mapped as points.  
By comparing these vectors, the system can identify nuanced similarities and generate richer recommendations.  
In essence, the genome provides a **unified language** for comparing items and users across multiple perspectives.  

### Product Genome Representation  

In this project, each product is represented using a **genome vector** – a compressed, multi-signal embedding that captures different aspects of the product.  

The genome is built from:  
1. **Textual features** – TF-IDF on product descriptions + aggregated reviews (reduced with TruncatedSVD).  
2. **Categorical features** – One-hot encoding of product categories.  
3. **Numeric features** – Price, discount, rating, rating count, and average review sentiment (scaled).  

These features are concatenated and compressed into a **latent vector space** (via SVD), producing a **genome embedding** for every product.  

This allows us to:  
- Compare products on multiple dimensions (not just text or ratings).  
- Represent users as intent vectors (preferences) in the same genome space.  
- Use cosine similarity to recommend products closest to the user’s genome profile.  

---

## Implemented Methods  

### 1. Content-Based Filtering (CBF)  
- TF-IDF vectorization on product names/descriptions.  
- Cosine similarity for nearest neighbors.  

### 2. Collaborative Filtering (CF)  
- User–Item rating matrix.  
- Matrix Factorization.  

### 3. Demographic Filtering (DF)  
- Rule-based matching on user demographics/preferences:  
  - Category interests  
  - Price sensitivity  
  - Brand affinity  

### 4. Hybrid Model  
- Combines CB + CF + DF into a **weighted fusion model**.  
- Produces more robust recommendations.  

---
##  Installation & Running the Project

Follow these steps to set up the project locally.

---
### 1. Clone the Repository

```bash
git clone git remote add origin https://github.com/buzzz341/product_recommendation_system.git
cd product_recommendation_system
```

### 2. Setup Python Backend (FastAPI)
a. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # On Linux / macOS
venv\Scripts\activate         # On Windows
```
b. Install dependencies:
```bash
pip install -r requirements.txt
```
c. Start the backend
```bash
uvicorn app.main:app --reload
```

### 3. Setup Frontend 
The frontend here is a demo app built with Streamlit that calls backend APIs.

a. Install Streamlit (if not already installed):
```bash
pip install streamlit requests
```
b. Run the frontend:
```bash
streamlit run frontend/streamlit_app.py
```

---

### ⭐ Final Note
This repository is intended as a **prototype & learning framework** for building hybrid recommendation systems with an experimental **genome representation**.  
It demonstrates how different paradigms — content-based, collaborative, demographic, and hybrid — can be integrated in a modular way.  
