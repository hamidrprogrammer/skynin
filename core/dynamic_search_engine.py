# core/dynamic_search_engine.py
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# --- Configuration ---
INDEX_PATH_V2 = 'saved_models/faiss_index_v2.bin'
DATA_MAP_PATH_V2 = 'saved_models/products_data_v2.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Load Models and Data ---
if not os.path.exists(INDEX_PATH_V2):
    raise FileNotFoundError("FAISS index v2 not found. Please run 'database_embedder.py' first.")

index = faiss.read_index(INDEX_PATH_V2)
with open(DATA_MAP_PATH_V2, 'rb') as f:
    product_data = pickle.load(f)
model = SentenceTransformer(MODEL_NAME)

def translate_metrics_to_query(metrics: dict) -> str:
    """
    Translates numeric skin metrics into a rich, descriptive text query for semantic search.
    This is the core of the dynamic system.
    """
    descriptions = []

    # Translate wrinkle score
    if metrics['wrinkle_score'] > 60:
        descriptions.append("powerful anti-aging and anti-wrinkle solutions for mature skin")
    elif metrics['wrinkle_score'] > 30:
        descriptions.append("products that target fine lines and improve skin texture")

    # Translate oiliness score
    if metrics['oiliness_score'] > 70:
        descriptions.append("strong oil control formulas for very oily skin, focusing on matte finish and pore minimization")
    elif metrics['oiliness_score'] > 40:
        descriptions.append("lightweight moisturizers for combination to oily skin that balance hydration and control shine")
    
    # Translate redness score
    if metrics['redness_score'] > 50:
        descriptions.append("soothing, calming ingredients for sensitive skin prone to high redness and irritation, like centella asiatica or niacinamide")
    elif metrics['redness_score'] > 20:
        descriptions.append("gentle formulas to reduce mild redness")

    # Check for dryness (inverse of oiliness)
    if metrics['oiliness_score'] < 20:
        descriptions.append("deeply hydrating and nourishing products for dry, dehydrated, or flaky skin, perhaps with oils or hyaluronic acid")

    # Build the final query
    if not descriptions:
        base_query = "General purpose, gentle skincare for normal, balanced skin."
    else:
        base_query = "Looking for skincare products that address the following concerns: " + ", ".join(descriptions) + "."

    base_query += f" The user's skin tone is {metrics['skin_tone_description']}."
    return base_query

def find_products_dynamically(skin_metrics: dict, top_k: int = 3) -> list:
    """
    Uses the dynamic query to perform a semantic search and find the best matching products.
    """
    # 1. Translate metrics into a descriptive query
    query_text = translate_metrics_to_query(skin_metrics)
    print(f"🧠 Dynamic Query Generated: '{query_text}'")

    # 2. Encode the query into a vector
    query_embedding = model.encode([query_text])
    query_embedding = np.array(query_embedding).astype('float32')

    # 3. Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # 4. Retrieve and return the results
    results = [product_data[i] for i in indices[0]]
    print(f"✅ Found {len(results)} relevant products from the new index.")
    return results

# --- For Standalone Testing ---
if __name__ == '__main__':
    # Simulate a dynamic output from our new analyzer
    sample_metrics = {
        'wrinkle_score': 65.7,
        'oiliness_score': 25.1,
        'redness_score': 55.4,
        'skin_tone_description': 'light with pink or ruddy undertones'
    }

    recommended_products = find_products_dynamically(sample_metrics, top_k=3)

    print("\n--- Top 3 Products Found Dynamically ---")
    for prod in recommended_products:
        print(f"\n  - Product: {prod['english_name']}")
        print(f"    Description: {prod['english_description']}")