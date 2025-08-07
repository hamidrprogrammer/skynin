# core/search_engine.py

import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# --- Define paths and load model and index ---
SAVE_DIR = 'saved_models'
INDEX_PATH = os.path.join(SAVE_DIR, 'faiss_index.bin')
DATA_MAP_PATH = os.path.join(SAVE_DIR, 'products_data.pkl')
MODEL_NAME = 'all-MiniLM-L6-v2'

# Check if required files exist
if not os.path.exists(INDEX_PATH) or not os.path.exists(DATA_MAP_PATH):
    raise FileNotFoundError(
        "Index or data files not found. Please run 'database_processor.py' first."
    )

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load product data
with open(DATA_MAP_PATH, 'rb') as f:
    product_data = pickle.load(f)

# Load embedding model (must be the exact same model used during indexing)
model = SentenceTransformer(MODEL_NAME)


def describe_skin_profile(features: dict) -> str:
    """
    Converts skin feature dictionary into a natural language query string.
    This function acts as our smart 'translator'.
    """
    descriptions = []

    # Priority is given to major skin concerns
    if features.get("redness_level") in ["medium", "high"]:
        descriptions.append(f"calming products for skin with {features['redness_level']} redness")

    if features.get("wrinkle_level") in ["slight wrinkles", "visible fine lines and wrinkles"]:
        descriptions.append(f"anti-aging solutions for {features['wrinkle_level']}")

    if features.get("skin_type") == "oily":
        descriptions.append("formulas for oily and acne-prone skin that control shine")
    elif features.get("skin_type") == "dry":
        descriptions.append("intense hydration for dry skin")

    base_query = f"Skincare recommendations for a person with {features.get('skin_type', 'normal')} skin. "

    if descriptions:
        base_query += "Key concerns are: " + ", ".join(descriptions) + "."

    base_query += f" The skin tone is described as {features.get('skin_tone_description', 'undefined')}."

    return base_query


def find_similar_products(skin_features: dict, top_k: int = 3) -> list:
    """
    Takes skin features, builds a query, and finds similar products from the index.

    Args:
        skin_features (dict): Output dictionary from the analyze_skin_features function.
        top_k (int): Number of top matching products to return.

    Returns:
        list: List of dictionaries representing similar products.
    """
    # 1. Convert feature dictionary to a text query
    query_text = describe_skin_profile(skin_features)
    print(f"🔍 Generated query for search: '{query_text}'")

    # 2. Create embedding for the query
    query_embedding = model.encode([query_text])
    query_embedding = np.array(query_embedding).astype('float32')

    # 3. Perform FAISS search
    # index.search returns:
    # D: distance (similarity score - lower is better)
    # I: index of the items found
    distances, indices = index.search(query_embedding, top_k)

    # 4. Retrieve found products
    results = []
    for i in indices[0]:
        results.append(product_data[i])

    print(f"\n✅ {len(results)} related product(s) found.")
    return results


# --- Example usage ---
if __name__ == '__main__':
    # Simulate sample output from skin analysis step
    sample_skin_features = {
        "skin_tone_description": "light with pink undertones",
        "skin_type": "combination",
        "redness_level": "medium",
        "wrinkle_level": "slight wrinkles"
    }

    # Search for related products
    recommended_products = find_similar_products(sample_skin_features, top_k=3)

    # Print results
    print("\n--- Recommended Products ---")
    for prod in recommended_products:
        print(f"  - Product Name: {prod['english_name']}")
        print(f"    Description: {prod['english_description']}\n")
