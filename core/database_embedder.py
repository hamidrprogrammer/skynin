# core/database_embedder.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# --- Configuration ---
DATA_PATH = 'data/mock_products.csv'
SAVE_DIR = 'saved_models'
INDEX_PATH = os.path.join(SAVE_DIR, 'faiss_index_v2.bin') # New name for the index
DATA_MAP_PATH = os.path.join(SAVE_DIR, 'products_data_v2.pkl') # New name for the data map
MODEL_NAME = 'all-MiniLM-L6-v2'

def create_product_embeddings():
    """
    Reads the advanced product database, creates semantic embeddings,
    and saves them into a new FAISS index.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load the enhanced dataset
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Product data file not found at '{DATA_PATH}'.")
        return

    # 2. Create a comprehensive text document for each product
    df['combined_text'] = (
        df['english_name'].fillna('') + ". " +
        df['english_description'].fillna('') + ". " +
        df['english_meta_title'].fillna('') + ". " +
        df['english_meta_description'].fillna('')
    )

    print("Loading embedding model and processing products...")

    # 3. Load the pre-trained sentence transformer model
    model = SentenceTransformer(MODEL_NAME)

    # 4. Generate embeddings for all product descriptions
    product_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
    product_embeddings = np.array(product_embeddings).astype('float32')

    print(f"Embedding creation complete. Vector shape: {product_embeddings.shape}")

    # 5. Build and save the FAISS index
    embedding_dimension = product_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(product_embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to: {INDEX_PATH}")

    # 6. Save the corresponding product data for later retrieval
    product_data = df.to_dict(orient='records')
    with open(DATA_MAP_PATH, 'wb') as f:
        pickle.dump(product_data, f)
    print(f"Product data map saved to: {DATA_MAP_PATH}")
    print("\nDatabase processing is complete.")

if __name__ == '__main__':
    create_product_embeddings()