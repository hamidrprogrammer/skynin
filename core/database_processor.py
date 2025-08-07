# core/database_processor.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# --- تعریف مسیرها ---
DATA_PATH = 'data/mock_products.csv'
SAVE_DIR = 'saved_models'
INDEX_PATH = os.path.join(SAVE_DIR, 'faiss_index.bin')
DATA_MAP_PATH = os.path.join(SAVE_DIR, 'products_data.pkl')

# نام مدل Embedding
# این مدل یکی از بهترین و سریع‌ترین مدل‌های چندمنظوره است.
MODEL_NAME = 'all-MiniLM-L6-v2'

def process_and_embed_products():
    """
    محصولات را از CSV خوانده، برای آنها Embedding ساخته و در یک ایندکس FAISS ذخیره می‌کند.
    """
    # اطمینان از وجود پوشه برای ذخیره‌سازی
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ۱. خواندن دیتا از CSV
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"خطا: فایل دیتابیس در مسیر '{DATA_PATH}' یافت نشد.")
        return

    # ۲. ترکیب فیلدهای متنی برای ساخت یک "سند" کامل برای هر محصول
    # این کار باعث می‌شود Embedding ما درک کامل‌تری از محصول داشته باشد.
    df['combined_text'] = (
        df['english_name'].fillna('') + '. ' +
        df['english_description'].fillna('') + '. ' +
        df['english_meta_title'].fillna('') + '. ' +
        df['english_meta_description'].fillna('')
    )
    
    print("شروع فرآیند ساخت Embedding برای محصولات...")

    # ۳. بارگذاری مدل Embedding
    model = SentenceTransformer(MODEL_NAME)

    # ۴. ساخت Embedding برای تمام محصولات
    # .encode تمام متن‌ها را به وکتورهای عددی تبدیل می‌کند
    product_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

    # تبدیل به فرمت مورد نیاز FAISS
    product_embeddings = np.array(product_embeddings).astype('float32')
    
    print(f"ساخت {len(product_embeddings)} وکتور با ابعاد {product_embeddings.shape[1]} کامل شد.")

    # ۵. ساخت ایندکس FAISS
    embedding_dimension = product_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    
    # اضافه کردن وکتورها به ایندکس
    index.add(product_embeddings)

    # ۶. ذخیره‌سازی ایندکس و دیتای محصولات
    print(f"ذخیره ایندکس FAISS در مسیر: {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    # ما به دیتای اصلی هم نیاز داریم تا بعد از جستجو، اطلاعات محصول را نمایش دهیم
    product_data = df.to_dict(orient='records')
    with open(DATA_MAP_PATH, 'wb') as f:
        pickle.dump(product_data, f)
    
    print("مرحله ۲ با موفقیت به پایان رسید.")
    print(f"ایندکس و دیتای محصولات در پوشه '{SAVE_DIR}' ذخیره شدند.")


if __name__ == '__main__':
    process_and_embed_products()