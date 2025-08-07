# core/ai_recommender.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# برای تست مستقل، توابع مراحل قبل را وارد می‌کنیم
from core.dynamic_search_engine import find_products_dynamically

# بارگذاری کلید API
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
import pyodbc

def get_connection():
    conn = pyodbc.connect(
        "Driver={SQL Server};"
        "Server=localhost;"
        "Database=LioxsaPlatform;"
        "Trusted_Connection=yes;"
    )
    return conn

def fetch_product_image_data(product_ids):
    import base64

    if not product_ids:
        return {}

    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ','.join(['?'] * len(product_ids))
    query = f"""
    SELECT [Id],
           [ProductId],
           [ImageId],
           [ThumbnailImageId],
           [ImageType],
           [CreateDateTime],
           [UpdateDateTime],
           [CreateUserId],
           [LastUpdateUserId]
    FROM [Product].[ProductImage]
    WHERE ProductId IN ({placeholders})
    """

    cursor.execute(query, product_ids)
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]

    result = {}
    for row in rows:
        row_dict = dict(zip(columns, row))
        pid = str(row_dict['ProductId'])

        # گرفتن اطلاعات فایل از جدول Blob.FileData
        image_id = row_dict.get("ImageId")
        if image_id:
            cursor.execute("""
                SELECT [Id], [Content], [FileAddress], [MimeType], [FileName], [FileSize],
                       [CreateDateTime], [UpdateDateTime], [CreateUserId], [LastUpdateUserId]
                FROM [Blob].[FileData]
                WHERE Id = ?
            """, (image_id,))
            file_row = cursor.fetchone()
            if file_row:
                file_columns = [column[0] for column in cursor.description]
                file_dict = dict(zip(file_columns, file_row))

                # تبدیل Content به base64
                if file_dict.get("Content"):
                    file_dict["Content"] = base64.b64encode(file_dict["Content"]).decode("utf-8")

                row_dict["FileData"] = file_dict

        if pid not in result:
            result[pid] = []
        result[pid].append(row_dict)

    conn.close()
    return result
def generate_ai_recommendation(skin_metrics: dict, product_list: list) -> dict:
    # product_list ممکنه لیست دیکشنری باشه، پس
    # اگر product_list دیکشنری است، idها را استخراج کن، وگرنه فرض کن خودش لیست id است.

    if product_list and isinstance(product_list[0], dict):
        product_ids = [p['id'] for p in product_list]
        product_info_map = {p['id']: p for p in product_list}
    else:
        product_ids = product_list
        product_info_map = {}

    product_images_data = fetch_product_image_data(product_ids)
    print("product_info_map keys:", list(product_info_map.keys()))
    print("product_images_data keys:", list(product_images_data.keys()))
    product_info_map = {str(p['id']): p for p in product_list}


    # اگر product_info_map وجود دارد، اطلاعات نام و توضیح را اضافه کن
    if product_info_map:
        for pid, images in product_images_data.items():
            info = product_info_map.get(pid)
            if info:
                for image in images:
                    image['english_name'] = info.get('english_name')
                    image['english_description'] = info.get('english_description')

    # --- **پرامپت مهندسی‌شده برای تحلیل داده (Data-Driven Prompt)** ---
    prompt = f"""
    **Your Role:** You are a highly-respected data scientist specializing in dermatology. Your analysis is based purely on quantitative data. You must be objective, precise, and justify every conclusion with the provided metrics.

    **User's Quantitative Skin Profile (Scale 0-100):**
    - **Wrinkle Index:** {skin_metrics['wrinkle_score']}
    - **Oiliness Index:** {skin_metrics['oiliness_score']}
    - **Redness/Irritation Index:** {skin_metrics['redness_score']}
    - **Analyzed Skin Tone:** {skin_metrics['skin_tone_description']}

    **Candidate Products (Identified by semantic search):**
    ---
    """
   

    for i, product in enumerate(product_list, 1):
        prompt += f"""
    **Product {i}: {product.get('english_name')}**
    - **Description:** {product.get('english_description')}
    ---
    """

    prompt += """
    **Your Mission:**
    1.  **Analyze the Data:** Briefly interpret the user's quantitative metrics. For example, a high Wrinkle Index suggests a need for anti-aging ingredients like retinol. A high Redness Index points to sensitivity. A low Oiliness Index indicates dryness.
    2.  **Select the Optimal Product:** Based on your data analysis, choose the SINGLE most suitable product from the list. Your choice must be directly justified by the metrics.
    3.  **Provide a Justification:** In a section called "Analytical Justification", explain *why* you chose that product by explicitly referencing the user's scores and the product's description. For example: "The user's Redness Index is high ({skin_metrics['redness_score']}), and Product X contains Centella Asiatica, an ingredient known for calming irritation."
    4.  **Format the Output:** Present the final recommendation in clear, professional English. Use Markdown for structure. The tone should be that of an expert providing a clinical-grade consultation.
    """

    print("\n🔬 Sending data-driven prompt to Gemini...")

    try:
        response = model.generate_content(prompt)
        return {
    "text": response.text,
    "images": product_images_data
}
    except Exception as e:
        return f"An error occurred while communicating with the AI model: {e}"

# --- For Standalone Testing ---
if __name__ == '__main__':
    # Simulate a dynamic profile from the analyzer
    sample_metrics = {
        'wrinkle_score': 65.7,
        'oiliness_score': 25.1,
        'redness_score': 55.4,
        'skin_tone_description': 'light with pink or ruddy undertones'
    }

    print("--- Running Dynamic Search to Find Candidates ---")
    candidate_products = find_products_dynamically(sample_metrics, top_k=3)

    print("\n--- Sending Profile and Candidates to AI for Final Verdict ---")
    final_verdict = generate_ai_recommendation(sample_metrics, candidate_products)

    print("\n" + "="*50)
    print("🤖 **AI Dermatologist Consultation Report** 🤖")
    print("="*50)
    print(final_verdict)