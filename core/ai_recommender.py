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

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# Map numeric scores (0-100) to qualitative severities so downstream prompts
# have explicit meaning for each value. This helps the language model reason
# about the user's skin state without guessing what a number represents.
def _severity(score: float) -> str:
    if score is None:
        return "unknown"
    if score < 33:
        return "low"
    if score < 66:
        return "moderate"
    return "high"

# Human‑readable labels for metrics returned from the analyzer
METRIC_LABELS = {
    "wrinkle_score": "Wrinkle Index",
    "oiliness_score": "Oiliness Index",
    "redness_score": "Redness/Irritation Index",
    "dryness_score": "Dryness Texture Index",
    "dark_spot_score": "Dark Spot Index",
    "pore_score": "Pore Visibility Index",
    "acne_score": "Acne/Blemish Index",
}

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

    # --- Build metric section with qualitative interpretation ---
    metric_lines = []
    for key, label in METRIC_LABELS.items():
        value = skin_metrics.get(key)
        if value is not None:
            metric_lines.append(f"- **{label}:** {value} ({_severity(value)})")
    tone_desc = skin_metrics.get('skin_tone_description')
    if tone_desc:
        metric_lines.append(f"- **Analyzed Skin Tone:** {tone_desc}")
    metrics_block = "\n".join(metric_lines)

    # --- **پرامپت مهندسی‌شده برای تحلیل داده (Data-Driven Prompt)** ---
    prompt = f"""
    **Your Role:** You are a highly-respected data scientist specializing in dermatology. Your analysis is based purely on quantitative data. You must be objective, precise, and justify every conclusion with the provided metrics.

    **Metric Interpretation Guide (0-100):**
    - 0-32 → low concern
    - 33-65 → moderate concern
    - 66-100 → high concern

    **User's Quantitative Skin Profile:**
    {metrics_block}

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
    1. **Analyze the Data:** Interpret the user's metrics using the guide above. Avoid making assumptions beyond the numbers.
    2. **Select the Optimal Product:** Choose the SINGLE best product and justify the choice directly with the metrics.
    3. **Provide a Justification:** In a section called "Analytical Justification", explicitly reference the user's scores and product descriptions.
    4. **Format the Output:** Present the final recommendation in clear, professional English with Markdown formatting.
    """

    print("\n🔬 Sending data-driven prompt to Gemini...")

    try:
        response = model.generate_content(prompt)
        return {
            "text": response.text,
            "images": product_images_data,
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
