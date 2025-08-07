# api/main.py

import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# --- وارد کردن تمام توابع اصلی از ماژول‌های core ---
# نکته: برای اینکه پایتون پوشه core را پیدا کند، ممکن است نیاز باشد
# پروژه را از پوشه ریشه (skin_care_recommender) اجرا کنید.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.skin_analyzer import analyze_skin_features
from core.search_engine import find_similar_products
from core.recommender import generate_recommendation

# --- راه‌اندازی اپلیکیشن FastAPI ---
app = FastAPI(
    title="💡 مشاور هوشمند پوست",
    description="""
    این API یک عکس چهره دریافت کرده، ویژگی‌های پوستی را تحلیل می‌کند و بهترین محصولات مراقبتی را 
    با استفاده از ترکیب جستجوی معنایی و هوش مصنوعی Gemini پیشنهاد می‌دهد.
    """,
    version="1.0.0"
)

# ایجاد یک پوشه موقت برای آپلودها
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.get("/", tags=["Status"])
def read_root():
    """یک پیام ساده برای اطمینان از بالا بودن سرویس."""
    return {"status": "Skincare Recommender API is running successfully!"}


@app.post("/recommendations/image", tags=["Recommendation"])
async def create_recommendation_from_image(image: UploadFile = File(...)):
    """
    یک فایل عکس آپلود کنید تا تحلیل پوست و پیشنهاد محصول را دریافت نمایید.
    """
    # --- مرحله ۱: ذخیره و پردازش عکس آپلود شده ---
    # ایجاد یک نام فایل منحصر به فرد برای جلوگیری از تداخل
    file_extension = os.path.splitext(image.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    temp_image_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)

    try:
        # ذخیره فایل آپلود شده در مسیر موقت
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # --- مرحله ۲: تحلیل پوست با استفاده از ماژول skin_analyzer ---
        skin_analysis_result = analyze_skin_features(temp_image_path)
        
        if skin_analysis_result.get("status") != "success":
            raise HTTPException(status_code=400, detail=skin_analysis_result.get("error", "خطا در تحلیل چهره."))
        
        skin_features = skin_analysis_result["detected_features"]

        # --- مرحله ۳: جستجوی معنایی برای یافتن محصولات مرتبط ---
        candidate_products = find_similar_products(skin_features, top_k=3)
        if not candidate_products:
            raise HTTPException(status_code=404, detail="هیچ محصول مرتبطی برای این پروفایل پوستی یافت نشد.")

        # --- مرحله ۴: تولید پیشنهاد نهایی با Gemini ---
        final_recommendation_text = generate_recommendation(skin_features, candidate_products)
        
        return JSONResponse(content={"recommendation": final_recommendation_text})

    except Exception as e:
        # در صورت بروز هرگونه خطا، یک پاسخ مناسب برمی‌گردانیم
        raise HTTPException(status_code=500, detail=f"یک خطای داخلی در سرور رخ داد: {str(e)}")
    
    finally:
        # --- پاک‌سازی ---
        # فایل موقت را پس از اتمام کار حذف می‌کنیم
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)