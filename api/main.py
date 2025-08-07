# api/main.py
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# اضافه کردن مسیر ریشه پروژه برای پیدا کردن ماژول‌های core
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.middleware.cors import CORSMiddleware

# --- وارد کردن تمام ماژول‌های پیشرفته و جدید ---
from core.advanced_skin_analyzer import analyze_skin_pro
from core.dynamic_search_engine import find_products_dynamically
from core.ai_recommender import generate_ai_recommendation

# --- راه‌اندازی اپلیکیشن FastAPI ---
app = FastAPI(
    title="💡 Advanced AI Skincare Advisor (v2.0)",
    description="A fully dynamic API that performs a deep, metric-based skin analysis from an image and provides an expert, data-driven product recommendation using Generative AI.",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در حالت production بهتره دامین خاص بدی
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "AI Skincare Advisor v2.0 is running."}

@app.post("/generate-dynamic-recommendation/", tags=["Core Functionality"])
async def get_dynamic_recommendation(image: UploadFile = File(...)):
    """
    Upload an image to receive a complete, dynamic, and data-driven skincare consultation.
    """
    # ایجاد یک نام فایل منحصر به فرد
    unique_filename = f"{uuid.uuid4()}{os.path.splitext(image.filename)[1]}"
    temp_image_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)

    try:
        # ذخیره فایل آپلود شده
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # --- **مرحله ۱: تحلیل عمیق و پویای پوست** ---
        skin_analysis = analyze_skin_pro(temp_image_path)
        print(skin_analysis)
        if skin_analysis.get("status") != "success":
            raise HTTPException(status_code=400, detail=skin_analysis.get("message"))
        
        skin_metrics = skin_analysis["metrics"]

        # --- **مرحله ۲: جستجوی پویا برای یافتن محصولات مرتبط** ---
        candidate_products = find_products_dynamically(skin_metrics, top_k=3)
        if not candidate_products:
            raise HTTPException(status_code=404, detail="No suitable products found for the analyzed skin profile.")

        # --- **مرحله ۳: تولید پیشنهاد نهایی توسط تحلیلگر هوش مصنوعی** ---
        final_recommendation = generate_ai_recommendation(skin_metrics, candidate_products)

        # ترکیب تمام نتایج در یک پاسخ کامل و شفاف
        return JSONResponse(content={
            "status": "success",
            "analyzed_metrics": skin_metrics,
            "ai_consultation_report": final_recommendation["text"],
            "image": final_recommendation["images"]
        })

    except Exception as e:
        # مدیریت خطاهای احتمالی
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

    finally:
        # پاک‌سازی فایل موقت
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)