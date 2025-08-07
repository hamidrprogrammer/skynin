# core/skin_analyzer.py

import cv2
import dlib
import numpy as np
import os

# --- راه‌اندازی اولیه مدل‌های تشخیص چهره ---
# این فایل باید دانلود شده و در کنار اسکریپت قرار گیرد
# لینک دانلود: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(PREDICTOR_PATH):
    print("="*50)
    print(f"لطفاً فایل '{PREDICTOR_PATH}' را از لینک زیر دانلود کرده و در کنار اسکریپت قرار دهید:")
    print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("پس از دانلود، فایل را از حالت فشرده خارج کنید.")
    print("="*50)
    # exit() # در حالت واقعی برنامه را متوقف کنید

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError:
    predictor = None # اگر فایل موجود نباشد، ادامه نمی‌دهیم


def analyze_skin_features(image_path: str) -> dict:
    """
    یک تصویر از چهره را تحلیل کرده و ویژگی‌های پوستی را به صورت ساختاریافته باز می‌گرداند.
    این یک شبیه‌ساز هوشمند است که در آینده می‌تواند با یک مدل Deep Learning واقعی جایگزین شود.

    Args:
        image_path (str): مسیر فایل تصویر.

    Returns:
        dict: یک دیکشنری شامل ویژگی‌های تحلیل‌شده پوست.
    """
    if not predictor:
        raise FileNotFoundError(f"مدل '{PREDICTOR_PATH}' یافت نشد. لطفاً فایل را دانلود کنید.")
        
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"تصویر در مسیر '{image_path}' یافت نشد یا قابل خواندن نیست.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return {"error": "هیچ چهره‌ای در تصویر تشخیص داده نشد."}

    # فرض می‌کنیم بزرگترین چهره، چهره اصلی است
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    landmarks = predictor(gray, face)

    # --- شبیه‌سازی استخراج ویژگی‌ها ---

    # ۱. تحلیل رنگ پوست (Skin Tone)
    # با میانگین‌گیری رنگ در ناحیه گونه‌ها
    cheek_left_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(1, 5)])
    cheek_right_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(12, 16)])
    
    cheek_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(cheek_mask, [cheek_left_pts, cheek_right_pts], 255)
    
    # استفاده از ماسک برای محاسبه میانگین رنگ پوست
    mean_skin_color_bgr = cv2.mean(img, mask=cheek_mask)
    
    # تبدیل به HSV برای تحلیل بهتر رنگ
    mean_skin_color_hsv = cv2.cvtColor(np.uint8([[mean_skin_color_bgr[:3]]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # شبیه‌سازی دسته‌بندی تناژ پوست
    hue = mean_skin_color_hsv[0]
    if 5 < hue < 25:
        skin_tone_desc = "fair to light with neutral or yellow undertones"
    elif 25 <= hue < 40:
        skin_tone_desc = "medium tan or olive"
    else:
        skin_tone_desc = "light with pink undertones"

    # ۲. تخمین میزان قرمزی (Redness Level)
    # بررسی کانال "a" در فضای رنگی LAB که نماینده قرمزی-سبزی است
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab_img[:,:,1]
    forehead_pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(19, 22)])
    forehead_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(forehead_mask, [forehead_pts], 255)
    
    redness_value = cv2.mean(a_channel, mask=forehead_mask)[0]
    
    if redness_value > 135:
        redness_level = "high"
    elif 130 < redness_value <= 135:
        redness_level = "medium"
    else:
        redness_level = "low"

    # ۳. تخمین میزان چین و چروک (Wrinkle Level) - با تحلیل بافت
    # استفاده از واریانس لاپلاسین در اطراف چشم‌ها
    eye_region_left = (
        landmarks.part(36).x, landmarks.part(37).y,
        landmarks.part(39).x - landmarks.part(36).x,
        landmarks.part(41).y - landmarks.part(37).y
    )
    eye_region_right = (
        landmarks.part(42).x, landmarks.part(43).y,
        landmarks.part(45).x - landmarks.part(42).x,
        landmarks.part(47).y - landmarks.part(43).y
    )
    
    x, y, w, h = eye_region_left
    left_eye_gray = gray[y:y+h, x:x+w]
    laplacian_var_left = cv2.Laplacian(left_eye_gray, cv2.CV_64F).var() if h > 0 and w > 0 else 0
    
    x, y, w, h = eye_region_right
    right_eye_gray = gray[y:y+h, x:x+w]
    laplacian_var_right = cv2.Laplacian(right_eye_gray, cv2.CV_64F).var() if h > 0 and w > 0 else 0

    wrinkle_metric = (laplacian_var_left + laplacian_var_right) / 2
    
    if wrinkle_metric > 150:
        wrinkle_level = "visible fine lines and wrinkles"
    elif 80 < wrinkle_metric <= 150:
        wrinkle_level = "slight wrinkles"
    else:
        wrinkle_level = "smooth skin"

    # ۴. شبیه‌سازی تشخیص نوع پوست (چرب، خشک، مختلط) - با تحلیل درخشندگی
    # استفاده از روشنایی در ناحیه پیشانی
    brightness = cv2.mean(gray, mask=forehead_mask)[0]
    
    if brightness > 160:
        skin_type = "oily"
    elif brightness < 110:
        skin_type = "dry"
    else:
        skin_type = "combination"


    return {
        "status": "success",
        "detected_features": {
            "skin_tone_description": skin_tone_desc,
            "skin_type": skin_type,
            "redness_level": redness_level,
            "wrinkle_level": wrinkle_level,
        }
    }

# --- مثال نحوه استفاده ---
if __name__ == '__main__':
    # یک عکس نمونه را برای تست قرار دهید
    # می‌توانید یک عکس چهره در ریشه پروژه با نام 'sample_face.jpg' قرار دهید
    sample_image = 'sample_face.jpg' 
    if os.path.exists(sample_image):
        features = analyze_skin_features(sample_image)
        import json
        print(json.dumps(features, indent=4, ensure_ascii=False))
    else:
        print(f"فایل نمونه '{sample_image}' برای تست یافت نشد. لطفاً یک عکس در مسیر پروژه قرار دهید.")