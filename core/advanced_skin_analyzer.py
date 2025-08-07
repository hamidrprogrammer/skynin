# core/pro_advanced_skin_analyzer.py

import os
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from skimage.feature import local_binary_pattern

# --- Load Face Landmark Model ---
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"{PREDICTOR_PATH} not found.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# --- Load BiSeNet Face Parsing Model ---
# Download from: https://drive.google.com/file/d/1N3WQpOpI6A3ym1qKsYdcxZn5gF-w3bER/view
BISENET_PATH = "79999_iter.pth"

if not os.path.exists(BISENET_PATH):
    raise FileNotFoundError("BiSeNet model not found. Download from: https://github.com/zllrunning/face-parsing.PyTorch")

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        from core.model import BiSeNet as Net
  # assume model.py is in same dir
        self.model = Net(n_classes=n_classes)
        self.model.load_state_dict(torch.load(BISENET_PATH, map_location='cpu'))
        self.model.eval()

    def forward(self, image: np.ndarray) -> np.ndarray:
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        with torch.no_grad():
            image = cv2.resize(image, (512, 512))
            tensor = to_tensor(image).unsqueeze(0)
            out = self.model(tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            return cv2.resize(parsing.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

# --- Color Normalization (Gray World) ---
def normalize_color(img):
    b, g, r = cv2.split(img)
    r_avg, g_avg, b_avg = np.mean(r), np.mean(g), np.mean(b)
    gray_avg = (r_avg + g_avg + b_avg) / 3
    r = np.clip(r * gray_avg / (r_avg + 1e-6), 0, 255).astype(np.uint8)
    g = np.clip(g * gray_avg / (g_avg + 1e-6), 0, 255).astype(np.uint8)
    b = np.clip(b * gray_avg / (b_avg + 1e-6), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

# --- Extract Region by Mask Label ---
def extract_region(image, mask, label):
    region = cv2.bitwise_and(image, image, mask=(mask == label).astype(np.uint8) * 255)
    return region

# --- LBP Wrinkle Detection ---
def wrinkle_score_from_lbp(gray_roi):
    lbp = local_binary_pattern(gray_roi, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return min(100, max(0, hist.var() * 300))  # adjusted scaling

# --- Oiliness Detection via Bright HSV Pixels ---
def oiliness_score_from_hsv(img_hsv, mask):
    v_channel = img_hsv[:, :, 2]
    masked_v = cv2.bitwise_and(v_channel, v_channel, mask=mask)
    bright_pixels = cv2.inRange(masked_v, 220, 255)
    ratio = cv2.countNonZero(bright_pixels) / (cv2.countNonZero(mask) + 1e-6)
    return min(100, ratio * 150)  # aggressive scaling

# --- Redness Detection via LAB a* Channel ---
def redness_score_from_lab(lab_img, mask):
    a_channel = lab_img[:, :, 1]
    masked_a = cv2.bitwise_and(a_channel, a_channel, mask=mask)
    avg_a = np.mean(masked_a[mask > 0])
    return max(0, min(100, (avg_a - 128.0) * 4.0))

# --- Skin Tone Classification via HSV Hue ---
def describe_skin_tone(bgr_image, mask):
    mean_bgr = cv2.mean(bgr_image, mask=mask)[:3]
    hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hue = hsv[0]
    if 5 < hue < 23:
        return "fair to light with neutral or yellow undertones"
    elif 23 <= hue < 35:
        return "medium, tan, or olive"
    else:
        return "light with pink or ruddy undertones"
segmenter = BiSeNet()

# --- Main Function ---
def analyze_skin_pro(image_path: str) -> dict:
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image not found."}

    image = cv2.imread(image_path)
    if image is None:
        return {"status": "error", "message": "Unreadable image."}

    image = normalize_color(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    faces = detector(gray)
    if len(faces) == 0:
        return {"status": "error", "message": "No face detected."}

    face = max(faces, key=lambda r: r.width() * r.height())
    landmarks = predictor(gray, face)

    face_mask = segmenter.forward(image)

    forehead_mask = (face_mask == 1).astype(np.uint8) * 255
    nose_mask = (face_mask == 10).astype(np.uint8) * 255

    if forehead_mask.shape != gray.shape:
        forehead_mask = cv2.resize(forehead_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    if nose_mask.shape != gray.shape:
        nose_mask = cv2.resize(nose_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    tzone_mask = cv2.bitwise_or(forehead_mask, nose_mask)

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    wrinkle_roi = cv2.bitwise_and(gray, gray, mask=forehead_mask)
    wrinkle = wrinkle_score_from_lbp(wrinkle_roi)
    oiliness = oiliness_score_from_hsv(hsv_img, tzone_mask)
    redness = redness_score_from_lab(lab_img, forehead_mask)
    tone = describe_skin_tone(image, forehead_mask)

    return {
        "status": "success",
        "metrics": {
            "wrinkle_score": round(wrinkle, 2),
            "oiliness_score": round(oiliness, 2),
            "redness_score": round(redness, 2),
            "skin_tone_description": tone
        }
    }

# For testing
if __name__ == '__main__':
    test_image = "sample_face.jpg"
    result = analyze_skin_pro(test_image)
    import json
    print(json.dumps(result, indent=4))
