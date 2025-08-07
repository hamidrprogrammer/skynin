# core/pro_advanced_skin_analyzer.py

import os
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=False)
except Exception:
    _mtcnn = None

# Optional Hugging Face semantic segmentation for refined face masking
try:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    _hf_processor = AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    _hf_model = AutoModelForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    _hf_available = True
except Exception:
    _hf_processor = None
    _hf_model = None
    _hf_available = False

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

# --- Color Normalization + CLAHE Enhancement ---
def normalize_and_equalize(img):
    img = normalize_color(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# --- Extract Region by Mask Label ---
def extract_region(image, mask, label):
    region = cv2.bitwise_and(image, image, mask=(mask == label).astype(np.uint8) * 255)
    return region

# --- Basic LBP Implementation (no skimage dependency) ---
def _lbp_histogram(gray_roi: np.ndarray) -> np.ndarray:
    """Compute LBP histogram using 8 neighbors and radius 1."""
    gray = gray_roi.astype(np.uint8)
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp |= ((shifted >= gray) << i)
    hist, _ = np.histogram(lbp[1:-1, 1:-1].ravel(), bins=np.arange(257), density=True)
    return hist

def wrinkle_score_from_lbp(gray_roi: np.ndarray) -> float:
    hist = _lbp_histogram(gray_roi)
    return min(100, max(0, hist.var() * 300))

# --- Additional Gabor Based Wrinkle Analysis ---
def wrinkle_score_from_gabor(gray_roi: np.ndarray) -> float:
    responses = []
    for theta in np.linspace(0, np.pi, 4, endpoint=False):
        kernel = cv2.getGaborKernel((9, 9), sigma=4.0, theta=theta,
                                    lambd=10.0, gamma=0.5, psi=0)
        fimg = cv2.filter2D(gray_roi, cv2.CV_32F, kernel)
        responses.append(fimg)
    response = np.max(responses, axis=0)
    return min(100, np.var(response) * 0.02)

def combined_wrinkle_score(gray_roi: np.ndarray) -> float:
    lbp = wrinkle_score_from_lbp(gray_roi)
    gabor = wrinkle_score_from_gabor(gray_roi)
    return min(100, 0.5 * lbp + 0.5 * gabor)

# --- Oiliness Detection via Specular Highlights ---
def oiliness_score_from_hsv(img_hsv, mask):
    h, s, v = cv2.split(img_hsv)
    masked_s = cv2.bitwise_and(s, s, mask=mask)
    masked_v = cv2.bitwise_and(v, v, mask=mask)
    low_sat = cv2.inRange(masked_s, 0, 50)
    high_val = cv2.inRange(masked_v, 220, 255)
    highlights = cv2.bitwise_and(low_sat, high_val)
    kernel = np.ones((3, 3), np.uint8)
    highlights = cv2.morphologyEx(highlights, cv2.MORPH_OPEN, kernel)
    ratio = cv2.countNonZero(highlights) / (cv2.countNonZero(mask) + 1e-6)
    return min(100, ratio * 200)

# --- Redness Detection via LAB a* Channel ---
def redness_score_from_lab(lab_img, mask):
    a_channel = lab_img[:, :, 1]
    masked_a = cv2.bitwise_and(a_channel, a_channel, mask=mask)
    avg_a = np.mean(masked_a[mask > 0])
    return max(0, min(100, (avg_a - 128.0) * 4.0))

# --- Dark Spot Detection via Morphological Blackhat ---
def dark_spot_score_from_lab(lab_img, mask):
    l_channel = lab_img[:, :, 0]
    masked_l = cv2.bitwise_and(l_channel, l_channel, mask=mask)
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(masked_l, cv2.MORPH_BLACKHAT, kernel)
    _, spots = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ratio = cv2.countNonZero(spots) / (cv2.countNonZero(mask) + 1e-6)
    return min(100, ratio * 300)

# --- Acne Detection via LAB Color Blemishes ---
def acne_score_from_lab(lab_img, mask):
    a = lab_img[:, :, 1]
    b = lab_img[:, :, 2]
    redness = cv2.inRange(a, 150, 255)
    yellowness = cv2.inRange(b, 120, 200)
    blemish = cv2.bitwise_and(redness, yellowness)
    blemish = cv2.bitwise_and(blemish, mask)
    blemish = cv2.morphologyEx(blemish, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ratio = cv2.countNonZero(blemish) / (cv2.countNonZero(mask) + 1e-6)
    return min(100, ratio * 500)

# --- Pore Detection via Difference of Gaussians ---
def pore_score_from_dog(gray_img, mask):
    blur_small = cv2.GaussianBlur(gray_img, (3, 3), 0)
    blur_large = cv2.GaussianBlur(gray_img, (9, 9), 0)
    dog = cv2.subtract(blur_small, blur_large)
    _, thresh = cv2.threshold(dog, 5, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    ratio = cv2.countNonZero(thresh) / (cv2.countNonZero(mask) + 1e-6)
    return min(100, ratio * 800)

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


def hf_face_segmentation(image: np.ndarray) -> np.ndarray | None:
    """Return a binary face mask using a Hugging Face segmentation model."""
    if not _hf_available:
        return None
    inputs = _hf_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _hf_model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    parsing = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    face_id = None
    for idx, label in _hf_model.config.id2label.items():
        if "face" in label.lower():
            face_id = int(idx)
            break
    if face_id is None:
        return None
    return (parsing == face_id).astype(np.uint8) * 255

# --- Texture Based Dryness Metric ---
def dryness_score_from_texture(gray_roi):
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    score = (lap.var() + grad_mag.var()) / 10.0
    return min(100, score)

# --- Main Function ---
def analyze_skin_pro(image_path: str) -> dict:
    if not os.path.exists(image_path):
        return {"status": "error", "message": "Image not found."}

    image = cv2.imread(image_path)
    if image is None:
        return {"status": "error", "message": "Unreadable image."}

    image = normalize_and_equalize(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    # Prefer MTCNN for face detection if available
    face_box = None
    if _mtcnn is not None:
        boxes, _ = _mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            face_box = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))

    if face_box is None:
        faces = detector(gray)
        if len(faces) == 0:
            return {"status": "error", "message": "No face detected."}
        face_box = max(faces, key=lambda r: r.width() * r.height())

    landmarks = predictor(gray, face_box)

    face_mask = segmenter.forward(image)
    hf_mask = hf_face_segmentation(image)
    if hf_mask is not None:
        face_mask = np.where(hf_mask > 0, face_mask, 0)
    if face_mask.shape != gray.shape:
        face_mask = cv2.resize(face_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    forehead_mask = (face_mask == 1).astype(np.uint8) * 255
    nose_mask = (face_mask == 10).astype(np.uint8) * 255

    if forehead_mask.shape != gray.shape:
        forehead_mask = cv2.resize(forehead_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    if nose_mask.shape != gray.shape:
        nose_mask = cv2.resize(nose_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    tzone_mask = cv2.bitwise_or(forehead_mask, nose_mask)
    skin_mask = (face_mask == 1).astype(np.uint8) * 255
    cheeks_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(tzone_mask))

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    wrinkle_roi = cv2.bitwise_and(gray, gray, mask=forehead_mask)
    wrinkle = combined_wrinkle_score(wrinkle_roi)
    oiliness = oiliness_score_from_hsv(hsv_img, tzone_mask)
    redness = redness_score_from_lab(lab_img, forehead_mask)
    tone = describe_skin_tone(image, forehead_mask)
    dryness_roi = cv2.bitwise_and(gray, gray, mask=cheeks_mask)
    dryness = dryness_score_from_texture(dryness_roi)
    dark_spots = dark_spot_score_from_lab(lab_img, cheeks_mask)
    pores = pore_score_from_dog(gray, cheeks_mask)
    acne = acne_score_from_lab(lab_img, skin_mask)

    return {
        "status": "success",
        "metrics": {
            "wrinkle_score": round(wrinkle, 2),
            "oiliness_score": round(oiliness, 2),
            "redness_score": round(redness, 2),
            "dryness_score": round(dryness, 2),
            "dark_spot_score": round(dark_spots, 2),
            "pore_score": round(pores, 2),
            "acne_score": round(acne, 2),
            "skin_tone_description": tone
        }
    }

# For testing
if __name__ == '__main__':
    test_image = "sample_face.jpg"
    result = analyze_skin_pro(test_image)
    import json
    print(json.dumps(result, indent=4))
