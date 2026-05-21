import cv2
import numpy as np
import os

# ==============================
# AYARLAR
# ==============================
INPUT_IMAGE = "resim.jpg"
OUTPUT_DIR = "ciktilar"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# Yardımcı Fonksiyonlar
# ==============================

def resize_image(image, width=900):
    h, w = image.shape[:2]
    ratio = width / w
    new_size = (width, int(h * ratio))
    resized = cv2.resize(image, new_size)
    return resized, ratio


def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def enhance_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    blur = cv2.GaussianBlur(contrast, (3, 3), 0)

    cleaned = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        12
    )

    return cleaned


def find_document_contour(image):
    resized, ratio = resize_image(image, width=900)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 10000:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            document_contour = approx
            break

    if document_contour is None:
        raise Exception("Evrak köşeleri bulunamadı. Daha net ve zıt zeminli fotoğraf kullan.")

    document_contour = document_contour / ratio
    return document_contour.astype("float32"), edges, resized


# ==============================
# Ana Program
# ==============================

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise FileNotFoundError("Görüntü bulunamadı. INPUT_IMAGE adını kontrol et.")

original = image.copy()

# 1. Evrak sınırlarını ve köşelerini bul
document_contour, edges, resized_debug = find_document_contour(image)

# 2. Maskeleme işlemi - evrak dışı beyaz yapılır
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, [document_contour.astype(np.int32)], -1, 255, -1)

white_background = np.ones_like(image) * 255

document_only = cv2.bitwise_and(image, image, mask=mask)

inverse_mask = cv2.bitwise_not(mask)
white_outside = cv2.bitwise_and(
    white_background,
    white_background,
    mask=inverse_mask
)

masked_document = cv2.add(document_only, white_outside)

# 3. Köşeleri görselleştir
corner_image = image.copy()

for point in document_contour:
    x, y = point.ravel()
    cv2.circle(corner_image, (int(x), int(y)), 12, (0, 0, 255), -1)

cv2.drawContours(
    corner_image,
    [document_contour.astype(np.int32)],
    -1,
    (0, 255, 0),
    4
)

# 4. Yamuk / trapezoid perspektifi düzelt
warped = four_point_transform(original, document_contour)

# 5. Kontrast iyileştirme ve beyaz zemin
enhanced = enhance_document(warped)

# 6. Çıktıları kaydet
cv2.imwrite(f"{OUTPUT_DIR}/1_kenar_algilama.jpg", edges)
cv2.imwrite(f"{OUTPUT_DIR}/2_maskelenmis_evrak_beyaz_arka_plan.jpg", masked_document)
cv2.imwrite(f"{OUTPUT_DIR}/3_koseler_belirlendi.jpg", corner_image)
cv2.imwrite(f"{OUTPUT_DIR}/4_perspektif_duzeltilmis.jpg", warped)
cv2.imwrite(f"{OUTPUT_DIR}/5_kontrast_iyilestirilmis_beyaz.jpg", enhanced)

print("İşlem tamamlandı.")
print("Çıktılar 'ciktilar' klasörüne kaydedildi.")