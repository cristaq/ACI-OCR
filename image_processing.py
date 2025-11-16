import cv2
import numpy as np


def apply_clahe(gray_img):
    """Apply CLAHE contrast normalization."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def binarize_adaptive(gray_img):
    """Adaptive thresholding."""
    return cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 11
    )

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center
def deskew(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def crop_text_region(image):
    """Find and crop the main text block."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image  # fallback

    # largest contour = main text region
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return image[y:y+h, x:x+w]


def process_image(input_path, invert_colors=False):
    img = cv2.imread(input_path)

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE
    clahe_img = apply_clahe(gray)

    # 3. Noise reduction
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # 4. Adaptive thresholding
    bin_img = binarize_adaptive(clahe_img)

    # 5. Optional inversion
    if invert_colors:
        bin_img = cv2.bitwise_not(bin_img)

    # Prepare for deskewing: work on original but with new mask
    pre_deskew = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    angle = getSkewAngle(img)
    # 6. Deskew
    deskewed = deskew(img, -1 * angle)

    # 7. Crop text block
    cropped = crop_text_region(deskewed)

    return {
        "gray": gray,
        "clahe": clahe_img,
        "blurred": blurred,
        "binarized": bin_img,
        "deskewed": deskewed,
        "cropped": cropped
    }


if __name__ == "__main__":
    input_image = "p24.jpg"
    results = process_image(input_image, invert_colors=True)

    # Save results
    for key, img in results.items():
        cv2.imwrite(f"{key}.png", img)

    print("Processing complete! Output images saved.")
