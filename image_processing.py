import cv2
import numpy as np


def apply_clahe(gray_img):
    # Apply CLAHE contrast normalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def binarize_adaptive(gray_img):
    # Adaptive thresholding
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
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle
    angle = minAreaRect[-1]
    print(f"Detected skew angle: {angle:.2f} degrees")

    # Correction for negative angles
    if angle > 45:
        angle = 270 + angle

    return angle


def deskew(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def process_image(input_path, clahe, invert_colors):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE
    clahe_img = gray

    if clahe:
        print("Applying CLAHE...")
        clahe_img = apply_clahe(gray)

    # 3. Noise reduction
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # 4. Adaptive thresholding
    bin_img = binarize_adaptive(blurred)

    # 5. Optional inversion
    if invert_colors:
        bin_img = cv2.bitwise_not(bin_img)

    # Get deskew angle
    pre_deskew = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    angle = getSkewAngle(pre_deskew)

    # 6. Deskew
    deskewed = deskew(pre_deskew, angle)

    return {
        "gray": gray,
        "clahe": clahe_img,
        "blurred": blurred,
        "binarized": bin_img,
        "deskewed": deskewed
    }


if __name__ == "__main__":
    import argparse
    import os

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process an image and prepare outputs for OCR.")
    parser.add_argument("input_image", nargs="?", default="test_images\\document-p24.jpg",
                        help="Path to the input image (default: test_images\\document-p24.jpg)")

    # Use flags to enable/disable CLAHE
    parser.add_argument("--no-clahe", dest="clahe", action="store_false",
                        help="Disable CLAHE contrast normalization.")
    parser.set_defaults(clahe=True)

    # Use a flag for invert
    parser.add_argument("--invert", action="store_true", default=False,
                        help="Invert colors after binarization (default: False)")

    args = parser.parse_args()

    if not os.path.isfile(args.input_image):
        raise SystemExit(f"Input file not found: {args.input_image}")
    
    # Process image
    results = process_image(args.input_image, clahe=args.clahe, invert_colors=args.invert)

    # Save results
    for key, img in results.items():
        out_path = os.path.join("results", f"{key}.png")
        cv2.imwrite(out_path, img)

    print(f"Processing complete! Output images saved to: results")