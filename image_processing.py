import cv2
import numpy as np
import pytesseract
from pathlib import Path


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
    clahe_img = apply_clahe(gray) if clahe else gray

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

    # 6. Deskew binarized image
    deskewed = deskew(pre_deskew, angle)

    # Deskew gray image
    final = deskew(clahe_img, angle)

    return {
        "gray": gray,
        "clahe": clahe_img,
        "blurred": blurred,
        "binarized": bin_img,
        "deskewed": deskewed,
        "final": final
    }


if __name__ == "__main__":
    import argparse
    import os

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process an image and prepare outputs for OCR.")
    parser.add_argument("input_image", nargs="?", default="test_images\\document-p24.jpg",
                        help="Path to the input image (default: test_images\\document-p24.jpg)")

    # Use flags to enable/disable CLAHE and inversion
    parser.add_argument("--no-clahe", dest="clahe", action="store_false",
                        help="Disable CLAHE contrast normalization if the image is clear enough.")
    parser.set_defaults(clahe=True)

    parser.add_argument("--invert", action="store_true", default=False,
                        help="Invert colors after binarization (default: False)")

    args = parser.parse_args()

    if not os.path.isfile(args.input_image):
        raise SystemExit(f"Input file not found: {args.input_image}")
    
    # Clear characters directory
    [f.unlink() for f in Path("characters").iterdir() if f.is_file()]
    
    # Process image
    results = process_image(args.input_image, clahe=args.clahe, invert_colors=args.invert)

    # Save results
    for key, img in results.items():
        out_path = os.path.join("results", f"{key}.png")
        cv2.imwrite(out_path, img)

    print(f"\nProcessing complete! Output images saved to: results")

    # Tesseract OCR using image_to_string
    final_img = cv2.cvtColor(results["final"], cv2.COLOR_BGR2RGB)
    ocr_result = pytesseract.image_to_string(final_img)

    with open(os.path.join("results", "text_result.txt"), "w") as text_file:
        text_file.write(ocr_result)

    print(f"OCR result saved to: results/text_result.txt\n")

    # Detecting individual characters with bounding boxes
    h, w = final_img.shape[:2]
    boxes = pytesseract.image_to_boxes(final_img)
    boxed_img = final_img.copy()

    for idx, b in enumerate(boxes.splitlines()):
        b = b.split(' ')
        char, x1, y1, x2, y2 = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(boxed_img, (x1, h - y2), (x2, h - y1), (0, 255, 0), 2)
        cv2.putText(boxed_img, char, (x1, h - y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Save an image for each character
        char_img = final_img[h - y2 - 5:h - y1 + 5, x1 - 5:x2 + 5]
        char_out_path = os.path.join("characters", f"{idx}.png")
        cv2.imwrite(char_out_path, char_img)

    cv2.imwrite(os.path.join("results", "boxed_characters.png"), boxed_img)
    print(f"Character bounding box image saved to: results/boxed_characters.png\n")

    # Load character images
    characters_dir = "characters"

    files = sorted(
        os.listdir(characters_dir),
        key=lambda x: int(os.path.splitext(x)[0])
    )

    images = [
        cv2.imread(os.path.join(characters_dir, f))
        for f in files
    ]
