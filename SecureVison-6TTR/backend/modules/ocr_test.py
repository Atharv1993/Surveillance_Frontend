import cv2
import pytesseract
import numpy as np
import re

# Configure Tesseract path
tesseract_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# --------------------------
# Main OCR Function
# --------------------------
def crop_id_card(frame):
    """Detect and crop the ID card from the image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 20, 80)  # Edge detection

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return frame  # No contours found

    largest_contour = max(contours, key=cv2.contourArea, default=None)

    if largest_contour is not None and cv2.contourArea(largest_contour) > 5000:
        x, y, w, h = cv2.boundingRect(largest_contour)
        return frame[y:y+h, x:x+w]
    else:
        return frame  # Return original frame if no ID detected


def extract_ocr_data(frame, id_type):
    """Extract text from the ID card based on the selected ID type."""
    cropped_frame = crop_id_card(frame)
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    ocr_text = pytesseract.image_to_string(gray)
    if id_type == "aadhar":
        return extract_aadhar_data(ocr_text)
    elif id_type == "license":
        return extract_license_data(ocr_text)
    elif id_type == "college":
        return extract_college_id_data(ocr_text)
    else:
        return {"name": None, "id": None}

# --------------------------
# Extraction Logic for Aadhar Card
# --------------------------
def extract_aadhar_data(ocr_text):
    """Extract Name and Aadhar Number from Aadhar Card text."""
    name, aadhar_number = None, None
    lines = ocr_text.split("\n")

    for i, line in enumerate(lines):
        if "DOB" in line or "Year of Birth" in line:
            if i > 0:
                name = lines[i - 1].strip()
            break

    aadhar_match = re.search(r"\b\d{12}\b", ocr_text.replace(" ", ""))
    if aadhar_match:
        aadhar_number = aadhar_match.group()

    return {"name": name, "id": aadhar_number}

# --------------------------
# Extraction Logic for Driving License
# --------------------------
def extract_license_data(ocr_text):
    """Extract Name and License Number from Driving License text."""
    name, license_number = None, None
    lines = ocr_text.split("\n")

    license_match = re.search(r"\b[A-Z]{2}\d{2}\s?\d{11}\b", ocr_text.replace(" ", ""))
    if license_match:
        license_number = license_match.group()

    for line in lines:
        if "Name:" in line or "NAME:" in line:
            name = line.split("Name:")[-1].strip().rstrip(".")
            break

    return {"name": name, "id": license_number}

# --------------------------
# Extraction Logic for College ID
# --------------------------
def extract_college_id_data(ocr_text):
    """Extract Name and Roll Number from College ID text."""
    name, roll_number = None, None
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]

    roll_match = re.search(r"Roll\s*No[:]*\s*([\d]+)", ocr_text, re.IGNORECASE)
    if roll_match:
        roll_number = roll_match.group(1)

    for line in lines:
        if re.match(r"^[A-Z\s]+$", line) and 5 < len(line) < 50:
            name = line.strip()
            break

    return {"name": name, "id": roll_number}

# Load the test image manually
image_path ="C:/Users/Atharva/OneDrive/Pictures/Screenshots/test2.jpg"  # Replace with your test image path
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Image not found or path incorrect.")
    exit()

processed_img = extract_ocr_data(frame, 'license')
print(processed_img)