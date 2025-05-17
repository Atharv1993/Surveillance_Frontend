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
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    
    # Looking for Aadhar number - 12 digits
    aadhar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b|\b\d{12}\b"
    aadhar_match = re.search(aadhar_pattern, ocr_text.replace(" ", ""))
    if aadhar_match:
        aadhar_number = aadhar_match.group().replace(" ", "")
    
    # Name extraction based on patterns in Aadhar cards
    for i, line in enumerate(lines):
        # Skip lines that are definitely not names
        if (len(line) < 5 or "HRHRR" in line or "Government" in line or 
            "GITER" in line or "ssue Date" in line or "Issue Date" in line or
            "Ute" in line or "AT" in line or "3TER" in line):
            continue
            
        # Check if line contains "/DB" or other DOB markers
        if "/DB" in line or "DOB" in line or "Date of Birth" in line:
            # The name is typically in the line before DOB reference
            if i > 0:
                name = lines[i-1].strip()
                break
                
        # Look for lines with proper name format (first letter caps, multiple words)
        if not name and re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$', line):
            name = line
            break
            
        # Look for lines with proper name format (two words)
        if not name and re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', line):
            name = line
            break
            
        # Look for "Male" or "Female" indicators - name usually appears before gender
        if "/Male" in line or "/Female" in line or "Male" in line or "Female" in line:
            if i > 0:
                potential_name = lines[i-1].strip()
                # Verify it's not a header or other text
                if (len(potential_name.split()) >= 2 and 
                    "Government" not in potential_name and 
                    "Date" not in potential_name):
                    name = potential_name
                    break

    return {"name": name, "id": aadhar_number}
# --------------------------
# Improved Extraction Logic for Driving License
# --------------------------
def extract_license_data(ocr_text):
    """Extract Name and License Number from Driving License text."""
    name, license_number = None, None
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    
    # License number pattern for Indian driving licenses (e.g., GA0420230000595)
    license_pattern = r"\b[A-Z]{2}\d{2,12}\b|\b[A-Z]{2}\d{2}\d{10,11}\b|\b[A-Z]{2}\d{2,4}2023\d{7}\b"
    
    # Find all potential license numbers
    license_matches = re.findall(license_pattern, ocr_text.replace(" ", ""))
    if license_matches:
        # Use the longest match as it's likely to be the full license number
        license_number = max(license_matches, key=len)
    
    # Name extraction - look for patterns commonly seen in driving licenses
    for i, line in enumerate(lines):
        # Name might be after "Name:" or similar
        if ("Name:" in line or "NAME:" in line) and ":" in line:
            name_part = line.split(":", 1)[1].strip()
            if name_part and len(name_part) > 3:  # Valid name should be longer than 3 chars
                name = name_part
                break
        
        # In many Indian licenses, name appears in all caps after certain keywords
        if "Indian Union Driving Licence" in line and i + 1 < len(lines):
            potential_name = lines[i + 1].strip()
            if re.match(r'^[A-Z][A-Z\s]+$', potential_name):
                name = potential_name
                break
        
        # Name is often a line with 2-3 words in caps with no numbers
        if (not name and re.match(r'^[A-Z][A-Z\s]+$', line) and 
            len(line.split()) >= 2 and len(line.split()) <= 4 and
            "LICENCE" not in line and "DRIVING" not in line and 
            "GOVERNMENT" not in line and "GOA" not in line):
            name = line
    
    return {"name": name, "id": license_number}

# --------------------------
# Improved Extraction Logic for College ID
# --------------------------
def extract_college_id_data(ocr_text):
    """Extract Name and Roll Number from College ID text."""
    name, roll_number = None, None
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    
    # Look for roll number patterns (e.g., 211106010)
    roll_patterns = [
        r"Roll\s*No[.:]*\s*(\d{9})",  # Roll No: 211106010
        r"Roll\s*No[.:]*\s*(\d+)",    # Roll No: with any number of digits
        r"\bRol\s*No[.:]*\s*(\d+)",   # Rol No: with any number of digits (typo)
        r"\b(\d{9})\b"                # Just 9-digit number which is common for roll numbers
    ]
    
    for pattern in roll_patterns:
        roll_match = re.search(pattern, ocr_text, re.IGNORECASE)
        if roll_match:
            roll_number = roll_match.group(1)
            break
    
    # If no roll number found with patterns, try looking for standalone numbers
    if not roll_number:
        # Look for standalone 9-digit numbers that often represent roll numbers
        numbers = re.findall(r'\b\d{9}\b', ocr_text)
        if numbers:
            roll_number = numbers[0]
    
    # Name extraction - look for patterns commonly seen in college IDs
    for i, line in enumerate(lines):
        # Skip lines that are definitely not names
        if ("College" in line or "Government" in line or "Valid" in line or 
            "Principal" in line or "@" in line or ".ac.in" in line or 
            "Engineering" in line or "Roll" in line or "DOB" in line):
            continue
        
        # Look for all-caps names that are common in college IDs
        if re.match(r'^[A-Z][A-Z\s]+$', line) and 3 <= len(line.split()) <= 4:
            name = line
            break
    
    return {"name": name, "id": roll_number}