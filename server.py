import cv2
import pytesseract
import numpy as np
import urllib.request
import requests
import time
import re

# ==========================================
# Configuration
# ==========================================
# Replace with the IP address printed in your Arduino Serial Monitor
ESP32_IP = "192.168.1.100" 
ESP32_CAPTURE_URL = f"http://{ESP32_IP}/capture"
ESP32_GATE_URL = f"http://{ESP32_IP}/open-gate"

# Windows Users: Uncomment and modify the line below if pytesseract is not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Database of authorized plates (Make sure to format them without spaces)
AUTHORIZED_PLATES = [
    "TN01AB1234",
    "KA05XY9876",
    "MH12CD5678"
]

def clean_text(text):
    """Removes special characters and spaces from the OCR output."""
    # Keep only alphanumeric characters and convert to uppercase
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def process_frame(frame):
    """Processes the image to improve OCR accuracy."""
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply bilateral filter to reduce noise while keeping edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # 3. Edge detection (optional, useful if you want to find contours of the plate first)
    # edged = cv2.Canny(bfilter, 30, 200)
    
    # 4. Thresholding to make the text stand out
    # Using Otsu's thresholding
    _, thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def main():
    print(f"Connecting to ESP32-CAM at {ESP32_IP}...")
    
    while True:
        try:
            # 1. Fetch the frame from ESP32-CAM
            img_resp = urllib.request.urlopen(ESP32_CAPTURE_URL, timeout=5)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)
            
            if frame is None:
                print("Failed to decode frame.")
                continue

            # 2. Pre-process the image for OCR
            processed_frame = process_frame(frame)
            
            # 3. Run OCR using Pytesseract
            # --psm 8: Treat the image as a single word (good for license plates)
            # --psm 11: Sparse text. Try 11 if 8 doesn't work well.
            custom_config = r'--oem 3 --psm 8'
            raw_text = pytesseract.image_to_string(processed_frame, config=custom_config)
            
            # 4. Clean up the extracted text
            plate_text = clean_text(raw_text)
            
            # 5. Display the feed and the result
            cv2.putText(frame, f"Read: {plate_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live Feed", frame)
            # cv2.imshow("Processed for OCR", processed_frame) # Uncomment to see the B&W threshold image
            
            # 6. Verification Logic
            if len(plate_text) >= 6: # Ignore random short strings like "A" or "12"
                print(f"Detected Plate: {plate_text}")
                
                # Check against database
                if any(auth_plate in plate_text for auth_plate in AUTHORIZED_PLATES):
                    print(f"ACCESS GRANTED for {plate_text}. Opening gate...")
                    
                    try:
                        # Send signal to ESP32 to open the servo
                        response = requests.get(ESP32_GATE_URL, timeout=5)
                        if response.status_code == 200:
                            print("Gate opened successfully.")
                    except Exception as e:
                        print(f"Failed to communicate with gate: {e}")
                    
                    # Pause to let the vehicle pass and avoid multiple triggers for the same car
                    print("Waiting for vehicle to pass...")
                    time.sleep(5) 
                    print("Resuming monitoring...\n")
                else:
                    print(f"ACCESS DENIED for {plate_text}")

            # Press 'q' to quit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error fetching stream: {e}")
            time.sleep(1) # Wait a bit before retrying if network drops

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()