import cv2
import pandas as pd
import easyocr
from ultralytics import YOLO
import os
# Load YOLOv8 model (you can use a custom model trained for license plates or a general model)
model = YOLO('models/yolov8n.pt')  # Replace with your custom model path if available
reader = easyocr.Reader(['en'])
# CSV output
csv_file = 'CSV_Files/detected_plates_yolo.csv'
detected_data = []
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # Optional: Only process "license plate" class if using custom model
            # if cls != LICENSE_PLATE_CLASS_ID:
            #     continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # OCR
            ocr_result = reader.readtext(roi)
            if ocr_result:
                plate_text = ocr_result[0][-2].strip()
                if plate_text:
                    # Draw bounding box and overlay
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    # Overlay cropped plate (optional)
                    roi_resized = cv2.resize(roi, (x2 - x1, y2 - y1))
                    overlay_y = y1 - (y2 - y1)
                    if overlay_y > 0:
                        frame[overlay_y:y1, x1:x2] = roi_resized
                    # Save to CSV
                    detected_data.append({'frame': frame_count, 'plate_text': plate_text})
        out.write(frame)
    cap.release()
    out.release()
    pd.DataFrame(detected_data).to_csv(csv_file, index=False)
    print(f"Done! Output saved as '{output_path}' and plates in '{csv_file}'.")
# Run
process_video('videos/vehicles2.mp4', 'videos/output2.mp4')