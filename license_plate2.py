import cv2
import pandas as pd
import easyocr
from ultralytics import YOLO
import numpy as np
# Load models
model = YOLO('models/yolov8n.pt')  # Replace with your custom plate detection model if available
reader = easyocr.Reader(['en'])
# CSV output
csv_file = 'CSV_Files/cropped_plate_output.csv'
detected_data = []
def overlay_plate(frame, cropped_plate, plate_text, top_left):
    # Resize the cropped plate for better visibility
    plate_zoom = cv2.resize(cropped_plate, (250, 80))
    # Create white background for overlay
    plate_canvas = np.ones((120, 250, 3), dtype=np.uint8) * 255
    plate_canvas[10:90, :] = plate_zoom
    # Draw text
    cv2.putText(plate_canvas, plate_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    # Determine overlay position
    x, y = top_left
    if y - 120 < 0:
        y = y + 10  # shift below the car if top space not available
    # Overlay on original frame
    overlay_y = max(0, y - 120)
    overlay_x = x
    h, w = plate_canvas.shape[:2]
    # Clip if goes outside frame
    if overlay_x + w > frame.shape[1]:
        overlay_x = frame.shape[1] - w
    frame[overlay_y:overlay_y + h, overlay_x:overlay_x + w] = plate_canvas
    return frame
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Output writer
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
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            ocr_result = reader.readtext(roi)
            if ocr_result:
                plate_text = ocr_result[0][-2].strip()
                if plate_text:
                    # Draw box around detected plate
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Overlay cropped plate and text
                    frame = overlay_plate(frame, roi, plate_text, (x1, y1))
                    # Store in CSV
                    detected_data.append({'frame': frame_count, 'plate_text': plate_text})
        out.write(frame)
    cap.release()
    out.release()
    pd.DataFrame(detected_data).to_csv(csv_file, index=False)
    print(f"Done! Output: {output_path}, CSV: {csv_file}")
# Call the function
process_video('videos/vehicles2.mp4', 'videos/output2_1.mp4')