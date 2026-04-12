import cv2
import numpy as np
# Web cam or video file
cap = cv2.VideoCapture('videos/vehicles3.mp4')
# Minimum rectangle dimensions
min_width_react = 80
min_height_react = 80
# Line position
count_line_position = 550
# Background Subtractor
algo = cv2.createBackgroundSubtractorMOG2()
# Output video writer setup
output_file = 'videos/output3.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(3))  # Width of video
frame_height = int(cap.get(4))  # Height of video
out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
# Function to get center of rectangle
def center_handle(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return (cx, cy)
detect = []
offset = 6
counter = 0
while True:
    ret, frame1 = cap.read()
    if not ret:
        break  # End of video
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 255, 0), -1)
    for (x, y) in detect:
        if (count_line_position - offset) < y < (count_line_position + offset):
            counter += 1
            detect.remove((x, y))
            print("Vehicle Counter:", counter)
    cv2.putText(frame1, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    # Write the processed frame to output video
    out.write(frame1)
# Cleanup
cap.release()
out.release()
print(f"Video saved to: {output_file}")