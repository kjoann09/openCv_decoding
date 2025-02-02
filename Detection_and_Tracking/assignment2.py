import cv2
import numpy as np
import time
import serial
cap = cv2.VideoCapture(0)
background_subtractor = cv2.createBackgroundSubtractorMOG2()
# Initialize motion tracking variables
motion_detected = False
prev_centroids = []
centered_x_angle = 90
centered_y_angle = 90
# Threshold for significant movement
movement_threshold = 10
frame_center = (320, 240)
tolerance_zone = 20
# Connect to Arduino
arduino = serial.Serial('COM3', 9600)
motion_log = []
# Function to send centered angles to Arduino
def send_centered_angles(x_angle, y_angle):
    command = f"{x_angle},{y_angle}\n"
    arduino.write(command.encode())
    print(f"Sent to Arduino: {command.strip()}")
def save_motion_log(log):
    with open("motion_log.csv", "w") as file:
        file.write("Timestamp,CX,CY\n")
        for entry in log:
            file.write(f"{entry['timestamp']},{entry['cx']},{entry['cy']}\n")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Resize for consistent processing
    frame = cv2.resize(frame, (640, 480))
    mask = background_subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignore small movements
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2  # Calculate the centroid
            current_centroids.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
    # positional offset
    if current_centroids:
        # Use the first detected object's centroid
        cx, cy = current_centroids[0]
        offset_x = cx - frame_center[0]
        offset_y = cy - frame_center[1]
        if abs(offset_x) > movement_threshold:
            centered_x_angle -= int(offset_x / 50)  # Scale offset to angle adjustment
        if abs(offset_y) > movement_threshold:
            centered_y_angle += int(offset_y / 50)
        centered_x_angle = max(0, min(180, centered_x_angle))
        centered_y_angle = max(0, min(180, centered_y_angle))
        send_centered_angles(centered_x_angle,centered_y_angle)
        motion_log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cx": cx,
            "cy": cy
        })
    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
save_motion_log(motion_log)
cap.release()
cv2.destroyAllWindows()
