import cv2
import numpy as np
from ultralytics import YOLO

# Initialize the YOLOv12 model
model = YOLO('my_model.pt')  # Path to your YOLOv12 model

# Define the camera source (USB camera)
cap = cv2.VideoCapture('usb0')

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run inference using YOLOv12
    results = model(frame)  # Direct inference

    # Extract bounding boxes and labels
    for result in results.pred[0]:  # YOLOv12 prediction format
        # The format of result in YOLOv12 is different, here we get the box and confidence score
        xyxy = result[:4].cpu().numpy().astype(int)  # Bounding box coordinates
        conf = result[4].item()  # Confidence score
        cls = int(result[5].item())  # Class ID

        if conf > 0.5:  # Threshold for displaying boxes
            # Draw bounding box
            x1, y1, x2, y2 = xyxy
            label = f'{model.names[cls]} {conf:.2f}'
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('YOLOv12 Detection', frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
