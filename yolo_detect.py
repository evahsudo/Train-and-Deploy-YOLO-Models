import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class KalmanBoxTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, w, h) and 2 measurements (x, y)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.statePre = np.zeros((4, 1), np.float32)
        self.kf.statePost = np.zeros((4, 1), np.float32)

    def update(self, x, y, w, h):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        predicted = self.kf.predict()
        px, py, pw, ph = predicted[0][0], predicted[1][0], predicted[2][0], predicted[3][0]
        return int(px), int(py), int(pw), int(ph)

# Store Kalman filters for each class
trackers = defaultdict(KalmanBoxTracker)

# YOLO Model
model = YOLO('runs/detect/train/weights/best.pt')

# Read from camera or video
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, verbose=False)

    for det in results[0].boxes:
        xyxy = det.xyxy[0].cpu().numpy().astype(int)
        conf = det.conf[0].item()
        cls = int(det.cls[0].item())

        if conf > 0.5:
            x, y, x2, y2 = xyxy
            w, h = x2 - x, y2 - y

            # Kalman filter update
            if cls not in trackers:
                trackers[cls] = KalmanBoxTracker()
            x, y, w, h = trackers[cls].update(x, y, w, h)

            # Clamp values to avoid overflow
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Show FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display output
    cv2.imshow('YOLOv11 Detection', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
