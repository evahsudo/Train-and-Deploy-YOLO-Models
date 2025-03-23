import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# Store smoothing state for each object
class BoxSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev_box = None
        self.prev_conf = None
        self.frames_since_seen = 0

    def smooth(self, box, conf):
        if self.prev_box is None:
            self.prev_box = box
            self.prev_conf = conf
        else:
            # Exponential moving average for box coordinates
            self.prev_box = [
                int(self.alpha * box[i] + (1 - self.alpha) * self.prev_box[i])
                for i in range(4)
            ]
            # Smooth confidence
            self.prev_conf = self.alpha * conf + (1 - self.alpha) * self.prev_conf

        return self.prev_box, self.prev_conf
    
    def update_seen(self):
        self.frames_since_seen = 0

    def age(self):
        self.frames_since_seen += 1
        return self.frames_since_seen

# Initialize YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# Trackers for each class
trackers = defaultdict(BoxSmoother)

# Minimum confidence threshold (with hysteresis)
CONF_THRESHOLD = 0.5
HYSTERESIS = 0.05

# Capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO inference
    results = model(frame, verbose=False)

    # Track seen objects
    seen_classes = set()

    for det in results[0].boxes:
        xyxy = det.xyxy[0].cpu().numpy().astype(int)
        conf = det.conf[0].item()
        cls = int(det.cls[0].item())

        if conf > (CONF_THRESHOLD - HYSTERESIS):
            seen_classes.add(cls)

            # Initialize tracker if not already present
            if cls not in trackers:
                trackers[cls] = BoxSmoother()

            # Smooth box and confidence
            x, y, x2, y2 = xyxy
            smoothed_box, smoothed_conf = trackers[cls].smooth([x, y, x2, y2], conf)
            trackers[cls].update_seen()

            if smoothed_conf > CONF_THRESHOLD:
                color = (0, 255, 0)
                cv2.rectangle(frame, (smoothed_box[0], smoothed_box[1]), (smoothed_box[2], smoothed_box[3]), color, 2)
                label = f"{model.names[cls]}: {smoothed_conf:.2f}"
                cv2.putText(frame, label, (smoothed_box[0], smoothed_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Remove old detections (age out unseen objects)
    for cls in list(trackers.keys()):
        if cls not in seen_classes:
            age = trackers[cls].age()
            if age > 5:  # Remove if not seen for 5 frames
                del trackers[cls]

    # Show FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display results
    cv2.imshow('YOLOv11 Detection', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
