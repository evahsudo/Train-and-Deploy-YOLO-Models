import os
import sys
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source: image file, folder, video, USB camera ("usb0"), or PiCamera ("picamera0")', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold (example: "0.4")', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH format (example: "640x480")', default=None)
parser.add_argument('--record', help='Record results to "output.avi"', action='store_true')
args = parser.parse_args()

# Parse arguments
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Validate model path
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(0)

# Load the YOLO model
model = YOLO(model_path)

# Set up resolution
resize = False
if user_res:
    resW, resH = map(int, user_res.split('x'))
    resize = True

# Handle recording setup
if record and not user_res:
    print('Please specify resolution to record video at.')
    sys.exit(0)

if record:
    record_name = 'output.avi'
    record_fps = 20
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Handle image source
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mp4', '.mkv']

if os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print('Unsupported file format.')
        sys.exit(0)
elif os.path.isdir(img_source):
    source_type = 'folder'
    imgs_list = sorted([os.path.join(img_source, f) for f in os.listdir(img_source) if f.endswith(tuple(img_ext_list))])
elif img_source.startswith('usb'):
    source_type = 'usb'
    cap = cv2.VideoCapture(int(img_source[3:]))
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"size": (resW, resH)}))
    cap.start()
else:
    print('Invalid source type.')
    sys.exit(0)

# Set up FPS calculation
fps_buffer = []
fps_avg_len = 30

# Color for bounding boxes
bbox_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# Inference loop
img_count = 0
while True:
    start_time = time.perf_counter()

    # Load frame
    if source_type == 'image':
        frame = cv2.imread(img_source)
    elif source_type == 'folder':
        if img_count >= len(imgs_list):
            print("Finished processing images.")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()

    if frame is None:
        print("Failed to capture frame.")
        break

    # Resize frame if needed
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Draw detections
    for det in results[0].boxes:
        xyxy = det.xyxy[0].cpu().numpy().astype(int)
        conf = det.conf[0].item()
        cls = int(det.cls[0].item())

        if conf > min_thresh:
            color = bbox_colors[cls % len(bbox_colors)]
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display FPS
    end_time = time.perf_counter()
    fps = 1 / (end_time - start_time)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_fps = sum(fps_buffer) / len(fps_buffer)
    cv2.putText(frame, f'FPS: {avg_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display results
    cv2.imshow('YOLOv11 Detection', frame)
    if record:
        recorder.write(frame)

    # Key control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

# Cleanup
if source_type in ['video', 'usb']:
    cap.release()
if source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()

print(f"Average FPS: {avg_fps:.2f}")
