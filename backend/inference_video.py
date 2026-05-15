from ultralytics import YOLO
import cv2
import os

os.makedirs("backend/outputs", exist_ok=True)

# Load your trained model
model = YOLO('backend/models/salamander.pt')

# Open the source video
cap = cv2.VideoCapture('uploads/Salamander.mp4')

# Get video properties to match the output video format
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the output video writer
out = cv2.VideoWriter(
    'backend/outputs/annotated_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    for box in results[0].boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        conf = box.conf[0]
        print(f"Frame {frame_count}: salamander at ({cx:.1f}, {cy:.1f}), confidence {conf:.2f}")

    frame_count += 1

cap.release()
out.release()
print(f"Done. Processed {frame_count} frames.")
print("Saved to backend/outputs/annotated_video.mp4")