from ultralytics import YOLO
import cv2
import os

os.makedirs("backend/outputs", exist_ok=True)

# Load the trained model
model = YOLO('backend/models/salamander.pt')

# Run inference on a single frame
results = model('uploads/frames/frame_0001.jpg')

# Draw the bounding boxes and save the output
for r in results:
    annotated = r.plot()  # draws boxes directly onto the image
    cv2.imwrite('backend/outputs/test_single.jpg', annotated)
    
    # Print detection info to terminal
    for box in r.boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        conf = box.conf[0]
        print(f"Detected salamander at center ({cx:.1f}, {cy:.1f}) with confidence {conf:.2f}")

print("Saved to backend/outputs/test_single.jpg")