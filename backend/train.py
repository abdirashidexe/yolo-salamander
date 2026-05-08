# This is the training file. Run "python train.py" in terminal to train.
# OR ALTERNATIVELY run CLI version:
# yolo detect train model=yolov8n.pt data=backend/dataset/data.yaml epochs=50 imgsz=640

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="backend/dataset/data.yaml",
    epochs=50,
    imgsz=640
)