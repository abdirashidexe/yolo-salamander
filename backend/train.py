from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='/home/shawn003/yolo-salamander/backend/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='salamander_v1'
)