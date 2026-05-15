from collections import defaultdict
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
from ultralytics import YOLO

app = FastAPI(title="Salamander Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set paths based on your repo structure
BASE_DIR = Path(__file__).resolve().parent.parent 

VIDEOS_DIR = BASE_DIR / "uploads"
VIDEOS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Serve both folders over HTTP
app.mount("/uploads", StaticFiles(directory=str(VIDEOS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Load YOLO model at startup (so it only happens once)
MODEL_PATH = BASE_DIR / "models" / "salamander.pt"
model = YOLO(str(MODEL_PATH))

@app.get("/")
def root():
    return {"ok": True}

@app.post("/track")
def start_track(video: UploadFile = File(...)):
    # 1. Save incoming video
    input_path = VIDEOS_DIR / "input.mp4"
    input_path.write_bytes(video.file.read())
    
    # 2. Open video with OpenCV to prep for YOLO
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Starting inference: fps={fps} dims={width}x{height} frames={total}")

    # Set up the output video writer ('avc1' is best for web browser playback)
    output_path = OUTPUTS_DIR / "output.webm"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"vp80"),
        int(fps),
        (width, height)
    )

# --- NEW: Initialize metric dictionaries before the loop ---
    frames_seen = defaultdict(int)
    label_for = {}

    # 3. Loop through frames and run YOLO tracking
    for frame_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
            
        result = model.track(frame, persist=True, verbose=False)[0]
        writer.write(result.plot())
        
        # --- NEW: Extract IDs and count frames per individual ---
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            # boxes.id and boxes.cls are tensors, convert to standard lists
            for tid, cls_id in zip(boxes.id.tolist(), boxes.cls.tolist()):
                frames_seen[int(tid)] += 1
                label_for[int(tid)] = model.names[int(cls_id)]
        
        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}/{total}")

    cap.release()
    writer.release()
    
    # --- NEW: Convert frame counts to seconds for the frontend ---
    tracks = [
        {
            "track_id": tid,
            "time_on_screen_s": round(count / fps, 2),
            "label": label_for[tid],
        }
        for tid, count in frames_seen.items()
    ]

    # 4. Return the URL and the new tracks data
    return {
        "status": "done",
        "video_url": f"http://localhost:8000/outputs/output.webm?t={int(time.time())}",
        "tracks": tracks,
    }