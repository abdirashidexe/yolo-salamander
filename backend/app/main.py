import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
from ultralytics import YOLO
from collections import defaultdict
from threading import Thread

app = FastAPI(title="Salamander Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent 

VIDEOS_DIR = BASE_DIR / "uploads"
VIDEOS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(VIDEOS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

MODEL_PATH = BASE_DIR / "models" / "salamander.pt"
model = YOLO(str(MODEL_PATH))

# --- NEW: Shared state dict to track job progress ---
job = {"status": "idle"}

@app.get("/")
def root():
    return {"ok": True}

# --- NEW: The Background Worker Function ---
def run_track_job():
    try:
        input_path = VIDEOS_DIR / "input.mp4"
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = OUTPUTS_DIR / "output.webm"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"vp80"),
            int(fps),
            (width, height)
        )

        frames_seen = defaultdict(int)
        label_for = {}

        for frame_idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
                
            result = model.track(frame, persist=True, verbose=False)[0]
            writer.write(result.plot())
            
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for tid, cls_id in zip(boxes.id.tolist(), boxes.cls.tolist()):
                    frames_seen[int(tid)] += 1
                    label_for[int(tid)] = model.names[int(cls_id)]
            
            # --- NEW: Update progress percentage every frame ---
            job["percent"] = int((frame_idx + 1) / total * 100)

        cap.release()
        writer.release()
        
        tracks = [
            {
                "track_id": tid,
                "time_on_screen_s": round(count / fps, 2),
                "label": label_for[tid],
            }
            for tid, count in frames_seen.items()
        ]

        # --- NEW: Mark job as done and save the payload ---
        job.clear()
        job["status"] = "done"
        job["percent"] = 100
        job["result"] = {
            "video_url": f"http://localhost:8000/outputs/output.webm?t={int(time.time())}",
            "tracks": tracks,
        }

    except Exception as e:
        print(f"error: {e}", flush=True)
        job.clear()
        job["status"] = "error"
        job["message"] = str(e)


# --- UPDATED: POST endpoint now starts a thread and returns immediately ---
@app.post("/track")
def start_track(video: UploadFile = File(...)):
    input_path = VIDEOS_DIR / "input.mp4"
    input_path.write_bytes(video.file.read())
    
    # Reset job state
    job.clear()
    job["status"] = "processing"
    job["percent"] = 0
    
    # Fire off the worker in the background
    Thread(target=run_track_job, daemon=True).start()
    
    return {"status": "processing"}

# --- NEW: GET endpoint for the frontend to poll ---
@app.get("/track")
def get_track():
    return job