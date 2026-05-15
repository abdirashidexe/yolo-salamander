import numpy as np
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

# Turns the folders into mini web servers so the React frontend can directly request the video URLs and heatmap image
app.mount("/uploads", StaticFiles(directory=str(VIDEOS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

MODEL_PATH = BASE_DIR / "models" / "salamander.pt"
model = YOLO(str(MODEL_PATH))

# Shared state dict to track job progress
job = {"status": "idle"}

@app.get("/")
def root():
    return {"ok": True}

# The Background Worker Function
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
        
        # Capture the first frame to use as our background
        ok, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to start for the tracking loop

        # HEATMAP: Create a blank grayscale canvas
        heatmap_data = np.zeros((height, width), dtype=np.float32)

        for frame_idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
                
            result = model.track(frame, persist=True, verbose=False)[0]
            writer.write(result.plot())
            
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                # We need xywh (x-center, y-center, width, height) for the heatmap
                for box, tid, cls_id in zip(boxes.xywh.tolist(), boxes.id.tolist(), boxes.cls.tolist()):
                    frames_seen[int(tid)] += 1
                    label_for[int(tid)] = model.names[int(cls_id)]
                    
                    # Draw a point at the center of the salamander
                    cx, cy = int(box[0]), int(box[1])
                    cv2.circle(heatmap_data, (cx, cy), 15, (1,), -1)
            
            job["percent"] = int((frame_idx + 1) / total * 100)

        cap.release()
        writer.release()
        
        # HEATMAP: Process and save the final image
        heatmap_blurred = cv2.GaussianBlur(heatmap_data, (51, 51), 0)
        heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # HEATMAP: Blend the heat onto the actual video background
        # 1. Create a mask where there is actual "heat" (ignore the empty zero areas)
        mask = heatmap_norm > 5 
        
        # 2. Expand mask to 3 color channels so it matches the image format
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # 3. Blend the background frame and the colored heatmap (40% background, 60% heat)
        blended = cv2.addWeighted(first_frame, 0.4, heatmap_colored, 0.6, 0)
        
        # 4. Paste the blended heat onto the original frame ONLY where the salamander moved
        final_overlay = np.where(mask_3d, blended, first_frame)
        
        heatmap_path = OUTPUTS_DIR / "heatmap.jpg"
        cv2.imwrite(str(heatmap_path), final_overlay)

        tracks = [
            {
                "track_id": tid,
                "time_on_screen_s": round(count / fps, 2),
                "label": label_for[tid],
            }
            for tid, count in frames_seen.items()
        ]

        job.clear()
        job["status"] = "done"
        job["percent"] = 100
        job["result"] = {
            "video_url": f"http://localhost:8000/outputs/output.webm?t={int(time.time())}",
            "heatmap_url": f"http://localhost:8000/outputs/heatmap.jpg?t={int(time.time())}",
            "tracks": tracks,
        }

    except Exception as e:
        print(f"error: {e}", flush=True)
        job.clear()
        job["status"] = "error"
        job["message"] = str(e)


# POST endpoint starts a thread and returns status immediately
@app.post("/track")
def start_track(video: UploadFile = File(...)):
    input_path = VIDEOS_DIR / "input.mp4"
    input_path.write_bytes(video.file.read())
    
    # Reset job state
    job.clear()
    job["status"] = "processing"
    job["percent"] = 0
    
    # Start the worker in the background
    Thread(target=run_track_job, daemon=True).start()
    
    return {"status": "processing"}

# GET endpoint for the frontend to poll
@app.get("/track")
def get_track():
    return job