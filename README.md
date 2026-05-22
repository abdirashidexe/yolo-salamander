# Salamander Detector

### Requirements
- Python 3.12.3
- Node.js v20.20.2
- WSL (Ubuntu) recommended for Windows users

---

### Run Instructions

#### Backend
1. `cd backend`
2. Create and activate the virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate        # Mac/Linux/WSL
    venv\Scripts\activate           # Windows (CMD)
    ```
3. `pip install -r requirements.txt`
4. `uvicorn app.main:app --reload`

The backend will be running at `http://127.0.0.1:8000`

#### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`

Browser location:  `http://localhost:5173`

---

### Dataset Details
- Total labeled frames: ~140 frames sampled from a 14-second video clip at 30fps (every 3rd frame extracted using ffmpeg)
- Labeling tool: Label Studio with a single class: `salamander`
- Train/val split: 80/20 (~112 training frames, ~28 validation frames)
- Label format: YOLO format (normalized bounding box coordinates)
- Dataset variety: clear centered shots, motion blur, partial occlusion, edge-of-frame detections, and low contrast conditions

---

### Comparison Paragraph
When using color masking, thats going through each pixel and it's able to locate the salamander depending on the specifc color. When using YOLO it's taking in hundrens of labeled images of salamanders instead of just taking the color. This lets the YOLO method to be a lot more accurate than the color masking method. The benefit of using the color masking method is the speed of how fast it is tracking the salamander (if it ends up being accurate). Personally I feel as though we diffinetly prefer the YOLO method a lot more since accuracy is a more valuable than speed.
