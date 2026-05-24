import { useState, useRef } from 'react';
import './index.css';

export default function App() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [percent, setPercent] = useState(0);
  const [tracks, setTracks] = useState([]);
  
  const pollInterval = useRef(null);

  const pollStatus = async () => {
    try {
      const res = await fetch("http://localhost:8000/track");
      const data = await res.json();

      if (data.status === "processing") {
        setPercent(data.percent || 0);
      } else if (data.status === "done") {
        setPercent(100);
        setVideoUrl(data.result.video_url);
        setHeatmapUrl(data.result.heatmap_url);
        setTracks(data.result.tracks || []);
        setIsProcessing(false);
        clearInterval(pollInterval.current);
      } else if (data.status === "error") {
        console.error("Backend error:", data.message);
        setIsProcessing(false);
        clearInterval(pollInterval.current);
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsProcessing(true);
    setVideoUrl(null);
    setHeatmapUrl(null);
    setTracks([]);
    setPercent(0);

    const form = new FormData();
    form.append("video", file);

    try {
      await fetch("http://localhost:8000/track", { 
        method: "POST", 
        body: form 
      });
      pollInterval.current = setInterval(pollStatus, 1000);
    } catch (error) {
      console.error("Error uploading video:", error);
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Salamander Tracker</h1>
        <p style={{ color: 'var(--text-muted)' }}>AI-Powered YOLO Detection & Metrics</p>
        
        {/* Route Links opening in new tabs */}
        <div className="nav-links">
          <a href="https://github.com/abdirashidexe/yolo-salamander/blob/main/README.md" target="_blank" rel="noopener noreferrer">README.md</a>
          <a href="http://localhost:8000/" target="_blank" rel="noopener noreferrer">Backend Health</a>
          <a href="http://localhost:8000/track" target="_blank" rel="noopener noreferrer">API Polling (/track)</a>
        </div>
      </header>
      
      <main>
        <div className="card upload-form">
          <form onSubmit={handleUpload} style={{ display: 'flex', alignItems: 'center' }}>
            <input 
              type="file" 
              accept="video/*"
              onChange={(e) => setFile(e.target.files[0])} 
            />
            <button className="btn" type="submit" disabled={isProcessing}>
              {isProcessing ? "Processing..." : "Upload & Track"}
            </button>
          </form>

          {isProcessing && (
            <div className="progress-container">
              <p className="progress-text">Running YOLO tracking: {percent}%</p>
              <progress value={percent} max={100} />
            </div>
          )}
        </div>

        {videoUrl && (
          <div className="results-grid">
            <div className="card">
              <h3>Annotated Video</h3>
              <video src={videoUrl} controls />
            </div>

            {heatmapUrl && (
              <div className="card">
                <h3>Position Heatmap</h3>
                <img src={heatmapUrl} alt="Salamander Position Heatmap" className="heatmap-img" />
              </div>
            )}
            
            {tracks.length > 0 && (
              <div className="card" style={{ gridColumn: '1 / -1' }}>
                <h3>Detection Metrics</h3>
                <table className="metrics-table">
                  <thead>
                    <tr>
                      <th>Track ID</th>
                      <th>Label</th>
                      <th>Time on Screen (s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tracks.map(track => (
                      <tr key={track.track_id}>
                        <td>{track.track_id}</td>
                        <td>{track.label}</td>
                        <td>{track.time_on_screen_s}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}