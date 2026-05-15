import { useState } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [tracks, setTracks] = useState([]); // <-- NEW: State for metrics

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsProcessing(true);
    setVideoUrl(null);
    setTracks([]); // Clear old metrics

    const form = new FormData();
    form.append("video", file);

    try {
      const response = await fetch("http://localhost:8000/track", { 
        method: "POST", 
        body: form 
      });
      const data = await response.json();
      setVideoUrl(data.video_url);
      
      // <-- NEW: Save the tracks data to state
      if (data.tracks) {
        setTracks(data.tracks);
      }
    } catch (error) {
      console.error("Error uploading video:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Salamander Tracker</h1>
      
      <form onSubmit={handleUpload} style={{ marginBottom: '20px' }}>
        <input 
          type="file" 
          accept="video/*"
          onChange={(e) => setFile(e.target.files[0])} 
        />
        <button type="submit" disabled={isProcessing}>
          {isProcessing ? "Processing..." : "Upload & Track"}
        </button>
      </form>

      {isProcessing && <p style={{ color: "blue" }}>Running YOLO inference... Check backend terminal for progress!</p>}

      {videoUrl && (
        <div style={{ display: 'flex', gap: '40px', flexWrap: 'wrap' }}>
          <div>
            <h3>Annotated Video</h3>
            <video src={videoUrl} controls width="600" />
          </div>
          
          {/* NEW: Render the metrics table */}
          {tracks.length > 0 && (
            <div>
              <h3>Detection Metrics</h3>
              <table border="1" cellPadding="10" style={{ borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f0f0f0' }}>
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
    </div>
  );
}