import { useState } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsProcessing(true); // Disable button and show message
    setVideoUrl(null); // Clear the old video from the screen

    const form = new FormData();
    form.append("video", file);

    try {
      const response = await fetch("http://localhost:8000/track", { 
        method: "POST", 
        body: form 
      });
      const data = await response.json();
      setVideoUrl(data.video_url); // Load the new annotated video
    } catch (error) {
      console.error("Error uploading video:", error);
    } finally {
      setIsProcessing(false); // Re-enable the button
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
        <div>
          <h3>Annotated Video Playback</h3>
          <video src={videoUrl} controls width="600" />
        </div>
      )}
    </div>
  );
}