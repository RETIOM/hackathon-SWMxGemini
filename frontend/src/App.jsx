import React from 'react';
import UploadZone from './components/UploadZone';
import VideoPlayer from './components/VideoPlayer';
import NarrationLog from './components/NarrationLog';
import { useProcessing } from './hooks/useProcessing';
import { RefreshCw } from 'lucide-react';
import './index.css';

function App() {
  const { 
    status, 
    errorMsg, 
    segments, 
    narrations, 
    fileName,
    processVideo, 
    reset 
  } = useProcessing();

  return (
    <div className="app-container">
      <header className="header">
        <h1>Third Ear</h1>
        <p>Upload an MP4 and get an AI-narrated version streamed back in real-time.</p>
      </header>

      {status === 'idle' ? (
        <UploadZone onUpload={processVideo} />
      ) : (
        <div className="workspace">
          <div style={{display: 'flex', flexDirection: 'column', gap: '1rem'}}>
            <VideoPlayer segments={segments} status={status} />
            
            <div className="controls-bar">
              <div>
                <strong>File:</strong> {fileName}
              </div>
              <button className="secondary" onClick={reset}>
                <RefreshCw size={18} /> Start Over
              </button>
            </div>

            {errorMsg && (
              <div style={{color: 'var(--error)', padding: '1rem', border: '1px solid var(--error)', borderRadius: '8px', backgroundColor: 'rgba(248, 81, 73, 0.1)'}}>
                <strong>Error: </strong> {errorMsg}
              </div>
            )}
          </div>
          
          <NarrationLog narrations={narrations} />
        </div>
      )}
    </div>
  );
}

export default App;
