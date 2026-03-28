import React, { useEffect, useRef, useState } from 'react';

export default function VideoPlayer({ segments, status }) {
  const videoRef = useRef(null);
  const mediaSourceRef = useRef(null);
  const sourceBufferRef = useRef(null);
  const [mseReady, setMseReady] = useState(false);
  const processedCountRef = useRef(0);
  const queueRef = useRef([]);

  // Initialize MSE
  useEffect(() => {
    if (!videoRef.current) return;
    
    const ms = new MediaSource();
    mediaSourceRef.current = ms;
    videoRef.current.src = URL.createObjectURL(ms);

    const onSourceOpen = () => {
      try {
        // Standard fMP4 MIME with H.264 + AAC
        const mimeCodec = 'video/mp4; codecs="avc1.4d401e,mp4a.40.2"';
        
        if (MediaSource.isTypeSupported(mimeCodec)) {
          const sb = ms.addSourceBuffer(mimeCodec);
          sb.mode = 'sequence'; // Ensure segments play exactly one after another
          sourceBufferRef.current = sb;
          setMseReady(true);
          
          sb.addEventListener('updateend', () => {
            if (queueRef.current.length > 0 && !sb.updating) {
              const nextBuf = queueRef.current.shift();
              try {
                sb.appendBuffer(nextBuf);
              } catch (e) {
                console.error("Append error", e);
              }
            } else if (status === 'done' && queueRef.current.length === 0 && ms.readyState === 'open') {
              ms.endOfStream();
            }
            
            // Auto play if paused but we just got data
            if (videoRef.current && videoRef.current.paused && videoRef.current.buffered.length > 0) {
              videoRef.current.play().catch(e => console.log("Autoplay blocked", e));
            }
          });
        } else {
          console.error("MSE MIME type not supported:", mimeCodec);
        }
      } catch (e) {
        console.error("MSE setup error", e);
      }
    };

    ms.addEventListener('sourceopen', onSourceOpen);

    return () => {
      ms.removeEventListener('sourceopen', onSourceOpen);
    };
  }, []);

  // Handle new segments
  useEffect(() => {
    if (!mseReady || !sourceBufferRef.current) return;
    
    // Check if new segments arrived
    if (segments.length > processedCountRef.current) {
      const newSegments = segments.slice(processedCountRef.current);
      processedCountRef.current = segments.length;
      
      for (const seg of newSegments) {
        if (!sourceBufferRef.current.updating && queueRef.current.length === 0) {
          try {
            sourceBufferRef.current.appendBuffer(seg.buffer);
          } catch (e) {
            console.error("Immediate append error", e);
            queueRef.current.push(seg.buffer);
          }
        } else {
          queueRef.current.push(seg.buffer);
        }
      }
    }
  }, [segments, mseReady, status]);

  // Handle stream end
  useEffect(() => {
    if (status === 'done' && mseReady && mediaSourceRef.current) {
      const ms = mediaSourceRef.current;
      const sb = sourceBufferRef.current;
      if (ms.readyState === 'open' && (!sb || !sb.updating) && queueRef.current.length === 0) {
        ms.endOfStream();
      }
    }
  }, [status, mseReady]);

  return (
    <div className="video-container">
      <div className="video-header">
        <div className="video-title">
          Live Generation
        </div>
        <div className={`status-badge ${status}`}>
          {status === 'processing' && <span style={{display: 'flex', alignItems: 'center', gap: '6px'}}><div className="loader" style={{width: 12, height: 12, borderWidth: 2}}></div> Processing</span>}
          {status === 'done' && 'Done'}
          {status === 'error' && 'Error'}
        </div>
      </div>
      <video 
        ref={videoRef} 
        controls 
        autoPlay 
        muted // Needed for some browsers to autoplay
      />
    </div>
  );
}
