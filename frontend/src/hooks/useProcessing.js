import { useState, useCallback } from 'react';

const API_URL = 'http://localhost:8000/api';

export function useProcessing() {
  const [status, setStatus] = useState('idle'); // idle, processing, done, error
  const [errorMsg, setErrorMsg] = useState('');
  const [segments, setSegments] = useState([]);
  const [narrations, setNarrations] = useState([]);
  const [fileName, setFileName] = useState('');

  const reset = useCallback(() => {
    setStatus('idle');
    setErrorMsg('');
    setSegments([]);
    setNarrations([]);
    setFileName('');
  }, []);

  const processVideo = useCallback(async (file) => {
    setStatus('processing');
    setErrorMsg('');
    setSegments([]);
    setNarrations([]);
    setFileName(file.name);

    // Define handleEvent inside to avoid hook dependency issues
    const handleEvent = (type, data) => {
      if (type === 'status') {
        console.log("Status:", data.message);
      } else if (type === 'segment') {
        if (data.text) {
          setNarrations(prev => [...prev, {
            index: data.index,
            text: data.text,
            time: new Date().toLocaleTimeString()
          }]);
        }
        
        if (data.data) {
          const binaryStr = atob(data.data);
          const bytes = new Uint8Array(binaryStr.length);
          for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
          }
          
          setSegments(prev => {
            // Only add if we don't already have this segment
            if (prev.find(s => s.index === data.index)) return prev;
            return [...prev, { index: data.index, buffer: bytes.buffer }];
          });
        }
      } else if (type === 'done') {
        setStatus('done');
      } else if (type === 'error') {
        setStatus('error');
        setErrorMsg(data.message);
      }
    };

    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_URL}/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      if (!response.body) {
        throw new Error("No response body to stream");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const eventSource of events) {
          const lines = eventSource.split('\n');
          let eventType = 'message';
          let dataStr = '';

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataStr += line.slice(5).trim();
            }
          }

          if (dataStr) {
            try {
              const data = JSON.parse(dataStr);
              handleEvent(eventType, data);
            } catch (e) {
              console.error("Failed to parse event data", e, dataStr);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setStatus('error');
      setErrorMsg(err.message || "An error occurred during processing");
    }
  }, []);

  return {
    status,
    errorMsg,
    segments,
    narrations,
    fileName,
    processVideo,
    reset
  };
}
