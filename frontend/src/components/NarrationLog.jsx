import React, { useEffect, useRef } from 'react';

export default function NarrationLog({ narrations }) {
  const listRef = useRef(null);

  // Auto-scroll to bottom when new narrations arrive
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [narrations]);

  return (
    <div className="narration-panel">
      <div className="narration-header">
        Narration Log
      </div>
      <div className="narration-list" ref={listRef}>
        {narrations.length === 0 ? (
          <div style={{color: 'var(--text-muted)', textAlign: 'center', marginTop: '2rem'}}>
            Waiting for AI descriptions...
          </div>
        ) : (
          narrations.map((narration, idx) => (
            <div key={idx} className="narration-item">
              <div className="narrator-timestamp">{narration.time} - Chunk {narration.index}</div>
              <div className="narrator-text">{narration.text}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
