import React, { useState, useRef } from 'react';
import { UploadCloud } from 'lucide-react';

export default function UploadZone({ onUpload }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type === 'video/mp4') {
        onUpload(file);
      } else {
        alert("Please upload an MP4 file.");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (file.type === 'video/mp4') {
        onUpload(file);
      } else {
        alert("Please upload an MP4 file.");
      }
    }
  };

  return (
    <div 
      className={`upload-zone ${isDragging ? 'dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input 
        type="file" 
        accept="video/mp4" 
        style={{ display: 'none' }} 
        ref={fileInputRef}
        onChange={handleFileChange}
      />
      <UploadCloud className="upload-icon" />
      <div className="upload-text">Drag & drop your MP4 video here</div>
      <div className="upload-subtext">or click to browse from your computer</div>
    </div>
  );
}
