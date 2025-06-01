import React from 'react';
import './ProgressBar.css'; // optional, for styling

type ProgressBarProps = {
  progress: number; // 0 to 100
};

const ProgressBar: React.FC<ProgressBarProps> = ({ progress }) => {
  const clampProgress = Math.min(100, Math.max(0, progress));

  return (
    <div className="progress-bar-container">
      <div
        className="progress-bar-fill"
        style={{ width: `${clampProgress}%` }}
      />
    </div>
  );
};

export default ProgressBar;
