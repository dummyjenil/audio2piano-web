import React from 'react';
import './ProgressBar.css'; // optional, for styling

type ProgressBarProps = {
  progress: number; // 0 to 100
};

const ProgressBar: React.FC<ProgressBarProps> = ({ progress }) => {
  const clampProgress = Math.floor(progress);

  return (
    <div>
      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{ width: `${clampProgress}%` }}
        />
      </div>
      {progress > 0 && progress < 100 && <p>Processing: {progress.toFixed(0)}%</p>}
    </div>
  );
};

export default ProgressBar;
