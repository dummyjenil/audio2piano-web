import React from 'react';

type FileUploaderProps = {
  onFileUpload: (fileName: string, blobUrl: string) => void;
};

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload }) => {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const fileName = file.name;
    const blobUrl = URL.createObjectURL(file);

    onFileUpload(fileName, blobUrl);

    // Optional: Revoke the object URL when it's no longer needed
    // URL.revokeObjectURL(blobUrl);
  };

  return (
    <div>
      <h2>Upload a File</h2>
      <input type="file" onChange={handleFileChange} />
    </div>
  );
};

export default FileUploader;
