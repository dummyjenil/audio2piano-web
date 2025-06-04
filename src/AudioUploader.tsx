import React, { useRef, useState } from 'react';

type AudioUploaderProps = {
  onAudioReceived: (fileName: string, blobUrl: string) => void;
};

const AudioUploader: React.FC<AudioUploaderProps> = ({ onAudioReceived }) => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const audioChunksRef = useRef<Blob[]>([]);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioFileName, setAudioFileName] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const blobUrl = URL.createObjectURL(file);
    setAudioUrl(blobUrl);
    setAudioFileName(file.name);
    onAudioReceived(file.name, blobUrl);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const blobUrl = URL.createObjectURL(audioBlob);
        setAudioUrl(blobUrl);
        setAudioFileName('recorded_audio.webm');
        onAudioReceived('recorded_audio.webm', blobUrl);
        audioChunksRef.current = [];
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Microphone access denied or error occurred:', err);
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  return (
    <div>
      <h3>Upload or Record Audio</h3>
      <input type="file" accept="audio/*" onChange={handleFileChange} />
      <div style={{ marginTop: '10px' }}>
        {!isRecording ? (
          <button onClick={startRecording}>🎙️ Start Recording</button>
        ) : (
          <button onClick={stopRecording}>🛑 Stop Recording</button>
        )}
      </div>

      {audioUrl && (
        <div style={{ marginTop: '20px' }}>
          <h4>Audio Preview: {audioFileName}</h4>
          <audio controls src={audioUrl} />
        </div>
      )}
    </div>
  );
};

export default AudioUploader;
