import './App.css'
import prepare_audio from './audio_utils';
import pred_to_midi from "./toMidi";
import Basic_Pitch_tflite from './basic_pitch_tflite';
// import Basic_Pitch_tf from './basic_pitch_tf';
import FileUploader from './FileUploader';
import ProgressBar from './ProgressBar';
import { useState } from 'react';


const App: React.FC = async () => {
  const ml_model = new Basic_Pitch_tflite();
  await ml_model.load()
  const [progress, setProgress] = useState(0);
  async function handleFileUpload(name: string, audio_url: string) {
    const result = await ml_model.predict(await prepare_audio(audio_url), (per) => setProgress(per));
    const url = URL.createObjectURL(new Blob([pred_to_midi(result, name)], { type: 'audio/midi' }));
    const iframe = document.getElementById("midiviz") as HTMLIFrameElement;
    iframe.contentWindow?.postMessage({
      type: "load-midi",
      url: url,
      name: name
    }, "*");
  }

  return (
    <div>
      <h1>File Upload Example</h1>
      <FileUploader onFileUpload={handleFileUpload} />
      <ProgressBar progress={progress} />
      {progress > 0 && progress < 100 && <p>Processing: {progress.toFixed(0)}%</p>}
      <iframe src='vizulizer.html' width={"100%"} height={"500px"} id='midiviz'></iframe>
    </div>
  );
};

export default App;
