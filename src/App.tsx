import './App.css'
import prepare_audio from './audio_utils';
import { pred_to_midi, type detectionOptions } from "./toMidi";
import Basic_Pitch_tflite from './basic_pitch_tflite';
// import Basic_Pitch_tf from './basic_pitch_tf';
import AudioUploader from './AudioUploader';
import ProgressBar from './ProgressBar';
import { useState, useEffect } from 'react';
import ConfigComponent from './ConfigComponent';

const App: React.FC = () => {
  const [progress, setProgress] = useState(0);
  const [melody, setMelody] = useState({
    frames: [] as number[][],
    onsets: [] as number[][],
    contours: [] as number[][],
  });
  const [file_name, setFile_name] = useState<string>("output.midi");
  const [audio_url, setAudio_url] = useState<string>("");
  const [mlModel, setMlModel] = useState<Basic_Pitch_tflite | null>(null);
  const [process_audio_click, setProcess_audio_click] = useState(true);
  const [download_midi_click, setDownload_midi_click] = useState(true);
  const [convert_midi_click, setConvert_midi_click] = useState(true);
  const [midi_url, setMidi_url] = useState<string>("");
  const [midi_config, setMidi_config] = useState<detectionOptions>();
  const initialConfig = {
    onsetThresh: 0.5,
    frameThresh: 0.3,
    minNoteLen: 127.70,
    inferOnsets: true,
    maxFreq: null,
    minFreq: null,
    melodiaTrick: true,
    energyTolerance: 11,
  }
  function uuidToNumber() {
    return parseInt(crypto.randomUUID().replace(/\D/g, '').slice(0, 15));
  }

  useEffect(() => {
    (async () => {
      const model = new Basic_Pitch_tflite();
      await model.load();
      setMlModel(model);
    })();
  }, []);

  useEffect(() => {
    if (!mlModel) return;
    (async () => {
      setMelody(await mlModel.predict(await prepare_audio(audio_url), per => setProgress(per)));
      setProgress(0);
    })();
  }, [process_audio_click])

  useEffect(() => {
    if (melody.frames.length === 0) return;
    (async () => {
      const midiBlob = new Blob([await pred_to_midi(melody, file_name, per => setProgress(per), midi_config)], { type: 'audio/midi' });
      setProgress(0);
      setMidi_url(URL.createObjectURL(midiBlob));
    })();
  }, [convert_midi_click]);

  useEffect(() => {
    if (!midi_url) return;
    const iframe = document.getElementById("midiviz") as HTMLIFrameElement;
    iframe.contentWindow?.postMessage({
      type: "load-midi",
      url: midi_url,
      name: file_name + uuidToNumber() + ".mid"
    }, "*");
  }, [midi_url])

  useEffect(() => {
    if (!midi_url) return;
    const a = document.createElement('a');
    a.href = midi_url;
    a.download = file_name + ".mid";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [download_midi_click])


  return (
    <div>
      {!mlModel && "Loading model..."}
      <AudioUploader onAudioReceived={async (name: string, audio_url: string) => {
        setFile_name(name);
        setAudio_url(audio_url);
      }
      } />
      <br />
      <button onClick={() => { setProcess_audio_click(!process_audio_click) }}>process audio</button>
      <br />
      <ProgressBar progress={progress} />
      <br />
      <ConfigComponent initialConfig={initialConfig} onChange={(config) => {
        setMidi_config(config);
      }} />
      <br />
      <button onClick={() => { setConvert_midi_click(!convert_midi_click) }}>convert midi</button>
      <br />
      <button onClick={() => { setDownload_midi_click(!download_midi_click) }}>download midi</button>
      <br />
      <iframe src='vizulizer.html' width={"100%"} height={"500px"} id='midiviz'></iframe>
    </div>
  );
};

export default App;
