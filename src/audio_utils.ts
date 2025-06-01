function convertToMono(audioCtx: AudioContext, audioBuffer: AudioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  if (numChannels === 1) {
    return audioBuffer;
  }
  const length = audioBuffer.length;
  const sampleRate = audioBuffer.sampleRate;
  const monoBuffer = audioCtx.createBuffer(1, length, sampleRate);
  const monoData = monoBuffer.getChannelData(0);
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      monoData[i] += channelData[i] / numChannels;
    }
  }
  return monoBuffer;
}

export default async function prepare_audio(audioURL: string) {
  const audioCtx = new AudioContext({ sampleRate: 22050 });
  const response = await fetch(audioURL);
  const arrayBuffer = await response.arrayBuffer();
  const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  return convertToMono(audioCtx, decodedBuffer).getChannelData(0);
}
