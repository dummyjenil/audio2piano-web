import { TFLiteModel, loadTFLiteModel } from "@tensorflow/tfjs-tflite";
import { tensor, Tensor } from "@tensorflow/tfjs";

const AUDIO_SAMPLE_RATE = 22050;
const FFT_HOP = 256;
const AUDIO_WINDOW_LENGTH = 2;
const AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP;
const N_OVERLAPPING_FRAMES = 30;
const OVERLAP_LEN = N_OVERLAPPING_FRAMES * FFT_HOP;
const HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN;
const ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE / FFT_HOP;

function windowAudio(audio: Float32Array): Float32Array[] {
    const windows: Float32Array[] = [];
    for (let i = 0; i < audio.length; i += HOP_SIZE) {
        let chunk = audio.slice(i, i + AUDIO_N_SAMPLES);
        if (chunk.length < AUDIO_N_SAMPLES) {
            const padded = new Float32Array(AUDIO_N_SAMPLES);
            padded.set(chunk);
            chunk = padded;
        }
        windows.push(chunk);
    }
    return windows;
}

function unwrapOutput(
    output: number[][][],
    audioOriginalLength: number,
    nOverlappingFrames: number
): number[][] {
    const nOlap = Math.floor(0.5 * nOverlappingFrames);

    let trimmedOutput = output;
    if (nOlap > 0) {
        trimmedOutput = output.map(frame => frame.slice(nOlap, frame.length - nOlap));
    }

    // Flatten the first two dimensions
    const unwrappedOutput: number[][] = [];
    for (const batch of trimmedOutput) {
        for (const frame of batch) {
            unwrappedOutput.push(frame);
        }
    }

    const nOutputFramesOriginal = Math.floor(audioOriginalLength * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE));
    return unwrappedOutput.slice(0, nOutputFramesOriginal);
}

async function predict(
    model: TFLiteModel,
    audioData: Float32Array,
    percentCallback: (percent: number) => void
) {
    const padded = new Float32Array(audioData.length + OVERLAP_LEN / 2);
    padded.set(audioData, OVERLAP_LEN / 2);
    const windows = windowAudio(padded);
    const output: Record<string, number[][][]> = {
        note: [],
        onset: [],
        contour: [],
    };
    const total_windows = windows.length;
    for (let i = 0; i < total_windows; i++) {
        const window = windows[i];
        const inputTensor = tensor(window, [1, AUDIO_N_SAMPLES]);
        const results = model.predict({ 'serving_default_input_2:0': inputTensor } as unknown as any) as unknown as Record<string, Tensor>;
        output.note.push(...(await results["StatefulPartitionedCall:1"].array() as unknown as number[][][]));
        output.onset.push(...(await results["StatefulPartitionedCall:2"].array() as unknown as number[][][]));
        output.contour.push(...(await results["StatefulPartitionedCall:0"].array() as unknown as number[][][]));
        inputTensor.dispose();
        percentCallback(i / total_windows * 100);
        await new Promise(resolve => setTimeout(resolve, 0));
    }

    const originalLength = audioData.length;

    return {
        frames: unwrapOutput(output.note, originalLength, N_OVERLAPPING_FRAMES),
        onsets: unwrapOutput(output.onset, originalLength, N_OVERLAPPING_FRAMES),
        contours: unwrapOutput(output.contour, originalLength, N_OVERLAPPING_FRAMES),
    };
}

export default class Basic_Pitch_tflite {
    model_path: string;
    model!: TFLiteModel;
    constructor(tflite_model_path: string = "model.tflite") {
        this.model_path = tflite_model_path
    }
    async load() {
        this.model = await loadTFLiteModel(this.model_path);
    }
    predict(InputBuffer: Float32Array, processcing_callback: (percent: number) => void) {
        return predict(this.model, InputBuffer, processcing_callback)
    }
}