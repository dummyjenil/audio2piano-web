import { Midi } from '@tonejs/midi';

type Optional<T> = T | null;

export type detectionOptions = {
  onsetThresh?: number;
  frameThresh?: number;
  minNoteLen?: number;
  maxFreq?: number | null;
  minFreq?: number | null;
  energyTolerance?: number;
  melodiaTrick?: boolean;
  inferOnsets?: boolean;
};

type prediction = {
  frames: number[][];
  onsets: number[][];
  contours: number[][];
};

type NoteEvent = {
  startFrame: number;
  durationFrames: number;
  pitchMidi: number;
  amplitude: number;
  pitchBends?: number[];
};

interface NoteEventTime {
  startTimeSeconds: number;
  durationSeconds: number;
  pitchMidi: number;
  amplitude: number;
  pitchBends?: number[];
}

const MIDI_OFFSET = 21;
const AUDIO_SAMPLE_RATE = 22050;
const AUDIO_WINDOW_LENGTH = 2;
const FFT_HOP = 256;
const ANNOTATIONS_FPS = Math.floor(AUDIO_SAMPLE_RATE / FFT_HOP);
const ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH;
const AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP;
const WINDOW_OFFSET =
  (FFT_HOP / AUDIO_SAMPLE_RATE) * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP) +
  0.0018; //  this is a magic number, but it's needed for this to align properly
const MAX_FREQ_IDX = 87;
const CONTOURS_BINS_PER_SEMITONE = 3;
const ANNOTATIONS_BASE_FREQUENCY = 27.5; // lowest key on a piano
const ANNOTATIONS_N_SEMITONES = 88; // number of piano keys
const N_FREQ_BINS_CONTOURS =
  ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE;

/** PORTED LIBROSA FUNCTIONS */

/**
 * Convert Hz to the appropriate MIDI pitch.
 * @param hz Frequency (in hz).
 * @returns The MIDI pitch.
 */
const hzToMidi = (hz: number): number =>
  12 * (Math.log2(hz) - Math.log2(440.0)) + 69;

/**
 * Converts a MIDI pitch to its respect
 * @param midi The MIDI pitch
 * @returns The frequency of the MIDI pitch in Hz.
 */
const midiToHz = (midi: number): number =>
  440.0 * 2.0 ** ((midi - 69.0) / 12.0);

/**
 * Converts from the model's "frame" time to seconds.
 * @param frame The model's "frame."
 * @returns The time the frame maps to in seconds.
 */
const modelFrameToTime = (frame: number): number =>
  (frame * FFT_HOP) / AUDIO_SAMPLE_RATE -
  WINDOW_OFFSET * Math.floor(frame / ANNOT_N_FRAMES);

/** PORTED NUMPY FUNCTIONS */

/**
 *
 * @param arr Input array.
 * @returns The location of the maximum element in the array.
 */
function argMax(arr: number[]): Optional<number> {
  return arr.length === 0
    ? null
    : arr.reduce(
      (maxIndex, currentValue, index) =>
        arr[maxIndex] > currentValue ? maxIndex : index,
      -1,
    );
}

/**
 *
 * @param arr Input array.
 * @returns The location of the maximum element in each row.
 */
const argMaxAxis1 = (arr: number[][]): number[] =>
  arr.map(row => argMax(row) as number);

/**
 *
 * @param arr2d The input array.
 * @param threshold The value below which we want to filter out.
 * @returns A pair of arrays with the first representing axis 0 and
 * the second representing axis 1. These arrays contain the locations
 * of arr2d which have values greater than threshold.
 */
function whereGreaterThanAxis1(
  arr2d: number[][],
  threshold: number,
): [number[], number[]] {
  const outputX = [];
  const outputY = [];

  for (let i = 0; i < arr2d.length; i++) {
    for (let j = 0; j < arr2d[i].length; j++) {
      if (arr2d[i][j] > threshold) {
        // This is what NumPy does but do we actually want this?
        outputX.push(i);
        outputY.push(j);
      }
    }
  }
  return [outputX, outputY];
}

/**
 * Calculate mean and standard deviation for a 2D-array
 * @param array Array to find mean and standard deviation for.
 * @returns A tuple with the mean and standard deviation.
 */
function meanStdDev(array: number[][]): [number, number] {
  const [sum, sumSquared, count] = array.reduce(
    (prev, row) => {
      // Calculate N * E[x], N * E[x^2] and N
      const [rowSum, rowSumsSquared, rowCount] = row.reduce(
        (p, value) => [p[0] + value, p[1] + value * value, p[2] + 1],
        [0, 0, 0],
      );
      return [prev[0] + rowSum, prev[1] + rowSumsSquared, prev[2] + rowCount];
    },
    [0, 0, 0],
  );
  // E[x]
  const mean = sum / count;
  // sqrt( (1 / (N - 1)) * (E[x^2] - E[x]^2 / N))
  const std = Math.sqrt((1 / (count - 1)) * (sumSquared - (sum * sum) / count));
  return [mean, std];
}

/**
 * Calculate the global max value in a 2D array. This is equivalent to numpy.max
 * @param array Array to calculate max over
 * @returns The maximum value in array.
 */
function globalMax(array: number[][]): number {
  return array.reduce((prev, row) => Math.max(prev, ...row), 0);
}

/**
 * Calculate the minimum over axis 0 for a 3D array
 * @param array Array to calculate min over
 * @returns A 2D array where each element represents the minimum for a fixed first dimension
 */
function min3dForAxis0(array: number[][][]): number[][] {
  const minArray = array[0].map(v => v.slice());
  // np.min axis=0
  for (let x = 1; x < array.length; ++x) {
    for (let y = 0; y < array[0].length; ++y) {
      for (let z = 0; z < array[0][0].length; ++z) {
        minArray[y][z] = Math.min(minArray[y][z], array[x][y][z]);
      }
    }
  }

  return minArray;
}

/**
 * Calculate the relative extrema in an array over axis 0 assuming clipped edges.
 * A TS implemenation of scipy.signal.argrelmax
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html
 *
 * Relative extrema are calculated by finding locations where data[n] > data[n+1:n+order+1]
 * is true.
 * @param data Array to find the relative maxima.
 * @param order How many points on each side to use for the comparison to consider comparator(n, n+x)
 *  to be true.
 * @returns Indices of the maxima. Each element represents indices of the location in data
 * This does not match scipy which returns an n-d tuple with each dimension representing an axis of the
 * data
 */
function argRelMax(array: number[][], order: number = 1): [number, number][] {
  const result: [number, number][] = [];
  // could really use a transpose op right now. But that would also be expensive
  for (let col = 0; col < array[0].length; ++col) {
    for (let row = 0; row < array.length; ++row) {
      let isRelMax = true;

      for (
        let comparisonRow = Math.max(0, row - order);
        isRelMax && comparisonRow <= Math.min(array.length - 1, row + order);
        ++comparisonRow
      ) {
        if (comparisonRow !== row) {
          isRelMax = isRelMax && array[row][col] > array[comparisonRow][col];
        }
      }
      if (isRelMax) {
        result.push([row, col]);
      }
    }
  }
  return result;
}

/**
 * Calculate the maximum over axis 0 for a 3D array
 * @param array Array to calculate min over
 * @returns A 2D array where each element represents the maximum for a fixed first dimension
 */
function max3dForAxis0(array: number[][][]): number[][] {
  const maxArray = array[0].map(v => v.slice());
  for (let x = 1; x < array.length; ++x) {
    for (let y = 0; y < array[0].length; ++y) {
      for (let z = 0; z < array[0][0].length; ++z) {
        maxArray[y][z] = Math.max(maxArray[y][z], array[x][y][z]);
      }
    }
  }
  return maxArray;
}

/** Helpers */

/**
 *
 * @param t Value to check for nullity.
 * @returns True if t is not null else false.
 */
function isNotNull<T>(t: Optional<T>): t is T {
  return t !== null;
}

/**
 * Mutate onsets and frames to have 0s outside of the frequency bounds.
 * @param onsets Onsets output from evaluateModel.
 * @param frames frames output from evaluateModel.
 * @param maxFreq Maximum non-0 frequency in Hz.
 * @param minFreq Minimum non-0 frequency in Hz.
 */
function constrainFrequency(
  onsets: number[][],
  frames: number[][],
  maxFreq: Optional<number>,
  minFreq: Optional<number>,
) {
  if (maxFreq) {
    const maxFreqIdx = hzToMidi(maxFreq) - MIDI_OFFSET;
    for (let i = 0; i < onsets.length; i++) {
      onsets[i].fill(0, maxFreqIdx);
    }
    for (let i = 0; i < frames.length; i++) {
      frames[i].fill(0, maxFreqIdx);
    }
  }

  if (minFreq) {
    const minFreqIdx = hzToMidi(minFreq) - MIDI_OFFSET;
    for (let i = 0; i < onsets.length; i++) {
      onsets[i].fill(0, 0, minFreqIdx);
    }
    for (let i = 0; i < frames.length; i++) {
      frames[i].fill(0, 0, minFreqIdx);
    }
  }
}

/**
 * Infer onsets from large changes in frame amplitudes
 * @param onsets Onsets output from evaluateModel.
 * @param frames frames output from evaluateModel.
 */
function getInferredOnsets(
  onsets: number[][],
  frames: number[][],
  nDiff: number = 2,
): number[][] {
  const diffs = Array.from(Array(nDiff).keys())
    .map(n => n + 1)
    .map(n => {
      const framesAppended: number[][] = Array(n)
        .fill(Array(frames[0].length).fill(0))
        .concat(frames);
      const nPlus = framesAppended.slice(n);
      const minusN = framesAppended.slice(0, -n);
      if (nPlus.length !== minusN.length) {
        throw new Error(
          `nPlus length !== minusN length: ${nPlus.length} !== ${minusN.length}`,
        );
      }
      return nPlus.map((row, r) => row.map((v, c) => v - minusN[r][c]));
    });

  let frameDiff = min3dForAxis0(diffs);

  // frame_diff[frame_diff < 0] = 0
  frameDiff = frameDiff.map(row => row.map(v => Math.max(v, 0)));
  // frame_diff[:n_diff, :] = 0
  frameDiff = frameDiff.map((row, r) => (r < nDiff ? row.fill(0) : row));

  // rescale to have the same max as onsets
  const onsetMax = globalMax(onsets);
  const frameDiffMax = globalMax(frameDiff);
  frameDiff = frameDiff.map(row => row.map(v => (onsetMax * v) / frameDiffMax));

  // use the max of the predicted onsets and the differences
  // max_onsets_diff = np.max([onsets, frame_diff], axis=0)
  return max3dForAxis0([onsets, frameDiff]);
}

/**
 * Decode raw model output to polyphonic note events
 * @param frames: frame activation matrix(n_times, n_freqs)
 * @param onsets: onset activation matrix(n_times, n_freqs)
 * @param onsetThresh: minimum amplitude of an onset activation to be considered an onset
 * @param frameThresh: minimum amplitude of a frame activation for a note to remain "on"
 * @param minNoteLen: minimum allowed note length in frames
 * @param inferOnsets: if True, add additional onsets when there are large differences in frame amplitudes
 * @param maxFreq: maximum allowed output frequency, in Hz
 * @param minFreq: minimum allowed output frequency, in Hz
 * @param melodiaTrick: remove semitones near a peak
 * @param energyTolerance: number of frames allowed to drop below 0
 * @param progress_callback: a callback on progress 
 * @returns List of tuples[(startTimeSeconds, durationSeconds, pitchMidi, amplitude)]
                where amplitude is a number between 0 and 1
 */
async function outputToNotesPoly(
  frames: number[][], onsets: number[][], onsetThresh: number = 0.5, frameThresh: number = 0.3, minNoteLen: number = 5, inferOnsets: boolean = true, maxFreq: Optional<number> = null, minFreq: Optional<number> = null, melodiaTrick: boolean = true, energyTolerance: number = 11, progress_callback: (percent: number) => void,
): Promise<NoteEvent[]> {
  let inferredFrameThresh = frameThresh;
  if (inferredFrameThresh === null) {
    // calculate mean and std deviation of a flattened frames
    const [mean, std] = meanStdDev(frames);
    inferredFrameThresh = mean + std;
  }

  const nFrames = frames.length;

  // Modifies onsets and frames in place.
  constrainFrequency(onsets, frames, maxFreq, minFreq);

  let inferredOnsets = onsets;
  if (inferOnsets) {
    inferredOnsets = getInferredOnsets(onsets, frames); // avoid no-param-reassign
  }

  // a hacky form of zeros-like
  const peakThresholdMatrix = inferredOnsets.map(o => o.map(() => 0));
  argRelMax(inferredOnsets).forEach(([row, col]) => {
    peakThresholdMatrix[row][col] = inferredOnsets[row][col];
  });

  const [noteStarts, freqIdxs] = whereGreaterThanAxis1(
    peakThresholdMatrix,
    onsetThresh,
  );

  noteStarts.reverse();
  freqIdxs.reverse();

  // Deep copy to remaining energy
  const remainingEnergy = frames.map(frame => frame.slice());

  const noteEvents = noteStarts
    .map((noteStartIdx, idx) => {
      const freqIdx = freqIdxs[idx];
      // if we're too close to the end of the audio, continue
      if (noteStartIdx >= nFrames - 1) {
        return null;
      }

      // find time index at this frequency band where the frames drop below an energy threshold
      let i = noteStartIdx + 1;
      let k = 0; // number of frames since energy dropped below threshold
      while (i < nFrames - 1 && k < energyTolerance) {
        if (remainingEnergy[i][freqIdx] < inferredFrameThresh) {
          k += 1;
        } else {
          k = 0;
        }
        i += 1;
      }

      i -= k; // go back to frame above threshold

      // if the note is too short, skip it
      if (i - noteStartIdx <= minNoteLen) {
        return null;
      }

      for (let j = noteStartIdx; j < i; ++j) {
        remainingEnergy[j][freqIdx] = 0;
        if (freqIdx < MAX_FREQ_IDX) {
          remainingEnergy[j][freqIdx + 1] = 0;
        }
        if (freqIdx > 0) {
          remainingEnergy[j][freqIdx - 1] = 0;
        }
      }

      // add the note
      const amplitude =
        frames
          .slice(noteStartIdx, i)
          .reduce((prev, row) => prev + row[freqIdx], 0) /
        (i - noteStartIdx);

      return {
        startFrame: noteStartIdx,
        durationFrames: i - noteStartIdx,
        pitchMidi: freqIdx + MIDI_OFFSET,
        amplitude: amplitude,
      };
    })
    .filter(isNotNull);

  if (melodiaTrick === true) {
    let lastPercent = -1;

    const totalEnergyPoints = remainingEnergy.reduce(
      (sum, row) => sum + row.filter(val => val > inferredFrameThresh).length,
      0,
    );
    let processedPoints = 0;

    while (globalMax(remainingEnergy) > inferredFrameThresh) {
      // Find the (row, column) with the largest value in remainingEnergy
      const [iMid, freqIdx] = remainingEnergy.reduce(
        (prevCoord, currRow, rowIdx) => {
          const colMaxIdx = argMax(currRow)!;
          return currRow[colMaxIdx] >
            remainingEnergy[prevCoord[0]][prevCoord[1]]
            ? [rowIdx, colMaxIdx]
            : prevCoord;
        },
        [0, 0],
      );

      if (remainingEnergy[iMid][freqIdx] > inferredFrameThresh) processedPoints++;
      remainingEnergy[iMid][freqIdx] = 0;

      // Forward pass
      let i = iMid + 1;
      let k = 0;
      while (i < nFrames - 1 && k < energyTolerance) {
        if (remainingEnergy[i][freqIdx] < inferredFrameThresh) {
          k += 1;
        } else {
          k = 0;
        }

        if (remainingEnergy[i][freqIdx] > inferredFrameThresh) processedPoints++;
        remainingEnergy[i][freqIdx] = 0;

        if (freqIdx < MAX_FREQ_IDX) {
          if (remainingEnergy[i][freqIdx + 1] > inferredFrameThresh) processedPoints++;
          remainingEnergy[i][freqIdx + 1] = 0;
        }
        if (freqIdx > 0) {
          if (remainingEnergy[i][freqIdx - 1] > inferredFrameThresh) processedPoints++;
          remainingEnergy[i][freqIdx - 1] = 0;
        }

        i += 1;
      }
      const iEnd = i - 1 - k;

      // Backward pass
      i = iMid - 1;
      k = 0;
      while (i > 0 && k < energyTolerance) {
        if (remainingEnergy[i][freqIdx] < inferredFrameThresh) {
          k += 1;
        } else {
          k = 0;
        }

        if (remainingEnergy[i][freqIdx] > inferredFrameThresh) processedPoints++;
        remainingEnergy[i][freqIdx] = 0;

        if (freqIdx < MAX_FREQ_IDX) {
          if (remainingEnergy[i][freqIdx + 1] > inferredFrameThresh) processedPoints++;
          remainingEnergy[i][freqIdx + 1] = 0;
        }
        if (freqIdx > 0) {
          if (remainingEnergy[i][freqIdx - 1] > inferredFrameThresh) processedPoints++;
          remainingEnergy[i][freqIdx - 1] = 0;
        }

        i -= 1;
      }
      const iStart = i + 1 + k;

      if (iStart < 0) {
        throw new Error(`iStart is not positive! value: ${iStart}`);
      }

      if (iEnd >= nFrames) {
        throw new Error(`iEnd is past end of times. (iEnd, times.length): (${iEnd}, ${nFrames})`);
      }

      // Calculate amplitude
      const amplitude =
        frames.slice(iStart, iEnd).reduce((sum, row) => sum + row[freqIdx], 0) /
        (iEnd - iStart);

      if (iEnd - iStart <= minNoteLen) {
        // Note is too short or too quiet, skip it
        continue;
      }

      // Add the note
      noteEvents.push({
        startFrame: iStart,
        durationFrames: iEnd - iStart,
        pitchMidi: freqIdx + MIDI_OFFSET,
        amplitude: amplitude,
      });
      const percent = Math.floor((processedPoints / totalEnergyPoints) * 100);
      if (percent !== lastPercent) {
        progress_callback(percent);
        lastPercent = percent;
        await new Promise(r => setTimeout(r, 0)); // yield for UI
      }
    }

  }

  return noteEvents;
}

/**
 * Return a symmetric gaussian window. Based on scipy.signal.gaussian. The gaussian window is defined as
 *   w(n) = exp(-1/2 * (n / sigma)^2)
 * @param M Number of points in the output window. If zero or less, an empty array is returned.
 * @param std The standard deviation, sigma.
 * @returns The window, with the maximum value normalized to 1
 */
const gaussian = (M: number, std: number): number[] =>
  Array.from(Array(M).keys()).map(n =>
    Math.exp((-1 * (n - (M - 1) / 2) ** 2) / (2 * std ** 2)),
  );

const midiPitchToContourBin = (pitchMidi: number): number =>
  12.0 *
  CONTOURS_BINS_PER_SEMITONE *
  Math.log2(midiToHz(pitchMidi) / ANNOTATIONS_BASE_FREQUENCY);

function addPitchBendsToNoteEvents(
  contours: number[][],
  notes: NoteEvent[],
  nBinsTolerance: number = 25,
): NoteEvent[] {
  const windowLength = nBinsTolerance * 2 + 1;
  const freqGaussian = gaussian(windowLength, 5);
  return notes.map(note => {
    const freqIdx = Math.floor(
      Math.round(midiPitchToContourBin(note.pitchMidi)),
    );
    const freqStartIdx = Math.max(freqIdx - nBinsTolerance, 0);
    const freqEndIdx = Math.min(
      N_FREQ_BINS_CONTOURS,
      freqIdx + nBinsTolerance + 1,
    );

    const freqGuassianSubMatrix = freqGaussian.slice(
      Math.max(0, nBinsTolerance - freqIdx),
      windowLength -
      Math.max(0, freqIdx - (N_FREQ_BINS_CONTOURS - nBinsTolerance - 1)),
    );
    const pitchBendSubmatrix = contours
      .slice(note.startFrame, note.startFrame + note.durationFrames)
      .map(d =>
        d
          .slice(freqStartIdx, freqEndIdx)
          .map((v, col) => v * freqGuassianSubMatrix[col]),
      );

    const pbShift = nBinsTolerance - Math.max(0, nBinsTolerance - freqIdx);
    const bends = argMaxAxis1(pitchBendSubmatrix).map(v => v - pbShift);
    return {
      ...note,
      pitchBends: bends,
    };
  });
}

const noteFramesToTime = (notes: NoteEvent[]): NoteEventTime[] =>
  notes.map(note => {
    return {
      pitchMidi: note.pitchMidi,
      amplitude: note.amplitude,
      pitchBends: note.pitchBends,
      startTimeSeconds: modelFrameToTime(note.startFrame),
      durationSeconds:
        modelFrameToTime(note.startFrame + note.durationFrames) -
        modelFrameToTime(note.startFrame),
    };
  });

// function generateFileData(notes: NoteEventTime[]): Buffer {
//   const midi = new Midi();
//   const track = midi.addTrack();
//   notes.forEach(note => {
//     track.addNote({
//       midi: note.pitchMidi,
//       time: note.startTimeSeconds,
//       duration: note.durationSeconds,
//       velocity: note.amplitude,
//     });
//     if (note.pitchBends !== undefined && note.pitchBends !== null) {
//       note.pitchBends.forEach((bend, i) => {
//         track.addPitchBend({
//           time:
//             note.startTimeSeconds +
//             (i * note.durationSeconds) / note.pitchBends!.length,
//           value: bend,
//         });
//       });
//     }
//   });
//   return Buffer.from(midi.toArray());
// }

// Define the options for audio detection


export async function pred_to_midi(
  model_output: prediction,
  name: string,
  progress_callback: (percent: number) => void,
  options?: detectionOptions
) {
  const {
    onsetThresh = 0.5,
    frameThresh = 0.3,
    minNoteLen = 127.70,
    inferOnsets = true,
    maxFreq = null,
    minFreq = null,
    melodiaTrick = true,
    energyTolerance = 11,
  } = options || {};
  const midi = new Midi();
  const track = midi.addTrack();
  track.name = name;
  const rawNotes = await outputToNotesPoly(
    model_output.frames,
    model_output.onsets,
    onsetThresh,
    frameThresh,
    Math.round((minNoteLen / 1000) * (AUDIO_SAMPLE_RATE / FFT_HOP)),
    inferOnsets,
    maxFreq,
    minFreq,
    melodiaTrick,
    energyTolerance,
    progress_callback
  );
  const timedNotes = noteFramesToTime(
    addPitchBendsToNoteEvents(model_output.contours, rawNotes)
  );
  // ⚡ Minimize what you store and sort
  const events = timedNotes.map(n => ({
    pitch: n.pitchMidi,
    duration: n.durationSeconds,
    onset: n.startTimeSeconds,
    velocity: n.amplitude,
    bends: n.pitchBends ?? null,
  }));
  // ✅ Must sort: stable and precise
  events.sort((a, b) => a.onset - b.onset || a.pitch - b.pitch);
  const total = events.length;
  let lastPercent = -1;
  for (let i = 0; i < total; i++) {
    const n = events[i];
    track.addNote({
      midi: n.pitch,
      duration: n.duration,
      time: n.onset,
      velocity: n.velocity,
    });
    if (n.bends) {
      const len = n.bends.length;
      for (let j = 0; j < len; j++) {
        track.addPitchBend({
          time: n.onset + (n.duration * j) / len,
          value: n.bends[j],
        });
      }
    }
    const percent = Math.floor((i / total) * 100);
    if (percent !== lastPercent) {
      progress_callback(percent);
      lastPercent = percent;
      await new Promise(r => setTimeout(r, 0)); // yield for UI
    }
  }
  return midi.toArray();
}
