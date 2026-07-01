// Hand-tracking worker. Runs MediaPipe HandLandmarker off the main thread so
// inference never blocks rendering. The main thread sends camera frames as
// transferable ImageBitmaps; we reply with palm points in normalised coords.
//
// NOTE: this is a *classic* worker (created without { type: "module" }).
// MediaPipe's WASM loader calls importScripts() internally, which is forbidden
// in module workers — so we stay classic and pull the ESM bundle in with a
// dynamic import() instead of a top-level import.

const LIB_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const PALM = [0, 5, 9, 13, 17]; // wrist + finger MCP joints → stable palm center

let landmarker = null;
let ts = 0;

async function init(numHands) {
  const { HandLandmarker, FilesetResolver } = await import(LIB_URL);
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    numHands: numHands || 2,
    runningMode: "VIDEO",
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  let delegate = "GPU";
  try {
    landmarker = await HandLandmarker.createFromOptions(vision, opts("GPU"));
  } catch (e) {
    delegate = "CPU";
    landmarker = await HandLandmarker.createFromOptions(vision, opts("CPU"));
  }
  postMessage({ type: "ready", delegate });
}

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === "init") {
    try {
      await init(msg.numHands);
    } catch (err) {
      postMessage({
        type: "error",
        error: String(err && err.message ? err.message : err),
      });
    }
    return;
  }

  if (msg.type === "frame") {
    const bmp = msg.bitmap;
    if (!landmarker) {
      if (bmp && bmp.close) bmp.close();
      postMessage({ type: "result", hands: [] });
      return;
    }
    ts = Math.max(ts + 1, msg.ts || performance.now());

    const hands = [];
    try {
      const res = landmarker.detectForVideo(bmp, ts);
      if (res && res.landmarks) {
        for (const lm of res.landmarks) {
          let sx = 0;
          let sy = 0;
          for (const i of PALM) {
            sx += lm[i].x;
            sy += lm[i].y;
          }
          hands.push({ x: sx / PALM.length, y: sy / PALM.length });
        }
      }
    } catch (err) {
      // Ignore transient inference errors; report no hands this frame.
    }
    if (bmp && bmp.close) bmp.close();
    postMessage({ type: "result", hands });
  }
};
