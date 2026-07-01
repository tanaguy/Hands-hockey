# Hand Hockey 🏒 — Web Worker edition

A two-player air-hockey game you play with your **hands** in front of a webcam.
The camera feed is the table; white lines mark the rink and goals; a glowing
LED scoreboard keeps score.

This is the **Web Worker** build: MediaPipe hand tracking runs on a background
thread, so inference never stalls the render loop. It's the smoothest version.

- **Player 1** controls the **left** mallet, **Player 2** the **right** mallet.
- Move your hand — the mallet follows your palm in real time.
- First to **7** wins. Press **R** to reset, **Space** to pause, **B** for
  background mode (webcam / blurred / black).

## Run it

The webcam only works from a served origin (`localhost` counts, `file://`
does **not**), so start the tiny bundled server:

```bash
cd "Sideproject004 air hockey webworker"
python3 server.py
```

Then open **http://localhost:5176** in Chrome, Edge or Safari and click
**START**. Allow camera access when prompted.

> No camera? The game still runs in a **mouse / touch** fallback so you can try
> the physics on a laptop. On a touchscreen, two fingers drive both mallets.

## How the worker version differs

| | Main-thread version | This (worker) version |
| --- | --- | --- |
| Where inference runs | main thread (blocks render ~10–25 ms/detection) | **Web Worker** (render never waits) |
| Frame hand-off | `detectForVideo(video)` | `createImageBitmap(video)` transferred to the worker |
| Back-pressure | none | one frame in flight at a time (`workerBusy`) |

The main thread grabs each camera frame as a transferable `ImageBitmap` and
posts it to `vision.worker.js`. The worker runs HandLandmarker and posts back
just the palm points. Because only one frame is ever in flight, the render loop
stays at 60 fps no matter how long a detection takes.

Open the console after START — you'll see `HandLandmarker ready in worker — GPU
delegate`. If it says **CPU**, enable hardware acceleration for a big speed-up.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Markup + start overlay |
| `style.css` | Layout, LED/glow styling |
| `app.js` | Camera, game loop, physics, rendering, scoreboard, worker glue |
| `vision.worker.js` | Background thread: MediaPipe HandLandmarker |
| `server.py` | Static server with correct MIME types on port 5176 |
