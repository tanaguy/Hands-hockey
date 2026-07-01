# Hand Hockey 🏒

A two-player air-hockey game you play with your **hands** in front of a webcam.
The camera feed is the table; white lines mark the rink and goals; a glowing
LED scoreboard keeps score.

- **Player 1** controls the **left** mallet, **Player 2** the **right** mallet.
- Move your hand — the mallet follows your palm in real time.
- First to **7** wins. Press **R** to reset, **Space** to pause.

## Run it

The webcam only works from a served origin (`localhost` counts, `file://`
does **not**), so start the tiny bundled server:

```bash
cd "Sideproject004 air hockey"
python3 server.py
```

Then open **http://localhost:5175** in Chrome, Edge or Safari and click
**START**. Allow camera access when prompted.

> No camera? The game still runs in a **mouse-mode** fallback — the cursor drives
> whichever mallet's half it's in — so you can try the physics on a laptop.

## How to set up two players

Stand side by side facing the webcam so both of you are in frame. Keep one hand
raised in your own half of the picture (left player on the left, right player on
the right). The mirrored view means the on-screen mallet moves the same way your
hand does.

## Tech / performance notes

- Hand tracking: **MediaPipe HandLandmarker** (GPU delegate), loaded from CDN.
- Zero build step — native ES modules.
- The webcam is a CSS background (no per-frame `drawImage`); only vector game
  graphics are redrawn each frame on a separate canvas.
- Hand detection runs only when a new camera frame arrives; physics + rendering
  run on `requestAnimationFrame`. Device-pixel-ratio is capped at 2. This keeps
  input-to-mallet latency low and the frame rate high.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Markup + start overlay |
| `style.css` | Layout, LED/glow styling |
| `app.js` | Camera, hand tracking, physics, rendering, scoreboard |
| `server.py` | Static server with correct MIME types on port 5175 |
