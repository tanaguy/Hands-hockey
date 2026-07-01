// Hand Hockey — webcam air-hockey where each player's hand is a mallet.
// Native ES modules, no build step. MediaPipe HandLandmarker runs on the GPU.

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
const WIN_SCORE = 7;
const COUNTDOWN_MS = 2100; // 3 · 2 · 1 before each serve
const GOAL_FREEZE_MS = 1000; // "GOAL" flash before the countdown
const WINNER_MS = 4200; // winner banner before the match resets
const GOAL_RATIO = 0.34; // goal-mouth height as a fraction of the field
const RESTITUTION = 0.98; // wall bounciness
const PADDLE_BOUNCE = 1.06; // mallet bounciness (slightly lively)
const INPUT_LERP = 34; // mallet responsiveness (higher = snappier)
const PUCK_DIAMETER_RATIO = 1 / 7; // puck diameter as a fraction of screen height
const MALLET_TO_PUCK_RATIO = 3.8 / 2.5; // real-world mallet : puck diameter ratio

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const video = document.getElementById("cam");
const gameCanvas = document.getElementById("game");
const sbCanvas = document.getElementById("scoreboard");
const ctx = gameCanvas.getContext("2d");
const sbCtx = sbCanvas.getContext("2d");
const overlay = document.getElementById("overlay");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const msgEl = document.getElementById("message");
const noticeEl = document.getElementById("notice");
const bgBtn = document.getElementById("bgBtn");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let W = 0,
  H = 0,
  dpr = 1;
let dim = {}; // derived sizes (radii, goal, speeds) recomputed on resize
let handLandmarker = null;
let useCamera = true;
let mouseMode = false;
let lastVideoTime = -1;
let detectTs = 0;

let gameState = "menu"; // menu | countdown | playing | goal | over | paused
let stateUntil = 0;
let pausedFrom = "playing";
let serveDir = 1;
let tNow = 0;

const score = [0, 0]; // [P1 (left), P2 (right)]

const players = [
  makePlayer(0), // left
  makePlayer(1), // right
];

const puck = { x: 0, y: 0, vx: 0, vy: 0, r: 14, trail: [] };

function makePlayer(side) {
  return { side, x: 0, y: 0, px: 0, py: 0, vx: 0, vy: 0, tx: 0, ty: 0, r: 40 };
}

// ---------------------------------------------------------------------------
// Layout / sizing
// ---------------------------------------------------------------------------
function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth;
  H = window.innerHeight;

  gameCanvas.width = Math.round(W * dpr);
  gameCanvas.height = Math.round(H * dpr);
  gameCanvas.style.width = W + "px";
  gameCanvas.style.height = H + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  computeDimensions();
  clampPlayer(players[0]);
  clampPlayer(players[1]);
  layoutScoreboard();
  drawScoreboard();
}

function computeDimensions() {
  const inset = Math.round(Math.min(W, H) * 0.015) + 4;
  const goalH = H * GOAL_RATIO;
  dim = {
    inset,
    goalTop: (H - goalH) / 2,
    goalBot: (H + goalH) / 2,
    creaseR: goalH * 0.62,
    centerR: Math.min(W, H) * 0.13,
    serveSpeed: H * 0.72,
    maxPuckSpeed: H * 1.95,
  };
  puck.r = (H * PUCK_DIAMETER_RATIO) / 2; // diameter = H / 7
  players[0].r = players[1].r = puck.r * MALLET_TO_PUCK_RATIO;

  // Sensible spawn positions the first time we have a real size.
  if (players[0].x === 0) {
    resetPositions();
  }
}

function resetPositions() {
  const p1 = players[0];
  const p2 = players[1];
  p1.x = p1.px = p1.tx = W * 0.22;
  p1.y = p1.py = p1.ty = H / 2;
  p2.x = p2.px = p2.tx = W * 0.78;
  p2.y = p2.py = p2.ty = H / 2;
  puck.x = W / 2;
  puck.y = H / 2;
  puck.vx = puck.vy = 0;
  puck.trail.length = 0;
}

function clampPlayer(p) {
  const i = dim.inset;
  const r = p.r;
  const minY = i + r;
  const maxY = H - i - r;
  if (p.side === 0) {
    p.x = clamp(p.x, i + r, W / 2 - r);
    p.tx = clamp(p.tx, i + r, W / 2 - r);
  } else {
    p.x = clamp(p.x, W / 2 + r, W - i - r);
    p.tx = clamp(p.tx, W / 2 + r, W - i - r);
  }
  p.y = clamp(p.y, minY, maxY);
  p.ty = clamp(p.ty, minY, maxY);
}

// ---------------------------------------------------------------------------
// Camera + hand tracking
// ---------------------------------------------------------------------------
function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);
}

async function initCamera() {
  const stream = await withTimeout(
    navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    }),
    6000
  );
  video.srcObject = stream;
  await video.play();
  await new Promise((res) => {
    if (video.videoWidth) return res();
    video.onloadedmetadata = () => res();
  });
}

async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    numHands: 4, // leftmost drives P1, rightmost drives P2; extras ignored
    runningMode: "VIDEO",
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, opts("GPU"));
  } catch (e) {
    handLandmarker = await HandLandmarker.createFromOptions(vision, opts("CPU"));
  }
}

// Map a landmark (normalised to the camera frame) to a screen pixel, taking
// object-fit: cover cropping and the mirrored display into account.
function mapToScreen(nx, ny) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  const scale = Math.max(W / vw, H / vh);
  const dispW = vw * scale;
  const dispH = vh * scale;
  const offX = (W - dispW) / 2;
  const offY = (H - dispH) / 2;
  let x = offX + nx * dispW;
  const y = offY + ny * dispH;
  x = W - x; // the video is shown mirrored
  return { x, y };
}

// Stable palm center: average of wrist + finger MCP joints.
const PALM = [0, 5, 9, 13, 17];
function palmPoint(lm) {
  let sx = 0,
    sy = 0;
  for (const i of PALM) {
    sx += lm[i].x;
    sy += lm[i].y;
  }
  return mapToScreen(sx / PALM.length, sy / PALM.length);
}

function detectHands() {
  if (!handLandmarker || !video.videoWidth) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  detectTs = Math.max(detectTs + 1, performance.now());

  const res = handLandmarker.detectForVideo(video, detectTs);
  const pts = [];
  if (res && res.landmarks) {
    for (const lm of res.landmarks) {
      const p = palmPoint(lm);
      if (p) pts.push(p);
    }
  }
  assignHands(pts);
}

// Left-most hand drives Player 1, right-most drives Player 2.
function assignHands(pts) {
  if (pts.length === 0) return;
  pts.sort((a, b) => a.x - b.x);
  if (pts.length === 1) {
    const p = pts[0];
    setTarget(p.x < W / 2 ? players[0] : players[1], p);
  } else {
    setTarget(players[0], pts[0]);
    setTarget(players[1], pts[pts.length - 1]);
  }
}

function setTarget(p, pt) {
  p.tx = pt.x;
  p.ty = pt.y;
  clampPlayer(p);
}

// ---------------------------------------------------------------------------
// Game loop
// ---------------------------------------------------------------------------
let lastTime = 0;
function loop(now) {
  tNow = now;
  const dt = lastTime ? Math.min(0.033, (now - lastTime) / 1000) : 0.016;
  lastTime = now;

  if (useCamera) detectHands();
  update(dt);
  render();
  requestAnimationFrame(loop);
}

function update(dt) {
  updatePaddles(dt);

  if (gameState === "playing") {
    updatePuck(dt);
  } else if (gameState === "countdown") {
    if (tNow >= stateUntil) {
      gameState = "playing";
      setMessage("");
      launchPuck(serveDir);
    } else {
      const n = Math.ceil((stateUntil - tNow) / (COUNTDOWN_MS / 3));
      setMessage(String(Math.max(1, n)));
    }
  } else if (gameState === "goal") {
    if (tNow >= stateUntil) startCountdown(serveDir);
  } else if (gameState === "over") {
    if (tNow >= stateUntil) resetMatch();
  }
}

function updatePaddles(dt) {
  const a = 1 - Math.exp(-INPUT_LERP * dt);
  for (const p of players) {
    p.px = p.x;
    p.py = p.y;
    p.x += (p.tx - p.x) * a;
    p.y += (p.ty - p.y) * a;
    clampPlayer(p);
    p.vx = (p.x - p.px) / dt;
    p.vy = (p.y - p.py) / dt;
    clampSpeed(p, dim.maxPuckSpeed * 1.4);
  }
}

function updatePuck(dt) {
  const speed = Math.hypot(puck.vx, puck.vy);
  const maxStep = puck.r * 0.75;
  const steps = Math.max(1, Math.ceil((speed * dt) / maxStep));
  const sdt = dt / steps;

  for (let i = 0; i < steps; i++) {
    puck.x += puck.vx * sdt;
    puck.y += puck.vy * sdt;
    if (handleWalls()) return; // goal scored
    collidePaddle(players[0]);
    collidePaddle(players[1]);
  }

  // Very light friction so rallies eventually calm down.
  const f = Math.exp(-0.08 * dt);
  puck.vx *= f;
  puck.vy *= f;
  clampSpeed(puck, dim.maxPuckSpeed);

  puck.trail.unshift({ x: puck.x, y: puck.y });
  if (puck.trail.length > 10) puck.trail.pop();
}

function handleWalls() {
  const i = dim.inset;
  const r = puck.r;
  // top / bottom
  if (puck.y - r < i) {
    puck.y = i + r;
    puck.vy = Math.abs(puck.vy) * RESTITUTION;
  } else if (puck.y + r > H - i) {
    puck.y = H - i - r;
    puck.vy = -Math.abs(puck.vy) * RESTITUTION;
  }
  const inGoal = puck.y > dim.goalTop && puck.y < dim.goalBot;
  // left wall / goal
  if (inGoal) {
    if (puck.x < i) {
      scoreGoal(1); // P2 scores in the left goal
      return true;
    }
  } else if (puck.x - r < i) {
    puck.x = i + r;
    puck.vx = Math.abs(puck.vx) * RESTITUTION;
  }
  // right wall / goal
  if (inGoal) {
    if (puck.x > W - i) {
      scoreGoal(0); // P1 scores in the right goal
      return true;
    }
  } else if (puck.x + r > W - i) {
    puck.x = W - i - r;
    puck.vx = -Math.abs(puck.vx) * RESTITUTION;
  }
  return false;
}

function collidePaddle(p) {
  let dx = puck.x - p.x;
  let dy = puck.y - p.y;
  let dist = Math.hypot(dx, dy);
  const minD = puck.r + p.r;
  if (dist >= minD) return;
  if (dist < 0.001) {
    dx = 0.001;
    dy = 0;
    dist = 0.001;
  }
  const nx = dx / dist;
  const ny = dy / dist;

  // Push the puck out of the mallet.
  puck.x = p.x + nx * minD;
  puck.y = p.y + ny * minD;

  const rvx = puck.vx - p.vx;
  const rvy = puck.vy - p.vy;
  const vn = rvx * nx + rvy * ny;
  if (vn < 0) {
    const j = -(1 + PADDLE_BOUNCE) * vn;
    puck.vx += j * nx;
    puck.vy += j * ny;
  }

  // Guarantee a lively minimum launch so the puck never sticks.
  const sp = Math.hypot(puck.vx, puck.vy);
  const minLaunch = dim.serveSpeed * 0.55;
  if (sp < minLaunch) {
    puck.vx = nx * minLaunch;
    puck.vy = ny * minLaunch;
  }
  clampSpeed(puck, dim.maxPuckSpeed);
}

function launchPuck(dir) {
  const ang = (Math.random() * 2 - 1) * 0.35;
  const sp = dim.serveSpeed;
  puck.x = W / 2;
  puck.y = H / 2;
  puck.vx = dir * sp * Math.cos(ang);
  puck.vy = sp * Math.sin(ang) * (Math.random() < 0.5 ? 1 : -1);
  puck.trail.length = 0;
}

function scoreGoal(scorer) {
  score[scorer]++;
  drawScoreboard();
  puck.vx = puck.vy = 0;
  serveDir = scorer === 1 ? -1 : 1; // serve toward the player who conceded

  if (score[scorer] >= WIN_SCORE) {
    gameState = "over";
    stateUntil = tNow + WINNER_MS;
    setMessage("PLAYER " + (scorer + 1) + " WINS", "small");
  } else {
    gameState = "goal";
    stateUntil = tNow + GOAL_FREEZE_MS;
    setMessage("GOAL");
  }
}

function startCountdown(dir) {
  serveDir = dir;
  puck.x = W / 2;
  puck.y = H / 2;
  puck.vx = puck.vy = 0;
  puck.trail.length = 0;
  gameState = "countdown";
  stateUntil = tNow + COUNTDOWN_MS;
}

function resetMatch() {
  score[0] = score[1] = 0;
  drawScoreboard();
  resetPositions();
  startCountdown(Math.random() < 0.5 ? -1 : 1);
}

// ---------------------------------------------------------------------------
// Rendering — the rink, mallets and puck
// ---------------------------------------------------------------------------
function render() {
  ctx.clearRect(0, 0, W, H);
  if (gameState === "menu") return;
  drawRink();
  drawTrail();
  drawPuck();
  drawPaddle(players[0]);
  drawPaddle(players[1]);
}

// Stroke the current path with a dark underlay so white lines read on any
// background, then the bright white line (optionally glowing).
function strokeWhite(w, glow) {
  // Dark underlay so white lines read on any background.
  ctx.lineWidth = w + 4;
  ctx.strokeStyle = "rgba(0,0,0,0.4)";
  ctx.stroke();
  // Bright white line, optionally glowing (double-stroked to bloom).
  ctx.lineWidth = w;
  ctx.strokeStyle = "rgba(255,255,255,0.98)";
  if (glow) {
    ctx.shadowColor = "rgba(255,255,255,0.95)";
    ctx.shadowBlur = 16;
    ctx.stroke();
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawRink() {
  const i = dim.inset;

  // Boundary
  ctx.beginPath();
  ctx.roundRect(i, i, W - 2 * i, H - 2 * i, Math.min(28, H * 0.03));
  strokeWhite(4, true);

  // Center line (dashed)
  ctx.save();
  ctx.setLineDash([14, 14]);
  ctx.beginPath();
  ctx.moveTo(W / 2, i);
  ctx.lineTo(W / 2, H - i);
  strokeWhite(4, true);
  ctx.restore();

  // Center circle + face-off dot
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, dim.centerR, 0, Math.PI * 2);
  strokeWhite(4, true);
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, 6, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.shadowColor = "rgba(255,255,255,0.95)";
  ctx.shadowBlur = 14;
  ctx.fill();
  ctx.shadowBlur = 0;

  drawGoal(0);
  drawGoal(1);
}

function drawGoal(side) {
  const i = dim.inset;
  const x = side === 0 ? i : W - i;
  const dir = side === 0 ? 1 : -1;

  // Crease (half circle bulging into the field)
  ctx.beginPath();
  ctx.arc(
    x,
    H / 2,
    dim.creaseR,
    side === 0 ? -Math.PI / 2 : Math.PI / 2,
    side === 0 ? Math.PI / 2 : (3 * Math.PI) / 2
  );
  strokeWhite(4, true);

  // Goal posts — short bright marks at the mouth ends
  for (const gy of [dim.goalTop, dim.goalBot]) {
    ctx.beginPath();
    ctx.moveTo(x, gy);
    ctx.lineTo(x + dir * Math.min(30, W * 0.024), gy);
    strokeWhite(8, true);
  }
}

function drawPaddle(p) {
  // faint fill
  ctx.beginPath();
  ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.06)";
  ctx.fill();

  // outer ring
  ctx.beginPath();
  ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(0,0,0,0.35)";
  ctx.stroke();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#fff";
  ctx.shadowColor = "rgba(255,255,255,0.8)";
  ctx.shadowBlur = 12;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // inner knob + center
  ctx.beginPath();
  ctx.arc(p.x, p.y, p.r * 0.5, 0, Math.PI * 2);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
}

function drawTrail() {
  for (let k = puck.trail.length - 1; k >= 1; k--) {
    const t = puck.trail[k];
    const a = (1 - k / puck.trail.length) * 0.28;
    ctx.beginPath();
    ctx.arc(t.x, t.y, puck.r * (1 - k * 0.05), 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255," + a + ")";
    ctx.fill();
  }
}

function drawPuck() {
  ctx.beginPath();
  ctx.arc(puck.x, puck.y, puck.r, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.shadowColor = "rgba(255,255,255,0.95)";
  ctx.shadowBlur = 18;
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(0,0,0,0.25)";
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// LED dot-matrix scoreboard (seven-segment made of dots)
// ---------------------------------------------------------------------------
// Segment membership for every dot on a 5x9 grid. Corner dots belong to two
// segments so the digit reads as a continuous shape.
const DOTS = [
  [0, 0, "fa"], [1, 0, "a"], [2, 0, "a"], [3, 0, "a"], [4, 0, "ab"],
  [0, 1, "f"], [4, 1, "b"],
  [0, 2, "f"], [4, 2, "b"],
  [0, 3, "f"], [4, 3, "b"],
  [0, 4, "feg"], [1, 4, "g"], [2, 4, "g"], [3, 4, "g"], [4, 4, "bcg"],
  [0, 5, "e"], [4, 5, "c"],
  [0, 6, "e"], [4, 6, "c"],
  [0, 7, "e"], [4, 7, "c"],
  [0, 8, "ed"], [1, 8, "d"], [2, 8, "d"], [3, 8, "d"], [4, 8, "cd"],
];
const SEG = {
  0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc",
  5: "afgcd", 6: "afgedc", 7: "abc", 8: "abcdefg", 9: "abcdfg",
};

let sb = {}; // scoreboard layout metrics

function layoutScoreboard() {
  const s = clamp(H * 0.0125, 6, 11);
  const pad = 3 * s;
  const labelH = 2.4 * s;
  const digitW = 4 * s;
  const digitStep = 5.4 * s;
  const groupW = digitStep + digitW; // two digits
  const sepW = 4 * s;
  const totalW = pad * 2 + groupW * 2 + sepW;
  const totalH = pad * 2 + labelH + 8 * s;

  sb = {
    s,
    pad,
    labelH,
    digitStep,
    groupW,
    sepW,
    totalW,
    totalH,
    digitTop: pad + labelH,
    g1x: pad,
    sepx: pad + groupW + sepW / 2,
    g2x: pad + groupW + sepW,
  };

  sbCanvas.width = Math.round(totalW * dpr);
  sbCanvas.height = Math.round(totalH * dpr);
  sbCanvas.style.width = totalW + "px";
  sbCanvas.style.height = totalH + "px";
  sbCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawDigit(value, ox, oy) {
  const s = sb.s;
  const segs = SEG[value] || "";
  for (const [c, r, ms] of DOTS) {
    let lit = false;
    for (const m of ms) if (segs.includes(m)) { lit = true; break; }
    const x = ox + c * s;
    const y = oy + r * s;
    sbCtx.beginPath();
    sbCtx.arc(x, y, s * 0.32, 0, Math.PI * 2);
    if (lit) {
      sbCtx.fillStyle = "#ffffff";
      sbCtx.shadowColor = "rgba(255,255,255,0.75)";
      sbCtx.shadowBlur = s * 0.55;
    } else {
      sbCtx.fillStyle = "rgba(255,255,255,0.05)";
      sbCtx.shadowBlur = 0;
    }
    sbCtx.fill();
  }
  sbCtx.shadowBlur = 0;
}

function drawTwoDigit(n, gx) {
  const tens = Math.floor(n / 10) % 10;
  const ones = n % 10;
  drawDigit(tens, gx, sb.digitTop);
  drawDigit(ones, gx + sb.digitStep, sb.digitTop);
}

function drawScoreboard() {
  if (!sb.totalW) return;
  const s = sb.s;
  sbCtx.clearRect(0, 0, sb.totalW, sb.totalH);

  // Housing
  sbCtx.beginPath();
  sbCtx.roundRect(0.5, 0.5, sb.totalW - 1, sb.totalH - 1, 10);
  sbCtx.fillStyle = "rgba(0,0,0,0.5)";
  sbCtx.fill();
  sbCtx.lineWidth = 1.5;
  sbCtx.strokeStyle = "rgba(255,255,255,0.18)";
  sbCtx.stroke();

  // Labels
  sbCtx.fillStyle = "rgba(255,255,255,0.5)";
  sbCtx.font = "600 " + Math.round(s * 1.7) + "px Arial, sans-serif";
  sbCtx.textAlign = "center";
  sbCtx.textBaseline = "middle";
  sbCtx.fillText("P1", sb.g1x + sb.groupW / 2, sb.pad + sb.labelH * 0.45);
  sbCtx.fillText("P2", sb.g2x + sb.groupW / 2, sb.pad + sb.labelH * 0.45);

  // Digits
  drawTwoDigit(score[0], sb.g1x);
  drawTwoDigit(score[1], sb.g2x);

  // Colon separator
  for (const ry of [2.5, 5.5]) {
    sbCtx.beginPath();
    sbCtx.arc(sb.sepx, sb.digitTop + ry * s, s * 0.32, 0, Math.PI * 2);
    sbCtx.fillStyle = "#ffffff";
    sbCtx.shadowColor = "rgba(255,255,255,0.75)";
    sbCtx.shadowBlur = s * 0.55;
    sbCtx.fill();
  }
  sbCtx.shadowBlur = 0;
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function setMessage(txt, cls) {
  if (!txt) {
    msgEl.className = "hidden";
    msgEl.textContent = "";
    return;
  }
  msgEl.textContent = txt;
  msgEl.className = cls === "small" ? "small" : "";
}

function setNotice(txt) {
  noticeEl.textContent = txt || "";
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function clampSpeed(o, max) {
  const s = Math.hypot(o.vx, o.vy);
  if (s > max) {
    const k = max / s;
    o.vx *= k;
    o.vy *= k;
  }
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------
async function start() {
  startBtn.disabled = true;
  try {
    statusEl.textContent = "Requesting camera…";
    await initCamera();
    statusEl.textContent = "Loading hand tracking…";
    await initHandLandmarker();
    useCamera = true;
    setNotice("R reset · Space pause · B background · P1 left · P2 right");
  } catch (err) {
    // No camera (or blocked): fall back to mouse control so the game still runs.
    console.warn("Camera/model unavailable, using mouse mode:", err);
    useCamera = false;
    mouseMode = true;
    setNotice("Camera off — mouse / touch controls the mallet in each half · B background");
  }

  overlay.classList.add("hidden");
  resize();
  resetPositions();
  startCountdown(Math.random() < 0.5 ? -1 : 1);
  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", start);
window.addEventListener("resize", resize);

// Pointer control: mouse, pen and multitouch. Each active touch fires its own
// pointer event, so two fingers (or two players on a touchscreen) drive both
// mallets at once — the one whose half the pointer is in. Always on alongside
// the camera as a fallback/second input.
function pointerControl(e) {
  if (gameState === "menu") return;
  if (e.target && e.target.closest && e.target.closest("button")) return;
  const p = e.clientX < W / 2 ? players[0] : players[1];
  p.tx = e.clientX;
  p.ty = e.clientY;
  clampPlayer(p);
}
window.addEventListener("pointermove", pointerControl, { passive: true });
window.addEventListener("pointerdown", pointerControl, { passive: true });

// ---- Background modes: webcam / blurred webcam / black -----------------------
const BG_MODES = ["clear", "blur", "black"];
const BG_LABEL = { clear: "BG · WEBCAM", blur: "BG · BLURRED", black: "BG · BLACK" };
let bgIndex = 0;
function applyBg() {
  const m = BG_MODES[bgIndex];
  document.body.classList.remove("bg-clear", "bg-blur", "bg-black");
  document.body.classList.add("bg-" + m);
  bgBtn.textContent = BG_LABEL[m];
}
function cycleBg() {
  bgIndex = (bgIndex + 1) % BG_MODES.length;
  applyBg();
}
bgBtn.addEventListener("click", cycleBg);
applyBg();

window.addEventListener("keydown", (e) => {
  if (e.key === "r" || e.key === "R") {
    if (gameState !== "menu") resetMatch();
  } else if (e.key === "b" || e.key === "B") {
    cycleBg();
  } else if (e.key === " ") {
    e.preventDefault();
    if (gameState === "playing") {
      pausedFrom = gameState;
      gameState = "paused";
      setMessage("PAUSED", "small");
    } else if (gameState === "paused") {
      gameState = pausedFrom;
      setMessage("");
    }
  }
});

// Keep the boot sizing sane before the first frame.
resize();
