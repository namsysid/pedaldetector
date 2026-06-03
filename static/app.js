const canvas = document.getElementById("visualizer");
const ctx = canvas.getContext("2d");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const pedalState = document.getElementById("pedalState");
const interDb = document.getElementById("interDb");
const rms = document.getElementById("rms");
const connectionState = document.getElementById("connectionState");

let audioContext = null;
let mediaStream = null;
let workletNode = null;
let socket = null;
let circles = [];
let background = [255, 255, 255];
let latest = null;
let animationId = null;

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function setConnection(text) {
  connectionState.textContent = text;
}

function wsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws/audio`;
}

async function start() {
  startButton.disabled = true;
  setConnection("Requesting mic");

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });

  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("/static/audio-worklet.js");

  socket = new WebSocket(wsUrl());
  socket.binaryType = "arraybuffer";

  socket.addEventListener("open", () => {
    setConnection("Connected");
    socket.send(JSON.stringify({
      type: "config",
      sampleRate: audioContext.sampleRate,
    }));
  });

  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "ready") {
      connectAudioGraph();
      stopButton.disabled = false;
      return;
    }
    if (data.type === "analysis") {
      latest = data;
      updateHud(data);
      maybeAddCircle(data);
    }
  });

  socket.addEventListener("close", () => {
    setConnection("Disconnected");
    stop();
  });

  socket.addEventListener("error", () => {
    setConnection("Socket error");
  });

  animate();
}

function connectAudioGraph() {
  const source = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "mic-processor");
  workletNode.port.onmessage = (event) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(event.data);
    }
  };

  source.connect(workletNode);
}

function stop() {
  startButton.disabled = false;
  stopButton.disabled = true;

  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  if (socket && socket.readyState <= WebSocket.OPEN) {
    socket.close();
  }
  socket = null;

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  setConnection("Idle");
}

function updateHud(data) {
  const isOn = data.pedal_on;
  pedalState.textContent = isOn ? "On" : "Off";
  pedalState.classList.toggle("on", isOn);
  interDb.textContent = Number.isFinite(data.inter_db) ? data.inter_db.toFixed(1) : "--";
  rms.textContent = Number.isFinite(data.global_rms) ? data.global_rms.toFixed(4) : "--";
}

function maybeAddCircle(data) {
  if (!data.onset) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  circles.push({
    x: Math.random() * rect.width,
    y: Math.random() * rect.height,
    r: Math.max(12, data.onset.r),
    color: data.onset.color,
    alpha: 0.9,
    midi: data.onset.midi,
  });

  if (circles.length > 120) {
    circles.shift();
  }
}

function animate() {
  const rect = canvas.getBoundingClientRect();
  const pedal = latest ? latest.pedal_state : 0;
  const active = latest ? latest.normalized_global_rms : 0;

  if (pedal > 0.5 && circles.length > 0) {
    const color = circles[circles.length - 1].color;
    background = background.map((value, index) => value + 0.02 * (color[index] - value));
  } else if (active > 0.007) {
    background = background.map((value) => value + 0.085 * (255 - value));
  } else {
    background = background.map((value) => value + 0.12 * (255 - value));
  }

  ctx.fillStyle = `rgb(${background.map((value) => Math.round(value)).join(",")})`;
  ctx.fillRect(0, 0, rect.width, rect.height);

  circles = circles.filter((circle) => {
    circle.r *= 0.985;
    circle.alpha *= 0.985;
    return circle.r > 2 && circle.alpha > 0.04;
  });

  for (const circle of circles) {
    ctx.beginPath();
    ctx.arc(circle.x, circle.y, circle.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${circle.color.join(",")},${circle.alpha})`;
    ctx.fill();
  }

  animationId = requestAnimationFrame(animate);
}

window.addEventListener("resize", resizeCanvas);
startButton.addEventListener("click", () => {
  start().catch((error) => {
    setConnection(error.message);
    stop();
  });
});
stopButton.addEventListener("click", stop);

resizeCanvas();
animate();
