const appState = {
  apiBase: "",
  streamUrl: "/stream.mjpg",
  screen: "dashboard",
  simulationMode: true,
  companionEnabled: true,
  deferredInstallPrompt: null,
  eventSource: null,
  connection: {
    detectorAlive: true,
    streamAlive: true,
    backendAlive: false,
  },
  status: {
    fps: 28.4,
    temperature: 74.2,
    pir: "Armed",
    system: "Active",
    uptime: "4h 12m",
    lastEventAge: "2m ago",
    source: "Mock still",
  },
  detections: [
    {
      label: "raccoon",
      confidence: 0.78,
      priority: "priority",
      time: "9:40 PM",
      ago: "2m ago",
      image: "/other/20260417_200655_359_frame3384_raccoon.jpg",
      summary: "North fence line, short pass near the trash bins.",
      box: { x1: 1020, y1: 262, x2: 1310, y2: 518 },
      frame_w: 1536,
      frame_h: 864,
    },
    {
      label: "deer",
      confidence: 0.83,
      priority: "routine",
      time: "9:36 PM",
      ago: "6m ago",
      image: "/other/20260417_202846_829_frame4190_deer.jpg",
      summary: "Stayed near the tree line for about 12 seconds.",
      box: { x1: 408, y1: 137, x2: 592, y2: 360 },
      frame_w: 1536,
      frame_h: 864,
    },
    {
      label: "bear",
      confidence: 0.91,
      priority: "critical",
      time: "8:58 PM",
      ago: "44m ago",
      image: "/other/20260417_203117_022_frame2677_bear.jpg",
      summary: "Large silhouette near the back gate. Critical class.",
      box: { x1: 640, y1: 180, x2: 1140, y2: 660 },
      frame_w: 1536,
      frame_h: 864,
    },
  ],
  chat: [
    {
      role: "assistant",
      text: "Today looked calm overall. Two deer passes, one raccoon visit, and one older bear sighting in the archive.",
    },
    {
      role: "user",
      text: "Was the raccoon just passing through?",
    },
    {
      role: "assistant",
      text: "The latest snapshot suggests a short pass near the bins instead of a long stop. Once backend summaries land, this answer will come from the real event log.",
    },
  ],
  companion: {
    state: "idle",
    x: 26,
    y: 92,
    facing: "left",
    activeLabel: "idle",
  },
};

const screens = Array.from(document.querySelectorAll("[data-screen]"));
const tabs = Array.from(document.querySelectorAll("[data-nav-target]"));
const streamImage = document.getElementById("stream-image");
const streamStage = document.getElementById("stream-stage");
const streamLabel = document.getElementById("stream-label");
const mockBox = document.getElementById("mock-box");
const dashboardFeed = document.getElementById("dashboard-feed");
const detectionsList = document.getElementById("detections-list");
const chatThread = document.getElementById("chat-thread");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const companion = document.getElementById("kanna-companion");
const companionToggle = document.getElementById("companion-toggle");
const companionStateLabel = document.getElementById("kanna-state-label");
const installButton = document.getElementById("install-button");
const kannaImage = document.getElementById("kanna-image");
const tabbar = document.querySelector(".tabbar");

const kannaAssetMap = {
  idle: "./assets/kanna/idle.png",
  jump: "./assets/kanna/jump.png",
  "point-left": "./assets/kanna/point_left.png",
  "point-right": "./assets/kanna/point_right.png",
  alert: "./assets/kanna/alert.png",
  sit: "./assets/kanna/sit.png",
  fall: "./assets/kanna/fall.png",
  "fall-ready": "./assets/kanna/fall_ready.png",
  recover: "./assets/kanna/recover.png",
  think: "./assets/kanna/think.png",
};

const kannaFrameSets = {
  idle: [
    "./assets/kanna/idle.png",
  ],
  jump: [
    "./assets/kanna/jump/jump_01.png",
    "./assets/kanna/jump/jump_02.png",
    "./assets/kanna/jump/jump_03.png",
    "./assets/kanna/jump/jump_04.png",
    "./assets/kanna/jump/jump_05.png",
    "./assets/kanna/jump/jump_06.png",
  ],
  "walk-left": [
    "./assets/kanna/walk-left/walk_01.png",
    "./assets/kanna/walk-left/walk_02.png",
    "./assets/kanna/walk-left/walk_03.png",
    "./assets/kanna/walk-left/walk_04.png",
    "./assets/kanna/walk-left/walk_05.png",
    "./assets/kanna/walk-left/walk_06.png",
    "./assets/kanna/walk-left/walk_07.png",
    "./assets/kanna/walk-left/walk_08.png",
    "./assets/kanna/walk-left/walk_09.png",
    "./assets/kanna/walk-left/walk_10.png",
  ],
  "walk-right": [
    "./assets/kanna/walk-right/walk_01.png",
    "./assets/kanna/walk-right/walk_02.png",
    "./assets/kanna/walk-right/walk_03.png",
    "./assets/kanna/walk-right/walk_04.png",
    "./assets/kanna/walk-right/walk_05.png",
    "./assets/kanna/walk-right/walk_06.png",
    "./assets/kanna/walk-right/walk_07.png",
    "./assets/kanna/walk-right/walk_08.png",
    "./assets/kanna/walk-right/walk_09.png",
    "./assets/kanna/walk-right/walk_10.png",
    "./assets/kanna/walk-right/walk_11.png",
    "./assets/kanna/walk-right/walk_12.png",
    "./assets/kanna/walk-right/walk_13.png",
    "./assets/kanna/walk-right/walk_14.png",
    "./assets/kanna/walk-right/walk_15.png",
    "./assets/kanna/walk-right/walk_16.png",
    "./assets/kanna/walk-right/walk_17.png",
    "./assets/kanna/walk-right/walk_18.png",
  ],
  fall: [
    "./assets/kanna/fall/fall_01.png",
    "./assets/kanna/fall/fall_02.png",
    "./assets/kanna/fall/fall_03.png",
    "./assets/kanna/fall/fall_04.png",
    "./assets/kanna/fall/fall_05.png",
    "./assets/kanna/fall/fall_06.png",
    "./assets/kanna/fall/fall_06.png",
    "./assets/kanna/fall/fall_06.png",
    "./assets/kanna/fall/fall_07.png",
    "./assets/kanna/fall/fall_07.png",
    "./assets/kanna/fall/fall_08.png",
    "./assets/kanna/fall/fall_08.png",
    "./assets/kanna/fall/fall_09.png",
    "./assets/kanna/fall/fall_09.png",
  ],
};

let kannaFrameInterval = null;
let kannaFrameIndex = 0;
let kannaWalkScheduler = null;
let kannaWalkMotion = null;
let kannaActionTimeout = null;
let kannaFrameTimeout = null;
const kannaJumpOffsets = [0, 18, 34, 28, 14, 0];

function render() {
  renderNavigation();
  renderConnectionStrip();
  renderDashboard();
  renderDetections();
  renderChat();
  renderLiveCard();
  renderCompanion();
}

function renderNavigation() {
  screens.forEach((screen) => {
    screen.classList.toggle("is-active", screen.dataset.screen === appState.screen);
  });

  tabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.navTarget === appState.screen);
  });
}

function renderConnectionStrip() {
  document.getElementById("status-detector").textContent = appState.connection.detectorAlive
    ? "Detector live"
    : "Detector offline";
  document.getElementById("status-detector").className = `status-pill ${appState.connection.detectorAlive ? "online" : "pending"}`;

  document.getElementById("status-stream").textContent = appState.connection.streamAlive
    ? "Stream ready"
    : "Stream stalled";
  document.getElementById("status-stream").className = `status-pill ${appState.connection.streamAlive ? "online" : "pending"}`;

  document.getElementById("status-backend").textContent = appState.connection.backendAlive
    ? "Backend linked"
    : "Backend pending";
  document.getElementById("status-backend").className = `status-pill ${appState.connection.backendAlive ? "online" : "pending"}`;

  document.getElementById("status-mode").textContent = appState.simulationMode
    ? "Simulation mode"
    : "Backend mode";
}

function renderDashboard() {
  const latest = appState.detections[0];
  document.getElementById("hero-summary").textContent =
    `Last activity was a ${latest.label} ${latest.ago}. ${latest.summary}`;
  document.getElementById("metric-system").textContent = appState.status.system;
  document.getElementById("metric-uptime").textContent = `Uptime ${appState.status.uptime}`;
  document.getElementById("metric-fps").textContent = String(appState.status.fps);
  document.getElementById("metric-temp").textContent = `${appState.status.temperature.toFixed(1)} F`;
  document.getElementById("metric-pir").textContent = appState.status.pir;

  dashboardFeed.innerHTML = appState.detections
    .slice(0, 3)
    .map((item) => {
      return `
        <article class="feed-item">
          <img src="${item.image}" alt="${item.label} detection">
          <div class="feed-copy">
            <h4>${capitalize(item.label)} - ${(item.confidence * 100).toFixed(0)}%</h4>
            <p>${item.summary}</p>
            <p>${item.time} - ${item.ago}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderDetections() {
  detectionsList.innerHTML = appState.detections
    .map((item) => {
      return `
        <article class="detection-item">
          <img src="${item.image}" alt="${item.label} snapshot">
          <div class="detection-copy">
            <h4>${capitalize(item.label)} - ${(item.confidence * 100).toFixed(0)}%</h4>
            <p>${item.summary}</p>
            <p>${item.time} - ${item.priority}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderChat() {
  chatThread.innerHTML = appState.chat
    .map((message) => {
      return `
        <article class="chat-bubble ${message.role}">
          <strong>${message.role === "assistant" ? "Field assistant" : "You"}</strong>
          <p>${message.text}</p>
        </article>
      `;
    })
    .join("");
}

function renderLiveCard() {
  const latest = appState.detections[0];
  document.getElementById("stream-source").textContent = appState.status.source;
  document.getElementById("stream-resolution").textContent = `${latest.frame_w} x ${latest.frame_h}`;
  document.getElementById("stream-last-event").textContent = `${latest.label} ${latest.ago}`;
  streamLabel.textContent = `${capitalize(latest.label)} ${latest.confidence.toFixed(2)}`;

  const mapped = mapBoxToStage(latest.box, latest.frame_w, latest.frame_h, streamStage);
  mockBox.style.left = `${mapped.left}px`;
  mockBox.style.top = `${mapped.top}px`;
  mockBox.style.width = `${mapped.width}px`;
  mockBox.style.height = `${mapped.height}px`;
  mockBox.className = `detection-box ${latest.priority}`;
}

function setPreviewSource(url) {
  streamImage.src = url;
}

function renderCompanion() {
  companion.hidden = !appState.companionEnabled;
  companion.style.left = `${appState.companion.x}px`;
  companion.style.bottom = `${appState.companion.y}px`;
  companion.className = `kanna-companion kanna-${appState.companion.state}`;

  const src = currentKannaFrame();
  if (src) {
    kannaImage.src = src;
    kannaImage.hidden = false;
    kannaImage.alt = `Kanna ${appState.companion.activeLabel}`;
    kannaImage.style.transform = "scaleX(1)";
  } else {
    kannaImage.hidden = true;
  }
}

function currentKannaFrame() {
  const frames = kannaFrameSets[appState.companion.state];
  if (frames && frames.length > 0) {
    return frames[kannaFrameIndex % frames.length];
  }
  return kannaAssetMap[appState.companion.state] || kannaAssetMap.idle;
}

function frameDelayForState(state) {
  if (state === "idle") {
    return 320;
  }
  if (state === "jump") {
    return 120;
  }
  if (state === "fall") {
    return 165;
  }
  if (state === "walk-left" || state === "walk-right") {
    return 210;
  }
  return 160;
}

function frameDelayForIndex(state, index) {
  if (state !== "fall") {
    return frameDelayForState(state);
  }

  if (index >= 5) {
    return 2000;
  }

  return 190;
}

function isLoopingState(state) {
  return state === "idle" || state === "walk-left" || state === "walk-right";
}

function syncKannaFramePlayback() {
  if (kannaFrameInterval) {
    window.clearInterval(kannaFrameInterval);
    kannaFrameInterval = null;
  }
  if (kannaFrameTimeout) {
    window.clearTimeout(kannaFrameTimeout);
    kannaFrameTimeout = null;
  }

  kannaFrameIndex = 0;
  const frames = kannaFrameSets[appState.companion.state];
  if (!frames) {
    kannaWalkMotion = null;
    renderCompanion();
    return;
  }

  if (frames.length === 1) {
    kannaWalkMotion = null;
    renderCompanion();
    return;
  }

  renderCompanion();
  if (appState.companion.state === "fall") {
    const advanceFallFrame = () => {
      if (appState.companion.state !== "fall") {
        kannaFrameTimeout = null;
        return;
      }

      if (kannaFrameIndex < frames.length - 1) {
        kannaFrameIndex += 1;
        renderCompanion();
        kannaFrameTimeout = window.setTimeout(
          advanceFallFrame,
          frameDelayForIndex("fall", kannaFrameIndex)
        );
        return;
      }

      kannaFrameTimeout = null;
    };

    kannaFrameTimeout = window.setTimeout(
      advanceFallFrame,
      frameDelayForIndex("fall", kannaFrameIndex)
    );
    return;
  }

  const frameDelay = frameDelayForState(appState.companion.state);
  kannaFrameInterval = window.setInterval(() => {
    const canLoop = isLoopingState(appState.companion.state);
    if (canLoop) {
      kannaFrameIndex = (kannaFrameIndex + 1) % frames.length;
    } else if (kannaFrameIndex < frames.length - 1) {
      kannaFrameIndex += 1;
    } else {
      window.clearInterval(kannaFrameInterval);
      kannaFrameInterval = null;
      renderCompanion();
      return;
    }

    if ((appState.companion.state === "walk-left" || appState.companion.state === "walk-right") && kannaWalkMotion) {
      const elapsed = Date.now() - kannaWalkMotion.startTime;
      const progress = Math.min(elapsed / kannaWalkMotion.duration, 1);
      appState.companion.x = Math.round(
        kannaWalkMotion.startX +
        ((kannaWalkMotion.endX - kannaWalkMotion.startX) * progress)
      );
    }
    if (appState.companion.state === "jump") {
      appState.companion.y = defaultCompanionBottom() + (kannaJumpOffsets[kannaFrameIndex] || 0);
    }
    renderCompanion();
  }, frameDelay);
}

function clearCompanionActionTimeout() {
  if (kannaActionTimeout) {
    window.clearTimeout(kannaActionTimeout);
    kannaActionTimeout = null;
  }
  if (kannaFrameTimeout) {
    window.clearTimeout(kannaFrameTimeout);
    kannaFrameTimeout = null;
  }
}

function resetCompanionToIdle() {
  clearCompanionActionTimeout();
  appState.companion.state = "idle";
  appState.companion.activeLabel = "idle";
  appState.companion.y = defaultCompanionBottom();
  kannaWalkMotion = null;
  syncKannaFramePlayback();
  scheduleNextWalk();
}

function playFallSequence() {
  clearCompanionActionTimeout();
  if (!appState.companionEnabled || appState.companion.state === "jump" || appState.companion.state === "alert") {
    resetCompanionToIdle();
    return;
  }

  appState.companion.state = "fall";
  appState.companion.activeLabel = "oops";
  appState.companion.y = defaultCompanionBottom();
  kannaWalkMotion = null;
  syncKannaFramePlayback();

  const duration = (5 * 190) + (Math.max(kannaFrameSets.fall.length - 5, 0) * 2000) + 600;
  kannaActionTimeout = window.setTimeout(() => {
    resetCompanionToIdle();
  }, duration);
}

function scheduleNextWalk() {
  if (kannaWalkScheduler) {
    window.clearTimeout(kannaWalkScheduler);
  }

  const delay = 4000 + Math.floor(Math.random() * 1200);
  kannaWalkScheduler = window.setTimeout(() => {
    if (!appState.companionEnabled || appState.companion.state !== "idle") {
      scheduleNextWalk();
      return;
    }

      const nextX = randomWalkwayX(appState.companion.x);
    const walkFacing = nextX < appState.companion.x ? "left" : "right";
    const walkState = `walk-${walkFacing}`;
    const walkFrames = kannaFrameSets[walkState] || [];
    const walkDuration = walkDurationMs(appState.companion.x, nextX, walkFrames.length);
    kannaWalkMotion = {
      startX: appState.companion.x,
      endX: nextX,
      startTime: Date.now(),
      duration: walkDuration,
    };
    appState.companion.facing = walkFacing;
    appState.companion.state = walkState;
    appState.companion.activeLabel = "strolling";
    appState.companion.y = defaultCompanionBottom();
    syncKannaFramePlayback();

      clearCompanionActionTimeout();
      kannaActionTimeout = window.setTimeout(() => {
        if (kannaWalkMotion) {
          appState.companion.x = kannaWalkMotion.endX;
        }
          if (Math.random() < 0.42) {
            playFallSequence();
            return;
          }
        resetCompanionToIdle();
      }, walkDuration);
  }, delay);
}

function mapBoxToStage(box, frameW, frameH, stageElement) {
  const stageRect = stageElement.getBoundingClientRect();
  const stageWidth = stageRect.width;
  const stageHeight = stageRect.height;
  const stageAspect = stageWidth / stageHeight;
  const frameAspect = frameW / frameH;

  let drawWidth = stageWidth;
  let drawHeight = stageHeight;
  let offsetX = 0;
  let offsetY = 0;

  if (frameAspect > stageAspect) {
    drawHeight = stageWidth / frameAspect;
    offsetY = (stageHeight - drawHeight) / 2;
  } else {
    drawWidth = stageHeight * frameAspect;
    offsetX = (stageWidth - drawWidth) / 2;
  }

  return {
    left: offsetX + (box.x1 / frameW) * drawWidth,
    top: offsetY + (box.y1 / frameH) * drawHeight,
    width: ((box.x2 - box.x1) / frameW) * drawWidth,
    height: ((box.y2 - box.y1) / frameH) * drawHeight,
  };
}

function defaultCompanionBottom() {
  if (!tabbar) {
    return 92;
  }
  const rect = tabbar.getBoundingClientRect();
  return Math.max(window.innerHeight - rect.top + 10, 92);
}

function defaultCompanionLeft() {
  if (!tabbar) {
    return 26;
  }
  const rect = tabbar.getBoundingClientRect();
  return Math.round(rect.left + 10);
}

function walkwayRight() {
  if (!tabbar) {
    return Math.max(window.innerWidth - 180, 26);
  }
  const rect = tabbar.getBoundingClientRect();
  return Math.max(Math.round(rect.right - 146), defaultCompanionLeft());
}

function randomWalkwayX(exceptX = null) {
  const left = defaultCompanionLeft();
  const right = walkwayRight();
  if (right <= left) {
    return left;
  }

  let candidate = left + Math.round(Math.random() * (right - left));
  if (exceptX === null) {
    return candidate;
  }

  let attempts = 0;
  while (Math.abs(candidate - exceptX) < 54 && attempts < 8) {
    candidate = left + Math.round(Math.random() * (right - left));
    attempts += 1;
  }
  return candidate;
}

function walkDurationMs(startX, endX, frameCount) {
  const distance = Math.abs(endX - startX);
  const frameDelay = frameDelayForState(endX < startX ? "walk-left" : "walk-right");
  const pixelsPerFrame = 16;
  const requiredFrames = Math.max(frameCount, Math.ceil(distance / pixelsPerFrame));
  return requiredFrames * frameDelay;
}

function triggerCompanionForDetection(detection) {
  if (!appState.companionEnabled) {
    return;
  }

  const mapped = mapBoxToStage(detection.box, detection.frame_w, detection.frame_h, streamStage);
  const stageRect = streamStage.getBoundingClientRect();
  const targetCenterX = stageRect.left + mapped.left + (mapped.width / 2);
  const targetOnLeft = targetCenterX < window.innerWidth / 2;
  const targetX = targetOnLeft
    ? Math.min(targetCenterX + 18, walkwayRight())
    : Math.max(defaultCompanionLeft(), targetCenterX - 136);

  appState.companion.x = targetX;
  appState.companion.y = defaultCompanionBottom();
  appState.companion.state = detection.priority === "critical" ? "alert" : "jump";
  appState.companion.activeLabel = `tracking ${detection.label}`;
  appState.companion.facing = targetOnLeft ? "left" : "right";
  kannaWalkMotion = null;
  appState.companion.y = defaultCompanionBottom();
  clearCompanionActionTimeout();
  syncKannaFramePlayback();

  kannaActionTimeout = window.setTimeout(() => {
    appState.companion.y = defaultCompanionBottom();
    appState.companion.state = targetOnLeft ? "point-left" : "point-right";
    appState.companion.activeLabel = `${targetOnLeft ? "point left" : "point right"}`;
    syncKannaFramePlayback();
    kannaActionTimeout = window.setTimeout(resetCompanionToIdle, 1500);
  }, 900);
}

function pushDetection(label) {
  const seed = detectionSeed(label);
  const detection = {
    ...seed,
    ago: "just now",
    time: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
  };

  appState.detections.unshift(detection);
  appState.detections = appState.detections.slice(0, 12);
  appState.status.lastEventAge = "just now";
  appState.status.source = "Simulation stream";
  triggerCompanionForDetection(detection);
  render();
}

function detectionSeed(label) {
  if (label === "bear") {
    return {
      label: "bear",
      confidence: 0.92,
      priority: "critical",
      image: "/other/20260417_203117_022_frame2677_bear.jpg",
      summary: "Critical class near the back gate.",
      box: { x1: 630, y1: 180, x2: 1126, y2: 660 },
      frame_w: 1536,
      frame_h: 864,
    };
  }

  if (label === "raccoon") {
    return {
      label: "raccoon",
      confidence: 0.79,
      priority: "priority",
      image: "/other/20260417_200655_359_frame3384_raccoon.jpg",
      summary: "Short pass near the bin line.",
      box: { x1: 1000, y1: 270, x2: 1290, y2: 522 },
      frame_w: 1536,
      frame_h: 864,
    };
  }

  return {
    label: "deer",
    confidence: 0.84,
    priority: "routine",
    image: "/other/20260417_202846_829_frame4190_deer.jpg",
    summary: "Browsing along the tree line.",
    box: { x1: 420, y1: 140, x2: 618, y2: 366 },
    frame_w: 1536,
    frame_h: 864,
  };
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function applyStatus(status) {
  appState.connection.backendAlive = true;
  appState.connection.detectorAlive = Boolean(status.detector_alive);
  appState.connection.streamAlive = Boolean(status.stream_alive);
  appState.simulationMode = Boolean(status.simulation_mode);
  appState.status.fps = Number(status.fps ?? appState.status.fps);
  appState.status.temperature = Number(status.temperature_f ?? appState.status.temperature);
  appState.status.pir = status.pir || appState.status.pir;
  appState.status.system = status.system || appState.status.system;
  appState.status.uptime = status.uptime || appState.status.uptime;
  appState.status.source = appState.simulationMode ? "Backend simulation" : "Live stream target";
  appState.streamUrl = status.stream_url || appState.streamUrl;
}

function normalizeDetection(item) {
  return {
    label: item.label,
    confidence: Number(item.confidence),
    priority: item.priority || "routine",
    time: item.time || "Unknown",
    ago: item.ago || "unknown",
    image: item.image || "/other/20260417_202846_829_frame4190_deer.jpg",
    summary: item.summary || `${item.label} sighting recorded.`,
    box: item.box || { x1: 0, y1: 0, x2: 0, y2: 0 },
    frame_w: item.frame_w || 1536,
    frame_h: item.frame_h || 864,
    timestamp: item.timestamp,
    sighting_id: item.sighting_id,
  };
}

async function loadBackendState() {
  try {
    const statusResponse = await fetch(`${appState.apiBase}/api/status`);
    if (!statusResponse.ok) {
      throw new Error("status failed");
    }
    const status = await statusResponse.json();
    applyStatus(status);

    const detectionsResponse = await fetch(`${appState.apiBase}/api/detections?limit=12`);
    if (detectionsResponse.ok) {
      const payload = await detectionsResponse.json();
      if (Array.isArray(payload.items) && payload.items.length > 0) {
        appState.detections = payload.items.map(normalizeDetection);
      }
    }

    connectEventStream();
  } catch (error) {
    appState.connection.backendAlive = false;
    appState.simulationMode = true;
    appState.status.source = "Mock still";
  }

  render();
}

function connectEventStream() {
  if (!window.EventSource) {
    return;
  }
  if (appState.eventSource) {
    appState.eventSource.close();
  }

  const source = new EventSource(`${appState.apiBase}/api/events`);
  source.onopen = () => {
    appState.connection.backendAlive = true;
    renderConnectionStrip();
  };
  source.onerror = () => {
    appState.connection.backendAlive = false;
    renderConnectionStrip();
  };
  source.onmessage = handleIncomingEvent;
  source.addEventListener("sighting_start", handleIncomingEvent);
  source.addEventListener("sighting_update", handleIncomingEvent);
  source.addEventListener("sighting_end", handleIncomingEvent);
  appState.eventSource = source;
}

function handleIncomingEvent(event) {
  try {
    const payload = JSON.parse(event.data);
    const detection = normalizeDetection({
      ...payload,
      image: payload.snapshot,
      time: new Date(payload.timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
      ago: "just now",
      summary: payload.summary || `${payload.label} sighting recorded.`,
    });
    appState.detections.unshift(detection);
    appState.detections = appState.detections.slice(0, 12);
    appState.status.source = appState.simulationMode ? "Backend simulation" : "Live stream target";
    triggerCompanionForDetection(detection);
    render();
  } catch (error) {
    return;
  }
}

async function emitDetection(label) {
  if (!appState.connection.backendAlive) {
    pushDetection(label);
    return;
  }

  try {
    const response = await fetch(`${appState.apiBase}/api/dev/emit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    if (!response.ok) {
      throw new Error("emit failed");
    }
  } catch (error) {
    pushDetection(label);
  }
}

async function askBackend(question) {
  try {
    const response = await fetch(`${appState.apiBase}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      throw new Error("ask failed");
    }
    const payload = await response.json();
    return payload.answer;
  } catch (error) {
    return "Backend ask endpoint is not live yet, so this answer is still coming from the frontend fallback.";
  }
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.navTarget;
    if (!target) {
      return;
    }
    appState.screen = target;
    render();
  });
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();
  if (!question) {
    return;
  }

  appState.chat.push({ role: "user", text: question });
  chatInput.value = "";
  renderChat();

  askBackend(question).then((answer) => {
    appState.chat.push({ role: "assistant", text: answer });
    renderChat();
  });
});

companionToggle.addEventListener("change", () => {
  appState.companionEnabled = companionToggle.checked;
  renderCompanion();
});

document.querySelectorAll("[data-sim-label]").forEach((button) => {
  button.addEventListener("click", () => {
    emitDetection(button.dataset.simLabel || "deer");
  });
});

document.getElementById("simulate-event").addEventListener("click", () => {
  emitDetection("raccoon");
});

document.getElementById("refresh-stream").addEventListener("click", () => {
  if (!appState.connection.backendAlive) {
    setPreviewSource(appState.detections[0].image);
    appState.status.source = "Mock still";
  } else {
    setPreviewSource(`${appState.streamUrl}?t=${Date.now()}`);
    appState.status.source = "Live stream target";
  }
  appState.connection.streamAlive = true;
  renderConnectionStrip();
  renderLiveCard();
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && appState.connection.backendAlive) {
    setPreviewSource(`${appState.streamUrl}?t=${Date.now()}`);
  }
});

window.addEventListener("resize", () => {
  renderLiveCard();
  if (
    appState.companion.state === "idle" ||
    appState.companion.state === "walk-left" ||
    appState.companion.state === "walk-right"
  ) {
    appState.companion.x = Math.min(
      walkwayRight(),
      Math.max(defaultCompanionLeft(), appState.companion.x)
    );
    appState.companion.y = defaultCompanionBottom();
    syncKannaFramePlayback();
  }
});

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  appState.deferredInstallPrompt = event;
  installButton.hidden = false;
});

installButton.addEventListener("click", async () => {
  if (!appState.deferredInstallPrompt) {
    return;
  }
  appState.deferredInstallPrompt.prompt();
  await appState.deferredInstallPrompt.userChoice;
  appState.deferredInstallPrompt = null;
  installButton.hidden = true;
});

if ("serviceWorker" in navigator && window.location.protocol !== "file:") {
  navigator.serviceWorker.register("./service-worker.js").catch(() => {
    appState.connection.backendAlive = false;
    renderConnectionStrip();
  });
}

appState.companion.x = randomWalkwayX();
appState.companion.y = defaultCompanionBottom();

render();
syncKannaFramePlayback();
scheduleNextWalk();
loadBackendState();
