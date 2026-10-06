// 浏览器端：16kHz 麦克风采集 → WS 上行；WS 下行 PCM → WebAudio 绝对时间轴队列。
// 打断由服务端 VAD 判定后下发 vox.interrupt，本地同步清播放队列。
const SR = 16000;
const els = {
  dot: document.getElementById("dot"), state: document.getElementById("state"),
  stage: document.getElementById("stage"),
  stats: document.getElementById("stats"), btn: document.getElementById("btn"),
  log: document.getElementById("log"),
};

let ws = null, mic = null, running = false, starting = false, curTurn = null, micLevel = 0, micPeak = 0;
async function refreshBrain() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) return;
    const health = await response.json(), model = health.models?.llm || "";
    const label = document.getElementById("brain");
    const names = { "deepseek-v4-flash": "DeepSeek Flash", "deepseek-flash": "DeepSeek Flash", "deepseek-v4-pro": "DeepSeek V4 Pro" };
    const name = names[model] || (model.includes("Qwen3") ? "Qwen3 local" : model);
    label.textContent = name + (health.thinking_disabled === true ? " · thinking off" : "")
      + (health.llm_ready === false ? " · unavailable" : "");
    label.title = model;
    label.classList.toggle("unavailable", health.llm_ready === false);
  } catch (_) {}
}
refreshBrain();
setInterval(refreshBrain, 5000);
function sendControl(type, fields = {}) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type, ...fields }));
}
setInterval(() => {
  if (!running) return;
  const pct = Math.min(100, Math.round(micLevel * 400));
  // 用区间峰值判断是否"通"：瞬时值常落在词间静音上，会误报成麦克风没接
  const alive = micPeak > 0.004;
  micPeak *= 0.75;
  els.stats.textContent = (alive ? "mic " : "mic (no signal) ")
    + "█".repeat(Math.ceil(pct / 12)).padEnd(9, "·");
}, 100);

// ── 播放队列：绝对时间轴排程，块间无缝；flush 立即停止所有已排程源 ──────────
const player = {
  ctx: null, nextStart: 0, sources: new Set(),
  // 音画同步：每回合开始时先"扣住"音频，等本回合第一帧画面到了再一起开播（最多扣 maxHold）。
  // 之后音频块紧接排程，帧按 pts（相对本回合音频起点）显示。
  held: [], holding: false, turnStart: 0, _holdTimer: null,
  turn: null, generationDone: false, epoch: 0, samples: 0,
  started: 0, ended: 0, peak: 0, segments: [],
  beginTurn(id, maxHoldMs) {
    this.flush();
    this.turn = id; this.samples = 0; this.generationDone = false;
    this.holding = maxHoldMs > 0;
    if (this.holding) this._holdTimer = setTimeout(() => this.release(), maxHoldMs);
  },
  release() {
    if (!this.holding) return;
    this.holding = false; clearTimeout(this._holdTimer);
    for (const b of this.held) this._schedule(b);
    this.held = [];
    this.checkDrained();
  },
  turnTimeMs() {
    if (!this.ctx || !this.turnStart) return -1;
    if (this.ctx.currentTime < this.turnStart) return (this.ctx.currentTime - this.turnStart) * 1000;
    return this.playedMs();
  },
  ensure() {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
    return this.ctx;
  },
  feed(bytes) {
    if (this.turn === null) return;       // 打断之后迟到的音频不再排程
    if (this.holding) { this.held.push(bytes); return; }
    this._schedule(bytes);
  },
  _schedule(bytes) {
    const ctx = this.ensure();
    if (!bytes.byteLength || bytes.byteLength % 2) throw new Error("Invalid PCM audio packet");
    // 'A' 标记占一个字节，subarray(1) 的偏移不对齐。DataView 可从任意偏移读取 little-endian PCM。
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const f32 = new Float32Array(bytes.byteLength / 2);
    for (let i = 0; i < f32.length; i++) {
      f32[i] = view.getInt16(i * 2, true) / 32768;
      this.peak = Math.max(this.peak, Math.abs(f32[i]));
    }
    const buf = ctx.createBuffer(1, f32.length, SR);
    buf.copyToChannel(f32, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination);
    // 首块留 30ms 抖动余量；后续块紧接上一块结束时刻（关键：用 nextStart 而非 currentTime，
    // 否则每块都按"当前时刻+余量"排程，累积出可听的间隙）
    const t = Math.max(ctx.currentTime + 0.03, this.nextStart);
    src.start(t);
    if (!this.turnStart) this.turnStart = t;      // 本回合音频起点 = 帧 pts 的零点
    this.nextStart = t + buf.duration;
    this.segments.push({ start: t, duration: buf.duration });
    this.samples += f32.length; this.started++;
    this.sources.add(src);
    const epoch = this.epoch;
    src.onended = () => {
      this.sources.delete(src); this.ended++;
      if (epoch === this.epoch) this.checkDrained();
    };
  },
  playedMs() {
    if (!this.ctx) return 0;
    // 音频块之间的网络断流不算已经听过；字幕、口型和history使用同一个样本时钟。
    return this.segments.reduce((ms, segment) => ms + Math.max(0, Math.min(segment.duration, this.ctx.currentTime - segment.start)) * 1000, 0);
  },
  isPlaying() {
    // 有排程源不等于已经出声：首块前的余量与网络断流都要保持静默口型。
    if (this.turn === null || !this.ctx || this.ctx.state !== "running" || this.holding) return false;
    const now = this.ctx.currentTime;
    return this.segments.some(segment => segment.start <= now && now < segment.start + segment.duration);
  },
  finishGeneration(id) {
    if (id !== this.turn) return;
    this.generationDone = true;
    if (this.holding && !this.held.length) this.release();
    this.checkDrained();
  },
  checkDrained() {
    if (this.turn === null || !this.generationDone || this.holding || this.held.length || this.sources.size) return;
    const id = this.turn, played_ms = this.playedMs();
    this.turn = null;
    CAPTION.finish(); FACE.end();
    sendControl("playback_done", { id, played_ms });
    setState("listening");
  },
  flush() {
    // stop() 也触发 onended；先废弃世代，禁止把取消误报成正常播放结束。
    this.epoch++; this.turn = null; this.generationDone = false;
    for (const s of this.sources) { try { s.stop(); } catch (_) {} }
    this.sources.clear(); this.nextStart = 0;
    this.segments = [];
    this.held = []; this.holding = false; this.turnStart = 0; clearTimeout(this._holdTimer);
  },
  bufferedMs() {
    if (!this.ctx || !this.nextStart) return 0;
    return Math.max(0, (this.nextStart - this.ctx.currentTime) * 1000);
  },
};

// ── 数字人画面：服务端按音频现算的帧（JPEG + pts）→ canvas，按音频时钟显示 ──────
// 帧到得比音频晚时保持上一帧（嘴短暂定格）；比音频早 ≤100ms 允许先画（ITU-R BT.1359：
// 画面领先音频不易察觉，音频领先画面很刺眼）。静默时同步恢复缓存的中性待机图。
// ── 字幕：服务端按音频位置发来每段文字，按音频时钟到点显示，跟嘴同步 ──────────
const CAPTION = {
  items: [], turn: 0, bubble: null, text: "", finalText: "",
  begin(id) { this.turn = id; this.items = []; this.text = ""; this.finalText = ""; this.bubble = addTurn("CK", "", "ck"); curTurn = this.bubble; },
  push(turn, pts, text) { if (turn !== this.turn) return; this.items.push({ pts, text }); },
  tick() {
    if (!this.items.length || !this.bubble) return;
    const now = player.turnTimeMs();
    if (now < 0) return;
    let n = 0;
    while (n < this.items.length && this.items[n].pts <= now + 150) n++;
    if (!n) return;
    for (const it of this.items.splice(0, n)) this.text += (this.text ? " " : "") + it.text;
    this.bubble.querySelector(".body").textContent = this.text;
    els.log.scrollTop = els.log.scrollHeight;
  },
  finish() {
    // 只有真正播完才显示全文；生成完时仍保留按 pts 排程的字幕。
    if (this.bubble && this.finalText) this.bubble.querySelector(".body").textContent = this.finalText;
    this.items = []; this.text = this.finalText || this.text;
  },
  cut() { this.items = []; },
};

const IDLE_PTS = 0xFFFFFFFF;   // 服务端待机帧标记：立即画，不排时间轴
const FACE = {
  canvas: null, g: null, frames: [], turn: 0, fps: 20, enabled: false, visible: false,
  shown: 0, talkShown: 0, dropped: 0, inTurn: false, lastTalkPts: -1, epoch: 0,
  idleCanvas: null, idleReady: false, idleShowing: false,
  init(stage) {
    if (this.g) return;
    this.canvas = stage.querySelector("#face");
    if (!this.canvas) return;
    this.g = this.canvas.getContext("2d");
    const tick = () => { this.tick(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  },
  begin(id, fps, face) {
    this.epoch++; this.clear();
    this.enabled = !!face; this.turn = id; this.fps = fps || 20; this.inTurn = true; this.lastTalkPts = -1;
    player.beginTurn(id, face ? 2500 : 0);
    CAPTION.begin(id);
    this.restoreIdle();
  },
  push(turn, pts, bytes) {
    const idle = pts === IDLE_PTS;
    if (!this.enabled || (!idle && (!this.inTurn || turn !== this.turn))) { this.dropped++; return; }
    const epoch = this.epoch;
    createImageBitmap(new Blob([bytes], { type: "image/jpeg" })).then((bm) => {
      if (epoch !== this.epoch || !this.enabled) { bm.close(); return; }
      if (idle) {
        // 服务端提供静默中性图；保存一份可同步重画的像素，bitmap 随后关闭。
        this.cacheIdle(bm); bm.close();
        // 正在播音时只更新缓存，不用普通 idle 覆盖说话口型。
        if (!this.inTurn) this.restoreIdle(true);
        return;
      }
      if (!this.inTurn || turn !== this.turn) { bm.close(); return; }
      this.lastTalkPts = Math.max(this.lastTalkPts, pts);
      this.frames.push({ pts, bm });
      if (this.frames.length > 1 && pts < this.frames[this.frames.length - 2].pts) this.frames.sort((a, b) => a.pts - b.pts);
      if (player.holding) player.release();          // 第一帧到了：音频开播
    }).catch((e) => console.error("Video frame decoding failed", e));
  },
  tick() {
    CAPTION.tick();
    if (!this.g) return;
    if (!player.isPlaying()) { this.restoreIdle(); return; }
    if (!this.frames.length) return;
    const now = player.turnTimeMs();
    if (now < 0) return;
    let k = -1;
    for (let i = 0; i < this.frames.length; i++) { if (this.frames[i].pts <= now + 100) k = i; else break; }
    if (k < 0) return;
    const f = this.frames[k];
    for (let i = 0; i < k; i++) this.frames[i].bm.close();
    this.dropped += k; this.talkShown++;
    this.frames.splice(0, k + 1);
    this.draw(f.bm);
  },
  draw(bm) {
    this.idleShowing = false;
    this.paint(bm); bm.close();
  },
  paint(image) {
    if (this.canvas.width !== image.width || this.canvas.height !== image.height) {
      this.canvas.width = image.width; this.canvas.height = image.height;
    }
    this.g.drawImage(image, 0, 0); this.shown++;
    this.show(true);
  },
  cacheIdle(bm) {
    if (!this.idleCanvas) this.idleCanvas = document.createElement("canvas");
    this.idleCanvas.width = bm.width; this.idleCanvas.height = bm.height;
    this.idleCanvas.getContext("2d").drawImage(bm, 0, 0);
    this.idleReady = true;
  },
  restoreIdle(force = false) {
    if (!this.g) return;
    if (!this.enabled || !this.idleReady) { this.idleShowing = false; this.show(false); return; }
    if (this.idleShowing && !force) return;
    this.paint(this.idleCanvas); this.idleShowing = true;
  },
  show(v) {
    if (v === this.visible || !this.canvas) return;
    this.visible = v; this.canvas.style.opacity = v ? "1" : "0";
  },
  // 回合结束立即恢复中性图，不依赖后台渲染线程何时再次发待机帧。
  end() {
    this.epoch++; this.inTurn = false; this.clear(); this.restoreIdle();
  },
  cut() { this.end(); CAPTION.cut(); },
  clear() { for (const f of this.frames) f.bm.close(); this.frames = []; },
};

// ── UI ────────────────────────────────────────────────────────────────────
function setState(s) {
  els.state.textContent = s;
  els.dot.className = "dot " + (["listening", "thinking", "speaking", "transcribing"].includes(s) ? s : "");
  els.stage.classList.toggle("talking", s === "speaking");
}
function addTurn(who, text, cls) {
  const d = document.createElement("div");
  d.className = "turn " + cls;
  d.innerHTML = `<div class="who">${who}</div><div class="body"></div>`;
  d.querySelector(".body").textContent = text;
  els.log.appendChild(d); els.log.scrollTop = els.log.scrollHeight;
  return d;
}
function setMeta(el, s) {
  if (!el) return;
  let m = el.querySelector(".meta");
  if (!m) { m = document.createElement("div"); m.className = "meta"; el.appendChild(m); }
  m.textContent = s;
}

// ── 麦克风 ────────────────────────────────────────────────────────────────
const WORKLET = `
class PCMCapture extends AudioWorkletProcessor {
  process(inputs){ const i=inputs[0];
    if(i&&i[0]&&i[0].length) this.port.postMessage(i[0].slice(0));
    return true; }
}
registerProcessor("pcm-capture", PCMCapture);`;

async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({
    // echoCancellation 是必须的：扬声器播放的合成语音会被麦克风拾取，
    // 否则 VAD 会把 AI 自己的声音判成用户打断，形成自触发循环。
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext({ sampleRate: SR });
  await ctx.audioWorklet.addModule(
    URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" })));
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  node.port.onmessage = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const f = e.data, i16 = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) {
      const s = Math.max(-1, Math.min(1, f[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    ws.send(i16.buffer);                       // 二进制上行，不做 base64
    // 麦克风电平指示：让用户一眼看出有没有采到声音
    let sum = 0; for (let i = 0; i < f.length; i += 4) sum += f[i] * f[i];
    const rms = Math.sqrt(sum / Math.max(1, f.length / 4));
    micLevel = Math.max(rms, micLevel * 0.85);
    micPeak = Math.max(micPeak, rms);          // 区间峰值，供指示器判断
  };
  ctx.createMediaStreamSource(stream).connect(node);
  const g = ctx.createGain(); g.gain.value = 0;   // 必须连到 destination 否则 worklet 不被调度
  node.connect(g); g.connect(ctx.destination);
  mic = { ctx, stream, node };
}
function stopMic() {
  if (!mic) return;
  mic.node.disconnect(); mic.stream.getTracks().forEach(t => t.stop()); mic.ctx.close().catch(() => {});
  mic = null;
}

function cutPlayback() {
  if (player.turn !== null) sendControl("playback_progress", { id: player.turn, played_ms: player.playedMs() });
  player.flush(); FACE.cut();
  els.stage.classList.remove("talking");
}
function disconnect(sock = ws) {
  if (sock !== ws) return;
  cutPlayback(); stopMic();
  ws = null; running = false; micLevel = micPeak = 0;
  els.btn.textContent = "Start"; setState("disconnected");
  if (sock && sock.readyState < WebSocket.CLOSING) sock.close();
}

// ── 连接 ──────────────────────────────────────────────────────────────────
function connect() {
  const sock = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws = sock; sock.binaryType = "arraybuffer";
  sock.onopen = () => { if (ws !== sock) return; setState("listening"); addTurn("system", "New conversation started", "sys"); };
  sock.onclose = () => { if (ws === sock) disconnect(sock); };
  sock.onerror = () => { if (ws === sock) { addTurn("system", "Connection lost. Press Start to reconnect.", "sys"); disconnect(sock); } };
  sock.onmessage = (ev) => {
    if (ws !== sock) return;
    try {
    if (ev.data instanceof ArrayBuffer) {
      const u8 = new Uint8Array(ev.data);
      if (u8[0] === 0x41) player.feed(u8.subarray(1));                       // 'A' 音频
      else if (u8[0] === 0x56) {                                             // 'V' 视频帧
        if (u8.byteLength < 7) throw new Error("Invalid video packet");
        const dv = new DataView(ev.data);
        FACE.push(dv.getUint16(1, true), dv.getUint32(3, true), u8.subarray(7));
      }
      return;
    }
    const m = JSON.parse(ev.data);
    if (m.type === "state") {
      if (m.state !== "listening" || player.turn === null) setState(m.state);
    }
    else if (m.type === "turn") { cutPlayback(); FACE.begin(m.id, m.fps, m.face); }
    else if (m.type === "caption") CAPTION.push(m.turn, m.pts_ms, m.text);
    else if (m.type === "face") {
      FACE.enabled = m.ready !== false;
      els.stage.classList.toggle("face", FACE.enabled);
      if (!FACE.enabled) {
        FACE.show(false);
        document.querySelector("#stage-empty .big").textContent = "Video unavailable";
        document.querySelector("#stage-empty .small").textContent = "Voice conversation is available. Restart the service to retry video.";
        addTurn("system", "Video could not start. Voice is available.", "sys");
      }
    }
    else if (m.type === "user") addTurn("you", m.text, "you");
    else if (m.type === "restored") addTurn(m.role === "user" ? "you" : "CK", m.text,
      m.role === "user" ? "you" : "ck");
    else if (m.type === "assistant") {
      if (m.id !== undefined && m.id !== CAPTION.turn) return;
      if (CAPTION.bubble && (m.id === undefined || m.id === CAPTION.turn)) { CAPTION.finalText = m.text; curTurn = CAPTION.bubble; }
      else curTurn = addTurn("CK", m.text, "ck");
      setMeta(curTurn, `ASR ${m.asr_ms}ms · TTFT ${m.ttft_ms}ms · first audio ${m.tta_ms}ms · ${m.words}w`);
      els.stats.textContent = `end-to-end ${m.e2e_ms}ms`;
    }
    else if (m.type === "generation_done") {
      if (m.id === CAPTION.turn && m.text) CAPTION.finalText = m.text;
      player.finishGeneration(m.id);
    }
    else if (m.type === "interrupt") {
      if (m.id !== undefined && player.turn !== null && m.id !== player.turn) return;
      cutPlayback();
      setState("listening");
    }
    else if (m.type === "sys") addTurn("system", m.text, "sys");
    } catch (e) {
      console.error("Conversation playback failed", e);
      addTurn("system", "Playback failed: " + e.message, "sys");
      cutPlayback();
    }
  };
}

els.btn.onclick = async () => {
  if (starting) return;
  if (running) { disconnect(); return; }
  starting = true; els.btn.disabled = true;
  try {
    // 在点击手势内解锁输出 AudioContext；等麦克风授权之后才创建可能被自动播放策略挂起。
    await player.ensure().resume();
    await startMic();
    FACE.init(els.stage);
    connect();
    running = true; els.btn.textContent = "Stop";
  } catch (e) { disconnect(); addTurn("system", "Microphone could not start: " + e.message, "sys"); }
  finally { starting = false; els.btn.disabled = false; }
};
