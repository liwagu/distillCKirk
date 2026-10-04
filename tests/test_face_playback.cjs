// 回归真实 app.js 的音画准入/清队列协议；只用 Node 内置库和假的浏览器时钟。
// 不加载模型、不连接服务、不声称实际麦克风或人耳验收。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function harness() {
  const bitmaps = [];
  const canvas = () => ({ width: 100, height: 100, style: {}, imageLabel: null,
    getContext() { return { drawImage: image => { this.imageLabel = image.label ?? image.imageLabel; } }; } });
  const faceCanvas = canvas();
  function element() {
    const body = { textContent: "" }, classes = new Set();
    return { style: {}, textContent: "", children: [],
      classList: { toggle(name, on) { if (on) classes.add(name); else classes.delete(name); },
        remove(name) { classes.delete(name); } },
      querySelector(selector) { return selector === "#face" ? faceCanvas : selector === ".body" ? body : null; },
      appendChild(child) { this.children.push(child); } };
  }
  const ids = Object.fromEntries(["dot", "state", "stage", "stats", "btn", "log", "brain"].map(id => [id, element()]));
  class FakeAudioContext {
    constructor() { this.currentTime = 1; this.state = "running"; this.destination = {}; this.allSources = []; }
    resume() { return Promise.resolve(); }
    createBuffer(_channels, n, sr) { return { duration: n / sr, copyToChannel() {} }; }
    createBufferSource() {
      const source = { connect() {}, start(t) { this.startTime = t; }, stop() { this.stopped = true; } };
      this.allSources.push(source); return source;
    }
    advance(time) {
      this.currentTime = time;
      for (const source of this.allSources) {
        if (!source.stopped && !source.finished && source.startTime + source.buffer.duration <= time) {
          source.finished = true; source.onended?.();
        }
      }
    }
  }
  class FakeSocket {
    static OPEN = 1; static CLOSING = 2;
    constructor() { this.readyState = 1; this.sent = []; FakeSocket.last = this; }
    send(data) { this.sent.push(JSON.parse(data)); }
    close() { this.readyState = 3; }
  }
  const context = {
    document: { getElementById: id => ids[id], querySelector: () => element(),
      createElement: tag => tag === "canvas" ? canvas() : element() },
    fetch: async () => ({ ok: false }), setInterval() {}, setTimeout() { return 1; }, clearTimeout() {},
    requestAnimationFrame() {}, location: { protocol: "http:", host: "fixture" },
    AudioContext: FakeAudioContext, WebSocket: FakeSocket,
    Blob: class { constructor(parts) { this.parts = parts; } },
    createImageBitmap(blob) {
      const bytes = blob.parts[0];
      const bitmap = { label: bytes.label, width: 100, height: 100, closed: false,
        close() { this.closed = true; } };
      bitmaps.push(bitmap);
      if (bytes.deferred) return new Promise(resolve => { bytes.resolve = () => resolve(bitmap); });
      return Promise.resolve(bitmap);
    },
    ArrayBuffer, Uint8Array, Float32Array, DataView, Int16Array, console,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8")
    + ";globalThis.api={FACE,player,els,connect,disconnect};", context);
  const { FACE, player, els, connect, disconnect } = context.api;
  FACE.init(ids.stage); connect(); const socket = FakeSocket.last;
  const message = value => socket.onmessage({ data: JSON.stringify(value) });
  message({ type: "face", ready: true, fps: 20 });
  const image = (id, pts, label, deferred = false) => {
    const bytes = new Uint8Array([1]); bytes.label = label; bytes.deferred = deferred;
    FACE.push(id, pts, bytes); return bytes;
  };
  return { FACE, player, els, message, image, canvas: faceCanvas, socket, bitmaps, context, disconnect };
}
const settle = () => new Promise(resolve => setImmediate(resolve));
async function neutral(h) { h.image(0, 0xFFFFFFFF, "closed idle"); await settle(); }
async function talking(h, id = 1) {
  h.message({ type: "turn", id, fps: 20, face: true });
  h.player.feed(new Uint8Array(32000));
  h.image(id, 0, "open talking"); await settle();
  h.player.ctx.advance(h.player.turnStart + .02); h.FACE.tick();
  assert.equal(h.canvas.imageLabel, "open talking");
}

(async () => {
  {
    const h = harness(); await neutral(h);
    h.message({ type: "turn", id: 1, fps: 20, face: true });
    assert.equal(h.canvas.imageLabel, "closed idle");
    h.player.feed(new Uint8Array(32000));
    assert.equal(h.player.holding, true);
    h.image(1, 0, "open talking"); await settle();
    assert.equal(h.player.holding, false); // 首帧仍能释放 audiohold。
    h.FACE.tick(); assert.equal(h.canvas.imageLabel, "closed idle");
    assert.equal(h.FACE.talkShown, 0); // 首块真正开始前不画说话帧。
    h.player.ctx.advance(h.player.turnStart + .02); h.FACE.tick();
    assert.equal(h.canvas.imageLabel, "open talking");
    h.player.ctx.state = "suspended"; h.FACE.tick();
    assert.equal(h.canvas.imageLabel, "closed idle");
  }
  {
    const h = harness(); await neutral(h); await talking(h);
    // 播音时收到 ordinary idle 只能更新缓存，不能覆盖口型。
    h.image(0, 0xFFFFFFFF, "updated closed idle"); await settle();
    assert.equal(h.canvas.imageLabel, "open talking");
    h.message({ type: "generation_done", id: 1 });
    assert.equal(h.canvas.imageLabel, "open talking");
    assert.equal(h.socket.sent.filter(m => m.type === "playback_done").length, 0);
    h.player.ctx.advance(h.player.nextStart + .01);
    assert.equal(h.canvas.imageLabel, "updated closed idle");
    assert.equal(h.FACE.inTurn, false);
    const done = h.socket.sent.filter(m => m.type === "playback_done");
    assert.equal(done.length, 1); assert.equal(done[0].played_ms, 1000);
  }
  {
    const h = harness(); await neutral(h); await talking(h);
    h.player.ctx.advance(h.player.nextStart + .05); h.FACE.tick();
    assert.equal(h.player.sources.size, 0); assert.equal(h.player.generationDone, false);
    h.image(1, 1100, "future open mouth"); await settle(); h.FACE.tick();
    assert.equal(h.canvas.imageLabel, "closed idle"); // 冻结时钟+100 不再驱动未来口型。
    assert.equal(h.FACE.frames.length, 1);
    h.player.feed(new Uint8Array(32000)); h.FACE.tick();
    assert.equal(h.canvas.imageLabel, "closed idle"); // 下一块 30ms 余量仍静默。
    h.player.ctx.advance(h.player.nextStart - .98); h.FACE.tick();
    assert.equal(h.canvas.imageLabel, "future open mouth"); // 真正恢复播音才放行。
  }
  {
    const h = harness(); await neutral(h); await talking(h);
    const delayed = h.image(1, 100, "late old mouth", true);
    h.message({ type: "interrupt", id: 1 });
    assert.equal(h.canvas.imageLabel, "closed idle");
    assert.equal(h.player.sources.size, 0);
    assert.equal(h.socket.sent.filter(m => m.type === "playback_progress").length, 1);
    assert.equal(h.socket.sent.filter(m => m.type === "playback_done").length, 0);
    h.message({ type: "turn", id: 2, fps: 20, face: true });
    delayed.resolve(); await settle(); h.FACE.tick();
    assert.equal(h.canvas.imageLabel, "closed idle");
    assert.equal(h.FACE.frames.length, 0); assert.equal(h.player.holding, true);
    assert.equal(h.bitmaps.at(-1).closed, true); // 迟到旧世代 bitmap 被关闭。
  }
  {
    const h = harness(); await neutral(h); await talking(h);
    vm.runInContext("running=true;", h.context);
    await h.els.btn.onclick(); // 实际 Stop 按钮路径走 disconnect/cutPlayback。
    assert.equal(h.canvas.imageLabel, "closed idle"); assert.equal(h.player.turn, null);
    assert.equal(h.socket.readyState, 3); assert.equal(h.els.btn.textContent, "Start");
    assert.equal(h.socket.sent.filter(m => m.type === "playback_done").length, 0);
  }
  {
    const h = harness(); await talking(h); // 没收到中性 idle 就开始：结束时隐藏旧张嘴图。
    h.disconnect(); assert.equal(h.FACE.visible, false); assert.equal(h.canvas.style.opacity, "0");
  }
  console.log("PASS: 6 face/playback scenarios (first audio, suspension, drain, gaps, interruption, Stop, late frames)");
})().catch(error => { console.error(error); process.exitCode = 1; });
