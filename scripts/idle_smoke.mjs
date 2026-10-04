#!/usr/bin/env node
/** Observe idle video over the real browser/WS path with a silent simulated mic.
 * Never sends a spoken fixture or directly calls an LLM. Does not start services.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const options = { url: 'http://127.0.0.1:8000', duration: 4, timeout: 30000, headed: false, expectStatic: false };
for (let i = 2; i < process.argv.length; i++) {
  const arg = process.argv[i];
  if (arg === '--help') {
    console.log(`Usage: node scripts/idle_smoke.mjs [options]
  --url URL             Existing service (default http://127.0.0.1:8000)
  --duration SECONDS    Observe idle JPEGs for at least this duration (default 4)
  --out DIR             Report, received idle JPEGs and screenshots
  --timeout-ms N        Start/first-frame timeout (default 30000)
  --expect-static       Require constant idle JPEG and canvas pixels, including Stop
  --headed              Show Chromium
VOXCK_PLAYWRIGHT_PATH / NODE_PATH and CHROMIUM_PATH override installed runtimes.
Uses a silent simulated getUserMedia stream through the app AudioWorklet.
No physical microphone or human hearing claim; JPEG changes alone do not prove
mouth motion. Inspect the saved raw frames/canvas screenshots for that claim.`);
    process.exit(0);
  }
  if (arg === '--url') options.url = process.argv[++i];
  else if (arg === '--out') options.out = path.resolve(process.argv[++i]);
  else if (arg === '--duration') options.duration = Number(process.argv[++i]);
  else if (arg === '--timeout-ms') options.timeout = Number(process.argv[++i]);
  else if (arg === '--expect-static') options.expectStatic = true;
  else if (arg === '--headed') options.headed = true;
  else throw new Error(`Unknown argument: ${arg}`);
}
if (!Number.isFinite(options.duration) || options.duration < 4 || !Number.isFinite(options.timeout) || options.timeout <= 0) {
  throw new Error('duration must be at least 4 seconds and timeout must be positive');
}
options.out ||= path.resolve('logs', `idle-${new Date().toISOString().replace(/[:.]/g, '-')}`);
await fs.mkdir(path.join(options.out, 'idle-frames'), { recursive: true });

function loadPlaywright() {
  for (const candidate of [process.env.VOXCK_PLAYWRIGHT_PATH, 'playwright',
    path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')].filter(Boolean)) {
    try { return require(candidate); } catch {}
  }
  throw new Error('Set VOXCK_PLAYWRIGHT_PATH or NODE_PATH to an installed Playwright package');
}
const report = { started_at: new Date().toISOString(), url: options.url, mode: 'browser_idle_only',
  microphone: 'Silent simulated getUserMedia MediaStream; original app AudioWorklet/PCM uplink runs',
  output: 'WebAudio source starts and player queues observed; physical speakers/human hearing unverified',
  requested_duration_seconds: options.duration, expect_static_jpeg: options.expectStatic,
  status: 'running', browser_errors: [], console: [], idle_frames: [], screenshots: [], checks: [] };
const writes = [];
let browser, page;
function check(name, ok, detail) { report.checks.push({ name, ok: !!ok, detail }); }

try {
  const { chromium } = loadPlaywright();
  const executablePath = process.env.CHROMIUM_PATH || process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  browser = await chromium.launch({ headless: !options.headed, ...(executablePath ? { executablePath } : {}) });
  const context = await browser.newContext({ viewport: { width: 1280, height: 960 }, permissions: ['microphone'] });
  page = await context.newPage();
  page.on('console', m => { const entry = { type: m.type(), text: m.text() }; report.console.push(entry); if (m.type() === 'error') report.browser_errors.push(entry); });
  page.on('pageerror', e => report.browser_errors.push({ type: 'pageerror', text: e.message }));
  await page.exposeBinding('__idleSaveFrame', (_source, turn, pts, at, b64) => {
    const bytes = Buffer.from(b64, 'base64');
    const index = report.idle_frames.length;
    const file = path.join(options.out, 'idle-frames', `frame-${String(index).padStart(4, '0')}.jpg`);
    report.idle_frames.push({ index, turn, pts, browser_at_ms: at, bytes: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex'), file });
    const writing = fs.writeFile(file, bytes); writes.push(writing); return writing;
  });
  await page.addInitScript(() => {
    const s = window.__idleSmoke = { events: [], messages: [], idleFrames: 0, nonIdleFrames: 0,
      audioPackets: 0, audioBytes: 0, assistantMessages: 0, userMessages: 0, turnMessages: 0,
      uplinkPackets: 0, uplinkBytes: 0, uplinkPeak: 0, outputSourcesStarted: 0,
      getUserMediaCalls: [], drawEvents: [], transfers: [], canvasSamples: [] };
    const event = (type, data = {}) => s.events.push({ type, at: performance.now(), ...data });
    const b64 = bytes => { let str = ''; for (let i = 0; i < bytes.length; i += 16384) str += String.fromCharCode(...bytes.subarray(i, i + 16384)); return btoa(str); };
    navigator.mediaDevices.getUserMedia = async constraints => {
      s.getUserMediaCalls.push(constraints);
      const ctx = new AudioContext({ sampleRate: 16000 });
      const dest = ctx.createMediaStreamDestination();
      const silence = ctx.createConstantSource(); silence.offset.value = 0;
      silence.connect(dest); silence.start(); await ctx.resume();
      s.simulatedMic = { ctx, dest, silence };
      event('simulated_silent_microphone', { sampleRate: ctx.sampleRate, state: ctx.state });
      return dest.stream;
    };
    const createSource = AudioContext.prototype.createBufferSource;
    AudioContext.prototype.createBufferSource = function (...args) {
      const source = createSource.apply(this, args), start = source.start.bind(source);
      source.start = function (...startArgs) { s.outputSourcesStarted++; event('output_source_start'); return start(...startArgs); };
      return source;
    };
    const NativeWS = window.WebSocket;
    window.WebSocket = new Proxy(NativeWS, { construct(Target, args) {
      const socket = new Target(...args); s.socket = socket;
      socket.addEventListener('open', () => event('ws_open'));
      socket.addEventListener('close', e => event('ws_close', { code: e.code }));
      socket.addEventListener('message', e => {
        if (typeof e.data === 'string') {
          try {
            const m = JSON.parse(e.data); s.messages.push({ at: performance.now(), ...m });
            if (m.type === 'assistant') s.assistantMessages++;
            if (m.type === 'user') s.userMessages++;
            if (m.type === 'turn') s.turnMessages++;
          } catch { event('invalid_json'); }
          return;
        }
        if (!(e.data instanceof ArrayBuffer)) { event('unexpected_binary_type'); return; }
        const data = new Uint8Array(e.data);
        if (data[0] === 65) { s.audioPackets++; s.audioBytes += data.length - 1; }
        else if (data[0] === 86 && data.length >= 7) {
          const view = new DataView(e.data), turn = view.getUint16(1, true), pts = view.getUint32(3, true);
          if (pts !== 0xFFFFFFFF) { s.nonIdleFrames++; return; }
          s.idleFrames++; s.firstIdleAt ??= performance.now(); s.lastIdleAt = performance.now();
          s.transfers.push(window.__idleSaveFrame(turn, pts, performance.now(), b64(data.subarray(7))));
        } else event('unknown_packet', { bytes: data.length, tag: data[0] });
      });
      const send = socket.send.bind(socket);
      socket.send = function (data) {
        if (typeof data === 'string') event('client_control', { message: JSON.parse(data) });
        else {
          const buf = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
          s.uplinkPackets++; s.uplinkBytes += buf.byteLength;
          for (const value of new Int16Array(buf)) s.uplinkPeak = Math.max(s.uplinkPeak, Math.abs(value) / 32768);
        }
        return send(data);
      };
      return socket;
    } });
    s.installDrawProbe = () => {
      // Observe actual on-screen painting, including restoreIdle/paint paths.
      // Offscreen idle-cache canvases deliberately do not count as display draws.
      const drawImage = CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage = function (...args) {
        const result = drawImage.apply(this, args);
        if (this.canvas.id === 'face') s.drawEvents.push({ at: performance.now(), inTurn: FACE.inTurn, turn: FACE.turn });
        return result;
      };
    };
    s.sampleCanvas = async label => {
      if (!FACE.g || !FACE.canvas.width || !FACE.canvas.height) throw new Error('Canvas is not initialized');
      const pixels = FACE.g.getImageData(0, 0, FACE.canvas.width, FACE.canvas.height).data;
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', pixels));
      const row = { label, at: performance.now(), sha256: Array.from(digest, x => x.toString(16).padStart(2, '0')).join(''),
        width: FACE.canvas.width, height: FACE.canvas.height };
      s.canvasSamples.push(row); return row;
    };
    s.snapshot = () => ({ events: s.events, messages: s.messages, idleFrames: s.idleFrames,
      nonIdleFrames: s.nonIdleFrames, audioPackets: s.audioPackets, audioBytes: s.audioBytes,
      assistantMessages: s.assistantMessages, userMessages: s.userMessages, turnMessages: s.turnMessages,
      firstIdleAt: s.firstIdleAt, lastIdleAt: s.lastIdleAt, uplinkPackets: s.uplinkPackets,
      uplinkBytes: s.uplinkBytes, uplinkPeak: s.uplinkPeak, outputSourcesStarted: s.outputSourcesStarted,
      getUserMediaCalls: s.getUserMediaCalls, drawEvents: s.drawEvents, canvasSamples: s.canvasSamples,
      running: typeof running !== 'undefined' ? running : null,
      state: document.getElementById('state')?.textContent,
      canvas: { width: FACE.canvas?.width, height: FACE.canvas?.height, opacity: FACE.canvas?.style.opacity,
        enabled: FACE.enabled, visible: FACE.visible, inTurn: FACE.inTurn, frames: FACE.frames.length, shown: FACE.shown },
      player: { turn: player.turn, sources: player.sources.size, held: player.held.length,
        bufferedMs: player.bufferedMs(), started: player.started, peak: player.peak,
        contextState: player.ctx?.state }, pageText: document.body.innerText });
  });
  await page.goto(options.url, { waitUntil: 'networkidle', timeout: options.timeout });
  await page.evaluate(() => window.__idleSmoke.installDrawProbe());
  await page.locator('#btn').click();
  await page.waitForFunction(() => window.__idleSmoke.idleFrames > 0, null, { timeout: options.timeout });
  const first = await page.evaluate(() => window.__idleSmoke.snapshot()); report.first = first;
  console.log('Start clicked; collecting idle JPEGs with zero PCM microphone input.');
  const screen = async label => { const file = path.join(options.out, `${label}.png`); await page.screenshot({ path: file }); report.screenshots.push(file); await page.evaluate(label => window.__idleSmoke.sampleCanvas(label), label); };
  await screen('idle-start');
  const samples = 8;
  for (let i = 1; i <= samples; i++) {
    await page.waitForTimeout(options.duration * 1000 / samples);
    await screen(`idle-${i}`);
  }
  await page.waitForFunction(duration => window.__idleSmoke.lastIdleAt - window.__idleSmoke.firstIdleAt >= duration * 1000, options.duration, { timeout: options.timeout });
  await page.evaluate(async () => { await Promise.all(window.__idleSmoke.transfers); });
  await screen('idle-final');
  const final = await page.evaluate(() => window.__idleSmoke.snapshot()); report.observed = final;
  report.observed_duration_ms = final.lastIdleAt - final.firstIdleAt;
  report.idle_jpeg_sha256_set = [...new Set(report.idle_frames.map(f => f.sha256))];
  report.unique_idle_jpegs = report.idle_jpeg_sha256_set.length;
  report.idle_canvas_pixel_sha256_set = [...new Set(final.canvasSamples.map(c => c.sha256))];
  report.unique_idle_canvas_pixel_hashes = report.idle_canvas_pixel_sha256_set.length;
  report.interpretation = report.unique_idle_jpegs > 1
    ? 'Idle packets vary while no assistant/audio turn is present. Hash variation proves different JPEGs, not by itself mouth motion; inspect saved frames.'
    : 'All received idle packets contain identical JPEG bytes; this observed idle window has no source-frame animation.';
  check('real_start_and_simulated_getUserMedia', final.running && final.getUserMediaCalls.length === 1, final.getUserMediaCalls);
  check('original_audio_worklet_sent_silent_pcm', final.uplinkPackets > 0 && final.uplinkPeak === 0, { packets: final.uplinkPackets, bytes: final.uplinkBytes, peak: final.uplinkPeak });
  check('idle_observed_at_least_requested_duration', report.observed_duration_ms >= options.duration * 1000, report.observed_duration_ms);
  check('idle_jpegs_saved', report.idle_frames.length === final.idleFrames && final.idleFrames > 1, { saved: report.idle_frames.length, received: final.idleFrames });
  check('no_non_idle_video', final.nonIdleFrames === 0, final.nonIdleFrames);
  check('no_audio_packets', final.audioPackets === 0 && final.audioBytes === 0, { packets: final.audioPackets, bytes: final.audioBytes });
  check('no_conversation_or_assistant_turn', final.assistantMessages === 0 && final.userMessages === 0 && final.turnMessages === 0,
    { assistant: final.assistantMessages, user: final.userMessages, turn: final.turnMessages });
  check('listening_canvas_has_idle_draws', final.state === 'listening' && final.canvas.visible && !final.canvas.inTurn && final.drawEvents.length > 0,
    { state: final.state, canvas: final.canvas, draws: final.drawEvents.length });
  check('no_audio_sources_or_queues', final.outputSourcesStarted === 0 && final.player.sources === 0 && final.player.held === 0 && final.player.bufferedMs === 0, final.player);
  check('no_browser_errors', report.browser_errors.length === 0, report.browser_errors);
  if (options.expectStatic) {
    check('identical_idle_jpeg_bytes', report.unique_idle_jpegs === 1, report.unique_idle_jpegs);
    check('identical_idle_canvas_pixels', report.unique_idle_canvas_pixel_hashes === 1, report.unique_idle_canvas_pixel_hashes);
  }
  await page.locator('#btn').click();
  await screen('stopped');
  report.after_stop = await page.evaluate(() => window.__idleSmoke.snapshot());
  check('stop_disconnected_with_empty_audio_queue', !report.after_stop.running && report.after_stop.state === 'disconnected'
    && report.after_stop.player.sources === 0 && report.after_stop.player.held === 0 && report.after_stop.player.bufferedMs === 0,
    { running: report.after_stop.running, state: report.after_stop.state, player: report.after_stop.player });
  if (options.expectStatic) check('stop_preserves_same_idle_canvas_pixels', report.after_stop.canvasSamples.at(-1).sha256 === report.idle_canvas_pixel_sha256_set[0], report.after_stop.canvasSamples.at(-1));
  report.status = report.checks.every(c => c.ok) ? 'passed' : 'failed';
} catch (error) {
  report.status = 'failed'; report.error = { name: error.name, message: error.message };
  if (page) {
    try { report.failure_snapshot = await page.evaluate(() => window.__idleSmoke?.snapshot()); } catch {}
    try { const file = path.join(options.out, 'failure.png'); await page.screenshot({ path: file }); report.screenshots.push(file); } catch {}
  }
} finally {
  await Promise.allSettled(writes);
  if (browser) await browser.close();
  report.finished_at = new Date().toISOString();
  await fs.writeFile(path.join(options.out, 'report.json'), JSON.stringify(report, null, 2) + '\n');
}
console.log(`${report.status.toUpperCase()}: ${report.idle_frames.length} idle JPEGs, ${report.unique_idle_jpegs ?? '?'} unique; ${path.join(options.out, 'report.json')}`);
process.exitCode = report.status === 'passed' ? 0 : 1;
