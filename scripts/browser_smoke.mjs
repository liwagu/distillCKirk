#!/usr/bin/env node
/** Real browser acceptance with an explicitly simulated microphone.
 * Start/getUserMedia adapter -> MediaStream -> app AudioWorklet -> real WS;
 * returned PCM -> app WebAudio -> running output graph; JPEG -> real canvas.
 * This verifies rendering/scheduling, never that a person heard the speakers.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';

const require = createRequire(import.meta.url);
const options = { url: 'http://127.0.0.1:8000', turns: 2, timeout: 180000, fixtures: [], interrupt: true, headed: false, maxSkew: 1500, maxFreeze: 2000 };
for (let i = 2; i < process.argv.length; i++) {
  const arg = process.argv[i];
  if (arg === '--help') {
    console.log(`Usage: node scripts/browser_smoke.mjs [options]
  --url URL                 Existing service (never started by this script)
  --fixture WAV             Normal utterance; repeat for separate utterances
  --interrupt-fixture WAV   Barge-in utterance
  --turns N                 Complete N normal turns before barge-in (default 2)
  --no-interrupt            Skip the additional barge-in/reply exchange
  --out DIR                 JSON, received media and screenshots directory
  --timeout-ms N            Per-phase timeout (default 180000)
  --max-skew-ms N           Max drawn frame lag vs audio clock (default 1500)
  --max-freeze-ms N         Max interval between content draws (default 2000)
  --headed                  Show Chromium
Defaults create WAV fixtures using macOS say + ffmpeg, with no model download.
VOXCK_PLAYWRIGHT_PATH or NODE_PATH overrides Playwright module discovery.
CHROMIUM_PATH / PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH overrides Chromium.
Microphone is simulated; playback evidence means a running WebAudio graph,
not physical microphone, speaker or human-hearing acceptance.`);
    process.exit(0);
  }
  if (arg === '--fixture') options.fixtures.push(path.resolve(process.argv[++i]));
  else if (arg === '--interrupt-fixture') options.interruptFixture = path.resolve(process.argv[++i]);
  else if (arg === '--url') options.url = process.argv[++i];
  else if (arg === '--out') options.out = path.resolve(process.argv[++i]);
  else if (arg === '--turns') options.turns = Number(process.argv[++i]);
  else if (arg === '--timeout-ms') options.timeout = Number(process.argv[++i]);
  else if (arg === '--max-skew-ms') options.maxSkew = Number(process.argv[++i]);
  else if (arg === '--max-freeze-ms') options.maxFreeze = Number(process.argv[++i]);
  else if (arg === '--no-interrupt') options.interrupt = false;
  else if (arg === '--headed') options.headed = true;
  else throw new Error(`Unknown argument: ${arg}`);
}
if (!Number.isInteger(options.turns) || options.turns < 1 || !Number.isFinite(options.timeout) || options.timeout <= 0 || !Number.isFinite(options.maxSkew) || options.maxSkew <= 0 || !Number.isFinite(options.maxFreeze) || options.maxFreeze <= 0) throw new Error('Invalid turns/timeout/frame tolerance');

function loadPlaywright() {
  const candidates = [process.env.VOXCK_PLAYWRIGHT_PATH, 'playwright',
    path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')].filter(Boolean);
  for (const candidate of candidates) { try { return require(candidate); } catch {} }
  throw new Error('Playwright missing. Set VOXCK_PLAYWRIGHT_PATH or NODE_PATH to an installed Playwright package.');
}
const stamp = new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Shanghai', dateStyle: 'short', timeStyle: 'medium' }).format(new Date()).replace(/[ :]/g, '-');
options.out ||= path.resolve('logs', `acceptance-${stamp}`);
await fs.mkdir(options.out, { recursive: true });

async function makeFixture(name, text) {
  const aiff = path.join(options.out, `${name}.aiff`), wav = path.join(options.out, `${name}.wav`);
  execFileSync(process.env.VOXCK_SAY || 'say', ['-v', 'Samantha', '-o', aiff, text]);
  execFileSync(process.env.VOXCK_FFMPEG || 'ffmpeg', ['-y', '-loglevel', 'error', '-i', aiff, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav]);
  await fs.unlink(aiff);
  return wav;
}
if (!options.fixtures.length) {
  options.fixtures.push(await makeFixture('input-1', 'I think the wage gap proves systemic discrimination against women.'));
  options.fixtures.push(await makeFixture('input-2', 'Why do you think career choices explain the wage gap instead of discrimination?'));
}
if (options.interrupt && !options.interruptFixture) options.interruptFixture = await makeFixture('input-interrupt', 'Stop. I disagree. What evidence would change your mind?');
const fixture64 = await Promise.all(options.fixtures.map(async f => (await fs.readFile(f)).toString('base64')));
const interrupt64 = options.interruptFixture ? (await fs.readFile(options.interruptFixture)).toString('base64') : null;
const report = { started_at: new Date().toISOString(), url: options.url, out: options.out,
  microphone: 'simulated MediaStream returned by getUserMedia adapter; original app AudioWorklet runs',
  output: 'PCM decoded by app, connected to AudioContext destination and measured by a zero-gain analyser tap; human hearing unverified',
  fixtures: options.fixtures, interrupt_fixture: options.interruptFixture, checks: [], browser_errors: [], console: [], screenshots: [], media: {}, status: 'running' };
const audio = new Map(), frameCounts = new Map(), mediaWrites = [];
let browser, page;
function check(name, ok, detail) { report.checks.push({ name, ok: !!ok, detail }); if (!ok) throw new Error(`${name}: ${JSON.stringify(detail)}`); }
function progress(message) { console.log(message); }

try {
  const { chromium } = loadPlaywright();
  browser = await chromium.launch({ headless: !options.headed,
    ...(process.env.CHROMIUM_PATH || process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.CHROMIUM_PATH || process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {}) });
  const context = await browser.newContext({ viewport: { width: 1280, height: 960 }, permissions: ['microphone'] });
  page = await context.newPage();
  page.on('console', m => { const entry = { type: m.type(), text: m.text() }; report.console.push(entry); if (m.type() === 'error') report.browser_errors.push(entry); });
  page.on('pageerror', e => report.browser_errors.push({ type: 'pageerror', text: e.message }));
  await page.exposeBinding('__smokeSaveMedia', (_source, kind, turn, pts, b64) => {
    const bytes = Buffer.from(b64, 'base64'), key = String(turn);
    if (kind === 'audio') { if (!audio.has(key)) audio.set(key, []); audio.get(key).push(bytes); }
    else {
      const n = frameCounts.get(key) || 0; frameCounts.set(key, n + 1);
      const dir = path.join(options.out, `turn-${key}`);
      mediaWrites.push(fs.mkdir(dir, { recursive: true }).then(() => fs.writeFile(path.join(dir, `frame-${String(n).padStart(4, '0')}-${pts}ms.jpg`), bytes)));
      return mediaWrites.at(-1);
    }
  });
  await page.addInitScript(() => {
    const s = window.__voxckSmoke = { events: [], turns: [], sources: [], outputPeak: 0, outputCurrentPeak: 0, uplinkBytes: 0,
      uplinkPeak: 0, outputContexts: [], getUserMediaCalls: [], activeTurn: null, micFixtures: [], draws: [], idleFrames: 0, mediaTransfers: [], localReplay: false };
    const event = (type, fields = {}) => { const e = { type, at: performance.now(), ...fields }; s.events.push(e); return e; };
    const b64 = bytes => { let str = ''; for (let i = 0; i < bytes.length; i += 16384) str += String.fromCharCode(...bytes.subarray(i, i + 16384)); return btoa(str); };
    const unb64 = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
    const turn = id => { let t = s.turns.find(t => t.id === id); if (!t) { t = { id, audioBytes: 0, frames: 0, pts: [], receivedAt: performance.now() }; s.turns.push(t); } return t; };

    // Tap the same AudioBufferSources connected to the output destination.
    const originalCreateSource = AudioContext.prototype.createBufferSource;
    AudioContext.prototype.createBufferSource = function (...args) {
      const src = originalCreateSource.apply(this, args), ctx = this;
      if (ctx.__smokeMic) return src;
      if (!s.outputContexts.includes(ctx)) s.outputContexts.push(ctx);
      const rec = { index: s.sources.length, contextIndex: s.outputContexts.indexOf(ctx), phase: s.localReplay ? 'local-replay' : 'live', turn: null, started: false, ended: false, stopped: false, connectedToDestination: false, peak: 0 };
      s.sources.push(rec);
      const connect = src.connect.bind(src);
      src.connect = function (dest, ...rest) {
        if (dest === ctx.destination) {
          rec.connectedToDestination = true;
          if (!ctx.__smokeAnalyser) {
            const analyser = ctx.createAnalyser(), mute = ctx.createGain();
            analyser.fftSize = 2048; mute.gain.value = 0; analyser.connect(mute); mute.connect(ctx.destination);
            ctx.__smokeAnalyser = analyser;
            const data = new Float32Array(analyser.fftSize);
            const sample = () => { analyser.getFloatTimeDomainData(data); let peak = 0; for (const v of data) peak = Math.max(peak, Math.abs(v)); s.outputPeak = Math.max(s.outputPeak, peak); s.outputCurrentPeak = peak;
              for (const x of s.sources) if (x.contextIndex === s.outputContexts.indexOf(ctx) && x.started && !x.ended && !x.stopped && ctx.currentTime >= x.scheduledContextTime && ctx.currentTime < x.scheduledContextTime + x.duration) {
                x.peak = Math.max(x.peak, peak);
                if (peak > 0.001) { x.firstOutputAt ??= performance.now(); x.firstOutputContextTime ??= ctx.currentTime; }
              }
              requestAnimationFrame(sample); }; requestAnimationFrame(sample);
          }
          connect(ctx.__smokeAnalyser);
        }
        return connect(dest, ...rest);
      };
      const start = src.start.bind(src), stop = src.stop.bind(src);
      src.start = function (when = 0, ...rest) { rec.turn = s.activeTurn; rec.started = true; rec.startAt = performance.now(); rec.contextTimeAtStart = ctx.currentTime; rec.scheduledContextTime = Math.max(when, ctx.currentTime); rec.duration = src.buffer?.duration || 0;
        if (src.buffer) { let peak = 0; for (const v of src.buffer.getChannelData(0)) peak = Math.max(peak, Math.abs(v)); rec.decodedPeak = peak; }
        event('source_start', { index: rec.index, turn: rec.turn }); return start(when, ...rest); };
      src.stop = function (...args) { rec.stopped = true; rec.stoppedAt = performance.now(); event('source_stop', { index: rec.index, turn: rec.turn }); return stop(...args); };
      src.addEventListener('ended', () => { rec.ended = true; rec.endedAt = performance.now(); rec.contextTimeAtEnd = ctx.currentTime; event('source_ended', { index: rec.index, turn: rec.turn, stopped: rec.stopped }); });
      return src;
    };

    // Explicit simulated microphone. The app's getUserMedia call, stream source,
    // AudioWorklet, Int16 conversion and binary WS upload remain on its real path.
    navigator.mediaDevices.getUserMedia = async constraints => {
      s.getUserMediaCalls.push(constraints);
      const ctx = new AudioContext({ sampleRate: 16000 }); ctx.__smokeMic = true;
      const dest = ctx.createMediaStreamDestination(); await ctx.resume();
      s.micContext = ctx; s.micDestination = dest;
      event('simulated_microphone_created', { sampleRate: ctx.sampleRate, state: ctx.state });
      return dest.stream;
    };
    s.playMic = async text => {
      const ctx = s.micContext; if (!ctx) throw new Error('Start has not created simulated microphone');
      await ctx.resume(); const buf = await ctx.decodeAudioData(unb64(text).buffer);
      const src = ctx.createBufferSource(); src.buffer = buf; src.connect(s.micDestination);
      const record = { startAt: performance.now(), duration: buf.duration, sampleRate: buf.sampleRate }; s.micFixtures.push(record);
      await new Promise(resolve => { src.addEventListener('ended', resolve, { once: true }); src.start(); });
      record.endedAt = performance.now(); event('microphone_fixture_ended', record); return record;
    };

    const NativeWS = window.WebSocket;
    window.WebSocket = new Proxy(NativeWS, { construct(Target, args) {
      const socket = new Target(...args); s.socket = socket;
      socket.addEventListener('open', () => event('ws_open'));
      socket.addEventListener('close', e => event('ws_close', { code: e.code }));
      socket.addEventListener('message', e => {
        if (typeof e.data === 'string') {
          const m = JSON.parse(e.data);
          const entry = event('server_message', { messageType: m.type, id: m.id, turn: m.turn, state: m.state });
          if (m.type === 'turn') { s.activeTurn = m.id; turn(m.id).face = m.face; }
          if (m.type === 'assistant') { const t = turn(s.activeTurn); t.assistant = true; t.assistantText = m.text; t.words = m.words; t.tta_ms = m.tta_ms; }
          if (m.type === 'user') entry.text = m.text;
          if (m.type === 'generation_done') { const t = turn(m.id); t.generationDoneAt = performance.now(); t.generationAudioMs = m.audio_ms; if (m.text) t.assistantText = m.text; }
          if (m.type === 'interrupt') entry.interruptedTurn = s.activeTurn;
          return;
        }
        if (!(e.data instanceof ArrayBuffer)) return;
        const data = new Uint8Array(e.data);
        if (data[0] === 65) { const t = turn(s.activeTurn); t.audioBytes += data.length - 1; t.firstAudioAt ??= performance.now(); s.mediaTransfers.push(window.__smokeSaveMedia('audio', s.activeTurn, 0, b64(data.subarray(1)))); }
        else if (data[0] === 86) { const dv = new DataView(e.data), id = dv.getUint16(1, true), pts = dv.getUint32(3, true);
          if (pts === 0xFFFFFFFF) { s.idleFrames++; return; }
          const t = turn(id); t.frames++; t.pts.push(pts); t.firstFrameAt ??= performance.now(); s.mediaTransfers.push(window.__smokeSaveMedia('frame', id, pts, b64(data.subarray(7))));
        }
      });
      const send = socket.send.bind(socket);
      socket.send = function (data) {
        if (typeof data === 'string') { const m = JSON.parse(data); event('client_message', { messageType: m.type, id: m.id, played_ms: m.played_ms }); }
        else {
          const buf = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
          s.uplinkBytes += buf.byteLength; let peak = 0; for (const v of new Int16Array(buf)) peak = Math.max(peak, Math.abs(v) / 32768); s.uplinkPeak = Math.max(s.uplinkPeak, peak);
        }
        return send(data);
      };
      return socket;
    } });

    s.installFaceProbe = () => {
      if (typeof FACE === 'undefined') throw new Error('FACE is absent');
      const tick = FACE.tick.bind(FACE), draw = FACE.draw.bind(FACE);
      FACE.tick = function (...args) {
        const timeMs = player.turnTimeMs(), candidates = this.frames.filter(f => f.pts <= timeMs + 100), candidate = candidates[candidates.length - 1];
        s.drawCandidate = candidate ? { turn: this.turn, pts: candidate.pts, audioClockMs: timeMs } : null;
        try { return tick(...args); } finally { s.drawCandidate = null; }
      };
      FACE.draw = function (bm, ...args) { if (s.drawCandidate) s.draws.push({ at: performance.now(), ...s.drawCandidate }); return draw(bm, ...args); };
    };
    s.snapshot = () => ({ events: s.events, turns: s.turns, sources: s.sources, outputPeak: s.outputPeak, outputCurrentPeak: s.outputCurrentPeak,
      uplinkBytes: s.uplinkBytes, uplinkPeak: s.uplinkPeak, getUserMediaCalls: s.getUserMediaCalls,
      micFixtures: s.micFixtures, draws: s.draws, idleFrames: s.idleFrames,
      contexts: s.outputContexts.map(c => ({ state: c.state, time: c.currentTime, sampleRate: c.sampleRate })),
      player: typeof player !== 'undefined' ? { sources: player.sources.size, held: player.held.length, holding: player.holding, bufferedMs: player.bufferedMs(), turnStart: player.turnStart, turnTimeMs: player.turnTimeMs() } : null,
      face: typeof FACE !== 'undefined' ? { turn: FACE.turn, frames: FACE.frames.length, inTurn: FACE.inTurn, shown: FACE.shown, dropped: FACE.dropped, width: FACE.canvas?.width, height: FACE.canvas?.height } : null,
      running: typeof running !== 'undefined' ? running : null, state: document.getElementById('state')?.textContent });
  });
  await page.goto(options.url, { waitUntil: 'networkidle', timeout: 30000 });
  await page.evaluate(() => window.__voxckSmoke.installFaceProbe());
  await page.locator('#btn').click(); // Genuine browser click invokes Start/resume.
  await page.waitForFunction(() => window.__voxckSmoke.events.some(e => e.type === 'ws_open'), null, { timeout: 30000 });
  progress('Browser Start clicked; simulated microphone is feeding the app AudioWorklet.');
  const snapshot = () => page.evaluate(() => window.__voxckSmoke.snapshot());
  async function screen(label) { const file = path.join(options.out, `${label}.png`); await page.screenshot({ path: file }); report.screenshots.push(file); }
  async function newTurn(before, wav) {
    const pumping = page.evaluate(wav => window.__voxckSmoke.playMic(wav), wav);
    await page.waitForFunction(n => window.__voxckSmoke.turns.length > n && window.__voxckSmoke.turns[n].audioBytes > 0, before, { timeout: options.timeout });
    await pumping;
    return (await snapshot()).turns[before].id;
  }
  async function played(id) {
    await page.waitForFunction(id => window.__voxckSmoke.events.some(e => e.type === 'client_message' && e.messageType === 'playback_done' && e.id === id), id, { timeout: options.timeout });
    const snap = await snapshot(), t = snap.turns.find(t => t.id === id), sources = snap.sources.filter(x => x.turn === id && x.started), draws = snap.draws.filter(x => x.turn === id);
    check(`turn-${id}: returned PCM and video`, t.audioBytes > 0 && t.frames >= 2 && new Set(t.pts).size >= 2, t);
    check(`turn-${id}: native audio ended`, sources.length > 0 && sources.every(x => x.ended && !x.stopped && x.connectedToDestination && x.contextTimeAtEnd > x.contextTimeAtStart), sources);
    check(`turn-${id}: non-silent output rendered`, sources.some(x => x.peak > 0.001), { peak: Math.max(...sources.map(x => x.peak)), graph: snap.contexts });
    check(`turn-${id}: actual canvas advanced`, draws.length >= 2 && new Set(draws.map(x => x.pts)).size >= 2, { draws: draws.length, first: draws[0], last: draws.at(-1) });
    const skew = draws.map(x => x.audioClockMs - x.pts).sort((a, b) => a - b);
    t.observed_draw_skew_ms = { min: skew[0], median: skew[Math.floor(skew.length / 2)], max: skew.at(-1) };
    const durationMs = t.audioBytes / 32, done = snap.events.find(e => e.type === 'client_message' && e.messageType === 'playback_done' && e.id === id), scheduledMs = sources.reduce((v, x) => v + x.duration * 1000, 0);
    check(`turn-${id}: entire received PCM played`, !!done && Math.abs(done.played_ms - durationMs) < 150 && Math.abs(scheduledMs - durationMs) < 1, { receivedMs: durationMs, scheduledMs, acknowledgedMs: done?.played_ms });
    check(`turn-${id}: canvas covers response duration`, draws.length >= Math.min(20, Math.max(2, Math.floor(durationMs / 200))) && draws.at(-1).pts >= durationMs - 1500, { drawnFrames: draws.length, lastPts: draws.at(-1)?.pts, durationMs });
    const freeze = draws.slice(1).map((x, i) => x.at - draws[i].at);
    check(`turn-${id}: frame timeline stays within tolerance`, skew[0] >= -150 && skew.at(-1) <= options.maxSkew && Math.max(0, ...freeze) <= options.maxFreeze, { skew: t.observed_draw_skew_ms, maxFreezeMs: Math.max(0, ...freeze), maxSkewMs: options.maxSkew, allowedFreezeMs: options.maxFreeze });
    check(`turn-${id}: generation completion acknowledged`, !!t.generationDoneAt && !!done && done.played_ms > 0 && (t.generationAudioMs === undefined || Math.abs(t.generationAudioMs - durationMs) < 150), t);
    const neutral = await page.evaluate(() => {
      if (!FACE.idleReady || !FACE.idleCanvas) return { ready: false };
      const face = FACE.canvas, cached = FACE.idleCanvas;
      if (face.width !== cached.width || face.height !== cached.height) return { ready: true, sameSize: false };
      const actual = FACE.g.getImageData(0, 0, face.width, face.height).data;
      const expected = cached.getContext('2d').getImageData(0, 0, cached.width, cached.height).data;
      return { ready: true, sameSize: true, samePixels: actual.every((value, i) => value === expected[i]),
        inTurn: FACE.inTurn, playing: player.isPlaying(), idleShowing: FACE.idleShowing };
    });
    check(`turn-${id}: completed speech immediately restores idle pixels`, neutral.ready && neutral.sameSize && neutral.samePixels && !neutral.inTurn && !neutral.playing && neutral.idleShowing, neutral);
    const input = snap.micFixtures.filter(x => x.startAt < sources[0].startAt).at(-1);
    const wallStart = sources[0].startAt + (sources[0].scheduledContextTime - sources[0].contextTimeAtStart) * 1000;
    report.turn_summaries ||= [];
    report.turn_summaries.push({ ...t, drawn_frames: draws.length, started_sources: sources.length,
      ended_sources: sources.filter(x => x.ended).length, sampled_output_peak: Math.max(...sources.map(x => x.peak)),
      fixture_ended_event_to_scheduled_playback_ms: input ? wallStart - input.endedAt : null,
      fixture_ended_event_to_first_output_peak_ms: input && sources[0].firstOutputAt ? sources[0].firstOutputAt - input.endedAt : null,
      first_source_start: sources[0], first_draw: draws[0], last_draw: draws.at(-1) });
    await screen(`turn-${id}-played`); progress(`Turn ${id} rendered audio and video, then sent playback_done.`);
  }
  for (let i = 0; i < options.turns; i++) { const before = (await snapshot()).turns.length; const id = await newTurn(before, fixture64[i % fixture64.length]); await played(id); }
  if (options.interrupt) {
    const before = (await snapshot()).turns.length, id = await newTurn(before, fixture64[(options.turns - 1) % fixture64.length]);
    await page.waitForFunction(id => window.__voxckSmoke.turns.find(t => t.id === id)?.generationDoneAt, id, { timeout: options.timeout });
    const pre = await snapshot(), eventIndex = pre.events.length;
    check('barge-in window after generation_done', pre.player.sources > 0 && pre.player.bufferedMs > 300 && pre.state === 'speaking', pre.player);
    const pumping = page.evaluate(wav => window.__voxckSmoke.playMic(wav), interrupt64);
    await page.waitForFunction(n => window.__voxckSmoke.events.slice(n).some(e => e.type === 'server_message' && e.messageType === 'interrupt'), eventIndex, { timeout: 20000 });
    const cut = await snapshot(), oldSources = cut.sources.filter(x => x.turn === id && x.started);
    check('barge-in stops old scheduled audio', oldSources.every(x => x.ended || x.stopped), oldSources);
    const liveBefore = pre.sources.filter(x => x.turn === id && x.started && !x.ended);
    check('barge-in actively stops queued sources for the correct turn', cut.events.slice(eventIndex).some(e => e.type === 'server_message' && e.messageType === 'interrupt' && e.id === id) && liveBefore.some(x => oldSources.some(y => y.index === x.index && y.stopped)), { activeBefore: liveBefore.map(x => x.index), stoppedAfter: oldSources.filter(x => x.stopped).map(x => x.index) });
    check('barge-in clears old canvas queue', cut.face.frames === 0 && !cut.face.inTurn, cut.face);
    await screen(`turn-${id}-interrupted`); await pumping;
    await page.waitForFunction(n => window.__voxckSmoke.turns.length > n && window.__voxckSmoke.turns[n].audioBytes > 0, before + 1, { timeout: options.timeout });
    const replyId = (await snapshot()).turns[before + 1].id;
    await played(replyId);
    const recovered = await snapshot(), cutAt = cut.events.findLast(e => e.type === 'server_message' && e.messageType === 'interrupt' && e.id === id).at;
    check('interrupted turn does not resume in recovered playback', !recovered.sources.some(x => x.turn === id && x.startAt > cutAt) && !recovered.draws.some(x => x.turn === id && x.at > cutAt + 50), { cutAt, interruptedTurn: id });
    report.barge_in = { interrupted_turn: id, reply_turn: replyId, before: pre.player, after: cut.player, events_since_start: cut.events.slice(eventIndex) };
  }

  // Use recorded response media to make a real local player queue without an
  // additional model request, then click the actual Stop button.
  await page.evaluate(() => Promise.all(window.__voxckSmoke.mediaTransfers));
  await Promise.all(mediaWrites);
  const prior = await snapshot(), lastId = prior.turns.at(-1).id, pcm = Buffer.concat(audio.get(String(lastId))), dir = path.join(options.out, `turn-${lastId}`);
  const frameFile = (await fs.readdir(dir)).find(x => x.endsWith('.jpg'));
  const jpeg = await fs.readFile(path.join(dir, frameFile));
  await page.evaluate(({ pcm, jpeg, id }) => {
    const bytes = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
    window.__voxckSmoke.localReplay = true;
    player.beginTurn(id, 0); // Explicit local replay, after real playback_done.
    player.feed(bytes(pcm));
    FACE.inTurn = true; FACE.push(id, Math.max(0, player.turnTimeMs()) + 10000, bytes(jpeg));
  }, { pcm: pcm.toString('base64'), jpeg: jpeg.toString('base64'), id: lastId });
  await page.waitForFunction(() => player.sources.size > 0 && FACE.frames.length > 0, null, { timeout: 5000 });
  await page.waitForFunction(() => window.__voxckSmoke.sources.some(x => x.phase === 'local-replay' && x.firstOutputAt && !x.ended) && window.__voxckSmoke.outputCurrentPeak > 0.001, null, { timeout: 5000 });
  const beforeStop = await snapshot();
  await page.locator('#btn').click();
  await page.waitForFunction(() => !running && player.sources.size === 0 && player.held.length === 0 && FACE.frames.length === 0, null, { timeout: 5000 });
  await page.waitForTimeout(300);
  const stopped = await snapshot();
  check('Stop clears active PCM and future frame queues', !stopped.running && stopped.player.sources === 0 && stopped.player.held === 0 && stopped.face.frames === 0 && !stopped.player.holding, { before: beforeStop.player, beforeFrames: beforeStop.face.frames, after: stopped.player, afterFrames: stopped.face.frames });
  check('Stop silences the browser output graph', stopped.outputCurrentPeak < 0.001, { sampledFloat32PeakAfterStop: stopped.outputCurrentPeak });
  const replaySources = stopped.sources.filter(x => x.phase === 'local-replay');
  check('Stop actively cancels the non-silent local replay', replaySources.length > 0 && replaySources.every(x => x.firstOutputAt && x.stopped && x.ended), replaySources);
  report.stop_regression = { mode: 'local replay of received PCM/JPEG through app player, followed by actual Stop click; no additional model request', before: beforeStop.player, after: stopped.player };
  check('simulated microphone reaches server via original AudioWorklet', stopped.getUserMediaCalls.length === 1 && stopped.uplinkBytes > 0 && stopped.uplinkPeak > 0.001, { calls: stopped.getUserMediaCalls.length, bytes: stopped.uplinkBytes, peak: stopped.uplinkPeak });
  check('browser has no console or JavaScript errors', report.browser_errors.length === 0, report.browser_errors);
  await screen('final-stopped');
  report.browser = stopped; report.status = 'passed';
} catch (error) {
  report.status = 'failed'; report.error = error.stack || String(error);
  if (page) { try { report.browser = await page.evaluate(() => window.__voxckSmoke.snapshot()); await page.screenshot({ path: path.join(options.out, 'failure.png') }); report.screenshots.push(path.join(options.out, 'failure.png')); } catch {} }
} finally {
  await Promise.allSettled(mediaWrites);
  for (const [id, chunks] of audio) {
    const pcm = Buffer.concat(chunks), header = Buffer.alloc(44);
    header.write('RIFF'); header.writeUInt32LE(pcm.length + 36, 4); header.write('WAVEfmt ', 8); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22); header.writeUInt32LE(16000, 24); header.writeUInt32LE(32000, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34); header.write('data', 36); header.writeUInt32LE(pcm.length, 40);
    const dir = path.join(options.out, `turn-${id}`); await fs.mkdir(dir, { recursive: true }); const file = path.join(dir, 'audio.wav'); await fs.writeFile(file, Buffer.concat([header, pcm]));
    report.media[id] = { audio: file, audio_seconds: pcm.length / 32000, received_frames: frameCounts.get(id) || 0, frames_dir: dir };
  }
  report.finished_at = new Date().toISOString(); await fs.writeFile(path.join(options.out, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  if (browser) await browser.close();
}
console.log(`${report.status.toUpperCase()}: ${path.join(options.out, 'report.json')}`);
if (report.status !== 'passed') { console.error(report.error); process.exitCode = 1; }
