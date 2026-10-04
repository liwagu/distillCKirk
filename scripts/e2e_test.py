#!/usr/bin/env python
"""后端 fixture 测试：把 PCM 喂进 WS，收回文字、音频与 JPEG 帧。

验证后端路径：上行 PCM → VAD → SmartTurn → ASR → LLM → TTS → PCM/JPEG。
按实时速率喂音频（20ms 一帧），因为 VAD 的判停依赖真实时间节奏。
不采集真实麦克风、不播放音频、不绘制帧，不能验证人耳听见或浏览器同步。
浏览器与播放器验收使用 scripts/browser_smoke.mjs。

用法（先另起 run.py）:
    .venv/bin/python scripts/e2e_test.py /tmp/vadtest/a.pcm
"""
from __future__ import annotations

import asyncio
import argparse
import json
import struct
import time
from pathlib import Path

import aiohttp
import cv2
import numpy as np

SR = 16000
FRAME_MS = 20
FRAME_BYTES = SR * FRAME_MS // 1000 * 2
URL = "ws://127.0.0.1:8000/ws"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcm", nargs="?", default="/tmp/vadtest/a.pcm", help="16kHz mono signed Int16LE fixture")
    parser.add_argument("tail_ms", nargs="?", type=int, default=1500)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--report", type=Path, help="Write machine-readable backend evidence")
    args = parser.parse_args()
    pcm_path = Path(args.pcm)
    pcm = pcm_path.read_bytes()
    if not pcm or len(pcm) % 2:
        parser.error("fixture must contain nonempty Int16LE PCM")
    if args.tail_ms < 0 or args.timeout <= 0:
        parser.error("tail_ms must be nonnegative and timeout positive")
    tail_ms = args.tail_ms   # 尾部静音，触发判停
    pcm += b"\x00\x00" * (SR * tail_ms // 1000)

    audio_bytes = 0
    audio_peak = 0.0

    t_first_frame = None; frames = []; frame_bytes = 0; idle_frames = [0]; t_last_content = [time.perf_counter()]
    t_send_done = None
    t_first_audio = None
    got: dict = {}
    protocol_errors = []
    generation_done = False

    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(args.url, max_msg_size=8 << 20) as ws:
            print("模式: 后端 PCM fixture；不验证真实麦克风、扬声器或浏览器播放")
            print(f"已连接。喂 {len(pcm)/2/SR:.2f}s 音频（含尾部 {tail_ms}ms 静音），实时速率…")

            async def pump():
                nonlocal t_send_done
                t0 = time.perf_counter()
                for i in range(0, len(pcm), FRAME_BYTES):
                    await ws.send_bytes(pcm[i:i + FRAME_BYTES])
                    # 按挂钟对齐，误差不累积
                    nxt = t0 + (i / FRAME_BYTES + 1) * FRAME_MS / 1000
                    d = nxt - time.perf_counter()
                    if d > 0:
                        await asyncio.sleep(d)
                t_send_done = time.perf_counter()
                print(f"喂完（{t_send_done - t0:.2f}s）。等回复…")

            task = asyncio.create_task(pump())
            deadline = time.perf_counter() + args.timeout
            try:
                while time.perf_counter() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=5)
                    except asyncio.TimeoutError:
                        if generation_done and got.get("assistant") and audio_bytes:
                            break
                        continue
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        d = msg.data
                        if d[:1] == b"V":                      # 视频帧：'V' <H turn><I pts_ms> JPEG
                            if len(d) < 9:
                                protocol_errors.append("truncated video packet")
                                continue
                            turn, pts = struct.unpack_from("<HI", d, 1)
                            if pts == 0xFFFFFFFF:                  # 待机帧：只计数（它们永不停止，所以退出条件看"内容帧"）
                                idle_frames[0] += 1
                                if generation_done and got.get("assistant") and audio_bytes and time.perf_counter() - t_last_content[0] > 3:
                                    break
                                continue
                            t_last_content[0] = time.perf_counter()
                            if d[7:9] != b"\xff\xd8":
                                protocol_errors.append("video payload is not JPEG")
                                continue
                            decoded = cv2.imdecode(np.frombuffer(d[7:], dtype=np.uint8), cv2.IMREAD_COLOR)
                            if decoded is None or decoded.size == 0:
                                protocol_errors.append("JPEG could not be decoded")
                                continue
                            if got.get("turn") is not None and turn != got["turn"]:
                                protocol_errors.append(f"stale video turn {turn}, expected {got['turn']}")
                                continue
                            if frames and pts < frames[-1]:
                                protocol_errors.append("video pts moved backwards")
                            if t_first_frame is None:
                                t_first_frame = time.perf_counter()
                            frames.append(pts); frame_bytes += len(d) - 7
                            continue
                        if d[:1] == b"A":
                            d = d[1:]
                        else:
                            protocol_errors.append("unknown binary packet type")
                            continue
                        if not d or len(d) % 2:
                            protocol_errors.append("malformed Int16LE audio")
                            continue
                        audio_peak = max(audio_peak, float(np.max(np.abs(np.frombuffer(d, dtype="<i2").astype(np.int32)))) / 32768)
                        if t_first_audio is None:
                            t_first_audio = time.perf_counter()
                        t_last_content[0] = time.perf_counter()
                        audio_bytes += len(d)
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        m = json.loads(msg.data)
                        k = m.get("type")
                        if k == "state":
                            print(f"   [状态] {m['state']}")
                        elif k == "turn":
                            got["turn"] = m["id"]
                        elif k == "generation_done":
                            if got.get("turn") is not None and m.get("id") == got["turn"]:
                                generation_done = True
                            else:
                                protocol_errors.append("generation_done turn mismatch")
                        elif k == "user":
                            got["user"] = m["text"]
                            print(f"\n   [识别] {m['text']}")
                        elif k == "assistant":
                            got["assistant"] = m
                            print("\n   [回答]", " ".join(m["text"].split()))
                            print(f"   ASR {m['asr_ms']}ms · TTFT {m['ttft_ms']}ms · "
                                  f"首音 {m['tta_ms']}ms · 端到端 {m['e2e_ms']}ms · "
                                  f"{m['words']} 词 · 缓存 {m.get('cached')}/{m.get('prompt_tokens')}")
                        elif k == "sys":
                            print(f"   [系统] {m['text']}")
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    print("\n" + "=" * 70)
    if t_first_audio and t_send_done:
        print(f"用户停止说话 → 首个音频包: {(t_first_audio - t_send_done + tail_ms/1000)*1000:.0f}ms "
              f"（含 {tail_ms}ms 判停静音）")
    print(f"收到音频 {audio_bytes/2/SR:.2f}s ({audio_bytes} bytes)")
    if frames:
        span = (max(frames)-min(frames))/1000 + 0.05
        frame_delay = f"{(t_first_frame - t_first_audio)*1000:.0f}ms" if t_first_audio else "无音频"
        print(f"收到画面 {len(frames)} 帧（pts {min(frames)/1000:.1f}–{max(frames)/1000:.1f}s，{frame_bytes/1024/1024:.1f} MB）"
              f" · 首帧比首音晚 {frame_delay} · 帧率 {len(frames)/span:.1f} fps · 待机帧 {idle_frames[0]}")
    checks = {
        "turn_announced": got.get("turn") is not None,
        "user_transcribed": bool(got.get("user")),
        "assistant_returned": bool(got.get("assistant")),
        "non_silent_audio_received": audio_bytes >= SR * 2 // 10 and audio_peak > 0.001,
        "generation_completed": generation_done,
        "video_received_and_advanced": len(frames) >= 2 and len(set(frames)) >= 2,
        "video_covers_response": bool(frames) and max(frames) >= audio_bytes / (2 * SR) * 1000 - 1500
        and len(frames) >= min(20, max(2, int(audio_bytes / (2 * SR) * 5))),
        "protocol_valid": not protocol_errors,
    }
    ok = all(checks.values())
    print("结果:", "✓ 后端fixture通过（含视频）" if ok else "✗ 后端fixture不完整", checks)
    if protocol_errors:
        print("协议错误:", protocol_errors)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "mode": "backend_pcm_fixture", "human_microphone_verified": False,
            "browser_playback_verified": False, "human_hearing_verified": False,
            "ok": ok, "checks": checks, "protocol_errors": protocol_errors,
            "generation_done": generation_done, "turn": got.get("turn"),
            "audio_bytes": audio_bytes, "audio_peak": audio_peak, "video_frames": len(frames),
            "pts_ms": frames, "idle_frames": idle_frames[0],
            "first_frame_after_audio_ms": round((t_first_frame - t_first_audio) * 1000) if t_first_frame and t_first_audio else None,
        }, ensure_ascii=False, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
