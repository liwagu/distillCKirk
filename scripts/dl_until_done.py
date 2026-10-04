#!/usr/bin/env python
"""死磕到底的下载器：重试直到 snapshot 真正完整为止。

本机网络会在传输中途随机断（peer closed connection / read timeout），
但 hf 支持断点续传，所以只需足够执着地重试。
前提：hf_xet 必须卸载 —— 它是 Rust 实现，不读 HTTP_PROXY，会永久挂起。

用法: .venv/bin/python scripts/dl_until_done.py [模型...]
"""
from __future__ import annotations

import os
import sys
import time

from huggingface_hub import snapshot_download

DEFAULT = [
    "Qwen/Qwen3-ASR-1.7B",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
]


def complete(repo: str) -> tuple[bool, float]:
    """snapshot 是否完整 + 已落盘大小(MB)。断链的符号链接 = 未下完。"""
    try:
        p = snapshot_download(repo, local_files_only=True)
    except Exception:
        return False, _size(repo)
    bad = [f for f in os.listdir(p)
           if os.path.islink(os.path.join(p, f)) and not os.path.exists(os.path.join(p, f))]
    tot = sum(os.path.getsize(os.path.realpath(os.path.join(p, f)))
              for f in os.listdir(p) if os.path.exists(os.path.join(p, f)))
    return (not bad), tot / 1e6


def _size(repo: str) -> float:
    d = os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--"))
    tot = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e6


def main() -> int:
    try:
        import hf_xet  # noqa: F401
        print("ERROR: hf_xet 已安装会导致挂起。先卸载它。", file=sys.stderr)
        return 1
    except ImportError:
        pass

    repos = sys.argv[1:] or DEFAULT
    for repo in repos:
        for i in range(1, 201):
            ok, mb = complete(repo)
            if ok:
                print(f"✓ {repo} 完整 ({mb:.0f} MB)", flush=True)
                break
            print(f"[{time.strftime('%H:%M:%S')}] {repo} 第 {i} 次 (当前 {mb:.0f} MB)", flush=True)
            try:
                snapshot_download(repo, max_workers=4)
            except Exception as e:
                print(f"    {type(e).__name__}: {str(e)[:110]}", flush=True)
                time.sleep(3)
        else:
            print(f"✗ {repo} 200 次后仍未完整", flush=True)
            return 1
    print("ALL COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
