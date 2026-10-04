#!/usr/bin/env bash
# 串行下载，一次一个，不并发（并发会抢同一个文件锁互相饿死）。
cd "$(cd "$(dirname "$0")/.." && pwd)"
for M in "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit" "Qwen/Qwen3-ASR-1.7B" \
         "runanywhere/silero-vad-v5" "pipecat-ai/smart-turn-v3"; do
  for i in $(seq 1 25); do
    echo "[$(date +%H:%M:%S)] $M ($i)"
    .venv/bin/hf download "$M" --max-workers 4 2>&1 | tr '\r' '\n' | grep -vi "xet storage\|waiting to acquire" | tail -2
    sz=$(du -sm "$HOME/.cache/huggingface/hub/models--${M//\//--}" 2>/dev/null | cut -f1)
    echo "   → ${sz:-0} MB"
    .venv/bin/python - "$M" <<'PY' && break
import sys, os
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(sys.argv[1], local_files_only=True)
    missing = [f for f in os.listdir(p) if os.path.islink(os.path.join(p,f))
               and not os.path.exists(os.path.join(p,f))]
    sys.exit(1 if missing else 0)
except Exception:
    sys.exit(1)
PY
    sleep 3
  done
done
echo "ALL VOICE MODELS COMPLETE"
