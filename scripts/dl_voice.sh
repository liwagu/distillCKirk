#!/usr/bin/env bash
# Phase 1 语音管线权重。
#
# 关键坑（2026-09-21 实锤）：hf_xet 是 Rust 实现，**不读 HTTP_PROXY**。
# 本机直连 HF 会被 SSL UNEXPECTED_EOF 拒绝，必须走代理，所以启用 xet 时下载永久挂起，
# 且 HF_HUB_DISABLE_XET=1 在 huggingface_hub 1.30 下不生效。
# 唯一可靠解法：卸载 hf_xet，让 hub 回退到普通 HTTP（会打警告，忽略即可）。
# 卸载后实测 ~10 MB/s，比 xet 挂起前的 0.8 MB/s 快一个数量级。
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
.venv/bin/python -c "import hf_xet" 2>/dev/null && {
  echo "ERROR: hf_xet 已安装，会导致挂起。先跑: uv pip uninstall --python .venv/bin/python hf_xet" >&2
  exit 1; }
for M in "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit" \
         "Qwen/Qwen3-ASR-1.7B" \
         "runanywhere/silero-vad-v5" \
         "pipecat-ai/smart-turn-v3"; do
  for i in $(seq 1 30); do
    echo "[$(date +%H:%M:%S)] $M 尝试 $i"
    .venv/bin/hf download "$M" --max-workers 4 > /dev/null 2>&1 && { echo "  ✓ $M"; break; }
    sleep 5
  done
done
echo "=== 最终 ==="
for m in mlx-community--Qwen3-TTS-12Hz-1.7B-Base-8bit Qwen--Qwen3-ASR-1.7B \
         runanywhere--silero-vad-v5 pipecat-ai--smart-turn-v3; do
  printf "  %-46s %s MB\n" "$m" "$(du -sm "$HOME/.cache/huggingface/hub/models--$m" 2>/dev/null|cut -f1)"
done
echo "VOICE MODELS DONE"
