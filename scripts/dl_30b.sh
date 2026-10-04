#!/usr/bin/env bash
# 带重试的下载。本机网络不稳（SSL UNEXPECTED_EOF 会随机中断），
# 必须循环重试；hf download 自身支持断点续传。
# 注意：绝不能开 HF_HUB_ENABLE_HF_TRANSFER / xet —— Rust 实现不读 HTTP_PROXY，会永久挂起。
cd "$(cd "$(dirname "$0")/.." && pwd)"
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_DISABLE_XET=1
M="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
D="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit"
for i in $(seq 1 200); do
  SZ=$(du -sm "$D" 2>/dev/null | cut -f1 || echo 0)
  echo "[$(date +%H:%M:%S)] 尝试 $i  当前 ${SZ:-0} MB / 17200 MB"
  if [ "${SZ:-0}" -ge 16800 ]; then echo "COMPLETE"; break; fi
  .venv/bin/hf download "$M" --max-workers 4 > /dev/null 2>&1
  sleep 5
done
echo "[$(date +%H:%M:%S)] 最终 $(du -sm "$D" 2>/dev/null | cut -f1) MB"
