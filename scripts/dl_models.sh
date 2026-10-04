#!/usr/bin/env bash
# 串行下载。注意：绝不能开 HF_HUB_ENABLE_HF_TRANSFER —— 它是 Rust 实现、
# 不读 HTTP_PROXY，而本机直连 HF 会被 SSL 拒绝，开了就永久挂起。
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_DISABLE_XET=1          # xet 同样是 Rust 路径，同样绕过代理
for M in "mlx-community/Qwen3-1.7B-4bit" "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"; do
  echo "=== [$(date +%H:%M:%S)] $M ==="
  .venv/bin/hf download "$M" --max-workers 8 2>&1 | tr '\r' '\n' | grep -v "^\s*$" | tail -2
  echo "    [$(date +%H:%M:%S)] done"
done
echo "ALL DONE"
