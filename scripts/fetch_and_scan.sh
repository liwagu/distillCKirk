#!/bin/zsh
# 下载一个公开视频（可限定时间段）并扫描适合做底子的片段。
# 用法: scripts/fetch_and_scan.sh <url> <name> [起始秒 结束秒]   例: ... "https://youtu.be/xxx" show1 0 600
set -e
cd "$(dirname "$0")/.."
URL=$1; NAME=$2; T0=${3:-0}; T1=${4:-}
OUT=${OUTDIR:-logs/found}
mkdir -p "$OUT"
SEC=()
if [ -n "$T1" ]; then SEC=(--download-sections "*${T0}-${T1}"); fi
yt-dlp --no-update --proxy http://127.0.0.1:7897 -S "res:1080,ext:mp4" \
  --merge-output-format mp4 --no-playlist "${SEC[@]}" -o "$OUT/$NAME.%(ext)s" "$URL" 2>&1 | tail -3
F="$OUT/$NAME.mp4"; [ -f "$F" ] || { echo "下载失败"; exit 1; }
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$F" | cut -d. -f1)
echo "文件 $F 时长 ${DUR}s $(ffprobe -v error -select_streams v -show_entries stream=width,height -of csv=p=0 "$F")"
.venv/bin/python scripts/scan_slice.py "$F" --t0 0 --t1 "$DUR" --step 0.5 --max-yaw 0.14 --min-face 0.16 --min-run 4 --out "$OUT/scan_$NAME" 2>&1 | grep -vE "Warning|warn" | tail -14
