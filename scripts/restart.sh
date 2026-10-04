#!/bin/zsh
# 干净地重启服务：等旧进程真正退出（否则旧进程会继续往同一个日志写，把日志搅乱），日志按时间戳分文件。
cd "$(dirname "$0")/.." || exit 1
LOGDIR=${LOGDIR:-/tmp/voxck-logs}; mkdir -p "$LOGDIR"
OLD=$(lsof -nP -iTCP:8000 -sTCP:LISTEN | tail -n +2 | awk '{print $2}' | sort -u)
for p in $OLD; do kill "$p" 2>/dev/null; done
for i in {1..20}; do pgrep -f "python run.py" >/dev/null || break; sleep 0.5; done
pgrep -f "python run.py" >/dev/null && { pkill -9 -f "python run.py"; pkill -9 -f "multiprocessing.spawn"; sleep 1; }
LOG="$LOGDIR/service-$(date +%H%M%S).log"; ln -sf "$LOG" "$LOGDIR/service.log"
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 HF_HUB_OFFLINE=1 nohup .venv/bin/python run.py > "$LOG" 2>&1 &
echo "started pid $! → $LOG"
for i in {1..120}; do grep -q "warm in\|ready on" "$LOG" 2>/dev/null && break; sleep 2; done
grep -E "取景窗|face renderer ready|warm in|预热失败|Traceback" "$LOG" | tail -4 | cut -c1-160
