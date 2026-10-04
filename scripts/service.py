#!/usr/bin/env python3
"""Manage this checkout's run.py without starting models from status/help.

    .venv/bin/python scripts/service.py start
    .venv/bin/python scripts/service.py status
    .venv/bin/python scripts/service.py stop
    .venv/bin/python scripts/service.py restart
    .venv/bin/python scripts/service.py start --local-llm

Start waits at most 20 seconds by default, then reports starting and the log path.
run.py continues to load .env.local and apply its existing LLM/proxy overrides.
Only a recorded PID with the same process start time and this exact run.py is
eligible for a signal. An untracked listener is never stopped or adopted.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
RUN = ROOT / "run.py"
LOGS = ROOT / "logs"
STATE = LOGS / "service-state.json"
LOCK = LOGS / "service.lock"


def process_info(pid: int) -> dict | None:
    """Read PID identity without relying on kill(pid, 0) or a broad name match."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return None
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "stat=", "-o", "command="],
            capture_output=True, text=True, timeout=2, check=False,
        )
        parts = result.stdout.strip().split(None, 6)
        if result.returncode or len(parts) != 7 or parts[5].startswith("Z"):
            return None
        return {"started": " ".join(parts[:5]), "command": parts[6]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Failure to inspect is not evidence of exit: preserve the state and
        # refuse to signal/start when PID ownership cannot be established.
        raise RuntimeError(f"Could not inspect PID {pid}; no signal sent.") from exc


def belongs_here(info: dict) -> bool:
    try:
        argv = shlex.split(info["command"])
        return bool(argv) and Path(argv[0]).resolve() == PYTHON.resolve() and str(RUN) in argv[1:]
    except (KeyError, OSError, ValueError):
        return False


def owned(state: dict, info: dict | None = None) -> bool:
    info = process_info(state.get("pid")) if info is None else info
    return bool(info and belongs_here(info) and info["started"] == state.get("process_started"))


def read_state() -> dict | None:
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else None
    except (OSError, ValueError):
        return None


def write_state(state: dict) -> None:
    temporary = STATE.with_name(f"{STATE.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(state, out, ensure_ascii=False, indent=2)
        out.write("\n")
    temporary.replace(STATE)


@contextlib.contextmanager
def mutation_lock():
    LOGS.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a") as lock:
        # Refuse concurrent control commands instead of blocking behind a startup.
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another service command is running; use status and retry.")
        yield


def endpoint(config: Path) -> tuple[str, int, bool]:
    cfg = json.loads(config.read_text(encoding="utf-8"))
    host = cfg.get("host", "127.0.0.1")
    # A wildcard bind is probed through loopback, never through an external host.
    probe_host = "::1" if host == "::" else "127.0.0.1" if host in ("0.0.0.0", "localhost") else host
    return probe_host, int(cfg.get("port", 8000)), bool(cfg.get("face", {}).get("enabled", False))


def port_occupied(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


def health(host: str, port: int) -> dict | None:
    host_part = f"[{host}]" if ":" in host else host
    url = f"http://{host_part}:{port}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=1) as response:
            value = json.loads(response.read(65536).decode("utf-8"))
            return value if isinstance(value, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def snapshot(config: Path) -> dict:
    state = read_state()
    if state:
        config = Path(state.get("config", str(config)))
    host, port, face_expected = endpoint(config)
    host_part = f"[{host}]" if ":" in host else host
    result = {"status": "stopped", "owned": False, "url": f"http://{host_part}:{port}"}
    if state:
        result["llm_mode"] = state.get("llm_mode", "configured")
        result.update({k: state[k] for k in ("pid", "started_at", "log", "config") if k in state})
        info = process_info(state.get("pid"))
        if info and not owned(state, info):
            result.update(status="foreign_pid", detail="Recorded PID identity changed; no signal is allowed.")
            return result
        if info:
            result.update(status="starting", owned=True)
            probe = health(host, port)
            if probe:
                result["health"] = probe
                if isinstance(probe.get("remote_llm"), bool):
                    result["llm_mode"] = "remote" if probe["remote_llm"] else "local"
                if probe.get("ready") is True:
                    result["status"] = "degraded" if face_expected and probe.get("face_ready") is not True else "ready"
                    if probe.get("llm_ready") is False:
                        result["status"] = "degraded"
                        result["detail"] = "The selected brain has not returned a reply; inspect the service log."
            return result
    if port_occupied(host, port):
        result.update(status="port_occupied", detail="Port is occupied by an untracked process; it will not be stopped.")
    return result


def child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("NO_PROXY", "no_proxy"):
        previous = env.get(key, "").split(",")
        env[key] = ",".join(dict.fromkeys(["127.0.0.1", "localhost", "::1"] + [p for p in previous if p]))
    return env


def start(config: Path, wait_seconds: float, local_llm: bool = False) -> dict:
    current = snapshot(config)
    if current["owned"]:
        current["already_running"] = True
        return current
    if current["status"] in ("foreign_pid", "port_occupied"):
        return current
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise RuntimeError(f"Project Python is missing or not executable: {PYTHON}")
    # Recheck the requested config, because an old state can point at another port.
    host, port, _ = endpoint(config)
    if port_occupied(host, port):
        return {"status": "port_occupied", "owned": False, "detail": f"Refusing to start: {host}:{port} is occupied."}
    stamp = dt.datetime.now().astimezone()
    log_path = LOGS / f"service-{stamp.strftime('%Y%m%d-%H%M%S-%f')}.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "ab", buffering=0) as output:
        command = [str(PYTHON), "-u", str(RUN), str(config)]
        if local_llm:
            command.append("--local-llm")
        child = subprocess.Popen(
            command,
            cwd=ROOT, env=child_environment(), stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
    info = process_info(child.pid)
    if not info or not belongs_here(info):
        # Startup can fail before ps observes it. Do not write an unverified PID.
        return {"status": "failed", "owned": False, "pid": child.pid, "log": str(log_path),
                "detail": "Could not verify the new process identity; inspect the log."}
    state = {"pid": child.pid, "process_started": info["started"], "started_at": stamp.isoformat(),
             "config": str(config), "log": str(log_path), "llm_mode": "local" if local_llm else "configured"}
    write_state(state)
    deadline = time.monotonic() + wait_seconds
    while True:
        current = snapshot(config)
        if current["status"] in ("ready", "degraded"):
            return current
        if not current["owned"]:
            current["status"] = "failed"
            current["detail"] = "Service exited before readiness; inspect the log."
            return current
        if time.monotonic() >= deadline:
            current["detail"] = "Still warming up; run status again or follow the log."
            return current
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))


def stop(config: Path, wait_seconds: float) -> dict:
    state = read_state()
    if not state:
        return snapshot(config)
    info = process_info(state.get("pid"))
    if not info:
        STATE.unlink(missing_ok=True)
        return snapshot(config)
    if not owned(state, info):
        return {"status": "foreign_pid", "owned": False, "pid": state.get("pid"),
                "detail": "PID/start time/run.py identity mismatch; refused to signal."}
    pid = state["pid"]
    # SIGINT lets run.py execute its existing model/subprocess cleanup.
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        info = process_info(pid)
        if not info:
            STATE.unlink(missing_ok=True)
            return snapshot(config)
        if not owned(state, info):
            return {"status": "foreign_pid", "owned": False, "pid": pid,
                    "detail": "PID identity changed while stopping; no further signal sent."}
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    # Do not SIGKILL the supervisor and strand model children. Report the bounded
    # graceful shutdown honestly; a later stop/status can observe completion.
    return {"status": "stopping", "owned": True, "pid": pid, "log": state.get("log"),
            "detail": "SIGINT sent; shutdown has not finished yet. Run status again."}


def bounded_wait(value: str) -> float:
    seconds = float(value)
    if not 0 <= seconds <= 30:
        raise argparse.ArgumentTypeError("wait must be between 0 and 30 seconds")
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("start", "status", "stop", "restart"))
    parser.add_argument("--config", default="configs/assistant.json", help="Project-relative or absolute config path")
    parser.add_argument("--wait", type=bounded_wait, default=20.0, help="Readiness/shutdown wait, 0–30 seconds (default: 20)")
    parser.add_argument("--local-llm", action="store_true", help="For start/restart: use JSON's local Qwen settings instead of .env.local LLM overrides")
    args = parser.parse_args()
    if args.local_llm and args.action not in ("start", "restart"):
        parser.error("--local-llm applies only to start or restart")
    config = (ROOT / args.config).resolve()
    try:
        if args.action == "status":
            result = snapshot(config)
        else:
            with mutation_lock():
                if args.action == "stop":
                    result = stop(config, args.wait)
                elif args.action == "restart":
                    deadline = time.monotonic() + args.wait
                    result = stop(config, args.wait)
                    if result["status"] == "stopped":
                        result = start(config, max(0, deadline - time.monotonic()), args.local_llm)
                else:
                    result = start(config, args.wait, args.local_llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in ("ready", "starting", "stopping", "stopped") else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "detail": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
