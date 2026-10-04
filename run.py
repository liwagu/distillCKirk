#!/usr/bin/env python
"""启动全套：mlx_lm.server（大脑）→ orchestrator（管线 + 网页）。

用法:
    .venv/bin/python run.py                     # 默认配置
    .venv/bin/python run.py configs/assistant.json
然后浏览器打开 http://127.0.0.1:8000

停止: Ctrl-C（会一并收掉 LLM 子进程）
"""
from __future__ import annotations

import json
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run")


def wait_for_llm(url: str, timeout: int = 900) -> bool:
    """轮询 /v1/models 直到 server 就绪。17GB 权重首次加载要一会儿。"""
    probe = url.rstrip("/") + "/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本机不走代理
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with opener.open(probe, timeout=3):
                return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local voice and live-face conversation service.")
    parser.add_argument("config", nargs="?", default="configs/assistant.json")
    parser.add_argument("--local-llm", action="store_true", help="Use the downloaded local LLM from the JSON config; preserve cloud settings in .env.local.")
    args = parser.parse_args()
    # .env.local 存密钥（已 gitignore）。不加载的话语义守卫会静默降级为只用正则。
    env_file = ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # 大脑可切云端：.env.local 里给 LLM_URL / LLM_MODEL / LLM_API_KEY 就覆盖配置，
    # 三行删掉即回到本地 Qwen。云端时不再起 mlx_lm.server，GPU 全留给 ASR/TTS/脸。
    if not args.local_llm and os.environ.get("LLM_URL"):
        cfg["llm_url"] = os.environ["LLM_URL"]
        cfg["llm_model"] = os.environ.get("LLM_MODEL", cfg["llm_model"])
    cfg["llm_api_key"] = "" if args.local_llm else os.environ.get("LLM_API_KEY", "")
    remote = not any(h in cfg["llm_url"] for h in ("127.0.0.1", "localhost"))
    if remote and not cfg["llm_api_key"]:
        log.warning("LLM_URL 指向云端但没给 LLM_API_KEY")
    if remote and cfg["llm_url"].startswith("http://"):
        log.warning("LLM_URL 是明文 http，API key 会裸奔——中转站支持 https，建议改")

    py = str(ROOT / ".venv/bin/python")
    # 本机地址绝不能走代理：有全局代理时 websockets/http 会把 127.0.0.1 也路由过去，
    # 握手直接 EOF（参考实现记录过这个坑）。云端 URL 则照常走系统代理。
    # 云端大脑默认**直连**：实测同一中转站走系统代理首字 7.2s，直连 2.2s（代理绕了海外节点）。
    # aiohttp 的 trust_env 会尊重 NO_PROXY，所以把大脑主机加进去即可。
    # 若某家 API 直连不通（如 api.deepseek.com 在本机直连 SSL EOF），设 LLM_VIA_PROXY=1。
    bypass = ["127.0.0.1", "localhost", "::1"]
    if remote and not os.environ.get("LLM_VIA_PROXY"):
        from urllib.parse import urlparse
        bypass.append(urlparse(cfg["llm_url"]).hostname or "")
    for k in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(k, "")
        os.environ[k] = ",".join(x for x in bypass + [cur] if x)
    env = dict(os.environ)

    llm = None
    if remote:
        log.info("大脑走云端：%s @ %s", cfg["llm_model"], cfg["llm_url"])
    else:
        port = cfg["llm_url"].rsplit(":", 1)[-1]
        log.info("启动 LLM server（%s）…", cfg["llm_model"])
        llm = subprocess.Popen(
            [py, "-m", "mlx_lm.server", "--model", cfg["llm_model"],
             "--host", "127.0.0.1", "--port", port, "--log-level", "WARNING"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        if llm is not None:
            if not wait_for_llm(cfg["llm_url"]):
                log.error("LLM server 未就绪，放弃")
                return 1
            log.info("LLM server 就绪")

        from aiohttp import web
        from voxck.orchestrator import build_app
        app = build_app(cfg)
        log.info("打开 http://%s:%s", cfg["host"], cfg["port"])
        web.run_app(app, host=cfg["host"], port=cfg["port"],
                    print=None, access_log=None)
        return 0
    finally:
        if llm is not None:
            log.info("收 LLM server…")
            llm.send_signal(signal.SIGINT)
            try:
                llm.wait(timeout=30)
            except subprocess.TimeoutExpired:
                llm.kill()


if __name__ == "__main__":
    raise SystemExit(main())
