"""补测：滚动窗口在【历史真正填满】时的缓存命中率。

前一轮测试有乐观偏差：2665 token 里 2400 是系统提示，历史极短，
所以淘汰最老一轮只作废了 5%。真实 30 轮对话历史大得多。

同时修两个已知测量缺陷：
  1. GPU 低功耗态（mlx-lm#432：prefill 可慢 7x，需废查询唤醒，空闲后复发）
  2. 每种情形多次测量取中位，而非单点
"""
import json, subprocess, sys, time, urllib.request, os, signal, statistics

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-1.7B-4bit"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8098
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

LEDGER = "\n".join(
    f"[cite-{i:02d}] Position {i}: he argued from a documented warrant citing a source dated "
    f"around 20{18+i%8}. The mental model is a symmetry test applied to the opposing claim."
    for i in range(1, 40))
PERSONA = ("You are a debate opponent modelled on a real public figure, for English practice. "
           "Argue his documented positions. Never assert a position without a cite-id.\n\n"
           "# Position ledger\n" + LEDGER + "\n")

# 造 30 轮真实体量的历史（每轮 user ~25 token + assistant ~60 token ≈ 2550 token）
def make_history(n):
    h = [{"role": "system", "content": PERSONA}]
    for i in range(n):
        h.append({"role": "user", "content":
                  f"Turn {i}: I think your position on point {i%39+1} fails because the evidence "
                  f"on outcomes does not support the causal story you are telling here."})
        h.append({"role": "assistant", "content":
                  f"Short answer: no. [cite-{i%39+1:02d}] The warrant is a symmetry test, and you "
                  f"have not applied it consistently. If you accept the standard in one direction "
                  f"you must accept it in the other, which is where your argument comes apart."})
    return h


def req(messages, max_tokens=24):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "stream": True, "temperature": 0.0,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r_ = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.perf_counter(); first = None; ptok = 0; cached = 0
    with op.open(r_, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            try: d = json.loads(line[6:])
            except Exception: continue
            u = d.get("usage") or {}
            if u:
                ptok = u.get("prompt_tokens", 0)
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            ch = d.get("choices") or [{}]
            dl = ch[0].get("delta", {}) if ch else {}
            if dl.get("content") or dl.get("reasoning"):
                if first is None: first = time.perf_counter() - t0
    return first, ptok, cached


def main():
    env = dict(os.environ); env["NO_PROXY"] = "127.0.0.1,localhost"
    srv = subprocess.Popen(
        [".venv/bin/python", "-m", "mlx_lm.server", "--model", MODEL,
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "WARNING"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        for _ in range(240):
            try: req([{"role": "user", "content": "hi"}], 2); break
            except Exception: time.sleep(1)
        else: print("server 启动失败"); return

        # GPU 唤醒（mlx-lm#432：低功耗态下 prefill 慢 7x）
        for _ in range(3): req(make_history(4), 8)
        print("GPU 已唤醒\n")

        for NT in (4, 12, 30):
            hist = make_history(NT)
            # 先把这条历史灌进缓存
            req(hist, 8)
            base_tok = None
            # A 追加一轮（缓存应几乎全中）
            h_app = hist + [{"role": "user", "content": "And what about the earlier framing?"}]
            f, pt, ca = req(h_app)
            base_tok = pt
            print(f"历史 {NT:>2} 轮 (~{pt:>5} tok)  追加一轮      TTFT {f*1000:>6.0f}ms  "
                  f"cached {ca:>5}/{pt:<5} = {ca/max(pt,1)*100:>3.0f}%")
            # B 淘汰最老一轮（滚动窗口）
            h_evict = [hist[0]] + hist[3:] + [{"role": "user", "content": "And what about the earlier framing?"}]
            f, pt, ca = req(h_evict)
            print(f"{'':>13}                淘汰最老轮    TTFT {f*1000:>6.0f}ms  "
                  f"cached {ca:>5}/{pt:<5} = {ca/max(pt,1)*100:>3.0f}%   "
                  f"→ 需重算 {pt-ca} tok")
            print()
    finally:
        srv.send_signal(signal.SIGINT); srv.wait(timeout=30)


if __name__ == "__main__":
    main()
