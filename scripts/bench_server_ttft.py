"""决定性测试：mlx_lm.server 的自动前缀缓存，在真实对话/打断场景下的 TTFT。

server.py 内含 LRUPromptCache + fetch_nearest_cache()——自动找最长已缓存前缀，
只 prefill 剩余部分。这正是部署形态（VoxEMW 用 chat-completions 后端打本地服务）。

测四种情形：
  1 冷启动             首轮，缓存空 → 全量 prefill
  2 追加式续轮         历史只追加 → 应命中长前缀
  3 打断后重写历史     助手回复被截断改写 → 分叉点靠后，应仍命中大部分
  4 滚动窗口丢弃最老轮 → 前缀在很早处就变了，应大幅失效（Kimi 指出的隐蔽杀手）

用法: .venv/bin/python scripts/bench_server_ttft.py <model> [port]
"""
import json, subprocess, sys, time, urllib.request, os, signal

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-1.7B-4bit"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8099
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

LEDGER = "\n".join(
    f"[cite-{i:02d}] Position {i}: he argued from a documented warrant, citing a source "
    f"dated around 20{18+i%8}. The mental model is a symmetry test applied to the opposing "
    f"claim, deployed whenever an asymmetric standard is proposed. Counter that landed: "
    f"opponents point to tension with his earlier framing." for i in range(1, 40))
PERSONA = ("You are a debate opponent modelled on a real public figure, for English speaking "
           "practice. Argue his documented positions. Never assert a position without a cite-id.\n\n"
           "# Rules\n- Stay in character. Concede small points to win the frame.\n"
           "- First sentence short, then develop over two or three sentences.\n\n"
           "# Position ledger\n" + LEDGER + "\n")

USER_TURNS = [
    "I think the wage gap proves systemic discrimination against women.",
    "But the controlled gap still exists in several large studies.",
    "That ignores occupational segregation being itself a product of social pressure.",
    "You keep shifting the frame instead of answering the question.",
    "So you concede the premise but dispute the remedy?",
]


def ttft(messages, max_tokens=40):
    """流式请求，返回 (首 token 秒, 总秒, 生成文本)"""
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "stream": True, "temperature": 0.0,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; out = []; cached=[0]; ptok=[0]
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 本机不走代理
    with op.open(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "): continue
            if line == "data: [DONE]": break
            try: d = json.loads(line[6:])
            except Exception: continue
            u = d.get("usage") or {}
            if u: cached[0] = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0); ptok[0] = u.get("prompt_tokens", 0)
            ch = d.get("choices") or [{}]
            dl = ch[0].get("delta", {}) if ch else {}
            piece = dl.get("content") or dl.get("reasoning")
            if piece:
                if first is None: first = time.perf_counter() - t0
                out.append(piece)
    return first, time.perf_counter() - t0, "".join(out), ptok[0], cached[0]


def main():
    env = dict(os.environ); env["NO_PROXY"] = "127.0.0.1,localhost"
    srv = subprocess.Popen(
        [".venv/bin/python", "-m", "mlx_lm.server", "--model", MODEL,
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "WARNING"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        print(f"启动 server ({MODEL}) …")
        for _ in range(180):
            try:
                ttft([{"role":"system","content":"hi"},{"role":"user","content":"hi"}], 2)
                break
            except Exception: time.sleep(1)
        else:
            print("server 启动失败"); return
        print("就绪\n")

        sys_msg = {"role": "system", "content": PERSONA}
        print(f"{'情形':<40} {'TTFT':>9} {'总耗时':>8} {'prompt':>8} {'已缓存':>8} {'命中率':>7}")
        print("-" * 88)

        # 1 冷启动
        m = [sys_msg, {"role": "user", "content": USER_TURNS[0]}]
        f, tot, a1, pt, ca = ttft(m)
        print(f"{'1 冷启动（缓存空，全量 prefill）':<40} {f*1000:>8.0f}ms {tot:>7.2f}s {pt:>8} {ca:>8} {ca/max(pt,1)*100:>6.0f}%")

        # 2 追加式续轮
        hist = m + [{"role": "assistant", "content": a1}]
        for i in (1, 2):
            hist.append({"role": "user", "content": USER_TURNS[i]})
            f, tot, a, pt, ca = ttft(hist)
            hist.append({"role": "assistant", "content": a})
            print(f"{'2 追加式续轮 '+str(i)+'（应命中前缀）':<42} {f*1000:>8.0f}ms {tot:>8.2f}s")

        # 3 打断：把最后一条助手回复截断改写
        h3 = list(hist)
        h3[-1] = {"role": "assistant", "content": h3[-1]["content"][:len(h3[-1]["content"])//3]}
        h3.append({"role": "user", "content": USER_TURNS[3]})
        f, tot, _, pt, ca = ttft(h3)
        print(f"{'3 打断后重写历史（分叉点靠后）':<40} {f*1000:>8.0f}ms {tot:>7.2f}s {pt:>8} {ca:>8} {ca/max(pt,1)*100:>6.0f}%")

        # 4 滚动窗口：丢弃最早一轮 → 前缀在很早处即改变
        h4 = [sys_msg] + hist[3:] + [{"role": "user", "content": USER_TURNS[4]}]
        f, tot, _, pt, ca = ttft(h4)
        print(f"{'4 滚动窗口丢最老轮（前缀早处变）':<40} {f*1000:>8.0f}ms {tot:>7.2f}s {pt:>8} {ca:>8} {ca/max(pt,1)*100:>6.0f}%")

        # 5 重复情形 2 确认缓存仍在
        f, tot, _, pt, ca = ttft(hist)
        print(f"{'5 回到情形2的历史（确认缓存存活）':<40} {f*1000:>8.0f}ms {tot:>7.2f}s {pt:>8} {ca:>8} {ca/max(pt,1)*100:>6.0f}%")
    finally:
        srv.send_signal(signal.SIGINT); srv.wait(timeout=30)


if __name__ == "__main__":
    main()
