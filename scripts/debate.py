"""跟人设真刀真枪辩一场（纯文本，语音还没接）。

同时回答最后一个未验证的外推：MoE 的 gather_qmm 在 prefill 上效率如何。
用法: .venv/bin/python scripts/debate.py [模型]
"""
import json, os, re, signal, subprocess, sys, time, urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
PORT = 8097
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

raw = open("personas/ck-gender.md").read()
PERSONA = raw.split("---", 2)[2].strip()          # 去掉 YAML frontmatter

TURNS = [
  "My name is Wei. You told a room of teenage girls that a career-driven life is empty. Which survey?",
  "You can't name it. So it's a feeling, not a finding. Why should anyone act on it?",
  "Fine. Different question. In 2018 you said American women were thriving and the pay gap was a myth. "
  "Now you say female careerism is destroying them. Those can't both be true. Which one do you drop?",
  "Are you an AI?",
]

def ask(messages, max_tokens=180):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "stream": True, "temperature": 0.7,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    rq = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.perf_counter(); first = None; out = []; pt = ct = 0
    with op.open(rq, timeout=900) as r:
        for line in r:
            s = line.decode().strip()
            if not s.startswith("data: ") or s == "data: [DONE]": continue
            try: d = json.loads(s[6:])
            except Exception: continue
            u = d.get("usage") or {}
            if u:
                pt = u.get("prompt_tokens", 0)
                ct = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            ch = d.get("choices") or [{}]
            dl = ch[0].get("delta", {}) if ch else {}
            p = dl.get("content") or dl.get("reasoning")
            if p:
                if first is None: first = time.perf_counter() - t0
                out.append(p)
    return first, time.perf_counter() - t0, "".join(out).strip(), pt, ct

def main():
    env = dict(os.environ); env["NO_PROXY"] = "127.0.0.1,localhost"
    srv = subprocess.Popen([".venv/bin/python", "-m", "mlx_lm.server", "--model", MODEL,
                            "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "WARNING"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        print(f"加载 {MODEL} …")
        t0 = time.perf_counter()
        for _ in range(900):
            try: ask([{"role": "user", "content": "hi"}], 4); break
            except Exception: time.sleep(1)
        else: print("启动失败"); return
        print(f"就绪 ({time.perf_counter()-t0:.0f}s)\n")
        for _ in range(2): ask([{"role":"system","content":PERSONA},{"role":"user","content":"warm"}], 8)

        hist = [{"role": "system", "content": PERSONA}]
        print("="*100)
        for i, u in enumerate(TURNS, 1):
            hist.append({"role": "user", "content": u})
            f, tot, a, pt, ct = ask(hist)
            hist.append({"role": "assistant", "content": a})
            words = len(a.split()); first_sent = re.split(r'(?<=[.!?])\s', a)[0]
            print(f"\n[你] {u}\n")
            print(f"[CK] {a}\n")
            print(f"      ── TTFT {f*1000:.0f}ms · 全程 {tot:.1f}s · {words} 词 · "
                  f"首句 {len(first_sent.split())} 词 · prompt {pt} (缓存 {ct}, {ct/max(pt,1)*100:.0f}%)")
            leak = [t for t in re.findall(r'\bG\d\d\b|\[[HM]\]|§\d', a)]
            if leak: print(f"      ⚠ 标签泄漏: {leak}")
        print("\n" + "="*100)
    finally:
        srv.send_signal(signal.SIGINT); srv.wait(timeout=60)

main()
