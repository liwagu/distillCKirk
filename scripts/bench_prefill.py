"""测头号风险：LLM prefill / TTFT，以及打断导致缓存失效的真实代价。

背景：人设 prompt + 引用账本约 2000-4000 token，每轮恒定。
算术预估 3k token × 5B 激活 ≈ 30 TFLOP ÷ 20 TFLOPS ≈ 1.5s —— 若属实则预算爆炸。
唯一出路是前缀缓存。本脚本量化四种情形：

  A 冷启动无缓存         每轮全量 prefill（最坏情况 / 缓存永远不命中）
  B 缓存命中（只追加）   只 prefill 新增 token —— 目标形态
  C 打断后重写历史       缓存失效 → 全量重 prefill（VoxEMW 的做法）
  D 打断后 trim 缓存     回滚到分叉点，只重算修正后的尾巴

用法: .venv/bin/python scripts/bench_prefill.py [模型名]
"""
import sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-1.7B-4bit"

# ── 构造真实体量的人设 prompt（模拟人设正文 + 立场引用账本）──
LEDGER = "\n".join(
    f"[cite-{i:02d}] Position {i}: On this topic he argued from a specific documented "
    f"warrant, citing a source dated around 20{18+i%8}. The underlying mental model is "
    f"a symmetry test applied to the opposing claim, which he deploys whenever an "
    f"asymmetric standard is proposed. Counter that landed: opponents point to the "
    f"tension with his earlier framing." for i in range(1, 40))
PERSONA = (
    "You are a debate opponent modelled on a real public figure, for English speaking "
    "practice. Argue his documented positions. Never assert a position without a cite-id.\n\n"
    "# Rules\n- Stay in character. Concede small points to win the frame.\n"
    "- First sentence short. Then develop the argument over two or three sentences.\n"
    "- If the record does not cover it, say you would need to think about it.\n\n"
    "# Position ledger\n" + LEDGER + "\n")

TURNS = [
    "I think the wage gap proves systemic discrimination against women.",
    "But you are ignoring that the controlled gap still exists in many studies.",
    "That does not address occupational segregation being itself a product of pressure.",
]


def ttft_and_rates(model, tok, messages, cache, max_tokens=40):
    """返回 (TTFT秒, prompt_token数, prefill tok/s, decode tok/s, 生成文本)"""
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    t0 = time.perf_counter()
    first = None; last = None; text = []
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, prompt_cache=cache):
        if first is None: first = time.perf_counter() - t0
        last = r; text.append(r.text)
    return first, last.prompt_tokens, last.prompt_tps, last.generation_tps, "".join(text)


def main():
    print(f"模型: {MODEL}")
    t0 = time.perf_counter()
    model, tok = load(MODEL)
    print(f"加载 {time.perf_counter()-t0:.1f}s   峰值 {mx.get_peak_memory()/1e9:.2f} GB")
    ntok = len(tok.encode(PERSONA))
    print(f"人设 prompt = {ntok} token\n")

    base = [{"role": "system", "content": PERSONA}]

    # 预热（编译 kernel），不计入结果
    _ = ttft_and_rates(model, tok, base + [{"role": "user", "content": "hi"}],
                       make_prompt_cache(model), max_tokens=4)

    print(f"{'情形':<34} {'TTFT':>8} {'prompt tok':>11} {'prefill t/s':>12} {'decode t/s':>11}")
    print("-" * 82)

    # ── A 每轮全量 prefill（无缓存）──
    for i, u in enumerate(TURNS):
        msgs = base + [{"role": "user", "content": u}]
        ttft, pt, pts, gts, _ = ttft_and_rates(model, tok, msgs, make_prompt_cache(model))
        print(f"{'A 无缓存 全量prefill 轮'+str(i+1):<34} {ttft*1000:>7.0f}ms {pt:>11} {pts:>12.0f} {gts:>11.1f}")

    # ── B 缓存命中（只追加）──
    cache = make_prompt_cache(model)
    hist = list(base)
    for i, u in enumerate(TURNS):
        hist.append({"role": "user", "content": u})
        ttft, pt, pts, gts, out = ttft_and_rates(model, tok, hist, cache)
        hist.append({"role": "assistant", "content": out})
        print(f"{'B 缓存命中 只追加 轮'+str(i+1):<34} {ttft*1000:>7.0f}ms {pt:>11} {pts:>12.0f} {gts:>11.1f}")

    # ── C 打断后重写历史 → 缓存失效 ──
    cache2 = make_prompt_cache(model)
    hist2 = list(base) + [{"role": "user", "content": TURNS[0]}]
    _, _, _, _, full = ttft_and_rates(model, tok, hist2, cache2)
    hist2.append({"role": "assistant", "content": full[:len(full)//3]})   # 只听到三分之一就被打断
    hist2.append({"role": "user", "content": TURNS[1]})
    ttft, pt, pts, gts, _ = ttft_and_rates(model, tok, hist2, cache2)      # 复用旧缓存但前缀已变
    print(f"{'C 打断+重写历史（缓存脏）':<34} {ttft*1000:>7.0f}ms {pt:>11} {pts:>12.0f} {gts:>11.1f}")

    # 同样内容但用全新缓存 = 真正的全量重 prefill 代价
    ttft, pt, pts, gts, _ = ttft_and_rates(model, tok, hist2, make_prompt_cache(model))
    print(f"{'C2 同上但全新缓存(真代价)':<34} {ttft*1000:>7.0f}ms {pt:>11} {pts:>12.0f} {gts:>11.1f}")

    # ── D trim 回滚 ──
    c3 = make_prompt_cache(model)
    print(f"\ntrim 支持: {can_trim_prompt_cache(c3)}")
    hist3 = list(base) + [{"role": "user", "content": TURNS[0]}]
    _, pt_a, _, _, full3 = ttft_and_rates(model, tok, hist3, c3)
    off = c3[0].offset
    ntrim = 30
    got = trim_prompt_cache(c3, ntrim)
    print(f"D 缓存 offset {off} → trim {ntrim} → 实际回滚 {got}，新 offset {c3[0].offset}")

    print(f"\n峰值内存 {mx.get_peak_memory()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
