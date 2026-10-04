#!/usr/bin/env python3
"""
Prefix-cache / prefill probe for mlx-lm 0.31.3 + Qwen3-30B-A3B-Instruct-2507-4bit.

Every API used here was verified line-by-line against the INSTALLED source at
./.venv/lib/python3.12/site-packages/mlx_lm:

  cache.py:15   make_prompt_cache(model, max_kv_size=None) -> List[Any]
  cache.py:88   can_trim_prompt_cache(cache) -> bool
  cache.py:95   trim_prompt_cache(cache, num_tokens) -> int   (annotation says List; it lies)
  cache.py:377  KVCache.is_trimmable() -> True
  cache.py:380  KVCache.trim(n): n=min(offset,n); offset-=n; return n      (O(1))
  cache.py:542  RotatingKVCache.is_trimmable() -> offset < max_size        (the trap)
  generate.py:307 generate_step(prompt, model, *, max_tokens, sampler, logits_processors,
                    max_kv_size, prompt_cache, prefill_step_size, kv_bits, kv_group_size,
                    quantized_kv_start, prompt_progress_callback, input_embeddings)
  generate.py:657 stream_generate(model, tokenizer, prompt, max_tokens=256,
                    draft_model=None, **kwargs)   <- kwargs forwarded to generate_step
  generate.py:423-450  the prefill loop: NO prefix detection, mx.eval per chunk
  sample_utils.py:10   make_sampler(temp=0.0, top_p=0.0, min_p=0.0, ...)

Run:
  ./.venv/bin/python \
      ./scripts/prefix_cache_probe.py

NOTE: this DOES load the model and run the GPU. Do not run it while other GPU
timing measurements are in flight.
"""

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
PREFILL_STEP = 2048


# ----------------------------------------------------------------------------
# One live KV cache + an exact mirror of the token ids inside it.
# ----------------------------------------------------------------------------
class PrefixCache:
    """generate_step does zero prefix detection (generate.py:423-450). It blindly
    prefills whatever you hand it into whatever cache you hand it. So the caller
    owns prefix logic: keep the cache alive, keep a token mirror, feed the delta.

    max_kv_size stays None on purpose: passing it swaps KVCache for
    RotatingKVCache (cache.py:33-38), whose is_trimmable() is
    `offset < max_size` (cache.py:542) -- once the window wraps, rollback
    silently becomes a no-op. At 96 KiB/token, 32k context is only ~3 GB.
    """

    def __init__(self, model):
        self.model = model
        self.cache: List[Any] = make_prompt_cache(model)  # -> [KVCache()] * 48
        self.tokens: List[int] = []
        self.last_trimmed = 0

    def reset(self) -> None:
        self.cache = make_prompt_cache(self.model)
        self.tokens = []
        self.last_trimmed = 0

    @property
    def offset(self) -> int:
        return self.cache[0].offset

    def prepare(self, prompt: List[int]) -> List[int]:
        """Return the suffix that still has to be prefilled, after rolling the
        live cache back to the longest common prefix."""
        if not prompt:
            raise ValueError("empty prompt: generate_step raises on len(prompt)==0")

        n = min(len(self.tokens), len(prompt))
        keep = 0
        while keep < n and self.tokens[keep] == prompt[keep]:
            keep += 1

        # generate_step needs >= 1 token to run (generate.py:363-366), and its
        # prefill loop stops one token short, so never hand it an empty suffix.
        if keep == len(prompt):
            keep = len(prompt) - 1

        to_trim = len(self.tokens) - keep
        if to_trim > 0:
            if not can_trim_prompt_cache(self.cache):
                # Only reachable with RotatingKVCache; kept as a hard guard.
                self.reset()
                keep, to_trim = 0, 0
            else:
                trimmed = trim_prompt_cache(self.cache, to_trim)
                if trimmed != to_trim:
                    raise RuntimeError(
                        f"trim_prompt_cache returned {trimmed}, expected {to_trim}"
                    )
                del self.tokens[keep:]
        self.last_trimmed = to_trim
        return prompt[keep:]

    def commit(self, fed: List[int]) -> None:
        self.tokens.extend(fed)
        if self.offset != len(self.tokens):
            raise RuntimeError(
                f"cache/mirror desync: offset={self.offset} mirror={len(self.tokens)}"
            )


# ----------------------------------------------------------------------------
# One turn, with honest timing.
# ----------------------------------------------------------------------------
def run_turn(
    model,
    tokenizer,
    pc: PrefixCache,
    messages: List[Dict[str, str]],
    max_tokens: int,
    sampler,
) -> Tuple[str, List[int], Dict[str, Any]]:
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    prompt = list(prompt)
    rest = pc.prepare(prompt)

    # prompt_progress_callback fires immediately AFTER
    #   mx.eval([c.state for c in prompt_cache])          (generate.py:441-444)
    # so its timestamps are real GPU-completion times, not lazy submissions.
    # Marks look like: (t,0,total), (t,chunk1,total), ..., (t,total-1,total),
    # then a final (t,total,total) emitted at n==0 AFTER mx.eval(y)
    # (generate.py:461-463) -- i.e. the last mark also contains the final
    # 1-token step plus the first sample. Both numbers are reported.
    marks: List[Tuple[float, int, int]] = []

    def progress(processed: int, total: int) -> None:
        marks.append((time.perf_counter(), processed, total))

    t_call = time.perf_counter()
    fed: List[int] = []
    parts: List[str] = []
    last = None
    t_first: Optional[float] = None

    for r in stream_generate(
        model,
        tokenizer,
        prompt=rest,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=pc.cache,
        prompt_progress_callback=progress,
        prefill_step_size=PREFILL_STEP,
    ):
        if t_first is None:
            t_first = time.perf_counter()
        # Ordering in generate_step: _step(y_n) pushes token n into the cache
        # BEFORE `yield y.item()` (generate.py:456-472). stream_generate emits
        # every token exactly once: in-loop yields for the non-terminal tokens,
        # then one final GenerationResponse carrying the token that caused the
        # break (EOS, or the max_tokens-th token) -- generate.py:713-753.
        fed.append(r.token)
        parts.append(r.text)
        last = r

    # Hard barrier: force every cache write to land before we stop the clock.
    mx.eval([c.state for c in pc.cache])
    t_end = time.perf_counter()

    pc.commit(rest + fed)

    # Pure prefill: from the 0-mark to the last IN-LOOP mark. Excludes the
    # trailing 1-token step and the first sample. Only available when the
    # prefill loop actually ran (i.e. len(rest) >= 2 and it chunked).
    bulk_tok, bulk_s = 0, 0.0
    if len(marks) >= 3:
        bulk_tok = marks[-2][1]
        bulk_s = marks[-2][0] - marks[0][0]

    # Prefill + final 1-token step + first sample.
    p1_s = marks[-1][0] - marks[0][0] if len(marks) >= 2 else float("nan")
    decode_s = t_end - t_first if t_first is not None else float("nan")

    return (
        "".join(parts),
        fed,
        dict(
            total=len(prompt),
            cached=len(prompt) - len(rest),
            new=len(rest),
            trimmed=pc.last_trimmed,
            bulk_prefill_tok=bulk_tok,
            bulk_prefill_s=bulk_s,
            bulk_prefill_tps=(bulk_tok / bulk_s) if bulk_s > 0 else float("nan"),
            prefill1_s=p1_s,
            ttft_s=(t_first - t_call) if t_first is not None else float("nan"),
            decode_tokens=len(fed),
            decode_tps=((len(fed) - 1) / decode_s) if len(fed) > 1 and decode_s > 0
            else float("nan"),
            wall_s=t_end - t_call,
            # mlx-lm's own numbers, for contrast. prompt_tps is
            # prompt.size / (prefill + first sample) where prompt.size is the
            # array YOU passed (generate.py:717-720) -- meaningless on a hit.
            mlx_prompt_tokens=last.prompt_tokens,
            mlx_prompt_tps=last.prompt_tps,
            mlx_gen_tps=last.generation_tps,
            finish=last.finish_reason,
            peak_gb=last.peak_memory,
            offset=pc.offset,
        ),
    )


def decode_ids(tokenizer, ids: List[int]) -> str:
    """Decode generated ids back to message text.

    stream_generate never feeds EOS to the detokenizer (it breaks before
    add_token, generate.py:722-723), but the final GenerationResponse still
    carries the EOS id, so `fed` contains it. HF `decode` does NOT skip special
    tokens by default -- decoding it would splice a literal "<|im_end|>" into
    the message content and the chat template would then add another one,
    breaking the prefix. Strip the stop ids first.
    """
    stop = set(tokenizer.eos_token_ids)
    return tokenizer.decode([t for t in ids if t not in stop])


def report(label: str, s: Dict[str, Any]) -> None:
    bulk = (
        f"{s['bulk_prefill_s']*1e3:7.1f} ms {s['bulk_prefill_tps']:8.1f} tok/s"
        f" ({s['bulk_prefill_tok']} tok)"
        if s["bulk_prefill_tok"]
        else "        n/a (no chunked prefill)"
    )
    print(
        f"{label:<14} prompt={s['total']:>5} cached={s['cached']:>5} "
        f"new={s['new']:>5} trim={s['trimmed']:>5} | prefill {bulk} | "
        f"TTFT {s['ttft_s']*1e3:7.1f} ms | wall {s['wall_s']*1e3:7.1f} ms | "
        f"decode {s['decode_tokens']:>4} tok {s['decode_tps']:6.1f} tok/s | "
        f"off={s['offset']} peak={s['peak_gb']:.1f} GB"
    )


def build_persona(tokenizer, target: int) -> str:
    line = (
        "- Position {i}: the measured effect is smaller than the headline. "
        "Source: BLS ({y}). Confidence: high. Rebuttal hook: cite the {y} "
        "figure and demand their source before conceding anything.\n"
    )
    head = "You are a disciplined live debate opponent.\n\nLEDGER\n"
    per_line = len(tokenizer.encode(line.format(i=0, y=2015), add_special_tokens=False))
    n = max(1, (target - len(tokenizer.encode(head, add_special_tokens=False))) // per_line)
    return head + "".join(
        line.format(i=i, y=2015 + i % 10) for i in range(n)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--persona-tokens", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--skip-cold", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)  # greedy -> reproducible across phases

    persona = build_persona(tokenizer, args.persona_tokens)
    persona_tok = len(tokenizer.encode(persona, add_special_tokens=False))
    probe = make_prompt_cache(model)
    print(
        f"persona={persona_tok} tokens | cache={type(probe[0]).__name__} "
        f"x {len(probe)} | trimmable={can_trim_prompt_cache(probe)}"
    )
    del probe

    turns = [
        "Minimum wage hikes always cost jobs. Two sentences.",
        "That's a dodge. Name your source. Two sentences.",
        "Your source is outdated. Two sentences.",
        "So you concede? Two sentences.",
    ]
    base = [{"role": "system", "content": persona}]

    # ---- WARMUP -------------------------------------------------------------
    # MLX JIT-compiles Metal kernels per shape, and the first forward pass also
    # pages 17 GB of weights in. Without this, "turn 0" measures compilation and
    # disk I/O, not prefill. Warm on the SAME persona so the 2048-chunk and the
    # tail-chunk shapes are the ones we will later time; then warm the
    # small-delta shape too.
    if not args.no_warmup:
        w = PrefixCache(model)
        m = base + [{"role": "user", "content": turns[0]}]
        _, ids, _ = run_turn(model, tokenizer, w, m, 8, sampler)
        m = m + [{"role": "assistant", "content": decode_ids(tokenizer, ids)}]
        m = m + [{"role": "user", "content": turns[1]}]
        run_turn(model, tokenizer, w, m, 8, sampler)
        del w
        mx.clear_cache()
        print("warmup done")
    mx.reset_peak_memory()

    # ---- COLD: fresh cache every turn (the thing prefix caching must beat) ---
    if not args.skip_cold:
        print("\n=== COLD: fresh KV cache per turn (full re-prefill) ===")
        cold = PrefixCache(model)
        msgs = list(base)
        for t, q in enumerate(turns):
            cold.reset()
            m = msgs + [{"role": "user", "content": q}]
            text, ids, s = run_turn(model, tokenizer, cold, m, args.max_tokens, sampler)
            msgs = m + [{"role": "assistant", "content": text}]
            report(f"cold {t}", s)
        del cold
        mx.clear_cache()
        mx.reset_peak_memory()

    # ---- WARM: one live cache, delta-only feeding ---------------------------
    print("\n=== WARM: one live cache reused across turns ===")
    pc = PrefixCache(model)
    messages = list(base)
    turn_ids: List[List[int]] = []
    for t, q in enumerate(turns):
        messages.append({"role": "user", "content": q})
        text, ids, s = run_turn(model, tokenizer, pc, messages, args.max_tokens, sampler)
        messages.append({"role": "assistant", "content": text})
        turn_ids.append(ids)
        report(f"warm {t}", s)

    # ---- BARGE-IN: rewrite the last assistant turn, roll back, continue -----
    # Truncate on a TOKEN boundary using the ids we actually generated. Cutting
    # a decoded string at a character boundary can re-tokenize differently at
    # the seam and force a real re-prefill of the tail.
    print("\n=== BARGE-IN: token-boundary truncation of the last reply ===")
    heard = turn_ids[-1][: max(1, len(turn_ids[-1]) // 3)]
    messages[-1]["content"] = decode_ids(tokenizer, heard)
    messages.append({"role": "user", "content": "No, stop. Answer directly."})
    text, ids, s = run_turn(model, tokenizer, pc, messages, args.max_tokens, sampler)
    report("barge-in", s)
    print(
        f"  rolled back {s['trimmed']} tokens in O(1); re-prefilled {s['new']} "
        f"of {s['total']} ({s['cached']} reused)"
    )


if __name__ == "__main__":
    main()
