> Historical engineering record. For the current released architecture and setup, read [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Private logs and local session archives referenced below are not distributed.

# LLM prefill / TTFT 实测（2026-09-10）

第二轮复核把 LLM prefill 列为头号风险：人设 prompt ~3k token，算术估算
3k × 5B激活 ≈ 30 TFLOP ÷ 20 TFLOPS ≈ 1.5s，且怀疑打断会让缓存全废。
本文档给出实测答案。

## 1. 量化 prefill 效率：接近满血

模型 Qwen3-1.7B-4bit（稠密），2886 token 人设，直连 API（`scripts/bench_prefill.py`）：
prefill 实测 **2533–6497 tok/s**。峰值换算 2 × 1.7e9 × 6497 ≈ **22 TFLOPS**，
与实测的 20 TFLOPS bf16 峰值同量级。

**结论：量化 matmul 的 prefill 效率没有塌方，算术估算成立。**
外推：3B 激活 MoE 跑 3k token ≈ 0.9s；5.1B 激活 ≈ 1.5s。
→ 冷启动 prefill 确实会吃掉大半个延迟预算，**前缀缓存是必需品而非优化**。

## 2. mlx_lm.server 自带自动前缀缓存

`server.py` 内含 `LRUPromptCache` + `fetch_nearest_cache()`：自动匹配最长已缓存
前缀，只 prefill 剩余部分，且 LRU 可同时保存多条对话分支。
响应的 `usage.prompt_tokens_details.cached_tokens` 直接暴露命中量。

实测（`scripts/bench_server_ttft.py`，Qwen3-1.7B-4bit，2400+ token 人设）：

| 情形 | TTFT | prompt | cached | 命中率 |
|---|---|---|---|---|
| 1 冷启动（缓存空） | 884 ms | 2543 | 3 | 0% |
| 2 追加式续轮 | 495 / 420 ms | — | — | — |
| 3 **打断后重写历史** | 530 ms | 2702 | 2662 | **99%** |
| 4 **滚动窗口丢最老轮** | 365 ms | 2665 | 2523 | **95%** |
| 5 切回旧历史分支 | 305 ms | 2715 | 2706 | **100%** |

## 3. 对第二轮复核的三处修正

### 打断不会导致全量重 prefill —— 担忧不成立
我和 Kimi 都判断「重写历史 → 前缀改变 → 缓存失效 → 全量重算落在最不该慢的时刻」。
**实测命中 99%。** `fetch_nearest_cache` 按最长公共前缀匹配，只重算分叉后的尾巴。
→ **无需改成 append-only 历史**，VoxEMW 的 heard-prefix 写回可以照用。

### 滚动窗口也不是缓存杀手 —— 担忧不成立
预期：丢弃最老一轮会改变系统提示之后的前缀 → 每轮全废。
**实测命中 95%。** 原因：缓存命中的那 2523 token 恰好是常量人设前缀，
被作废的只是很短的可变历史。
→ 通则：**只要人设前缀在 token 数上占主导、历史保持短，损失就有界。**

### 多分支缓存存活
情形 5 切回旧历史仍 100% 命中，证明 LRU 同时保有多条分支——
打断后用户改口、或走不同话题分支，都不会互相踢掉。

## 4. 新发现的语音场景陷阱

**Qwen3 默认开启思考模式**，流式返回的是 `reasoning` 字段而非 `content`，
思考 token 全部排在可朗读内容之前 —— 对语音是直接的延迟杀手。

关闭方式（请求体字段名是 `chat_template_kwargs`，不是 CLI 的 `chat_template_args`）：
```json
{"chat_template_kwargs": {"enable_thinking": false}}
```
这正是 VoxEMW 配置里 `reasoning_effort: "none"` 注释「默认思考，先想 ~2s 才开口」的由来。

## 5. 环境坑

- 本机**必须走代理**（HTTP_PROXY=127.0.0.1:7897），直连 HF 报 SSL UNEXPECTED_EOF。
- **绝不能开 `HF_HUB_ENABLE_HF_TRANSFER` 或 xet**：Rust 实现不读 HTTP_PROXY，
  会永久挂起。必须 `HF_HUB_DISABLE_XET=1` 且不设 hf_transfer。
- 单连接 0.18–0.8 MB/s，8 并行约 1.52 MB/s。
- `mlx_lm.server` 按请求体 `model` 字段动态加载模型，名字必须精确。

## 6. 待办

- 换 Qwen3-30B-A3B-Instruct-2507-4bit（下载中）复测，确认 MoE 的 gather_qmm
  在 prefill 上是否同样接近满血 —— 这是唯一剩余的外推风险。

---

# 补测与修正（2026-09-10 第二轮）

## 修正 1：滚动窗口的伤害随历史增长——首轮测试有乐观偏差

首轮测出「淘汰最老轮仍命中 95%」，但那次历史极短（2665 token 里 2400 是系统提示）。
用真实体量历史重测（`scripts/bench_window.py`，每轮 user ~25 + assistant ~60 token）：

| 历史 | 总 token | 追加一轮命中 | 淘汰最老轮命中 | 需重算 | TTFT 追加→淘汰 |
|---|---|---|---|---|---|
| 4 轮 | 2006 | 96% | 85% | 281 tok | 248 → 196 ms |
| 12 轮 | 2723 | 97% | 72% | 733 tok | 276 → 332 ms |
| **30 轮** | **4361** | **98%** | **61%** | **1654 tok** | **399 → 607 ms** |

结论：
- **追加式续轮稳定 96–98% 命中**，这是常态路径，很好。
- **窗口填满后每轮淘汰会掉到 61%**，1.7B 上 +200ms，换算 2.73B 激活 MoE 约 **+330ms**。
- 但人设前缀始终命中，所以 Kimi 担心的「每轮固定 1.5s 重 prefill」不成立。

### 设计对策：批量压缩 + 移出关键路径

不要每轮淘汰一轮。改为：
1. 攒到阈值再**批量压缩**历史（摘要 + 保留最近 N 轮），把失效摊薄到多轮。
2. **在 TTS 播放期间执行压缩后的重 prefill** —— 一段辩论回复有 5–15s 音频，
   足够在用户听的时候把新前缀灌进缓存，下一轮直接命中热缓存。
   → 这 330ms 完全不落在关键路径上。

## 修正 2：激活参数量算错了

Qwen3-30B-A3B 实际激活 **2.73B**（从 config 精算：hidden 2048, 48 层, 128 专家,
top_k 8, moe_intermediate 768），不是此前假设的 5B —— 悲观了 1.8 倍。
冷启动 3k token prefill 重算 ≈ **0.8s**（而非 1.5s）。

## 修正 3：MoE prefill 是带宽受限，不是算力受限

单个 token 只激活 top_k=8 个专家，但一个 prefill chunk 里 N 个 token 各自独立路由，
被触及的专家比例 = 1-(1-8/128)^N：N=8 时 40%，N=32 时 87%，N=64 时 98.4%，
N≥128 时约 100%。

**即任何超过约 64 token 的 prefill chunk 都会流式读取几乎全部 17.2GB 权重。**
MoE 在 decode 上的带宽优势在 prefill 阶段不存在。所幸 prefill 摊薄到很多 token，
按 426 GB/s 算每个 2048-token chunk 约 40ms 纯权重传输，可接受。

## 必须避开的坑（源码级验证）

1. **绝不传 `max_kv_size`。** 会把 KVCache 换成 RotatingKVCache，其
   `is_trimmable()` 是 `offset < max_size`；窗口一绕回，`trim_prompt_cache`
   **静默返回 0 不报错**，打断回滚变成空操作。
2. **GPU 低功耗态**（mlx-lm#432）：空闲后 prefill 可慢 7 倍，需跑废查询唤醒。
   这很可能就是首轮直连测试中 prefill 在 2533–6497 tok/s 间大幅波动的原因。
   生产环境应在会话间保持 GPU 温热。
3. **`prompt_tps` 在缓存命中时是误导性的**：它是 `prompt.size / prompt_time`，
   其中 prompt.size 是你传入的 `rest`，而 prompt_time 还包含第一个采样 token。
   要测缓存效果只能看 `cached_tokens`，不能看 `prompt_tps`。
4. **不要整体重渲染 chat template。** Qwen3 模板是非局部的：位于
   `ns.last_query_index` 之后的 assistant 消息会被注入 `<think>` 块，
   而当后面再追加 user 轮时又会**不带该块重新渲染**——静默改变早先 token。
5. **生成的 id 里含 EOS**：`tokenizer.decode(fed)` 会把 `<|im_end|>` 字面量拼进去。
6. **绕过 `stream_generate` 就绕过了 `wired_limit()`**，macOS 可能把 17GB 权重换页。

---

# 修正 4（2026-09-23）：「淘汰最老轮 61%」是测试顺序造成的假象

`bench_window.py` 在**同一个 server、同一个 LRU** 里依次跑 4 → 12 → 30 轮，且每轮内容只取决于轮号。
于是 30 轮的淘汰请求 `[人设][t1..t29][q]` 恰好和前面 12 轮留下的淘汰分支 `[人设][t1..t11][q]`
共享一段 `[人设][t1..t11]` 前缀——命中的不只是人设，还有上一个测试剩下的历史。

用同一个 tokenizer 按测试顺序重放全部 prompt（LRU 取最近 10 条、按最长公共前缀匹配），
**逐位复现了当时的实测值**，再算出真实滚动窗口下的稳态（每轮都淘汰，只有人设能命中）：

| 历史 | 淘汰后总 tok | 重放·当时命中 | 实测记录 | **稳态命中** | **稳态需重算** |
|---|---|---|---|---|---|
| 4 轮 | 1917 | 85% / 281 tok | 85% / 281 | 85% | 286 tok |
| 12 轮 | 2634 | 72% / 733 tok | 72% / 733 | **62%** | **1003 tok** |
| 30 轮 | 4272 | 61% / 1654 tok | 61% / 1654 | **38%** | **2641 tok** |

（人设含 system 模板 = 1631 tok。）

结论：
- 「修正 1」里的 **61% / 1654 tok / +200ms（MoE 约 +330ms）低估了约 1.6 倍**。30 轮滚动窗口的真实代价是
  人设之后的历史**全部**重算，约 2641 tok。延迟需在 Qwen3-30B-A3B 上重测，不要沿用 +330ms。
- 「修正 1」的**设计对策不变且更重要**：批量压缩（把失效摊到多轮）+ 在 TTS 播放期间做重 prefill。
- 「人设前缀始终命中」这句本身没错，错在把 61% 当成了「只有人设命中」的结果——两者其实互相矛盾
  （1631/4272 = 38%）。
- 测试方法教训：测缓存时每个场景要**独立的 server 或先清空 LRU**，否则前一个场景的分支会污染后一个。
