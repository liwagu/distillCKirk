> Historical engineering record. For the current released architecture and setup, read [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Private logs and local session archives referenced below are not distributed.

# DeepSeek 大脑切换与复读修复

用户真人试用后反馈：不同问题被连续回复相同的 11 词句子。此前本地 Qwen 浏览器验收只证明播放、口型、插话链路；不能视为对话理解质量通过。用户先提供 Paycools，随后提供官方 DeepSeek 接入，要求 `deepseek-v4-flash`，明确关闭 thinking。

## 配置与边界

- `.env.local`：`LLM_URL=https://api.deepseek.com`、`LLM_MODEL=deepseek-v4-pro`，使用用户提供的官方密钥。用户随后询问 Pro / Flash 的对话效果；同一账户本次 Flash 无正文、Pro 返回正文，因此当前采用 Pro。保留 TypeSafe 专用代理，密钥不写入文档、测试报告或日志。
- DeepSeek Chat Completions 请求明确发送 `thinking: {"type": "disabled"}`；[官方开关说明](https://api-docs.deepseek.com/guides/thinking_mode/)。
- 启动用 `.venv/bin/python scripts/service.py restart`，不带 `--local-llm`。本地备用模型只有显式传入该参数才选用。
- `max_tokens` 由 140 提高到 300，声音回合仍受 90 词上限约束，减少正文被 token 限额截断后只剩一句的风险。
- 页面显示实际大脑、thinking 关闭状态及未成功返回回复的状态。健康检查和服务管理区分“语音/画面加载完成”与“大脑成功回复”。
- [官方当前模型说明](https://api-docs.deepseek.com/quick_start/pricing/)列出 `deepseek-flash` 与 `deepseek-v4-pro`；原 Flash 名称仍支持，实际路由到 V4.1 Flash。Pro 与 Flash 是两个型号。

## 官方接口现场结果

官方账户可用且有正余额；直连与代理的模型列表返回 200。Flash 的最小正文请求尚未成功；Pro 的最小请求在 0.69s 内返回 `Ready`。这一请求只验证 Pro 正文连通性，不能单独确认真人对话质量或实时延迟。

| 检查 | 结果 |
|---|---|
| 直连和代理 `/v1/models` | 200，列出 `deepseek-flash`、`deepseek-v4-pro` |
| `/user/balance` | 200，账户可用、有正余额；未记录余额数值 |
| 原 Flash 名称，直连流式 | 200 后等待 20.3s 超时，没有正文 |
| 当前 Flash 名称，直连原始流 | 200，90.34s 内每约 12s 返回 `: keep-alive`，没有任何正文 |
| 原 Flash 名称，代理非流式 `/chat/completions` | 200 后 60.32s 超时，没有正文 |
| `deepseek-v4-pro`，直连流式最小请求 | 200，0.69s 内返回正文 `Ready` |

所有以上生成请求均明确发送 `thinking.type=disabled`。[官方保活机制](https://api-docs.deepseek.com/quick_start/rate_limit/)说明这些注释在等待回复时可能出现；它们不表示模型已经开始生成。原始流中未收到 `data` 或 `content`，所以本次等待并非应用分句器丢弃了回答。

证据：`logs/deepseek-official-connectivity-20261002.json`、`logs/deepseek-official-reply-20261002.json`、`logs/deepseek-official-raw-stream-20261002.json`、`logs/deepseek-official-balance-and-proxy-20261002.json`、`logs/deepseek-official-pro-diagnostic-20261002.json`。这些报告不含密钥或 Authorization。

## Pro 实际对话与音画验收

两次三轮纯文本测试均使用生产人设与完整历史，实际返回模型均为 `deepseek-v4-pro`，思考关闭、未收到 reasoning，正常 `stop` 结束，没有相同或近似整轮复读。第二次首正文分别为 1598、1024、551ms，全文完成分别为 1823、2350、2003ms。问候、要求解释女权立场、工资差距条件反例得到不同且相关的回答。

初次文本的条件反例回答加上了犯罪和起诉定性，超出账本。已在 §1A 明确条件反例可以削弱解释，但不能补出新法律定性或刑事政策；26条账本、来源及 §0 未改变。修正后的文本复验没有这类定性。仍有边界：问候会编造角色当前现场，第三轮原始全文112词超过提示中的90词；语音编排另有句子边界词数预算。请求与复读检查通过不等于完整人设事实核查通过，也未做同提示的 Qwen / Pro A/B，不能把所有改善单独归因于模型。

文本报告：
- `logs/deepseek-official-pro-conversation-20261002.json`（修正前，保留法律外推证据）
- `logs/deepseek-official-pro-conversation-refined-20261002.json`（修正后）

实际浏览器使用明确标记的模拟麦克风，经 MediaStream / AudioWorklet → VAD / ASR → Pro → 克隆TTS → WebAudio / 动态口型。三轮分别自然播放完并提交播放ACK，**29/29检查通过，JavaScript错误0**。输入 ended 回调到实际排程首声分别约 **5.71、3.36、3.21s**；这些是包含ASR、TTS和等首帧的链路延迟，不能拿纯文本首字延迟代替。第三轮回答先承认反例的条件意义，再追问控制变量的数据，没有复读前两轮。

人设修正后重启，单独重验条件反例，**13/13检查通过，JavaScript错误0**；实际播放34词、未添加刑事定性，首声排程约4.09s。当前 `/health` 与服务管理均为ready，模型Pro、thinking关闭、脸ready；本地Qwen大脑进程已退出。浏览器报告：
- `logs/acceptance-20261002-deepseek-pro/report.json`
- `logs/acceptance-20261002-deepseek-pro-refined/report.json`

此轮没有重新执行Pro插话全套；先前Qwen生命周期检查和当前34项纯回归仍通过。模拟麦克风验收不代替真人对话、扬声器回声、声音相似度与嘴型观感验收。

## 已证实的复读缺口

最新 ASR 成功后会被追加为 `user` 并进入模型请求，未找到漏传最新问题的直接路径。人设第一条正是截图中的句子，旧首句明确豁免复读检测，所以模型只输出这句时能连续全部放行。

人设还有规则冲突：要求开场政治立场、禁止寒暄，同时又允许正常寒暄；缺少含糊转写先澄清的优先路由。本轮已明确寒暄、澄清、清晰辩论题的处理顺序，账本用于推理相关立场，不是逐字脚本；26 条立场与来源保持原样。

另外，旧 140 token 限额下，如果只有首句完整而剩余正文被截断，客户端会丢弃不完整尾句；这也能在 mock SSE 中复现为只有那 11 词。现有真人日志未记录完整原始生成，不能断言本次具体是哪一种路径。

Stop/Start 建立新 WebSocket，会重置模型历史；旧气泡仍在页面，但不代表新连接有它们的上下文。本轮连接提示明确为新会话。

长首句完全重复会被拦截，继续处理后续新句；整轮只有复读时最多重新生成一次，仍无新句就显示错误，不播放旧回复。用户明确要求重说或解释上一句时允许复述；新问题中的普通 `explain` 或抱怨复读不会放开重复检查。

远程请求还设有独立的首正文 20s 等待上限。SSE 保活不延长这一等待；收到正文后解除该限制，TTS 消费流的时间不计入首正文预算。无正文 EOF 或超时会清除 `llm_ready`，避免旧成功状态残留。

## 此前 Paycools 网关现场诊断

最小输入只要求回答 Ready。没有使用完整人设或音画模型，所以不是语音、口型或长历史引起的等待。除明确的最早协议对照外，思考模式关闭。

| 检查 | 结果 |
|---|---|
| HTTP/HTTPS `/v1/models` | 200，列出 `deepseek-v4-flash` |
| 无密钥 POST 对照 | 1.89s 返回 401 Invalid token |
| 有效密钥、HTTPS 流式、直连 | 45.9s 等待超时，没有正文 |
| 有效密钥、HTTPS 流式、本地代理 | 51.0s 等待超时，没有正文 |
| 有效密钥、最小非流式请求 | 64.35s 后 ServerDisconnectedError |
| HTTP 最小请求 | 30.79s 超时 |
| httpx2 + 显式本地代理 | 45.95s ReadTimeout |
| Anthropic 兼容接口 | 25.92s 超时 |
| 当前官方 Flash 别名诊断 | `deepseek-flash` 返回 503 `model_not_found / No available channel`；应用仍保留用户指定的原模型名 |

证据保存在 deepseek-connectivity-20261002.json（private local evidence; not distributed），不含密钥或 Authorization。它记录此前网关的生成失败；当前应用已改用官方入口。

源文件备份保存在 `.backups/codex-deepseek-20261002/`，不包含密钥文件。
