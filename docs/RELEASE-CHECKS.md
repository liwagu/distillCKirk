# Public source release checks

Publication audit: **2026-10-05**.

## Conversation-history update — 2026-10-06

The updated publication checkout passed **55 Python unit regressions**, the existing **6 frontend playback scenarios**, and `node --check web/app.js`. New regressions cover interruption-safe FIFO transcription, same-packet VAD events, empty/failed ASR, transport reset, shutdown draining, private atomic audio/text archives, and validated one-use restoration. These checks used the existing development environment against the exported source without launching a second avatar service.

The update also excludes the complete `conversations/` directory from Git. Real recordings, transcript journals, and resume seeds are runtime data and are not part of the source release. The earlier release results below remain historical. See [ASR-INPUT-FIX-2026-10-05.md](ASR-INPUT-FIX-2026-10-05.md) for the original defect and the development live-check boundaries.

## Checks executed on the public source snapshot

| Check | Result | What it establishes |
|---|---|---|
| Python unit regressions | **37 passed** | Brain request/parsing, no-thinking behavior, repetition handling, semantic-client mocks, playback/history lifecycle, cancellation, idle-face state |
| Frontend playback harness | **6 scenarios passed** | Audio start, suspension/drain/gaps, interruption/Stop, obsolete frame handling |
| Frontend syntax | `node --check web/app.js` passed | JavaScript parses |
| Python static parsing | **45 source files passed** | Public Python sources parse |
| Shell static parsing | **6 scripts passed** | Public shell scripts parse under Bash |
| Public JSON parsing | Passed | Configurations, research dossiers, and screenshot metadata remain valid JSON after path normalization |
| Gitleaks directory scan | No leaks found | Generic secret-pattern scan of the publication snapshot |
| Exact local-secret matching | No matches | Configured credential values are absent from the published files; values were never written to this report |
| Personal absolute-path scan | No matches after normalization | Developer home paths were removed from the public snapshot |

The tests used the existing development Python environment against the exported
source. They did **not** install all dependencies into a fresh environment, convert
new Core ML weights, or launch a second complete avatar instance. Expected mock
timeouts/error cases produce diagnostic logs; aiohttp also emits AppKey advisory
warnings. Those do not represent failed regressions.

## Current runtime and demonstration evidence

The existing development service's `/health` was inspected during publication:

```json
{
  "ready": true,
  "llm_ready": true,
  "face_ready": true,
  "remote_llm": true,
  "thinking_disabled": true,
  "models": {
    "asr": "Qwen/Qwen3-ASR-1.7B",
    "tts": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "llm": "deepseek-v4-pro"
  }
}
```

This is a readiness observation, not a new conversation benchmark or a guarantee
that subsequent API requests will succeed. The completed video demonstrates actual
recorded use; its selected frames at 03:45, 05:05, and 05:45 are in `docs/demo/`.
The source video is 3840×2160 and approximately 7 minutes 10 seconds. The avatar
renderer itself produces 640×480 at a configured target of 20 fps.

- [YouTube finished demo](https://www.youtube.com/watch?v=DtIkQjkBrZg)
- [Bilibili demo](https://www.bilibili.com/video/BV14oHv6qEDb)

## Historical checks are a separate evidence set

The October 2 records describe a local-Qwen browser stage (22/22 checks and a 13/13
single-turn follow-up), then a DeepSeek-Pro browser stage (29/29 checks and a 13/13
persona follow-up). The idle-face fix also has its own browser checks. Their reports
used explicitly synthetic microphone input and the original application transport.

Those results are preserved in the historical engineering documents. Private logs,
raw microphone fixtures, per-frame recordings, and local session archives are not
included in this release. The historical counts have not been rerun as a full
browser session during publication. They should not be described as fresh-clone,
real-speaker, or general-purpose performance acceptance.

## Publication boundary

The release uses an explicit source/media allowlist and a new Git history. It does
not upload `.env.local`, API credentials, logs, personal phone-transfer metadata,
private recordings/transcripts, source face/voice clips, virtual environments,
downloaded weights, or generated caches. Public research references and historical
helper defaults were normalized to repository-relative paths; private archive
references were removed or marked unavailable.

The four upstream Git submodules are pinned to the reviewed revisions, rather than
uploading their local agent state or copying their Git histories into this project.
Relevant upstream license notices are preserved in `licenses/` and the submodules;
see [ATTRIBUTION.md](ATTRIBUTION.md).

For reproduction requirements and remaining gaps, read
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
