# Reproducing the debate-avatar prototype

This is a macOS / Apple Silicon research prototype, with a working locally hosted
browser and speech/avatar pipeline. The demonstrated configuration uses **remote
DeepSeek V4 Pro** for conversation, with thinking disabled. ASR, TTS, turn detection,
and avatar rendering run locally. The checked-in JSON instead defaults to a local
Qwen3 model; an environment override selects the remote brain.

The source and public screenshots do not include model weights, the demonstration's
voice/video references, API credentials, or generated caches. A clean clone needs
the preparation below. These instructions were audited against source and installed
package metadata; a fresh-machine install and conversion have **not** been executed.

## Runtime and entry points

| Stage | Actual implementation | Location |
| --- | --- | --- |
| Browser input | `getUserMedia` + AudioWorklet, mono 16 kHz Int16 PCM over WebSocket | `web/app.js` |
| Turn detection | Silero VAD v5 + SmartTurn v3.2, CPU ONNX Runtime | `voxck/vad.py` |
| Speech recognition | `mlx-qwen3-asr`, `Qwen/Qwen3-ASR-1.7B`; live conversation uses English | `voxck/asr.py` |
| Conversation | OpenAI-compatible streaming API; remote DeepSeek or a child `mlx_lm.server` | `voxck/llm.py`, `run.py` |
| Persona and output checks | Sourced Markdown ledger, deterministic guards, optional remote TypeSafe semantic checks | `personas/`, `voxck/guard.py`, `voxck/semantic_guard.py` |
| Speech synthesis | MLX Qwen3-TTS Base 8-bit with a 24 kHz reference voice; resampled to 16 kHz PCM | `voxck/tts.py` |
| Avatar audio features / VAE | MuseTalk 1.5 MLX port, Metal GPU | `voxck/face.py`, `ref-musetalk-mlx/` |
| Avatar UNet | MuseTalk weights converted using Apple's ANE-friendly architecture; Core ML subprocess, `CPU_AND_NE` | `voxck/ane_worker.py`, `scripts/export_unet_ane.py` |
| Face crop and composition | SCRFD detector, OpenCV / NumPy; optional MODNet + YOLO background preparation | `ref-feathertalk/data_utils/`, `voxck/segment.py` |
| Playback | WebAudio sample clock, JPEG canvas frames and explicit playback acknowledgements | `web/app.js` |
| Supervision | aiohttp HTTP / WebSocket application, one MLX executor, cancellable turn lifecycle | `voxck/orchestrator.py` |

`generation_done` means that output generation finished. The browser separately
reports `playback_done` after audio drains, or `playback_progress` on interruption.
The server commits the portion heard to conversation history; new turns invalidate
old audio/video epochs. Idle uses a selected neutral frame, rather than playing an
unrelated talking video.

- `run.py [config] [--local-llm]`: foreground process; default config is
  `configs/assistant.json`.
- `scripts/service.py start|status|stop|restart [--config FILE] [--local-llm]`:
  checkout-specific process management. Status/help do not load models. Start uses
  offline Hugging Face caches and waits at most 20 seconds by default; startup can
  continue after that wait.
- `GET /health`: reports local readiness, `face_ready`, `llm_ready`, selected models,
  `remote_llm`, and thinking state. HTTP readiness alone is not proof of a usable
  conversation or a working face renderer.
- `GET /`: browser UI; `GET /ws`: streaming connection. Default bind is
  `127.0.0.1:8000`.

## Source dependencies and provenance

The project originally used separate upstream checkouts. Pinning those checkouts
as Git submodules preserves their provenance and avoids accidentally publishing
their internal Git history or local agent files. Initialize the published submodules:

```sh
git submodule update --init --recursive
```

The inspected revisions were:

| Directory | Upstream and pinned commit | Use | Inspected license evidence |
| --- | --- | --- | --- |
| `ref-musetalk-mlx` | [xocialize/musetalk-mlx](https://github.com/xocialize/musetalk-mlx), `c6eb30ebd1ed4d043983209813370153de9346bf` | Required runtime Python package | MIT declared in `pyproject.toml`; this revision has no separate license file |
| `ref-feathertalk` | [anliyuan/FeatherTalk](https://github.com/anliyuan/FeatherTalk), `ace1227ec0a367a983101bfc17d7e5fcf2bfb63f` | Required SCRFD detector module and bundled ONNX file | Apache-2.0 `LICENSE`; that source license is not a separate audit of the detector weight's terms |
| `ref-ml-stable-diffusion` | [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion), `ea2805dc1945be20561c77e5f6d1d9a5a637cda2` | Core ML conversion only | MIT `LICENSE.md` |
| `ref-VoxEMW` | [emwstudio/VoxEMW](https://github.com/emwstudio/VoxEMW), `63560df1512d4778dce86e975954bff5899de692` | Architecture/reference implementation; not imported by the runtime | MIT `LICENSE` |

There were no tracked source modifications in these inspected checkouts. Untracked
agent state is not part of the dependency. If a distribution uses source snapshots
instead, retain the above revisions, upstream notices/licenses, and any actual patch
files; exclude `.git`, `.omc`, downloaded weights, and caches. Do not replace the
MuseTalk port with an arbitrary package solely because its version also says `0.1.0`.

## Python and system tools

The observed interpreter was Python **3.12.9** on macOS. MLX requires Apple Silicon;
the ANE package exporter targets **macOS 15 or newer**. Other platforms and memory
sizes have not been validated. Performance documents contain measurements of one
development machine, not portable throughput or latency guarantees.

Install Python 3.12 and `uv` (or use a Python venv with pip). `ffmpeg` and `ffprobe`
are needed for preparing reference media and for the default browser acceptance
fixtures. Node.js is needed only for the JavaScript regressions / browser smoke tests.
The browser UI itself has no Node build step.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
```

`requirements.txt` pins the direct dependencies observed in the working runtime;
`requirements-coreml.txt` pins the separate export environment. Transitive dependency
resolution is not locked. The optional TypeSafe packages are listed because their
mock tests import the SDK; omitting `TYPESAFE_API_KEY` disables remote semantic checks
while deterministic checks remain active. No TypeSafe key is needed to launch.

## Download local models before offline service startup

Provision these public Hugging Face repositories into the standard cache with the
runtime environment. The explicit commands replace legacy workstation-specific
download shell scripts.

```sh
.venv/bin/hf download Qwen/Qwen3-ASR-1.7B
.venv/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
.venv/bin/hf download mlx-community/MuseTalk-1.5-fp16
.venv/bin/hf download runanywhere/silero-vad-v5 silero_vad.onnx
.venv/bin/hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx
```

For the local brain option, also download:

```sh
.venv/bin/hf download mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
```

Observed cache provenance:

| Model repository | Observed snapshot | Cached model-card license |
| --- | --- | --- |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `7278e1e70fe206f11671096ffdd38061171dd6e5` | Apache-2.0 |
| [mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) | `e7dd0585652209fa0d7783659aad4e8a324de11c` | Apache-2.0 |
| [mlx-community/MuseTalk-1.5-fp16](https://huggingface.co/mlx-community/MuseTalk-1.5-fp16) | `ad54104a0129121fe2ea67471250c0656c985284` | MIT |
| [runanywhere/silero-vad-v5](https://huggingface.co/runanywhere/silero-vad-v5) | `38a8e93669ea8fd4dd5d693bc90d86b7b667d36d` | MIT |
| [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) | `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` | BSD-2-Clause |
| [mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit) | `e9675aa3ca5f900ccef55267914466d55ab325fa` | Apache-2.0 |

These snapshot hashes describe the audited cache. Current runtime loaders ask for
the model ID's default revision, and the commands above retrieve current `main`;
they do not pin those snapshot hashes. An exact archival reproduction additionally
needs a recorded model-cache snapshot and dependency lock. Model-card declarations
are attribution evidence, not a clearance of all component weights or media rights.

## Supply your own reference media

| Required path / setting | Preparation and limitations |
| --- | --- |
| `face.base_video` | Provide your own readable face video and set its project-relative path. The demonstrated path is `web/media/base_fox_listen.mp4`; that clip is not distributed. |
| `face.pre_crop`, `expand`, `view_scale` | Tune for your video; the defaults crop the right side of a television split-screen and do not fit arbitrary media. |
| `face.idle_frame` | Select a visibly neutral, closed-mouth frame **after internal resampling to 25 fps**. The demo value 120 means 4.80 seconds in that reference; it is not a universal safe frame. |
| Persona `ref_wav` / `ref_text` | The included persona points to `assets/ck/ref.wav` and `assets/ck/ref.txt`; supply your own 24 kHz mono reference and matching verbatim transcript. Required to reproduce the configured reference-voice path. The wrapper warns about a default voice when files are missing, but the Base model has no built-in preset voices; unconditioned output is not an accepted substitute and was not validated in this audit. |
| `assets/coreml/musetalk_unet_ane_b1.mlpackage` | Required for face rendering; generate it below. The runtime path is currently fixed. |
| `ref-feathertalk/data_utils/scrfd_2.5g_kps.onnx` | Face detection weight tracked in the pinned upstream checkout. Preserve its provenance; upstream source license alone does not settle weight usage rights. |
| `assets/models/modnet.onnx`, `assets/models/yolo11n-seg.onnx` | Required only when background replacement is enabled. They are not distributed and the project has no complete pinned acquisition/export script for them. |

The default background pipeline uses MODNet matting and a YOLO11n-seg subject gate.
The source identifies the YOLO weights as Ultralytics AGPL-3.0. For the smallest
reproduction, set `face.background.enabled` to `false`; face lip-sync still works
without those two segmentation weights. Alternatively provision compatible models
at those paths and review their terms independently. `SEG_GATE=0` removes the YOLO
stage but still requires MODNet when background replacement is enabled.

To prepare a voice reference from media you can use, this helper writes the two
persona reference files and runs the local ASR model; review its chosen excerpt and
correct the transcript before use:

```sh
.venv/bin/python scripts/make_ref.py path/to/your-audio.wav --start 0 --len 8
```

Use media with the necessary permission for your purpose. Public screenshots show
the prototype's behavior; they do not grant a reusable voice or likeness license.

## Generate the Core ML UNet

Conversion is a separate setup task, not performed automatically by `run.py`. The
exporter imports Apple's source checkout directly; do not install its complete
upstream requirements, which contain older pins that differ from this environment.

```sh
uv venv --python 3.12 .venv-coreml
uv pip install --python .venv-coreml/bin/python -r requirements-coreml.txt
uv pip check --python .venv-coreml/bin/python
mkdir -p assets/coreml/ref
```

The exporter requires `unet_x.npy`, `unet_pe.npy`, and `unet_y.npy` for shape/parity
checking. This source-derived preparatory snippet produces synthetic activations
with the **actual downloaded MuseTalk UNet weights**, without loading private media:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_map
import numpy as np
from huggingface_hub import snapshot_download
from musetalk_mlx.models.unet import UNet2DConditionModel
from musetalk_mlx.utils.weights import load_native

snapshot = Path(snapshot_download('mlx-community/MuseTalk-1.5-fp16', local_files_only=True))
model = UNet2DConditionModel()
load_native(model, snapshot / 'unet.safetensors')
model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
model.eval()
rng = np.random.default_rng(0)
x = rng.standard_normal((1, 8, 32, 32)).astype(np.float32)
pe = rng.standard_normal((1, 50, 384)).astype(np.float32)
y = model(mx.array(x), mx.array([0]), mx.array(pe))
mx.eval(y)
out = Path('assets/coreml/ref')
out.mkdir(parents=True, exist_ok=True)
for name, value in [('x', x), ('pe', pe), ('y', np.asarray(y.astype(mx.float32)))]:
    np.save(out / f'unet_{name}.npy', value)
PY

.venv-coreml/bin/python scripts/export_unet_ane.py \
  --ref-dir assets/coreml/ref \
  --out assets/coreml/musetalk_unet_ane_b1.mlpackage
```

The preparatory snippet was checked against the pinned port's call signature but
has not been run as part of this publication audit. The export script checks Torch
and Core ML output against the MLX reference, prints numerical differences, warms
the model, benchmarks 200 predictions, and attempts a Core ML compute-plan report.
Inspect those results; it does not impose a strict automatic parity tolerance.

The exporter currently searches the standard `~/.cache/huggingface/hub` path with
a glob and selects the first MuseTalk snapshot. Custom cache roots or multiple
snapshots require resolving that path/revision explicitly before conversion. Keep
the reference tensors and exporter on the same model revision. The non-ANE
`export_unet_coreml.py` produces a different package name/layout and is **not** the
package selected by the current face runtime.

Generated `.mlpackage` files, reference tensors, face/alpha caches in `assets/cache`,
and ONNX Runtime Core ML caches are local artifacts, not repository source.

## Configure and launch

Copy `configs/assistant.json` to an ignored personal config if you need different
assets, ports, or persona settings. For an initial **voice-only** run set
`face.enabled` to `false`; to reproduce the face, provide the video and converted
UNet, and disable background replacement unless its models were provisioned.

For the demonstrated remote brain, create an ignored `.env.local` locally:

```dotenv
LLM_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-pro
LLM_API_KEY=replace-with-your-own-key
```

No credentials belong in JSON or Git. Existing shell environment values take
precedence over `.env.local`. The client requests `thinking.type=disabled` for
DeepSeek and ignores reasoning output; local Qwen uses
`chat_template_kwargs.enable_thinking=false`. For a different provider, use
`llm_extra_body` with its actual supported request fields.

```sh
# Start only after all required local models/assets are prepared:
.venv/bin/python scripts/service.py start
.venv/bin/python scripts/service.py status
curl --fail http://127.0.0.1:8000/health
```

Open `http://127.0.0.1:8000`, press Start, allow microphone access, and use headphones
for a real conversation / interruption check. Browser microphone access requires a
secure context; loopback HTTP is supported. Avoid exposing the server publicly: the
prototype has no authentication or multi-user deployment acceptance.

To force the JSON's local Qwen settings, ignoring remote environment overrides:

```sh
.venv/bin/python scripts/service.py restart --local-llm
```

For a custom configuration, use `--config configs/assistant.local.json` consistently
with service commands. Foreground debugging is available with
`.venv/bin/python run.py configs/assistant.local.json`; Ctrl-C performs cleanup.
`scripts/service.py stop` stops the service it recorded; it refuses unrelated PIDs.
Legacy download/restart scripts may contain development-machine paths and are not
the clean-clone entry points.

If face preparation fails, the application can start with voice only and reports
`face_ready=false`. If the brain fails warmup / returns no content, local components
can remain ready while `llm_ready=false` and service status is degraded. Remote
network/content wait defaults to 20 seconds, with a separate 12-second warmup bound.
A ready response does not imply every external API request will succeed.

## Verification and evidence boundaries

Lightweight regressions do not load models or call cloud services:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
node tests/test_face_playback.cjs
```

The brain-only smoke test uses the configured model and **makes actual API requests**
for a remote configuration. It does not silently choose a local fallback:

```sh
.venv/bin/python scripts/brain_smoke.py --expect-model deepseek-v4-pro
```

Backend acceptance needs a speech fixture you supply; it checks received PCM and
content video frames, not sound heard by a person:

```sh
ffmpeg -i path/to/utterance.wav -ar 16000 -ac 1 -f s16le fixture.pcm
.venv/bin/python scripts/e2e_test.py fixture.pcm 1500
```

Browser acceptance additionally needs Playwright and Chromium. They are test tools,
not runtime dependencies; no package/browser lock is currently included. Install
them in your own Node environment and set `VOXCK_PLAYWRIGHT_PATH` if the package is
not discoverable. With the prepared service running:

```sh
node scripts/browser_smoke.mjs --fixture path/to/utterance.wav \
  --interrupt-fixture path/to/barge-in.wav --out logs/browser-acceptance
```

That test supplies a simulated microphone `MediaStream`, exercises the app's real
AudioWorklet, WebSocket, WebAudio, canvas rendering, and playback acknowledgements.
It records evidence of software scheduling and frame drawing, not a real microphone,
physical speaker, human hearing, or flawless avatar quality. Without explicit
fixtures, its default path requires macOS `say` plus ffmpeg.

Remaining reproducibility gaps are a resolved dependency lock, model revision
selection in runtime/export paths, acquisition/permissions for reference media,
pinned segmentation-model preparation, browser test dependency pinning, and
fresh-machine end-to-end acceptance. Screenshots and earlier measurements support
the demonstrated prototype; they do not remove those setup requirements.
