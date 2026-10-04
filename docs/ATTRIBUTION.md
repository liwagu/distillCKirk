# Attribution and third-party provenance

## Primary inspiration: VoxEMW

This project was inspired by **[VoxEMW, created by emwstudio](https://github.com/emwstudio/VoxEMW)**. VoxEMW's implementation and engineering notes provided the starting reference for an interactive speaking avatar: voice activity detection, turn completion, speech recognition, an LLM, streaming speech synthesis, and audio-driven video.

We especially credit its discussion of streaming orchestration, interruption handling, audio/video scheduling against a clock, and recording only the portion of a reply the listener actually heard. The reference checkout reviewed for this project was commit [`63560df1512d4778dce86e975954bff5899de692`](https://github.com/emwstudio/VoxEMW/tree/63560df1512d4778dce86e975954bff5899de692).

VoxEMW is an architectural reference, rather than a package imported by the current `voxck` runtime. This project implements its Mac/browser pipeline around Qwen speech models, MuseTalk, MLX, and Core ML, and currently uses a DeepSeek API for conversation, with an explicit local Qwen option. It does not run VoxEMW's SoulX video server. VoxEMW's code is [MIT licensed](https://github.com/emwstudio/VoxEMW/blob/63560df1512d4778dce86e975954bff5899de692/LICENSE), copyright © 2026 emwstudio.

## Code used for rendering and conversion

| Upstream project | Contribution to this project | License/provenance |
|---|---|---|
| [MuseTalk, TMElyralab / Tencent Music Entertainment](https://github.com/TMElyralab/MuseTalk) | The pretrained audio-driven lip-sync model underlying the current renderer. | Upstream describes its code as MIT and publishes separate terms for its models and other model dependencies; see [Disclaimer/License](https://github.com/TMElyralab/MuseTalk#disclaimerlicense). |
| [musetalk-mlx, xocialize / MVS Collective](https://github.com/xocialize/musetalk-mlx) | Runtime MLX port: MuseTalk pipeline, VAE, and audio features. The project loads the published `mlx-community/MuseTalk-1.5-fp16` conversion. | MIT is declared in the [README](https://github.com/xocialize/musetalk-mlx#license) and package metadata. Reviewed commit: [`c6eb30ebd1ed4d043983209813370153de9346bf`](https://github.com/xocialize/musetalk-mlx/tree/c6eb30ebd1ed4d043983209813370153de9346bf). That checkout does not include a standalone `LICENSE` file; retain its declaration and upstream notices when redistributing it. |
| [FeatherTalk, anliyuan and contributors](https://github.com/anliyuan/FeatherTalk) | The current renderer imports its `data_utils/detect_face.py` SCRFD helper and uses the accompanying face detector for preprocessing. FeatherTalk was also evaluated as an alternative renderer. Its personalized talking-head training/inference network is not the current lip-sync backend. | [Apache-2.0](https://github.com/anliyuan/FeatherTalk/blob/ace1227ec0a367a983101bfc17d7e5fcf2bfb63f/LICENSE). Reviewed commit: [`ace1227ec0a367a983101bfc17d7e5fcf2bfb63f`](https://github.com/anliyuan/FeatherTalk/tree/ace1227ec0a367a983101bfc17d7e5fcf2bfb63f). Detector weights require a separate provenance check, described below. |
| [Apple ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) | Build/export dependency: `scripts/export_unet_ane.py` uses its Neural Engine-oriented UNet implementation to convert MuseTalk weights to Core ML. At runtime the renderer loads the converted Core ML artifact; it is not running the full Stable Diffusion image-generation pipeline. | [MIT](https://github.com/apple/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/LICENSE.md), copyright © 2024 Apple Inc. Reviewed commit: [`ea2805dc1945be20561c77e5f6d1d9a5a637cda2`](https://github.com/apple/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2). |

The Core ML conversion changes the execution format/implementation of an existing pretrained network. It is not training a new Charlie Kirk visual model.

## Models and inference libraries

Credit belongs to the original model authors and to the maintainers of their inference ports and conversions. Model licenses are separate from the license of this application's code.

| Component | Source and role | License information |
|---|---|---|
| Silero VAD | [Silero team / snakers4](https://github.com/snakers4/silero-vad); speech activity detection, using an ONNX model. | Upstream MIT. |
| Smart Turn v3 | [Pipecat / Daily](https://huggingface.co/pipecat-ai/smart-turn-v3); semantic end-of-turn detection. | Model card: BSD-2-Clause. |
| Qwen3-ASR-1.7B | [Qwen team](https://huggingface.co/Qwen/Qwen3-ASR-1.7B); speech recognition through [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr). | Qwen model card and the installed port's package metadata: Apache-2.0. |
| Qwen3-TTS-12Hz-1.7B-Base | [Qwen team](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base); reference-audio-conditioned synthesis through [mlx-audio](https://github.com/Blaizzy/mlx-audio). The configured conversion is [mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit). | Original model and conversion cards: Apache-2.0. `mlx-audio` package metadata: MIT. |
| MuseTalk 1.5 MLX | [MuseTalk authors](https://github.com/TMElyralab/MuseTalk) and [MVS Collective's MLX conversion](https://huggingface.co/mlx-community/MuseTalk-1.5-fp16); lip sync. | Conversion card: MIT, with dependency models retaining their own terms. |
| MuseTalk's VAE and audio encoder | [Stability AI sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) and [OpenAI Whisper](https://github.com/openai/whisper); used inside the MLX port, not developed or trained by this project. | Consult each original model distribution and its notices. The VAE card labels it MIT; Whisper's [source-code license](https://github.com/openai/whisper/blob/main/LICENSE) is separate from model-distribution metadata. |
| Local conversation option | [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), using an MLX-community 4-bit conversion with MLX-LM when explicitly selected. | Original model card: Apache-2.0; retain the conversion's own model card and notices. |

We also acknowledge the maintainers of [MLX](https://github.com/ml-explore/mlx), [MLX-LM](https://github.com/ml-explore/mlx-lm), [Core ML Tools](https://github.com/apple/coremltools), [ONNX Runtime](https://github.com/microsoft/onnxruntime), and the project's other Python/browser dependencies. This page records the main provenance relationships, rather than an exhaustive license inventory for every transitive package.

The configured DeepSeek conversation API and optional TypeSafe semantic guard are external services. Their service terms apply separately; using their APIs does not mean their model weights are included or licensed by this repository.

### SCRFD detector weights: provenance remains unresolved

FeatherTalk's Apache-2.0 code license does not establish a separate license for every bundled pretrained weight file. The exact origin and redistribution terms of `scrfd_2.5g_kps.onnx` used by this project have not been established here. [InsightFace's published license policy](https://github.com/deepinsight/insightface#license) distinguishes its MIT code from pretrained models restricted to non-commercial research. If this detector comes from those pretrained releases, that model policy must also be considered. Do not treat the helper's code license as a blanket commercial-use or redistribution grant for the detector.

## Persona research: a source ledger and prompts, not weight training

The persona is maintained in [`personas/ck-gender.md`](../personas/ck-gender.md). Its body is loaded by [`voxck/persona.py`](../voxck/persona.py) and supplied as a system prompt. It combines a public-statement ledger, confidence labels, rhetorical-style instructions, and limits on extrapolation.

The term “distillation” in this project refers to researching and organizing a persona for prompting. **This project has not fine-tuned an LLM on Charlie Kirk, produced Charlie-specific LLM weights, or trained a model to reproduce his mind.** Reference-audio voice conditioning and lip-sync inference are also distinct from LLM weight training.

The research dossiers [`ck-positions-gender.json`](../corpus/raw/ck-positions-gender.json) and [`ck-style-rhetoric.json`](../corpus/raw/ck-style-rhetoric.json) preserve source links. They draw on publicly available speeches, interviews, debate recordings, transcripts, and secondary reporting. Their links include Charlie Kirk's public posts, podcast transcripts, Oxford Union material, and reporting or analysis from outlets such as Axios, Fox News, Media Matters, and Snopes. Those publishers and original speakers retain ownership of their material; research citation is not a transfer of rights.

The persona-methodology research also consulted [nuwa-skill by alchaincyf](https://github.com/alchaincyf/nuwa-skill), particularly its extraction framework and fidelity scorecard. It is a methodology reference, not a dependency executed by this application; upstream declares MIT.

Ledger quotations, research summaries, and analyst-written “warrants” serve different purposes. A warrant is an interpretation of a reasoning pattern, not an attested quotation. Source confidence labels do not certify the underlying political/statistical claim as true. Generated replies are new model output, not newly discovered statements by Charlie Kirk, and the project does not claim his endorsement or affiliation.

## Media and license preservation

The public source export keeps the four reference projects above as Git submodules pinned to the recorded revisions. It does not bundle model weights, original reference clips, reference voice inputs, or local caches. The selected demo screenshots are frames from the project's published demonstration, rather than a distribution of its original source-media collection.

Reference video, voice samples, photographs, broadcast footage, quoted material, and demo recordings have their own provenance and rights. In particular, the Fox/Jesse Watters reference footage used during development is third-party media. This attribution page and any license for the application's original code do not grant rights in those recordings, a person's voice or likeness, or the resulting depiction. Do not label third-party reference media as original project assets merely because the processing code is published.

When redistributing upstream code or derived artifacts, include the applicable license texts and copyright notices, preserve relevant `NOTICE` files, and identify modifications where the upstream license requires it. A link or acknowledgement alone does not replace those notices. MIT dependencies retain their copyright and permission notices; Apache-2.0 material retains its license, relevant notices, and modification notices; BSD-2-Clause material retains its required notices.

This document is an acknowledgement and provenance record. It does not select a license for this project's original code, relicense third-party material, or assert that all dependencies and media are cleared for every form of reuse.

Copies of the inspected VoxEMW, FeatherTalk, and Apple license texts are retained in [`licenses/`](../licenses/), alongside the pinned submodules. Application-specific integration changes live in `voxck/` and `scripts/`; the pinned upstream checkouts themselves were not modified.
