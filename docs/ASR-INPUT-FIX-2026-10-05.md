# Completed speech survives reply interruption

The previous microphone route put ASR and answer generation in the same cancellable task. When speech resumed before ASR published its result, the preceding completed utterance vanished from both the UI and model context.

The live route now archives completed PCM before queueing it, transcribes each accepted clip in a separate FIFO worker, and only cancels the reply task when speech resumes. A new response starts after the queue drains and the user has stopped speaking. Consecutive user fragments are combined in the model request; real history objects remain separate so late playback acknowledgments retain their correct anchors. Complete local history is preserved; the model still uses a bounded recent context window.

`conversations/session-*/audio/*.wav` stores completed 16 kHz microphone segments. `events.jsonl` stores full transcripts and the parts of replies actually played, with user clip anchors for late updates. Disconnect drains accepted input. Empty/failed ASR is reported explicitly and cannot trigger an answer to a stale earlier fragment. Transport-close errors do not abort the input queue.

## Restoration and publication boundaries

A validated `conversations/resume-next.json` seed loads user/assistant messages into the next connection and displays them without starting a reply. The seed is consumed once and archived. Invalid seeds remain available for correction. See the README for the format. Session directories use mode 0700 and files use 0600; the entire `conversations/` directory is ignored by Git.

This implements local archives and explicit restoration, not automatic long-term memory across every future Stop/Start. Speech lost before the fix cannot be recreated if no recording remains. Real recordings, transcripts, resume seeds, and private validation evidence are excluded from this public repository.

## Validation

The development fix passed 55 Python tests and the existing 6-scenario frontend playback check. A live service check accepted two synthetic speech clips in one WebSocket packet, produced both complete Qwen3-ASR transcripts, then produced a DeepSeek reply plus audio and video packets. This verifies the input/cancellation path; it does not guarantee perfect recognition for arbitrary microphones or accents. The live check used the existing DeepSeek V4 Pro configuration with thinking disabled; its private logs are not bundled here.

Key files: `voxck/orchestrator.py`, `voxck/conversation.py`, `tests/test_asr_ingestion.py`, `tests/test_conversation_journal.py`, `web/app.js`, `web/index.html`.
