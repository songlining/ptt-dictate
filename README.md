# ptt-dictate

Hold-to-talk dictation on the local VibeVoice-ASR-Streaming 1.5B (MLX) — the
local replacement for a pre-existing dictation app's push-to-talk, with the same feel: hold the key,
the model transcribes as you speak, release and the text lands in whatever app
has focus.

One warm daemon, no cloud round-trip, ~0.3s from key release to pasted text,
plus a floating status pill while you speak.

## How it works

- A Quartz **listen-only** event tap watches one hotkey (no keystroke swallowed).
- On press: mic → 2.93s streaming steps into a model that stays resident, live
  partials printed as they land and shown in the status pill.
- On release: one padded flush step on the tail (~0.3s), then the text is put
  on the clipboard, Cmd-V is posted, and the previous clipboard is restored.
  Clipboard-based because CGEvent keyboard injection cannot type Chinese.

### The status pill

A borderless, non-activating `NSPanel`: dark rounded pill, bottom centre,
showing the prompt text until speech arrives, then the tail of the live
partial, with a 5-bar mic meter on the right (silence = flat dots, normal
speech = bars at ~80%). Click-through, above normal windows, and never takes
focus from the app you are dictating into.

The process runs as an **accessory** app (no Dock icon, never activated) — so
the panel needs `setHidesOnDeactivate_(False)` or it flashes on press and
vanishes, which is exactly what the default NSPanel behaviour does here.

## Requirements

Everything is reused from the existing local audio setup — nothing to install
except one pyobjc module:

- **venv**: `~/venv` (mlx + mlx-audio + sounddevice; shared
  with `~/models/another-captions-script` and the `another-skill` skill)
- **pyobjc**: `uv pip install --python ~/venv/bin/python pyobjc-framework-Quartz`
- **model**: `~/models/vibevoice-asr-streaming-1.5b-mlx-8bit` (2.8GB), or the 7B
  via `--model ~/models/VibeVoice-ASR-Streaming-7B-mlx-8bit`
- **permissions**: Accessibility + Microphone for the interpreter (TCC prompts
  on first use)

## Use

```bash
~/venv/bin/python ~/ptt-dictate/ptt_dictate.py --key right_option
```

Then hold the key, speak, release.

| flag | default | notes |
|------|---------|-------|
| `--key` | `right_option` | `left_option`, `right_command`, `left_command`, `right_shift`, `left_shift`, `right_control`, `left_control`, `fn`, or `f13`–`f19` |
| `--model` | 1.5B streaming | path to any streaming checkpoint |
| `--context` | `""` | hotwords/names, e.g. `"Kubernetes, Postgres, Terraform"` |
| `--live-file` | `""` | append live partials to a file |
| `--tail-ms` | `200` | extra mic time after release, to catch the last syllable |
| `--device` | system default | input device index or name |
| `--dry-run` | off | print the text instead of pasting it |
| `--no-overlay` | off | no status pill (headless / scripted use) |
| `--overlay-text` | `直接说` | pill text while waiting for speech |
| `--self-test` | — | window bookkeeping, text cleaning, meter curve — no model load |

Run it attended first (`--dry-run`) to confirm the hotkey and the transcript
before letting it paste into live apps.

## Runtime facts (measured, 1.5B 8-bit)

| | |
|---|---|
| sample rate | 24 kHz |
| window / advance | 3.47s / 2.93s (chunk 22 frames + 4 lookahead, ratio 3200) |
| cost per step | ~0.3s (file-direct, 3-chunk clip in 1.01s total) |
| model load | ~0.8s from disk (the daemon keeps it resident) |
| first partial | after ~3.5s of speech |
| mic RMS | silence ~0.001, speech p90 ~0.06 |

Text handling: the model prefixes chunks with `Speaker 0:` and emits
`[Silence]`/`[Noise]` markers — both are stripped, and nothing is pasted when
the result is empty.

Streaming output is content-accurate but not verbatim (chunk-granular
boundaries, light punctuation).

## Gotchas

- **a pre-existing dictation app must not hold the same key** — it ships bound to right Option, so quit
  it (or disable `~/Library/LaunchAgents/a pre-existing dictation app.plist`) before using
  `--key right_option`. The script warns when it detects a pre-existing dictation app running.
- **Modifier keys and the flag mask**: down/up is read from the event's modifier
  flags, so holding *both* option keys and releasing only the bound one will not
  register a release until both are up. Bind a non-modifier (`f13`) if that
  matters.
- **Accessibility**: the tap and the Cmd-V post both need it. Granted per
  interpreter path, so running under a LaunchAgent rather than a terminal may
  trigger a fresh prompt.
- **The window protocol matters**: window k must cover `[k*ADV, k*ADV+WIN)`.
  Feeding a wider overlap makes the model re-transcribe audio it already
  emitted — that bug is what the `feed`/`flush` split and the self-test guard.
- **Ctrl-C quits** — via a 0.5s idle timer, because the AppKit run loop would
  otherwise defer the signal forever and the daemon would look unkillable.
- **The pill is fixed-size** (380×44pt) and truncates long text from the left
  (`…` prefix) — the most recent words are the ones worth showing.

## Not implemented

- Autostart at login (a LaunchAgent + a 2.8GB resident model)
