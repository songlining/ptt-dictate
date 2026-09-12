# ptt-dictate

Hold-to-talk dictation on a local ASR model via MLX — hold the key, speak,
release, and the text lands in whatever app has focus.

One warm daemon, no cloud round-trip, ~0.3–0.5s from key release to pasted
text, plus a floating status pill while you speak.

Default model is **Qwen3-ASR-1.7B** (8-bit MLX) in **batch** mode: the whole
utterance is transcribed once on release. That is both more accurate and faster
than a streaming model for push-to-talk, because nothing has to wait for a
fixed audio window to fill. **Streaming** models (VibeVoice-ASR-Streaming) are
still supported via `--mode stream` — they trade accuracy for live partial text
in the pill, at the cost of a hard 3.5s floor before any text can appear.

## How it works

- A Quartz event tap watches one hotkey and **swallows** it, so an app that
  binds the same key for its own hold-to-talk does not also fire. (Some chat
  clients bind right Option for hold-to-talk: without swallowing, one key press
  gives you a voice message *and* the pasted text. Mature dictation tools ship
  the same behaviour, as a `block_keys`-style setting.)
  `--key-passthrough` opts out.
- On press: the mic stream opens on the *current* default input device and audio
  is buffered; the pill appears with a live mic meter.
- On release, either
  - **batch** (default): one `model.generate()` call over the buffered utterance
    (~0.3–0.5s for a 10s clip), or
  - **streaming** (`--mode stream`): a final padded step on the tail, with
    partials printed/shown as 2.93s windows land while you speak.
- The text is then put on the clipboard, Cmd-V is posted, and the previous
  clipboard is restored. Clipboard-based because CGEvent keyboard injection
  cannot type Chinese.

### The status pill

A borderless, non-activating `NSPanel`: dark rounded pill, bottom centre,
showing the prompt text, then (streaming mode only) the tail of the live
partial, with a 5-bar mic meter on the right (silence = flat dots, normal
speech = bars at ~80%). Click-through, above normal windows, and never takes
focus from the app you are dictating into.

In batch mode there is no partial text to show — the model only runs on release
— so the pill is a listening indicator plus the meter.

The process runs as an **accessory** app (no Dock icon, never activated) — so
the panel needs `setHidesOnDeactivate_(False)` or it flashes on press and
vanishes, which is exactly what the default NSPanel behaviour does here.

## Requirements

- **A Python env** with `mlx-audio[stt]`, `sounddevice` and `pyobjc-framework-Quartz`:

  ```bash
  uv venv ~/models/venv-mlx-audio
  uv pip install --python ~/models/venv-mlx-audio/bin/python \
      "mlx-audio[stt]" sounddevice pyobjc-framework-Quartz
  ```

  (Swap in any interpreter; point `PTT_PYTHON` at it for `install.sh`.)
- **Model**: any checkpoint `mlx-audio` can load. Default is
  `mlx-community/Qwen3-ASR-1.7B-8bit` (2.3GB, auto-downloaded on first run);
  streaming checkpoints (e.g. a VibeVoice-ASR-Streaming 8-bit MLX conversion)
  work too and are auto-detected. `--model` takes a path or an HF repo id.
- **Permissions**: Accessibility + Microphone for the interpreter (TCC prompts
  on first use).

## Use

```bash
./install.sh                    # login daemon, defaults to --key right_option
./install.sh --key f13          # or a key nothing else wants

# or run it in the foreground
~/models/venv-mlx-audio/bin/python ./ptt_dictate.py --key right_option
```

Then hold the key, speak, release.

| flag | default | notes |
|------|---------|-------|
| `--key` | `right_option` | `left_option`, `right_command`, `left_command`, `right_shift`, `left_shift`, `right_control`, `left_control`, `fn`, or `f13`–`f19` |
| `--key-passthrough` | off | let other apps see the hotkey too. Default (off) swallows it; turning this on costs you nothing functionally but re-introduces double-firing in apps that bind the same key |
| `--model` | `mlx-community/Qwen3-ASR-1.7B-8bit` | path or HF repo id; streaming checkpoints auto-switch to `--mode stream` |
| `--mode` | `auto` | `auto` \| `stream` \| `batch`. `auto` streams only for checkpoints with window/chunk metadata |
| `--context` | `""` | names/jargon. In batch mode these become **real hotwords** (`hotwords=[...]`) when the model supports it. Merges with `--context-file` |
| `--context-file` | `""` | file of hotwords, **re-read on every press** so edits apply without a restart |
| `--min-rms` | `0.002` | batch: skip a capture whose loudest 100ms is below this. Catches a muted mic; deliberately low so a quietly-spoken word is never dropped |
| `--transcribe-file` | `""` | transcribe a file and exit — smoke test, needs no hotkey and may run alongside the daemon |
| `--live-file` | `""` | append live partials to a file |
| `--tail-ms` | `200` | extra mic time after release, to catch the last syllable |
| `--device` | system default | input device index or name |
| `--dry-run` | off | print the text instead of pasting it |
| `--paste-delay` | `0.6` | seconds the transcript stays on the clipboard before the old one is restored |
| `--no-overlay` | off | no status pill (headless / scripted use) |
| `--overlay-text` | `直接说` | pill text while waiting for speech |
| `--self-test` | — | window bookkeeping, text cleaning, meter curve — no model load |

Run it attended first (`--dry-run`) to confirm the hotkey and the transcript
before letting it paste into live apps.

## Hotwords

ASR mis-hears two things above all: names, and English terms spoken inside a
Chinese sentence (measured: `README` came out as *rhythm*, `parameter` as
*Perimeter*, `Kubernetes` as *UberNitz*). Hotwords bias the decoder toward the
right spelling, and Qwen3-ASR takes them as a first-class argument rather than
as instructions embedded in a prompt.

They live in a plain file:

```
~/.config/ptt-dictate/hotwords.txt
```

```
# one term per line; # starts a comment
Alex Chen          # your name, colleagues, customers
Kubernetes
Postgres
```

Names of people and products are the entries that earn their place; anything the
model already spells correctly is noise.

The file is **re-read on every press**, so editing it applies to your very next
dictation — no reinstall, no daemon restart. Phrases work (`Vault Radar`);
splitting is on commas and newlines, not spaces, so a two-word term stays one
term. `--context "a, b"` merges with the file, `--context-file PATH` relocates
it, and `install.sh` seeds it from `--context` on first run (override the path
with `PTT_HOTWORDS`).

Keep the list to terms you actually dictate. A short, specific list is the point;
nobody has measured what a long one does, and a list of everything you might ever
say is unlikely to help.

## Runtime facts (measured on an M4 Pro, 8-bit MLX, 20s/10s clips)

| | Qwen3-ASR 1.7B *(default)* | Qwen3-ASR 0.6B | VibeVoice-Stream 1.5B | VibeVoice-Stream 7B |
|---|---|---|---|---|
| resident memory | 2.50GB | 1.09GB | 3.07GB | 7.57GB |
| 10.3s Chinese | **0.46s** (22×) | 0.24s (44×) | 1.23s (8×) | 2.54s (4×) |
| 20s English | 0.96s (21×) | 0.47s (42×) | 2.23s (9×) | 4.98s (4×) |
| 5s utterance, warm | 0.31s | 0.15s | 0.72s | 2.49s |
| live partials | no | no | yes (from ~3.5s) | yes |
| input rate | 16 kHz | 16 kHz | 24 kHz | 24 kHz |

Accuracy on the same audio: Qwen 1.7B got every technical term right
(`Terraform`, `Vault`, `dynamic secrets`, `Kubernetes`); the streaming 1.5B
produced `MakeSecrets` and `UberNitz`, and heard "parameter" as "Perimeter".
The 0.6B is 2× faster and 4× smaller but mis-spells the same vocabulary
(`Teraform`, `Volt`), so 1.7B is the default. These are clean TTS clips, which
flatter every model — treat the ranking as meaningful, the absolute scores as not.

**Warmup caveat:** the *first* inference after the daemon starts costs an extra
~1–2s (MLX compiles Metal kernels). It is paid once per process, not per press —
measured first-step times across three consecutive presses were 0.36/0.36/0.38s
(1.5B) and 0.88/0.88/1.00s (7B).

On silence: Qwen3-ASR returns an empty string for both digital silence and room
tone, so it invents nothing on a stray tap. Whisper-class models do hallucinate,
which is why the cheap `--min-rms` gate exists.

Text handling: the streaming models prefix chunks with `Speaker 0:` and emit
`[Silence]`/`[Noise]` markers — both are stripped, and nothing is pasted when the
result is empty. Streaming output is content-accurate but not verbatim
(chunk-granular boundaries, light punctuation).

## Install as a login daemon

`install.sh` generates `~/Library/LaunchAgents/local.ptt-dictate.plist` from
this repo's own location and `$HOME`, runs it at login, keeps it alive, and
logs to `~/Library/Logs/ptt-dictate/daemon.log`. The model stays resident
(~3GB). Nothing machine-specific is committed — `launchctl bootstrap` cannot
expand `~`, so the paths are written at install time.

```bash
./install.sh --key right_option --context "Kubernetes, Postgres"
./uninstall.sh                                                   # stop + remove

launchctl print gui/$(id -u)/local.ptt-dictate | grep -E 'state|pid'   # status
launchctl bootout gui/$(id -u)/local.ptt-dictate                       # stop only
```

Editing the script requires a bootout + bootstrap to take effect. Because
`KeepAlive` restarts it, `kill` is not how you stop it — use `bootout`.

## Gotchas

- **The hotkey is swallowed, so it is no longer usable as a modifier.** With
  right Option consumed by the daemon, Option+key on the *right* key no longer
  types special characters. That is the price of not double-firing in apps with
  their own hold-to-talk, and it is what other dictation tools do too. If you need the modifier
  back, use `--key-passthrough` and accept the collision — or bind a key nothing
  else wants (`--key f13`).

- **A Bluetooth headset coming or going changes the default input.** PortAudio
  caches the device list at init and never re-reads it, so a vanished default
  device made every press fail with `-10851 Invalid Property Value` and capture
  silence — which surfaced as a run of `→ (nothing)` with no explanation. The
  other half of the trap: once the headset returned we would have stayed on the
  built-in mic forever. A full device re-scan costs ~3ms, so every press now
  re-scans and opens whatever is *currently* the default input, with one retry
  after a second re-scan if the open fails. A press that still cannot get a mic
  aborts with `! mic unavailable — press skipped` instead of pretending, and the
  log prints `mic: <name>` whenever the device in use changes.
- **macOS can disable the event tap behind your back** — on callback timeout or
  during secure input it stops delivering events *silently*: the process looks
  healthy, idles at 0% CPU, and hears nothing until restarted. Symptom is
  "it worked, then stopped", which is easy to misdiagnose as a broken key or
  permission. The callback re-arms on `kCGEventTapDisabledByTimeout` /
  `...ByUserInput`, and a 0.5s health-check timer (the same one that keeps
  SIGINT alive) re-arms it if `CGEventTapIsEnabled` says otherwise. Both log
  `! tap disabled — re-armed` so the next occurrence is visible instead of
  silent.
- **Never name a method of an `NSObject` subclass `release`** (or `press`). It
  shadows `-release`, and since `performSelectorOnMainThread:` retains/releases
  its receiver, the override re-enters itself forever: a permanent ~100% CPU
  spin from the instant the object exists — even with the window never created
  or shown. This cost a long debugging session; the pill methods are `show_pill`
  / `hide_pill` for that reason.
- **pyobjc turns EVERY underscore into a colon** when deriving a selector:
  a method called via `performSelectorOnMainThread_` must map to exactly one
  colon for one argument (`hideNow_` → `hideNow:`). `do_hide_pill_` becomes
  `do:hide:pill:` and fails with `BadPrototypeError` at class-creation time.
  Leading-underscore names (`_build`) are not registered as selectors at all.
- **Something else on the same key**: only one app can own a hotkey. If a
dictation or input tool already holds it, quit it and disable its autostart, or
it takes the key back at next login — check `~/Library/LaunchAgents/` and any
input-method settings. On macOS the right-Option key is a popular choice, so a
pre-existing dictation tool is the usual culprit.
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

- On-screen editing of the transcript, per-app hotkeys, and a tray/menu item —
the pill shows state only, and text goes straight to the clipboard.
