# ptt-dictate

Hold-to-talk dictation on a local VibeVoice-ASR-Streaming model (MLX 8-bit) —
hold the key, the model transcribes as you speak, release and the text lands in
whatever app has focus.

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

- **A Python env** with `mlx-audio[stt]`, `sounddevice` and `pyobjc-framework-Quartz`:

  ```bash
  uv venv ~/venv
  uv pip install --python ~/venv/bin/python \
      "mlx-audio[stt]" sounddevice pyobjc-framework-Quartz
  ```

  (Swap in any interpreter; point `PTT_PYTHON` at it for `install.sh`.)
- **Model**: a VibeVoice ASR **streaming** checkpoint in MLX form (the local
  8-bit conversion used here is 2.8GB). `--model` takes a path or any HF repo id
  that mlx-audio can load as `vibevoice_asr`.
- **Permissions**: Accessibility + Microphone for the interpreter (TCC prompts
  on first use).

## Use

```bash
./install.sh                    # login daemon, defaults to --key right_option
./install.sh --key f13          # or a key nothing else wants

# or run it in the foreground
~/venv/bin/python ./ptt_dictate.py --key right_option
```

Then hold the key, speak, release.

| flag | default | notes |
|------|---------|-------|
| `--key` | `right_option` | `left_option`, `right_command`, `left_command`, `right_shift`, `left_shift`, `right_control`, `left_control`, `fn`, or `f13`–`f19` |
| `--model` | 1.5B streaming | path to any streaming checkpoint |
| `--context` | `""` | hotwords/names, e.g. `"Kubernetes, Postgres, Terraform"` — your own name, colleagues and customer names are the useful ones |
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

## Runtime facts (measured, 1.5B 8-bit)

| | |
|---|---|
| sample rate | 24 kHz |
| window / advance | 3.47s / 2.93s (chunk 22 frames + 4 lookahead, ratio 3200) |
| cost per step | ~0.3s (file-direct, 3-chunk clip in 1.01s total) |
| model load | ~0.8s from disk (the daemon keeps it resident) |
| first partial | after ~3.5s of speech |
| mic RMS | silence ~0.001, speech p90 ~0.06 |
| idle CPU | 0.0% (the pill costs ~5% while it is on screen) |

Text handling: the model prefixes chunks with `Speaker 0:` and emits
`[Silence]`/`[Noise]` markers — both are stripped, and nothing is pasted when
the result is empty.

Streaming output is content-accurate but not verbatim (chunk-granular
boundaries, light punctuation).

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
