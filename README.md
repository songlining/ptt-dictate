# ptt-dictate

Hold-to-talk dictation on a local ASR model via MLX — hold the key, speak,
release, and the text lands in whatever app has focus.

One warm daemon, no cloud round-trip. Release→paste is typically ~0.6–0.8s for a
short phrase and grows with how long you speak (~1.2s after 20s of continuous
dictation), because the whole utterance is transcribed on release. Plus a
floating status pill while you speak.

(Measured with the repo's own log line, `[release→paste …]`, on an M4 Pro. The
four-model table further down quotes **transcribe time only** — it excludes the
200ms tail, the audio teardown and the paste, so a model's row is not the
user-visible latency.)

Default model is **Qwen3-ASR-1.7B** (8-bit MLX) in **batch** mode: the whole
utterance is transcribed once on release. That is both more accurate and faster
than a streaming model for push-to-talk, because nothing has to wait for a
fixed audio window to fill. **Streaming** models (VibeVoice-ASR-Streaming) are
still supported via `--mode stream` — they trade accuracy for live partial text
in the pill, at the cost of a hard ~3.5s floor before the *first* text can appear
(the model needs a full 3.47s window before it can emit anything, and after that
partials land every 2.93s).

## How it works

- A Quartz event tap watches one hotkey and **swallows** it, so an app that
  binds the same key for its own hold-to-talk does not also fire. (Some chat
  clients bind right Option for hold-to-talk: without swallowing, one key press
  gives you a voice message *and* the pasted text. Mature dictation tools ship
  the same behaviour, as a `block_keys`-style setting.)
  `--key-passthrough` opts out, at the cost of that key no longer working as a
  modifier.
- On press: the mic stream opens on the *current* default input device and audio
  is buffered; the pill appears with a live mic meter.
- On release, either
  - **batch** (default): one `model.generate()` call over the buffered utterance
    (~0.7s for a 10s clip, scaling with utterance length), or
  - **streaming** (`--mode stream`): a final padded step on the tail, with
    partials printed/shown as 2.93s windows land while you speak.
- The text is then put on the clipboard, Cmd-V is posted, and the previous
  clipboard is restored. Clipboard-based because CGEvent keyboard injection
  cannot reliably type CJK in every target, whereas a paste uses the app's own
  input path.

### The status pill

A borderless, non-activating `NSPanel`: dark rounded pill, bottom centre,
showing the prompt text, then (streaming mode only) the tail of the live
partial, with a 5-bar mic meter on the right (silence = flat dots, normal
speech = bars at ~80%). Click-through, above normal windows, and never takes
focus from the app you are dictating into.

In batch mode there is no partial text to show — the model only runs on release
— so the pill is a listening indicator plus the meter. It is a fixed 380×44pt
and truncates long text from the left (`…`), keeping the most recent words.

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

Then hold the key, speak, release. **Double-press** it instead and recording runs
hands-free: the mic stays open, the pill counts up (`● REC 0:42`), and a single
press stops and transcribes. The first tap of a double-press is not thrown away —
it is the start of the dictation — and a *single* short tap still transcribes as
it always did, half a second later, because the daemon waits out the double-tap
window before deciding. `--no-double-tap` removes that wait. Note that with
`--key-passthrough` the foreground app sees *both* taps of the gesture.

Hands-free sessions stop themselves at 20 minutes (hold-to-talk at 5), and either
way what was captured is transcribed rather than discarded.

| flag | default | notes |
|------|---------|-------|
| `--key` | `right_option` | `left_option`, `right_command`, `left_command`, `right_shift`, `left_shift`, `right_control`, `left_control`, `fn`, or `f13`–`f19` |
| `--key-passthrough` | off | let other apps see the hotkey too. The default (off) swallows it, which is what stops a chat app's own hold-to-talk double-firing — but swallowing is also what costs you right Option as a modifier. Passthrough restores the modifier and re-exposes the collision |
| `--model` | `mlx-community/Qwen3-ASR-1.7B-8bit` | path or HF repo id; streaming checkpoints auto-switch to `--mode stream` |
| `--mode` | `auto` | `auto` \| `stream` \| `batch`. `auto` streams only for checkpoints with window/chunk metadata |
| `--context` | `""` | names/jargon. In batch mode these become **real hotwords** (`hotwords=[...]`) when the model supports it. Merges with `--context-file` |
| `--context-file` | `""` | file of hotwords, **re-read on every press** so edits apply without a restart |
| `--min-rms` | `0.002` | batch: **log** a `mic muted?` note when the loudest 100ms is below this. It does not skip anything — nothing is gated on loudness (see Troubleshooting) |
| `--transcribe-file` | `""` | transcribe a file and exit — smoke test, needs no hotkey and may run alongside the daemon |
| `--live-file` | `""` | append live partials to a file |
| `--tail-ms` | `200` | extra mic time after release, to catch the last syllable |
| `--device` | system default | input device index or name |
| `--no-double-tap` | off | no hands-free gesture: a press shorter than the window is transcribed at once instead of waiting to see whether it is half of a double-press |
| `--double-tap-ms` | system | double-press window in ms; default is the system double-click interval, clamped to 300–700 |
| `--polish-cmd` | `""` | rewrite the transcript through this command (stdin → stdout) — see below |
| `--polish-timeout` | `6.0` | seconds before the rewriter is abandoned and the raw transcript pasted |
| `--dry-run` | off | print the text instead of pasting it |
| `--paste-delay` | `0.6` | seconds the transcript stays on the clipboard before the old one is restored |
| `--no-overlay` | off | no status pill (headless / scripted use) |
| `--overlay-text` | `直接说` | pill text while waiting for speech |
| `--self-test` | — | window bookkeeping, text cleaning, meter curve, the double-press gate, the rewrite guard — no model load |

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
# one term per line; a line starting with # is a comment
Alex Chen
Kubernetes
Postgres
```

Names of people and products are the entries that earn their place; anything the
model already spells correctly is noise. (`Alex Chen` is a placeholder — use your
own name, colleagues, customers. Full-line comments only, so a term containing
`#` such as `C#` survives.)

The file is **re-read on every press**, so editing it applies to your very next
dictation — no reinstall, no daemon restart. Phrases work (`Vault Radar`);
splitting is on commas and newlines, not spaces, so a two-word term stays one
term. `--context "a, b"` merges with the file, `--context-file PATH` relocates
it, and `install.sh` seeds it from `--context` on first run (override the path
with `PTT_HOTWORDS`).

Keep the list to terms you actually dictate. A short, specific list is the point;
nobody has measured what a long one does, and a list of everything you might ever
say is unlikely to help.

## Rephrasing the transcript

`--polish-cmd` pipes the finished transcript through any command that reads text
on stdin and writes text on stdout, and pastes what comes back:

```bash
ptt_dictate.py --polish-cmd \
  'pi -p -nt --no-session --no-skills --no-extensions --no-context-files --no-mcp \
   --thinking off --model deepseek/deepseek-flash'
```

It fixes punctuation, capitalisation, filler words (`um`, false starts, repeated
words) and sentence/paragraph breaks, and repairs mis-hearings with the hotwords
list as a spelling glossary. Measured on this machine: 1.3–2.2s added to a
100–200 word dictation, with the pill reading `polishing…` in the meantime.
Anything under 8 words — or under 24 characters, since CJK is dense — is pasted
straight through; there is nothing to gain there.

Three things make it safe to leave on:

- **The transcript never reaches the command line.** It goes in on stdin, so it
  is not visible in `ps`, and `--polish-cmd` is the single knob that decides
  where it goes.
- **A rewrite is only a candidate.** It is pasted only if it still contains
  every number and every hotword the transcript had, sits within 0.70–1.25× its
  length, and does not open with `Here is…`. Under 40 words only the length is
  checked: "the Terraform version, no wait, the Vault version" is 25 words in
  and 13 out, so a retraction has to be allowed to delete what it retracted
  (measured — that rewrite is correct, and the first version of this guard
  rejected it). Anything else — a dropped clause, a timeout, a non-zero exit —
  is logged as `(polish skipped: <reason>)` and the raw transcript is pasted
  instead. Every model tried got something wrong on real dictation, so the guard
  is what makes this safe, not the choice of model.
- **The rules travel with the code.** The prompt (rules + glossary) lives in
  `ptt_dictate.py` and is sent on stdin ahead of the text, so swapping `--model`
  cannot quietly drop the guardrails. A lower-case entry in `hotwords.txt`
  cannot demote a name the transcript already capitalised: the text's casing
  wins (`nomad` in the file, `Nomad` in the text, `Nomad` out).

**A command that talks to a hosted model sends your dictation off the machine** —
customer names, versions, architecture talk. That is why the default is off, and
why a local rewriter (`ollama run …`, a local MLX model) needs no code at all:
same interface, different `--polish-cmd`.

Known limit: two dictations in quick succession paste in whichever order their
*rewrites* finish — transcribes are serialised behind one lock, rewrites are not,
and that is a wider window than it used to be.

## Runtime facts (measured on an M4 Pro, 8-bit MLX, 20s/10s clips)

| | Qwen3-ASR 1.7B *(default)* | Qwen3-ASR 0.6B | VibeVoice-Stream 1.5B | VibeVoice-Stream 7B |
|---|---|---|---|---|
| resident memory | 2.50GB | 1.09GB | 3.07GB | 7.57GB |
| 10.3s Chinese | **0.46s** (22×) | 0.24s (44×) | 1.23s (8×) | 2.54s (4×) |
| 20s English | 0.96s (21×) | 0.47s (42×) | 2.23s (9×) | 4.98s (4×) |
| 5s utterance, warm | 0.31s | 0.15s | 0.72s | 2.49s |
| live partials | no | no | yes (first at ~3.5s, then every 2.93s) | yes |
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
tone, so it invents nothing on a stray tap (observed repeatedly, not proven as a
guarantee — a loud room could still produce words, which is why `--min-rms` logs
a warning rather than enforcing one). Whisper-class models are known to
hallucinate, hence the belt-and-braces note.

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

The log holds your dictated text in cleartext, so it **size-rotates itself**
(5MB, keeping 3) from the daemon's own heartbeat: launchd owns the file
handle, so rotating means renaming and then `dup2`-ing a fresh file over
stdout/stderr. `--log-max-mb 0` disables it.

```bash
./install.sh --key right_option --context "Kubernetes, Postgres"
./uninstall.sh                                                   # stop + remove

launchctl print gui/$(id -u)/local.ptt-dictate | grep -E 'state|pid'   # status
launchctl bootout gui/$(id -u)/local.ptt-dictate                       # stop only
```

Editing the script requires a bootout + bootstrap to take effect. Because
`KeepAlive` restarts it, `kill` is not how you stop it — use `bootout`.

## Troubleshooting

Almost every failure mode here is *silent* — the daemon looks healthy and no
text appears; the exception is the clipped head below, where text does arrive.
Start with the log, `~/Library/Logs/ptt-dictate/daemon.log`: you want
`● listening` on press and `→ <text>` on release. Anything else is the clue.

**It worked, then stopped.** macOS disabled the event tap (callback timeout, or
secure input). The daemon re-arms it automatically and logs
`! tap disabled — re-armed`; if that line keeps repeating, the interpreter lacks
Accessibility — see Requirements. Note that idling at 0% CPU is *not* evidence
it is working; a disabled tap costs nothing.

**The pill appears but no text arrives.** Grep the log for `mic:` and for
`(very quiet: peak …)`. A capture is transcribed whatever its level — Qwen3-ASR
returns nothing for room tone rather than inventing words — so silence points at
the microphone, not the model. Bluetooth headsets come and go: a failed open
re-scans the device list, and an idle loop re-scans every ~30s so a reconnected
headset becomes the default again. The log prints `mic: <name>` when it changes.

**The first word or two goes missing.** Nothing in the pipeline trims the head:
in batch mode the whole press is a single `generate()` over a buffer that begins
at the first audio callback, streaming windows keep `[0, WIN)`, and Qwen3-ASR
only splits audio longer than 20 minutes (and only pads the tail). So the words
are not dropped downstream — they were never captured, which puts the loss in
the window between the key-down and the first callback. That window is
overwhelmingly the device's, not the code's: measured on a Bose QC Ultra 2 HP
already in HFP, the open returns in ~20ms and the first callback lands ~19ms
later (18–24ms, still so after 120s idle), far too little for a word (~300ms).
A Bluetooth headset sitting in A2DP — something playing through it — instead
has to switch to HFP/SCO when the input stream opens, on the order of hundreds
of ms during which the mic does not appear to be carrying audio; that switch was
never timed here, which is the part still open. Unconfirmed as the root cause of
the episodes that were actually reported: the symptom faded with no code change,
which is also what a link that is now parked in HFP looks like — so expect it
back when audio plays through the headset again.

Confirm cheaply with `--device "MacBook Pro Microphone"` (the name, not an
index — indices move as devices connect and disconnect): if the loss stops
there, it is the headset's profile switch, not the daemon. To time the window
directly, open the mic the way a press does:

```python
import time, sounddevice as sd
# add device="MacBook Pro Microphone" to time the mic your --device actually uses
t0, first = time.perf_counter(), {}
s = sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                   callback=lambda *a: first.setdefault("t", time.perf_counter()))
s.start(); time.sleep(1.0); s.stop(); s.close()
print(f"first callback {1000 * (first['t'] - t0):.0f} ms after the open began")
```

The log cannot see this class of loss: its `capture` figure is counted from the
release (tail sleep + teardown), so the head is invisible to it by construction.

**Nothing at all arrives: `(very quiet: peak 0.0000 — mic muted?)` then `(nothing)`.**
Digital silence from the *device*, per the paragraph above: worth confirming
outside the daemon before touching code — a 2s probe of that device returns
`peak=0.00000` while the built-in mic returns room tone in 60ms. The reason is
in `log show --last 12h --predicate 'process == "bluetoothd"'` (widen `--last`,
or use `--start/--end`, to reach the incident): on 2026-10-02 it shows 112,424
lines of `SCO connection handle is no longer valid. Dropping SCO data.` (≈270/s)
in a burst from 14:23:15 to 14:30:18, and 140,324 in a comparable burst on
2026-09-30. The headset's HFP voice channel (SCO) has a stale HCI handle while
macOS still enumerates `Bose QC Ultra 2 HP` as an input, so CoreAudio hands out
zeros. What the log shows *around* the break: HFP LinkQualityReport stays
healthy (RSSI ≈ -40, `eSCO scheduled pkts` ≈ 133/s, zero retransmits) right up to
the press, and the drops begin with the first press and stop when the stream
closes — so there is no disconnect event to detect, only silence once audio has
to flow. Whether an idle gap clears the stale handle is untested. Power-cycle
the headset (or toggle Bluetooth) to rebuild the link — or dictate on a wired or
built-in mic (`--device "MacBook Pro Microphone"` on this machine), which also
skips the per-press SCO setup described above.

**It never reacts to the key.** Accessibility hasn't been granted to the
interpreter — see Requirements.

**Text arrives twice, or the wrong app reacts.** The hotkey is swallowed by
design, which exists exactly for this. `--key-passthrough` re-exposes it to any
app that binds the same key for its own hold-to-talk. If a pre-existing input
tool owns the key outright, quit it and disable its autostart
(`~/Library/LaunchAgents/`, input-method settings) or it takes the key back at
next login.

**It doesn't stop when you release.** For modifier hotkeys the release is read
from the event's modifier flags, so holding *both* option keys and releasing only
the bound one won't register until both are up. Bind a non-modifier if that
matters.

For anyone changing the code rather than running it, three traps are documented
where they bite rather than here: the streaming window protocol (window k must
cover `[k*ADV, k*ADV+WIN)`, or the model re-transcribes audio it already
emitted), never naming an `NSObject` subclass method `release`, and pyobjc
turning inner underscores into selector colons.

## Not implemented

- On-screen editing of the transcript, per-app hotkeys, and a tray/menu item —
the pill shows state only, and text goes straight to the clipboard.
