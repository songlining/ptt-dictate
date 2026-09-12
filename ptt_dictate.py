#!/usr/bin/env python3
"""Hold-to-talk dictation on the local VibeVoice-ASR-Streaming 1.5B (MLX).

Hold the hotkey: the mic streams into a warm model, live partials print as they
land. Release: one final flush step (~0.3s), then the text is pasted into
whatever app has focus (clipboard + Cmd-V, so Chinese works).

    /path/to/venv/bin/python /path/to/ptt-dictate/ptt_dictate.py --key right_option

One-time setup: grant Accessibility + Microphone to the interpreter running
this (TCC prompts on first use), and make sure no other app is holding the
same hotkey.
"""

from __future__ import annotations

import argparse
import fcntl
import inspect
import os
import queue
import re
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import objc
import Quartz
import AppKit
import sounddevice as sd
import mlx.core as mx
from mlx_audio.stt import load

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"

# keycode -> (modifier flag mask or None for a normal key)
MODIFIERS = {
    "right_option": (61, Quartz.kCGEventFlagMaskAlternate),
    "left_option": (58, Quartz.kCGEventFlagMaskAlternate),
    "right_command": (54, Quartz.kCGEventFlagMaskCommand),
    "left_command": (55, Quartz.kCGEventFlagMaskCommand),
    "right_shift": (60, Quartz.kCGEventFlagMaskShift),
    "left_shift": (56, Quartz.kCGEventFlagMaskShift),
    "right_control": (62, Quartz.kCGEventFlagMaskControl),
    "left_control": (59, Quartz.kCGEventFlagMaskControl),
    "fn": (63, Quartz.kCGEventFlagMaskSecondaryFn),
}
PLAIN_KEYS = {"f13": 105, "f14": 107, "f15": 113, "f16": 106, "f17": 64, "f18": 79, "f19": 80}

NOISE = re.compile(r"\[(?:silence|noise|music|applause|laughter|inaudible)\]", re.I)
SPEAKER = re.compile(r"\s*speaker\s*\d+\s*:\s*", re.I)
METER_BARS = 5


def clean(text: str) -> str:
    """Strip the model's `Speaker 0:` prefix and `[Silence]`-style markers."""
    return re.sub(r"\s+", " ", NOISE.sub(" ", SPEAKER.sub(" ", text))).strip()


def ellipsize(text: str, limit: int = 40) -> str:
    """Keep the tail — the most recent words are what matters while dictating."""
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def meter_level(rms: float) -> float:
    """Map mic RMS to 0..1 bar height. Square-root curve: speech RMS sits
    around 0.01-0.1, so a linear scale leaves the bars flat at normal volume."""
    return min(1.0, max(0.0, rms) ** 0.5 * 3.2)


def _dedupe(words: list[str]) -> list[str]:
    """Case-insensitive dedupe, order preserved."""
    seen, out = set(), []
    for word in words:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            out.append(word)
    return out


def parse_hotwords(text: str) -> list[str]:
    """Hotwords from newline- or comma-separated text; `#` comments out the
    rest of a line.

    Splitting stops at commas and newlines — *not* whitespace — so a term may be
    a phrase ("Vault Radar", "Alex Chen") rather than being torn into two
    independent words.
    """
    words = []
    for line in text.splitlines():
        words += [w.strip() for w in line.split("#", 1)[0].split(",")]
    return _dedupe([w for w in words if w])


def result_text(out) -> str:
    """Text out of an mlx_audio STTOutput.

    STTOutput is attribute-only (not subscriptable), and the empty result is a
    real case — silence returns '' — so the obvious `or out["text"]` fallback
    raises TypeError exactly when there is nothing to say.
    """
    text = getattr(out, "text", None)
    if text is None:
        try:
            text = out["text"]
        except Exception:
            text = ""
    return text or ""


def peak_level(buf: np.ndarray, sr: int, block: float = 0.1) -> float:
    """Loudest 100ms-block RMS. Distinguishes "someone spoke" from room tone:
    a whole-capture average is dragged down by silence around the words, and on
    a quiet headset it sits below the room-tone floor of the loud parts."""
    n = max(1, int(sr * block))
    if buf.size < n:
        return float(np.sqrt(np.mean(np.square(buf)))) if buf.size else 0.0
    usable = buf[: buf.size - (buf.size % n)].reshape(-1, n)
    return float(np.sqrt(np.mean(np.square(usable), axis=1)).max())


def too_short(buf: np.ndarray, sr: int, min_seconds: float = 0.3) -> bool:
    """A mis-tap, not worth a model run.

    This is the *only* capture that gets skipped. There deliberately is no
    loudness gate: measured, Qwen3-ASR returns '' for digital silence and for
    room tone, so the model is its own authority on whether there was speech —
    and a level threshold cannot separate "mic muted" from "spoken quietly",
    which meant it silently ate real dictation once the input gain dropped.
    Loudness is logged instead of enforced.
    """
    return buf.size < sr * min_seconds


class MeterView(AppKit.NSView):
    """Rolling mic level as a row of bars — the pill's 'is it hearing me'."""

    def initWithFrame_(self, rect):
        self = objc.super(MeterView, self).initWithFrame_(rect)
        if self is None:
            return None
        self.bars = [0.0] * METER_BARS
        return self

    def setBars_(self, bars):
        self.bars = bars
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        AppKit.NSColor.whiteColor().set()
        width, height = self.bounds().size.width, self.bounds().size.height
        bar, gap = 2.0, 2.5
        for i, level in enumerate(self.bars):
            x = i * (bar + gap)
            if x + bar > width:
                break
            h = max(2.0, height * min(1.0, level))
            AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                AppKit.NSMakeRect(x, (height - h) / 2.0, bar, h), 1.0, 1.0
            ).fill()


class RepeatLimiter:
    """Rate-limit a repeated identical log line.

    A persistently disabled event tap (which is what an untrusted process looks
    like) is re-armed every 0.5s, so logging each attempt would write ~15MB/day
    of the same sentence. First occurrence always prints; after that, one in
    every `every` ticks, so the state stays visible without flooding.
    """

    def __init__(self, every: int = 120):
        self.every = every
        self.count = 0

    def tick(self) -> bool:
        self.count += 1
        return self.count == 1 or self.count % self.every == 0


class KeepAlive(AppKit.NSObject):
    """Idle timer doing two jobs: keep Python bytecode running (SIGINT is
    otherwise deferred forever by the AppKit run loop, which makes Ctrl-C look
    dead), and health-check the event tap. macOS disables taps on callback
    timeout or during secure input, which leaves this daemon looking perfectly
    healthy while hearing nothing at all until it is restarted."""

    def initWithCheck_(self, check):
        self = objc.super(KeepAlive, self).init()
        if self is None:
            return None
        self._check = check
        return self

    def noop_(self, timer):
        self._check()


class Overlay(AppKit.NSObject):
    """Floating status pill: prompt, live partial text, mic meter."""

    W, H = 380.0, 44.0

    def initWithPrompt_(self, prompt):
        self = objc.super(Overlay, self).init()
        if self is None:
            return None
        self.prompt = prompt
        self.recorder = None
        self.timer = None
        self._build()
        return self

    def _build(self):
        frame = AppKit.NSScreen.mainScreen().visibleFrame()
        rect = AppKit.NSMakeRect(
            frame.origin.x + (frame.size.width - self.W) / 2.0,
            frame.origin.y + 150.0,
            self.W,
            self.H,
        )
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(AppKit.NSFloatingWindowLevel)
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)  # never steal focus from the target app
        # An accessory app never activates, and NSPanel hides on deactivate by
        # default — without this the pill flashes on press and vanishes.
        panel.setHidesOnDeactivate_(False)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        content = panel.contentView()
        content.setWantsLayer_(True)
        content.layer().setBackgroundColor_(
            AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.06, 0.88).CGColor()
        )
        content.layer().setCornerRadius_(self.H / 2.0)
        content.layer().setMasksToBounds_(True)

        label = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(20.0, (self.H - 20.0) / 2.0, self.W - 88.0, 20.0)
        )
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setFont_(AppKit.NSFont.systemFontOfSize_(14.0))
        label.setTextColor_(AppKit.NSColor.whiteColor())
        label.setStringValue_(self.prompt)
        content.addSubview_(label)

        meter = MeterView.alloc().initWithFrame_(
            AppKit.NSMakeRect(self.W - 40.0, (self.H - 16.0) / 2.0, 20.0, 16.0)
        )
        content.addSubview_(meter)
        self.panel, self.label, self.meter = panel, label, meter

    def show_pill(self) -> None:
        self.label.setStringValue_(self.prompt)
        self.meter.setBars_([0.0] * METER_BARS)
        self.panel.orderFrontRegardless()
        if self.timer is not None:  # don't stack timers across rapid presses
            self.timer.invalidate()
        self.timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.06, self, "tick:", None, True
        )

    def hide_pill(self) -> None:
        """Called from the recorder thread — hop to the main thread for AppKit.

        NOT named `release`: that shadows NSObject's -release, and because
        performSelectorOnMainThread: retains/releases its receiver, the override
        re-enters itself forever — a permanent ~100% CPU spin from the moment
        the object exists. Same trap for `press`. Keep these names selector-free.

        Also note pyobjc turns EVERY underscore into a colon when deriving a
        selector, so a name called through performSelectorOnMainThread_ must map
        to exactly one colon for one argument (`hideNow_` -> `hideNow:`).
        """
        self.performSelectorOnMainThread_withObject_waitUntilDone_("hideNow:", None, False)

    def hideNow_(self, _):
        # A queued hide can land after a newer press — never hide a live one.
        if self.recorder is not None and getattr(self.recorder, "active", False):
            return
        if self.timer is not None:
            self.timer.invalidate()
            self.timer = None
        self.panel.orderOut_(None)

    def tick_(self, _):
        recorder = self.recorder
        level = meter_level(getattr(recorder, "level", 0.0))
        self.meter.setBars_(self.meter.bars[1:] + [level])
        text = getattr(recorder, "text_so_far", "")
        self.label.setStringValue_(ellipsize(text or self.prompt))


def paste(text: str, restore_delay: float = 0.6) -> None:
    """Put `text` on the clipboard, hit Cmd-V, then restore the old clipboard."""
    pb = AppKit.NSPasteboard.generalPasteboard()
    previous = pb.stringForType_(AppKit.NSPasteboardTypeString)
    pb.clearContents()
    pb.setString_forType_(text, AppKit.NSPasteboardTypeString)
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(src, 9, down)  # 9 = 'v'
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    # ponytail: fixed delay — the target app reads the pasteboard asynchronously,
    # so a slow app would paste the restored *old* contents. Raise --paste-delay
    # if that ever shows up; there is no "did you read it?" hook to poll.
    time.sleep(restore_delay)
    if previous is not None:
        pb.clearContents()
        pb.setString_forType_(previous, AppKit.NSPasteboardTypeString)


class Recorder:
    """One press-to-release dictation cycle against a loaded ASR model."""

    def __init__(self, model, context: str = "", live_file: str = "", dry_run: bool = False, device=None, overlay=None, paste_delay: float = 0.6, batch: bool = False, min_rms: float = 0.005, context_file: str = ""):
        self.model = model
        self.context = context
        self.context_file = context_file
        self.live_file = live_file
        self.dry_run = dry_run
        self.device = device
        self.overlay = overlay
        self.paste_delay = paste_delay
        self.batch = batch
        self.min_rms = min_rms
        self.last_device = None
        if overlay is not None:
            overlay.recorder = self
        self.SR = model.sample_rate
        # The streaming protocol's geometry only exists on streaming checkpoints.
        self.WIN = 0 if batch else model.streaming_window_samples
        self.ADV = 0 if batch else model.streaming_chunk_samples
        self.lock = threading.Lock()
        self.active = False
        self.busy = False
        self.reset()

    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self.steps = 0
        self.parts: list[str] = []
        self.text_so_far = ""
        self.level = 0.0
        self.state = None
        self.stop = threading.Event()

    def feed(self, chunk: np.ndarray) -> None:
        """Buffer new audio; step once per full window, sliding by the advance.

        Mirrors the model's own chunk iterator: window k covers
        [k*ADV, k*ADV+WIN), so the lookahead tail of one window is the head of
        the next. Feeding a wider overlap makes the model re-transcribe it.
        """
        self.buf = np.concatenate([self.buf, chunk])
        if self.batch:
            return  # one pass over the whole press, at release
        while self.buf.size >= self.WIN:
            self._step()
            self.buf = self.buf[self.ADV :]

    def flush(self) -> None:
        """Streaming: final padded step on the tail. Batch: transcribe it all."""
        if self.batch:
            self._transcribe()
        elif self.buf.size >= self.SR // 10:
            self._step()

    def _hotwords(self) -> list[str]:
        """--context plus the standing list in --context-file.

        The file is re-read on every press, so adding a word takes effect on the
        next dictation — no daemon restart, which matters because the useful
        list only emerges from watching what the model actually mis-hears.
        """
        words = parse_hotwords(self.context)
        if self.context_file:
            try:
                with open(self.context_file) as fh:
                    words += parse_hotwords(fh.read())
            except OSError:
                pass
        return _dedupe(words)

    def _bias_kwargs(self) -> dict:
        """Hotwords, when the model takes them (Qwen3-ASR does)."""
        words = self._hotwords()
        if words and "hotwords" in inspect.signature(self.model.generate).parameters:
            return {"hotwords": words}
        return {}

    def _transcribe(self) -> None:
        if too_short(self.buf, self.SR):
            print(f"  (skipped: only {self.buf.size / self.SR:.2f}s of audio)", flush=True)
            return
        peak = peak_level(self.buf, self.SR)
        if peak < self.min_rms:
            # Informational only — we still transcribe. Blocking here is how a
            # quiet speaker ends up with "stopped working".
            print(f"  (very quiet: peak {peak:.4f} — mic muted?)", flush=True)
        out = self.model.generate(self.buf, **self._bias_kwargs())
        piece = clean(result_text(out))
        if piece:
            self.parts.append(piece)
            self.text_so_far = piece
            print(f"  {piece}", flush=True)
            if self.live_file:
                with open(self.live_file, "a") as fh:
                    fh.write(piece + "\n")

    def _step(self) -> None:
        window = self.buf[: self.WIN]
        if window.size < self.WIN:
            window = np.pad(window, (0, self.WIN - window.size))  # pad the tail, not the head
        features = self.model.encode_speech(mx.array(window)[None, :])
        text, self.state = self.model.streaming_generate_step(features, self.state)
        self.steps += 1
        piece = clean(text)
        if piece:
            self.parts.append(piece)
            self.text_so_far = " ".join(self.parts)
            print(f"  {piece}", flush=True)
            if self.live_file:
                with open(self.live_file, "a") as fh:
                    fh.write(piece + "\n")

    def _on_audio(self, data, frames, time_info, status) -> None:
        self.level = float(np.sqrt(np.mean(np.square(data))))
        self.q.put(bytes(data))

    def _open_stream(self) -> None:
        """Open whatever is *currently* the default input device.

        PortAudio caches the device list at Pa_Initialize and never re-reads it,
        so a Bluetooth headset coming or going leaves us pointing at a device
        that may no longer exist (-10851 Invalid Property Value) — or silently
        stuck on the built-in mic after the headset is back. A full re-scan
        costs ~3ms, so we do it every press: always the live default.
        """
        sd._terminate()
        sd._initialize()
        for attempt in (1, 2):
            try:
                if self.device is None:
                    dev = sd.query_devices(sd.default.device[0])
                    if dev["name"] != self.last_device:
                        print(
                            f"mic: {dev['name']} (native {dev['default_samplerate']:.0f}Hz)",
                            flush=True,
                        )
                        self.last_device = dev["name"]
                self.stream = sd.InputStream(
                    device=self.device,
                    samplerate=self.SR,
                    channels=1,
                    dtype="float32",
                    callback=self._on_audio,
                )
                self.stream.start()
                return
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"! mic open failed ({exc}) — re-scanning audio devices", flush=True)
                sd._terminate()
                sd._initialize()

    def press(self) -> None:
        with self.lock:
            if self.active or self.busy:
                return
            self.active = True
        self.reset()
        # Batch models have no streaming state to prefill (nor init_streaming_state).
        if self.batch:
            self.state = None
        else:
            self.state = (
                self.model.init_streaming_state(context_info=self.context)
                if self.context
                else self.model.init_streaming_state()
            )
        self.q: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        try:
            self._open_stream()
        except Exception as exc:
            # Abort cleanly: never leave `active` set (which would swallow the
            # release) and never report the silence as an empty transcription.
            print(f"! mic unavailable — press skipped: {exc}", flush=True)
            self.stop.set()
            self.thread.join()
            with self.lock:
                self.active = False
            return
        if self.overlay is not None:
            self.overlay.show_pill()
        print("● listening", flush=True)

    def release(self, tail_ms: int = 200) -> None:
        with self.lock:
            if not self.active:
                return
            self.active = False
            self.busy = True
        try:
            time.sleep(tail_ms / 1000)  # catch the last syllable before the mic closes
            self.stream.stop()
            self.stream.close()
            self.stop.set()
            self.thread.join()
            text = clean(" ".join(self.parts))
            if text:
                print(f"→ {text}", flush=True)
                if not self.dry_run:
                    paste(text, self.paste_delay)
            else:
                print("→ (nothing)", flush=True)
        except Exception as exc:  # keep the daemon alive through a bad cycle
            print(f"release failed: {exc!r}", flush=True)
        finally:
            if self.overlay is not None:
                self.overlay.hide_pill()
            with self.lock:
                self.busy = False

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                block = self.q.get(timeout=0.05)
            except queue.Empty:
                continue
            self.feed(np.frombuffer(block, dtype=np.float32))
        while True:  # drain whatever arrived between the last read and the stop
            try:
                block = self.q.get_nowait()
            except queue.Empty:
                break
            self.feed(np.frombuffer(block, dtype=np.float32))
        if self.buf.size or self.steps == 0:
            self.flush()


class _FakeModel:
    """Self-test stand-in: WIN=3 samples, ADV=2 samples."""

    sample_rate = 10  # keeps the flush threshold at one sample
    streaming_window_samples = 3
    streaming_chunk_samples = 2

    def __init__(self):
        self.n = 0

    def init_streaming_state(self, **kwargs):
        return {}

    def encode_speech(self, x):
        return x

    def streaming_generate_step(self, features, state):
        self.n += 1
        return f" \n Speaker 0:chunk{self.n} [Silence]", state


def self_test() -> None:
    assert clean(" \n Speaker 0:Hello there ") == "Hello there"
    assert clean("Speaker 0:Hello Speaker 1:world") == "Hello world"
    assert clean("[Silence]") == ""
    assert clean("[Noise] [Silence]") == ""
    assert parse_hotwords("Terraform, Vault\n# a comment\nKubernetes") == ["Terraform", "Vault", "Kubernetes"]
    assert parse_hotwords("Vault, vault, VAULT") == ["Vault"]  # case-insensitive dedupe
    assert parse_hotwords("  \n\n") == []
    # a phrase must survive as one term, not be split into independent words
    assert parse_hotwords("Alex Chen\nVault Radar") == ["Alex Chen", "Vault Radar"]
    # a mis-tap is skipped; loudness never blocks a transcription
    assert too_short(np.zeros(4000, dtype=np.float32), 16000) is True    # 0.25s
    assert too_short(np.zeros(16000, dtype=np.float32), 16000) is False  # 1s
    assert peak_level(np.zeros(16000, dtype=np.float32), 16000) == 0.0
    quiet = np.full(16000, 0.0003, dtype=np.float32); quiet[8000:8800] = 0.02
    assert peak_level(quiet, 16000) > 0.01, "peak must find the loud part"
    # log rate limiting: first occurrence prints, then one in every `every`
    lim = RepeatLimiter(every=4)
    assert [lim.tick() for _ in range(9)] == [True, False, False, False, True, False, False, False, True]
    assert meter_level(0.0) == 0.0
    assert 0.3 < meter_level(0.02) < 0.7, meter_level(0.02)  # quiet speech still moves
    assert meter_level(0.5) == 1.0  # clamped, never overflows the bar

    rec = Recorder(_FakeModel(), dry_run=True)
    rec.reset()
    rec.state = rec.model.init_streaming_state()
    rec.feed(np.ones(3, dtype=np.float32))  # first step: full window
    assert rec.steps == 1, rec.steps
    rec.feed(np.ones(1, dtype=np.float32))  # below ADV -> no step
    assert rec.steps == 1, rec.steps
    rec.feed(np.ones(1, dtype=np.float32))  # ADV reached -> step
    assert rec.steps == 2, rec.steps
    rec.feed(np.ones(1, dtype=np.float32))  # tail, flushed on release
    assert rec.steps == 2, rec.steps
    rec.flush()
    assert rec.steps == 3, rec.steps
    assert rec.parts == ["chunk1", "chunk2", "chunk3"], rec.parts
    # window bookkeeping: window k = [k*ADV, k*ADV+WIN) — no wider overlap
    rec.reset()
    rec.state = rec.model.init_streaming_state()
    rec.buf = np.arange(7, dtype=np.float32)
    seen = []
    rec._step = lambda: seen.append(rec.buf[: rec.WIN].copy())
    rec.feed(np.zeros(0, dtype=np.float32))
    assert [w.tolist() for w in seen] == [[0, 1, 2], [2, 3, 4], [4, 5, 6]], seen
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key", default="right_option", help=f"{', '.join(MODIFIERS)} or f13..f19")
    ap.add_argument("--context", default="", help="names/jargon; real hotwords in batch mode")
    ap.add_argument("--context-file", default="",
                    help="file of hotwords (one per line, # comments); re-read every press"
                         " so edits apply without a restart")
    ap.add_argument("--mode", choices=("auto", "stream", "batch"), default="auto",
                    help="auto: streaming checkpoints stream, everything else batches")
    ap.add_argument("--min-rms", type=float, default=0.002,
                    help="batch: log a 'mic muted?' note below this level; still transcribes")
    ap.add_argument("--transcribe-file", default="", help="transcribe a file and exit (smoke test)")
    ap.add_argument("--live-file", default="", help="append live partials to this file")
    ap.add_argument("--tail-ms", type=int, default=200, help="extra mic time after key release")
    ap.add_argument("--device", default=None, help="input device index/name (default: system default)")
    ap.add_argument("--dry-run", action="store_true", help="print instead of pasting")
    ap.add_argument("--paste-delay", type=float, default=0.6, help="seconds before the old clipboard is restored")
    ap.add_argument("--no-overlay", action="store_true", help="skip the floating status pill")
    ap.add_argument("--overlay-text", default="直接说", help="pill text while waiting for speech")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    # Two daemons on the same key both paste — and the LaunchAgent keeps one
    # running. flock is released by the kernel on exit, so no stale pidfile.
    # --transcribe-file is a diagnostic: it owns no hotkey, so it may run
    # alongside the daemon.
    if not args.transcribe_file:
        lock = open(os.path.join(tempfile.gettempdir(), "ptt-dictate.lock"), "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit("another ptt_dictate is already running (launchctl bootout local.ptt-dictate)")

    if args.device and args.device.isdigit():
        args.device = int(args.device)
    if args.key in MODIFIERS:
        keycode, flag = MODIFIERS[args.key]
    elif args.key in PLAIN_KEYS:
        keycode, flag = PLAIN_KEYS[args.key], None
    else:
        sys.exit(f"unknown --key {args.key!r}")

    print(f"loading {args.model} ...", flush=True)
    model = load(args.model)
    batch = args.mode == "batch" or (
        args.mode == "auto" and not getattr(model, "is_streaming_model", False)
    )
    if args.mode == "stream" and not getattr(model, "is_streaming_model", False):
        sys.exit(f"{args.model} is not a streaming checkpoint (no window/chunk metadata)")
    if args.transcribe_file:
        print(result_text(model.generate(args.transcribe_file)))
        return
    try:
        dev = sd.query_devices(args.device if args.device is not None else sd.default.device[0])
        print(
            f"mic: {dev['name']} (native {dev['default_samplerate']:.0f}Hz)"
            f" — PortAudio asked for {model.sample_rate}Hz",
            flush=True,
        )
    except Exception as exc:
        print(f"mic: unavailable ({exc})", flush=True)

    # Accessory app: needed for the panel, but no Dock icon and never activated.
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: os._exit(0))
    overlay = None if args.no_overlay else Overlay.alloc().initWithPrompt_(args.overlay_text)
    recorder = Recorder(model, args.context, args.live_file, args.dry_run, args.device, overlay,
                        args.paste_delay, batch, args.min_rms, args.context_file)
    if batch:
        print(
            f"ready: {args.key} (keycode {keycode}) | BATCH | {recorder.SR}Hz input"
            f" | transcribes on release",
            flush=True,
        )
    else:
        print(
            f"ready: {args.key} (keycode {keycode}) | streaming | {model.sample_rate}Hz "
            f"window {model.streaming_window_samples / model.sample_rate:.2f}s "
            f"advance {model.streaming_chunk_samples / model.sample_rate:.2f}s",
            flush=True,
        )
    print(">>> HOLD the key to dictate, release to paste <<<", flush=True)

    mask = (
        Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp)
    )
    held = False
    tap_warn = RepeatLimiter()

    def callback(proxy, type_, event, refcon):
        nonlocal held
        if type_ in (
            Quartz.kCGEventTapDisabledByTimeout,
            Quartz.kCGEventTapDisabledByUserInput,
        ):
            Quartz.CGEventTapEnable(tap, True)
            held = False  # a disable can swallow the release
            if tap_warn.tick():
                print(f"! tap disabled by system (0x{type_ & 0xffffffff:x}) — re-armed", flush=True)
            return event
        code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        if code != keycode:
            return event
        if flag is not None:
            down = bool(Quartz.CGEventGetFlags(event) & flag)
        else:
            down = type_ == Quartz.kCGEventKeyDown
        if down == held:
            return event
        held = down
        if down:
            recorder.press()
        else:
            threading.Thread(
                target=recorder.release, args=(args.tail_ms,), daemon=True
            ).start()
        return event

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        mask,
        callback,
        None,
    )
    if tap is None:
        sys.exit("cannot create event tap — grant Accessibility to this interpreter")

    source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)

    def heartbeat() -> None:
        """Re-arm the tap if the system disabled it behind our back."""
        nonlocal held
        if Quartz.CGEventTapIsEnabled(tap):
            tap_warn.count = 0  # healthy again; a later failure should print
            return
        Quartz.CGEventTapEnable(tap, True)
        held = False
        if tap_warn.tick():
            print(
                f"! tap disabled — re-armed (x{tap_warn.count}). If it never stays enabled,"
                f" grant Accessibility + Microphone to {os.path.realpath(sys.executable)}",
                flush=True,
            )

    AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.5, KeepAlive.alloc().initWithCheck_(heartbeat), "noop:", None, True
    )
    app.run()


if __name__ == "__main__":
    main()
