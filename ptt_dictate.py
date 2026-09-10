#!/usr/bin/env python3
"""Hold-to-talk dictation on the local VibeVoice-ASR-Streaming 1.5B (MLX).

Hold the hotkey: the mic streams into a warm model, live partials print as they
land. Release: one final flush step (~0.3s), then the text is pasted into
whatever app has focus (clipboard + Cmd-V, so Chinese works).

    ~/venv/bin/python ~/ptt-dictate/ptt_dictate.py --key right_option

One-time setup: grant Accessibility + Microphone to the interpreter running
this (TCC prompts on first use). a pre-existing dictation app must not hold the same key.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import objc
import Quartz
import AppKit
import sounddevice as sd
import mlx.core as mx
from mlx_audio.stt import load

DEFAULT_MODEL = "~/models/vibevoice-asr-streaming-1.5b-mlx-8bit"

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
    """One press-to-release dictation cycle against a loaded streaming model."""

    def __init__(self, model, context: str = "", live_file: str = "", dry_run: bool = False, device=None, overlay=None, paste_delay: float = 0.6):
        self.model = model
        self.context = context
        self.live_file = live_file
        self.dry_run = dry_run
        self.device = device
        self.overlay = overlay
        self.paste_delay = paste_delay
        self.last_device = None
        if overlay is not None:
            overlay.recorder = self
        self.SR = model.sample_rate
        self.WIN = model.streaming_window_samples
        self.ADV = model.streaming_chunk_samples
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
        while self.buf.size >= self.WIN:
            self._step()
            self.buf = self.buf[self.ADV :]

    def flush(self) -> None:
        """Final step on the tail, right-padded — same as pad_last_chunk."""
        if self.buf.size >= self.SR // 10:
            self._step()

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
    ap.add_argument("--context", default="", help="hotwords / names fed to the model")
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

    if args.key == "right_option" and "other-dictation-app" in subprocess.run(
        ["pgrep", "-fl", "other-dictation-app"], capture_output=True, text=True
    ).stdout:
        print("warning: a pre-existing dictation app is running and also grabs right Option — quit it or pick another --key")

    print(f"loading {args.model} ...", flush=True)
    model = load(args.model)
    if not model.is_streaming_model:
        sys.exit("not a streaming checkpoint")
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
    recorder = Recorder(model, args.context, args.live_file, args.dry_run, args.device, overlay, args.paste_delay)
    print(
        f"ready: {args.key} (keycode {keycode}) | {model.sample_rate}Hz "
        f"window {model.streaming_window_samples / model.sample_rate:.2f}s "
        f"advance {model.streaming_chunk_samples / model.sample_rate:.2f}s"
    )
    print(">>> HOLD the key to dictate, release to paste <<<", flush=True)

    mask = (
        Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp)
    )
    held = False

    def callback(proxy, type_, event, refcon):
        nonlocal held
        if type_ in (
            Quartz.kCGEventTapDisabledByTimeout,
            Quartz.kCGEventTapDisabledByUserInput,
        ):
            Quartz.CGEventTapEnable(tap, True)
            held = False  # a disable can swallow the release
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
            return
        Quartz.CGEventTapEnable(tap, True)
        held = False
        print("! tap found disabled — re-armed", flush=True)

    AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.5, KeepAlive.alloc().initWithCheck_(heartbeat), "noop:", None, True
    )
    app.run()


if __name__ == "__main__":
    main()
