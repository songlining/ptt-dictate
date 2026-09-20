#!/usr/bin/env python3
"""Lifecycle tests for the Recorder/Session split — fake model, fake mic, fake
pasteboard. No real model load, no audio device, no clipboard.

Run: ~/models/venv-mlx-audio/bin/python test_lifecycle.py
"""

import threading
import time

import numpy as np

import ptt_dictate as pt


def wait_until(cond, timeout=5.0, what="condition"):
    t0 = time.perf_counter()
    while not cond():
        if time.perf_counter() - t0 > timeout:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.005)


class FakeStream:
    """Stands in for sd.InputStream — the tests never touch a real mic."""

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class FakeModel:
    """Batch/streaming stand-in. `generate` blocks on `gate` so a test can hold
    one session mid-transcribe — the window where the old code dropped presses."""

    sample_rate = 10
    streaming_window_samples = 3
    streaming_chunk_samples = 2

    def __init__(self, text="hello world", gate=None):
        self.text = text
        self.gate = gate if gate is not None else threading.Event()
        self.generate_calls = []
        self.n = 0

    def init_streaming_state(self, **kwargs):
        return {}

    def encode_speech(self, x):
        return x

    def streaming_generate_step(self, features, state):
        self.n += 1
        return f" \n Speaker 0:chunk{self.n} [Silence]", state

    def generate(self, buf, **kwargs):
        self.generate_calls.append(buf)
        self.gate.wait(timeout=10)
        return type("Out", (), {"text": self.text})()


class PasteSpy:
    """Counts paste calls instead of touching the real clipboard."""

    def __init__(self):
        self.lock = threading.Lock()
        self.starts = 0
        self.restores = 0

    def install(self):
        pt.paste_start = self.fake_start
        pt.paste_restore = self.fake_restore

    def fake_start(self, text):
        with self.lock:
            self.starts += 1
        return "old clipboard"

    def fake_restore(self, previous, delay):
        with self.lock:
            self.restores += 1


def install_fakes():
    pt.sd.InputStream = FakeStream
    pt.sd.query_devices = lambda *a, **k: {"name": "fake-mic", "default_samplerate": 48000.0}
    pt.sd.default = type("D", (), {"device": [0, 0]})()


def speak(session, n=1):
    """Push audio through the real consumer path (session.q -> _run -> feed)."""
    for _ in range(n):
        session.q.put(np.ones(3, dtype=np.float32).tobytes())


def test_press_accepted_while_previous_session_finalizes():
    """The headline bug: a press during a long fake generate() must be accepted.
    The old code held `busy` through transcribe+paste and silently dropped it."""
    spy = PasteSpy()
    spy.install()
    model = FakeModel()  # generate() blocks until the test opens the gate
    rec = pt.Recorder(model, dry_run=False, prewarm=False, batch=True)
    rec.press()
    a = rec.current
    assert a is not None and rec.active
    speak(a)
    rec.release(0)
    wait_until(lambda: len(model.generate_calls) == 1, what="A entering transcribe")
    assert not rec.active  # current is cleared synchronously, before finish ends
    rec.press()  # old code: busy -> silent drop, no pill, no log
    b = rec.current
    assert b is not None and b is not a, "press during finalize was dropped"
    assert rec.active
    speak(b)
    rec.release(0)
    model.gate.set()  # let both fake transcriptions complete
    wait_until(lambda: spy.starts == 2, what="both sessions pasting")
    wait_until(lambda: not rec.busy, what="sessions draining")
    assert rec.current is None
    assert len(model.generate_calls) == 2  # one transcribe per session


def test_duplicate_release_finishes_once():
    """Two near-simultaneous releases (real one + tap-disable) may spawn exactly
    one finish. The old race double-transcribed and double-pasted."""
    spy = PasteSpy()
    spy.install()
    model = FakeModel(gate=threading.Event())
    model.gate.set()  # transcribe returns immediately
    rec = pt.Recorder(model, dry_run=False, prewarm=False, batch=True)
    rec.press()
    speak(rec.current)
    t1 = threading.Thread(target=rec.release, args=(0,))
    t2 = threading.Thread(target=rec.release, args=(0,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    wait_until(lambda: not rec.busy, what="the one finish draining")
    assert spy.starts == 1, spy.starts
    assert len(model.generate_calls) == 1, len(model.generate_calls)


def test_streaming_round_trip():
    """Full session lifecycle in streaming mode: press -> start (state init,
    consumer, opener) -> partials from streaming_generate_step -> finish."""
    spy = PasteSpy()
    spy.install()
    model = FakeModel(text="streamed words")
    rec = pt.Recorder(model, dry_run=False, prewarm=False)  # streaming
    rec.press()
    s = rec.current
    assert s.state == {}  # init_streaming_state ran in start()
    speak(s, n=2)  # 6 samples -> two full windows
    wait_until(lambda: s.steps >= 2, what="streaming steps")
    rec.release(0)
    wait_until(lambda: not rec.busy, what="streaming finish")
    assert spy.starts == 1  # streaming pastes its joined partials too
    assert len(model.generate_calls) == 0  # never touched generate()
    assert s.parts and s.text_so_far


def test_release_with_nothing_recording_is_a_noop():
    spy = PasteSpy()
    spy.install()
    rec = pt.Recorder(FakeModel(), dry_run=False, prewarm=False, batch=True)
    rec.release(0)  # must not raise, spawn a finish, or paste
    assert not rec.active and not rec.busy
    assert spy.starts == 0


if __name__ == "__main__":
    install_fakes()
    test_press_accepted_while_previous_session_finalizes()
    test_duplicate_release_finishes_once()
    test_streaming_round_trip()
    test_release_with_nothing_recording_is_a_noop()
    print("lifecycle tests OK (4)")
