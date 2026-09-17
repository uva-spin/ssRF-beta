"""RF-button orchestration on the existing, unmodified pulse scheduler.

There are two distinct backend conditions: an enabled RF gate AND a started
program.  The main RF ON action must arrange both.  File loading must arrange
neither.  This module contains no population equations or pulse optimization.
It is independent of Qt so the exact playback path can be tested headlessly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PlaybackUpdate:
    completed: bool = False
    pause_requested: bool = False


class PulsePlaybackController:
    """Explicit start/replay, RF mute, completion and optional endpoint hold."""

    def __init__(self, model):
        self.model = model

    def _validate_start(self) -> None:
        preview = self.model.preview_rf_field()
        if self.model.program.gain <= 0:
            raise ValueError("Global RF gain is zero. Set it above zero before starting RF.")
        if (self.model._compiled.end_time <= 0 or not np.all(np.isfinite(preview))
                or not np.any(preview > 0)):
            raise ValueError("No enabled pulses with positive power and duration. "
                             "Load a program or enable pulse rows in the per-bin editor.")

    def rf_on(self) -> str:
        """Start a ready program, replay a finished one, or unmute an active one.

        The start epoch is the CURRENT simulation time, never absolute time 0.
        Replaying does not restore the population snapshot or reset P/Q.
        """
        self._validate_start()
        state = self.model.program_state
        if state in ("ready", "stopped", "finished"):
            self.model.start_program(turn_rf_on=True)
            return "started" if state == "ready" else "restarted"
        self.model.set_rf_enabled(True)
        return "resumed"

    def restart(self) -> str:
        self._validate_start()
        self.model.start_program(turn_rf_on=True)
        return "restarted"

    def rf_off(self) -> None:
        # Preserve historical mute behavior: the schedule advances if the
        # simulation keeps running. The separate Pause control freezes both.
        self.model.set_rf_enabled(False)

    def advance(self, n_steps: int, *, dnp_on=None,
                pause_at_end: bool = False) -> PlaybackUpdate:
        """Advance with the existing model.step; optionally stop at the endpoint.

        A short program can otherwise finish AND recover before the next GUI
        redraw. Endpoint hold captures n(T), without changing any pulse or
        physical coefficient. It shortens only the final numerical step.
        """
        if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        n_steps = int(n_steps)
        m = self.model
        prior = m.program_state
        active_schedule = prior in ("running", "waiting")
        endpoint = (float(m._event_times_absolute[-1])
                    if active_schedule else None)
        dt = float(m.params.dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # Far from the endpoint, retain exactly the original stepping path.
        if (pause_at_end and active_schedule
                and endpoint - m.t <= n_steps * dt + 8 * math.ulp(endpoint)):
            for _ in range(n_steps):
                remaining = endpoint - m.t
                if remaining <= 0:
                    break
                try:
                    m.params.dt = min(dt, remaining)
                    m.step(n_steps=1, dnp_on=dnp_on)
                finally:
                    m.params.dt = dt
        else:
            m.step(n_steps=n_steps, dnp_on=dnp_on)
        completed = active_schedule and m.program_state == "finished"
        if completed:
            # Do not leave a misleading RF ON indicator after a one-shot pulse.
            # Retain the schedule, epoch and exposure receipt for inspection.
            m.set_rf_enabled(False)
        return PlaybackUpdate(completed=completed,
                              pause_requested=bool(completed and pause_at_end))

    def button_text(self, *, paused=False) -> str:
        state = self.model.program_state
        if self.model.params.rf_enabled:
            if paused:
                return "RF ON — simulation paused"
            if state == "waiting":
                return "RF ARMED — waiting for pulse"
            return "RF ON — program running"
        if state == "finished":
            return "RF OFF — finished (click to replay)"
        if state in ("ready", "stopped"):
            return "RF OFF — click to start program"
        return "RF OFF — muted (click to resume)"

    def status_text(self, *, paused=False) -> str:
        m = self.model
        phase = m.program_state
        actual = m.applied_rf_field()
        count = int(np.count_nonzero(actual))
        exposed = int(np.count_nonzero(m.delivered_exposure > 0))
        if phase in ("ready", "stopped"):
            description = ("READY — press RF ON to start the loaded program"
                           if phase == "ready" else
                           "STOPPED — press RF ON to replay the loaded program")
        elif phase == "finished":
            description = "FINISHED — RF OFF; click RF ON to replay"
        elif not m.params.rf_enabled:
            description = "MUTED — schedule still advances unless simulation is paused"
        elif phase == "waiting":
            description = "WAITING — RF enabled; no pulse is scheduled at this instant"
        else:
            description = "RUNNING — pulse commands enabled"
        suffix = " [simulation paused]" if paused else ""
        return (f"{description}{suffix}\n"
                f"Program t={m.program_elapsed:.6g} / {m._compiled.end_time:.6g}\n"
                f"{count} bins commanded now; peak U={float(np.max(actual)):.6g}\n"
                f"Delivered this run: {exposed} bins; sum(U dt)="
                f"{float(np.sum(m.delivered_exposure)):.6g}")
