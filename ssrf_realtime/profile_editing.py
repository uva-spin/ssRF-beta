"""Pure, tested helpers for custom-bin editing. No dynamical changes."""
from __future__ import annotations
from dataclasses import replace
import csv
import io
import math
import re
from typing import Optional
import numpy as np
from .pulse_program import BinPulse, RFProfile


def nearest_bin(grid, value) -> int:
    grid = np.asarray(grid, dtype=float)
    x = float(value)
    if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise ValueError('A finite increasing grid is required.')
    if not math.isfinite(x) or x < grid[0] or x > grid[-1]:
        raise ValueError(f'R must be in [{grid[0]:g}, {grid[-1]:g}].')
    return int(np.argmin(np.abs(grid - x)))


def region_indices(grid, left, right):
    if float(left) > float(right):
        raise ValueError('Left R must not exceed Right R.')
    a, b = nearest_bin(grid, left), nearest_bin(grid, right)
    return range(a, b + 1)


def ensure_region_rows(profile: RFProfile, grid, left, right, *, start=0.0, duration=1.0) -> int:
    """Add every missing bin at zero RF power; preserve all existing pulses."""
    n = len(grid)
    existing = {p.bin_index for p in profile.pulses}
    additions = [BinPulse(k, 0.0, float(start), float(duration))
                 for k in region_indices(grid, left, right) if k not in existing]
    for pulse in additions:
        pulse.validate(n)
    profile.pulses.extend(additions)
    profile.pulses.sort(key=lambda p: (p.bin_index, p.start))
    return len(additions)


def update_rows(profile: RFProfile, rows, n_bins: int, *, rate: Optional[float] = None,
                start: Optional[float] = None, duration: Optional[float] = None) -> None:
    """Atomic exact-value changes, including multi-row updates."""
    rows = sorted(set(rows))
    if not rows:
        raise ValueError('Select at least one pulse row first.')
    if any(type(i) is not int or i < 0 or i >= len(profile.pulses) for i in rows):
        raise ValueError('Selected pulse row is outside the table.')
    changes = {name: value for name, value in (('rate', rate), ('start', start), ('duration', duration))
               if value is not None}
    if not changes:
        raise ValueError('Check at least one value to apply.')
    candidates = []
    for i in rows:
        pulse = replace(profile.pulses[i], **changes)
        pulse.validate(n_bins)
        candidates.append((i, pulse))
    for i, pulse in candidates:
        profile.pulses[i] = pulse


def custom_table_text(pulses, grid=None) -> str:
    """Four-column text suitable for copying into the explicit paste dialog."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['bin_index', 'rate', 'start', 'duration', 'enabled'])
    for p in pulses:
        writer.writerow([p.bin_index, repr(p.rate), repr(p.start), repr(p.duration), int(p.enabled)])
    return out.getvalue()


def parse_custom_table(text: str, grid):
    """Read bin_index,rate,start,duration or R,rate,start,duration.

    An optional fifth enabled column preserves disabled pulses on copy/paste.

    Accept comma-, tab-, or whitespace-separated rows. An R header is required
    to interpret the first column as frequency. Multiple rows per bin are
    deliberately allowed; this is how repeated pulses are represented.
    """
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if not lines:
        raise ValueError('The custom table is empty.')
    def split(line):
        if ',' in line:
            return next(csv.reader([line]))
        return re.split(r'\s+', line.strip())
    rows = [[field.strip() for field in split(line)] for line in lines]
    first = [field.lower() for field in rows[0]]
    mode = 'bin'
    offset = 1
    if first[0] in ('bin', 'bin_index', 'r'):
        if len(first) not in (4, 5) or first[1:4] != ['rate', 'start', 'duration'] or (len(first) == 5 and first[4] != 'enabled'):
            raise ValueError('Header must be bin_index,rate,start,duration or R,rate,start,duration, optionally followed by enabled.')
        mode = 'r' if first[0] == 'r' else 'bin'
        rows = rows[1:]
        offset = 2
    if not rows:
        raise ValueError('The custom table has no pulse rows.')
    pulses = []
    for line_number, fields in enumerate(rows, start=offset):
        try:
            if len(fields) not in (4, 5):
                raise ValueError('expected four columns, optionally followed by enabled')
            if mode == 'r':
                k = nearest_bin(grid, float(fields[0]))
            else:
                k = int(fields[0])  # int('12.5') is rejected, not silently rounded.
            enabled = True
            if len(fields) == 5:
                if fields[4].lower() not in ('0', '1', 'false', 'true'):
                    raise ValueError('enabled must be 0/1 or false/true')
                enabled = fields[4].lower() in ('1', 'true')
            pulse = BinPulse(k, float(fields[1]), float(fields[2]), float(fields[3]), enabled)
            pulse.validate(len(grid))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'Row {line_number}: {exc}') from exc
        pulses.append(pulse)
    return pulses
