"""Render the three live plot callbacks without Qt, and verify saved RF playback.

This uses the exact plotting/history method bodies from gui.py with a real
Matplotlib Agg canvas and small non-rendering Qt widget doubles. It is NOT a
native Qt screenshot. Run: python examples/preview_three_plots.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from gui_plot_harness import PlotHarness
from ssrf_realtime.material_config import PopulationSnapshot
from ssrf_realtime.pulse_program import PulseProgram
from ssrf_realtime.playback import PulsePlaybackController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/'display_validation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT/'outputs'/'default_701_design'
    report = json.loads((source/'optimization_report.json').read_text())
    window = PlotHarness(size=(6.85, 5.05))
    window.model = PopulationSnapshot.load(source/'starting_state.npz').make_model()
    window.model.params.dt = report['recommended_playback_dt']
    window.model.set_program(PulseProgram.load(source/'rf_program.json'))
    window._set_R(-.94)
    window._start_new_trace(record_now=True)
    controller = PulsePlaybackController(window.model)
    controller.rf_on()
    while True:
        result = controller.advance(12, pause_at_end=True)
        window._record_trace_point()
        if result.completed:
            break
    window._update_plots()
    window.canvas.draw()
    # Render at the actual compact canvas dimensions and at 2x for a legible preview.
    for name, dpi in [('three_panel_preview.png', 180), ('compact_canvas.png', 100)]:
        window.fig.savefig(args.output/name, dpi=dpi)
    with np.load(source/'predicted_endpoint.npz', allow_pickle=False) as data:
        print('Endpoint archive keys:', data.files)
        endpoint = data['n'].copy()
    error = float(np.max(np.abs(window.model.n-endpoint)))
    state_before = window.model.n.copy()
    t = window.model.t
    window._update_plots()
    np.testing.assert_array_equal(window.model.n, state_before)
    assert window.model.t == t
    info = {
        'test_type': 'unchanged saved-program replay plus actual GUI plotting callbacks on Agg',
        'native_Qt': False,
        'active_bins_delivered': int(np.count_nonzero(window.model.delivered_exposure)),
        'duration': window.model.program_elapsed,
        'state_max_abs_difference_from_saved_endpoint': error,
        'polarizations': window.model.polarizations(),
        'spectral_tensor_equals_displayed_Iplus_minus_Iminus': bool(np.array_equal(
            window.line_tensor.get_ydata(),
            window.line_Ip.get_ydata()-window.line_Im.get_ydata())),
        'monitor_samples': len(window.trace_t),
        'mirror_plus_samples': len(window.trace_Ip_minusR),
        'mirror_minus_samples': len(window.trace_Im_minusR),
        'monitor_R': window.model.params.rf_burn_R,
    }
    (args.output/'replay_and_display_check.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))
    if error != 0.0:
        raise AssertionError('Saved population endpoint changed')


if __name__ == '__main__':
    main()
