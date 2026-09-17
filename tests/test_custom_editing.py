"""Non-GUI tests for exact custom tables, atomic edits, and preserved dynamics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from ssrf_realtime.profile_editing import (
    nearest_bin, region_indices, ensure_region_rows, update_rows,
    parse_custom_table, custom_table_text,
)
from ssrf_realtime.pulse_program import RFProfile, BinPulse, PulseProgram
from ssrf_realtime.ideal_model import IdealBinModel, IdealBinParams

GRID = np.linspace(-3, 3, 701)


def test_region_adds_each_bin_without_overwriting_custom_values():
    profile = RFProfile('custom', pulses=[BinPulse(243, 7.125, .025, .034)])
    added = ensure_region_rows(profile, GRID, GRID[240], GRID[245], start=.125, duration=.4)
    assert added == 5
    assert [p.bin_index for p in profile.pulses] == list(range(240,246))
    retained = profile.pulses[3]
    assert (retained.rate, retained.start, retained.duration) == (7.125,.025,.034)
    assert all(p.rate == 0 for p in profile.pulses if p.bin_index != 243)
    assert ensure_region_rows(profile, GRID, GRID[240], GRID[245]) == 0


def test_region_exact_endpoints():
    assert list(region_indices(GRID, -3, -3)) == [0]
    assert list(region_indices(GRID, 3, 3)) == [700]
    assert len(region_indices(GRID, -3, 3)) == 701


@pytest.mark.parametrize('a,b', [(1,-1), (-3.001,0), (0,3.001), (float('nan'),1)])
def test_invalid_region_is_atomic(a,b):
    profile=RFProfile('test',pulses=[BinPulse(3,2,0,1)])
    with pytest.raises(ValueError):ensure_region_rows(profile,GRID,a,b)
    assert len(profile.pulses)==1


def test_update_exact_individual_and_selected_values():
    profile=RFProfile('test',pulses=[BinPulse(240,1,0,1),BinPulse(241,2,.1,.2)])
    update_rows(profile,[0],701,rate=1.234567890123,start=.00037,duration=.00041)
    assert profile.pulses[0].rate==1.234567890123
    assert profile.pulses[0].stop==pytest.approx(.00078)
    assert profile.pulses[1].rate==2
    update_rows(profile,[0,1],701,duration=.08765)
    assert profile.pulses[0].rate==1.234567890123
    assert profile.pulses[1].start==.1
    assert all(p.duration==.08765 for p in profile.pulses)


@pytest.mark.parametrize('kwargs', [{'rate':-1}, {'duration':float('nan')}, {'start':float('inf')}, {}])
def test_invalid_exact_update_is_atomic(kwargs):
    p=RFProfile('test',pulses=[BinPulse(20,1,0,1),BinPulse(21,2,0,1)])
    before=[vars(x).copy() for x in p.pulses]
    with pytest.raises(ValueError):update_rows(p,[0,1],701,**kwargs)
    assert [vars(x) for x in p.pulses]==before


@pytest.mark.parametrize('rows',[[],[-1],[2]])
def test_invalid_row_selection(rows):
    with pytest.raises(ValueError):update_rows(RFProfile('test',pulses=[BinPulse(20)]),rows,701,rate=2)


@pytest.mark.parametrize('text',[
    '240,1.35,0.0,0.25\n241,2.10,0.1,0.4',
    'bin_index,rate,start,duration\n240,1.35,0.0,0.25\n241,2.10,0.1,0.4',
    'bin\trate\tstart\tduration\n240\t1.35\t0.0\t0.25\n241\t2.10\t0.1\t0.4',
    '# demo\nbin_index rate start duration\n240 1.35 0 0.25\n241 2.10 0.1 0.4',
])
def test_four_column_paste(text):
    p=parse_custom_table(text,GRID)
    assert [x.bin_index for x in p]==[240,241]
    assert [x.rate for x in p]==[1.35,2.1]
    assert [x.duration for x in p]==[.25,.4]


def test_R_paste_snap_and_repeated_pulse():
    p=parse_custom_table('R,rate,start,duration\n-0.9,1.2,0,.3\n-0.9,2.4,1,.4',GRID)
    assert p[0].bin_index==p[1].bin_index==nearest_bin(GRID,-.9)
    assert p[1].start==1


@pytest.mark.parametrize('text',[
    '', 'bin_index,rate,start,duration', '1,2,3', '1,2,3,4,5',
    '701,2,0,1', '-1,2,0,1', '1.5,2,0,1', '2,nan,0,1',
    '2,1,0,-1', 'R,rate,start,duration\n4,2,0,1',
    'bin_index,duration,start,rate\n2,1,0,1',
])
def test_invalid_paste_is_rejected(text):
    with pytest.raises(ValueError):parse_custom_table(text,GRID)


def test_table_roundtrip_precision_and_scheduler(tmp_path):
    pulses=[BinPulse(240,1.234567890123,.00037,.00041),BinPulse(241,2.1,.2,.3),BinPulse(240,.8,.1,.07)]
    reconstructed=parse_custom_table(custom_table_text(pulses),GRID)
    assert [vars(p) for p in reconstructed]==[vars(p) for p in pulses]
    program=PulseProgram(profiles=[RFProfile('custom',pulses=reconstructed)])
    path=tmp_path/'program.json';program.save(path)
    loaded=PulseProgram.load(path)
    assert loaded.to_dict()==program.to_dict()
    model=IdealBinModel(IdealBinParams(diffusion_enabled=False,dnp_enabled=False,t1_rate=0))
    model.set_program(loaded);model.start_program();model.set_rf_enabled(True)
    while model.t<.52:model.step()
    expected=program.compile().exposure
    np.testing.assert_allclose(model.delivered_exposure,expected,rtol=0,atol=1e-13)


def test_physics_and_scheduler_source_unchanged():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'SOURCE_INTEGRITY.json').read_text())
    for relative, expected in manifest['unchanged_core_sha256'].items():
        actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
        assert actual==expected, f'Unrequested dynamics/scheduler modification: {relative}'


def test_copy_paste_preserves_disabled_rows():
    pulses=[BinPulse(3,2,.3,.4,False),BinPulse(5,1,0,1,True)]
    parsed=parse_custom_table(custom_table_text(pulses),GRID)
    assert [vars(x) for x in parsed]==[vars(x) for x in pulses]
