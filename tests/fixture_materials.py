"""Legacy materials used as physics-test fixtures (not shipped as user materials)."""
from pathlib import Path
from ssrf_realtime.material_config import MaterialConfig

FIXTURES = Path(__file__).resolve().parent/'fixtures'


def fixture_material(name='balanced_peak_selectivity'):
    return MaterialConfig.load(FIXTURES/'materials'/f'{name}.json')


def default_material():
    """The pre-production demo default that the regression tests were written against."""
    return fixture_material('balanced_peak_selectivity')
