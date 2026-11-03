"""Shipped material presets."""
from pathlib import Path
from .material_config import MaterialConfig

MATERIALS = Path(__file__).resolve().parents[1]/'materials'
DEFAULT = 'd_butanol_matched'


def material_path(name_or_path):
    """Resolve a short preset name (e.g. 'd_butanol_basic') or a file path."""
    p = Path(name_or_path)
    if p.suffix == '.json' and p.exists():
        return p
    candidate = MATERIALS/f'{name_or_path}.json'
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f'No material {name_or_path!r}; available: {", ".join(available())}')


def available():
    return sorted(p.stem for p in MATERIALS.glob('*.json'))


def default_material():
    return MaterialConfig.load(MATERIALS/f'{DEFAULT}.json')
