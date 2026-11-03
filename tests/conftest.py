"""Make tests/fixture_materials.py importable regardless of other installed 'tests' packages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
