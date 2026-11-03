"""Launch the ssRF simulator GUI.

    python3 run_app.py                         # experimental d-butanol material
    python3 run_app.py d_butanol_basic      # any preset name from materials/
    python3 run_app.py my_material.json --program programs/reference_0p1W_burn.json
    python3 run_app.py --list                  # list preset materials

The window opens paused with RF off. Other options (--state) are passed to the GUI.
"""
import sys
from ssrf_realtime.presets import available, material_path

if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] in ('--list', '-l', '-h', '--help'):
        print(__doc__); print('Materials:', ', '.join(available())); raise SystemExit(0)
    if args and not args[0].startswith('-'):
        args = ['--material', str(material_path(args[0]))] + args[1:]
    from ssrf_realtime.designer_gui import main
    raise SystemExit(main(args))
