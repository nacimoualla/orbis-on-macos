#!/usr/bin/env python3
"""Create a Desktop link to this checkout's current app bundle, never a copy."""
from pathlib import Path
import argparse


def install(bundle, desktop):
    bundle = bundle.resolve()
    if not (bundle / 'Contents/MacOS/ps4-gui').is_file():
        raise ValueError(f'Build the app first: {bundle}')
    desktop.mkdir(parents=True, exist_ok=True)
    for name in ('PS4 to macOS.app', 'PS4 to macOS (Latest Build).app'):
        shortcut = desktop / name
        if shortcut.is_symlink() and shortcut.resolve() == bundle:
            return shortcut
        if not shortcut.exists() and not shortcut.is_symlink():
            shortcut.symlink_to(bundle, target_is_directory=True)
            return shortcut
    raise ValueError('Desktop shortcut names are occupied by other files; nothing was replaced.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent.parent / 'build/ps4-gui.app')
    parser.add_argument('--desktop', type=Path, default=Path.home() / 'Desktop')
    args = parser.parse_args()
    print(install(args.bundle, args.desktop))
