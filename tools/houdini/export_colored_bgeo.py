#!/usr/bin/env python3
"""Import the master colored point cloud into Houdini as ``P`` + ``Cd`` points.

Runs inside Houdini/hython. The normal colorization backend never imports this
module or the ``hou`` package; Houdini is only required when BGEO export is
requested.

Usage::

    hython tools/houdini/export_colored_bgeo.py \
        --input colored_points.npz \
        --output colored_points.bgeo.sc

Attributes created:
    P                vector3 float (world position)
    Cd               vector3 float, range 0..1
    intensity        float
    color_confidence float
    color_count      int
"""
import argparse
from pathlib import Path


def _load_input(path):
    import numpy as np
    data = np.load(path, allow_pickle=False)
    required = ('points', 'Cd', 'color_confidence', 'color_count')
    for name in required:
        if name not in data.files:
            raise ValueError(f'{path.name} is missing the required "{name}" array')
    return data


def build_geometry(data):
    """Create Houdini point geometry; requires the ``hou`` module."""
    try:
        import hou
    except ImportError as error:
        raise RuntimeError(
            'Houdini is required for BGEO export. Run this script with hython '
            '(e.g. hython tools/houdini/export_colored_bgeo.py ...); the normal '
            'colorization pipeline does not require Houdini.') from error

    points = data['points'].astype('float32')
    cd = data['Cd'].astype('float32')
    intensity = data['intensity'] if 'intensity' in data.files else None
    confidence = data['color_confidence'].astype('float32')
    count = data['color_count'].astype('int32')

    geo = hou.Geometry()
    geo.createPoints(points.shape[0])
    geo.setPointFloatAttribValues('P', points.reshape(-1))
    geo.setPointFloatAttribValues('Cd', cd.reshape(-1))
    geo.setPointFloatAttribValues('color_confidence', confidence)
    geo.setPointIntAttribValues('color_count', count)
    if intensity is not None:
        geo.setPointFloatAttribValues('intensity', intensity)
    return geo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path,
                        help='colored_points.npz produced by the colorizer')
    parser.add_argument('--output', type=Path,
                        help='Output .bgeo.sc path (default: <input>.bgeo.sc)')
    args = parser.parse_args(argv)

    data = _load_input(args.input)
    geo = build_geometry(data)

    if args.output is None:
        args.output = args.input.with_suffix('.bgeo.sc')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    geo.saveToFile(str(output))
    print(output)


if __name__ == '__main__':
    main()
