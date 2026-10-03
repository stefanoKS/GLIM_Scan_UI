#!/usr/bin/env python3
"""Prepare paired NKSR inputs; --voxel-size uses meters (default 0.01)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ui/backend'))
from factory_mapping.reconstruction import main

if __name__ == '__main__':
    main()
