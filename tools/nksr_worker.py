#!/usr/bin/env python3
"""Run with NKSR_PYTHON, never the ROS interpreter."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ui/backend'))
from factory_mapping.nksr_worker import main
if __name__ == '__main__': sys.exit(main())
