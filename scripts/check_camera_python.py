#!/usr/bin/env python3
"""Validate the system PyGObject / vendor typelib route (no pip camera SDK)."""
import gi

gi.require_version('Gst', '1.0')
gi.require_version('Tcam', '1.0')
from gi.repository import Gst, Tcam
import cv2
import cv_bridge

Gst.init(None)
assert Gst.ElementFactory.find('tcambin'), 'tcambin unavailable; source scripts/env.sh'
print('Imaging Source Python: PyGObject', gi.__version__, 'Tcam 1.0,', Gst.version_string())
print('ROS/system OpenCV:', cv2.__version__, 'cv_bridge:', cv_bridge.__file__)
