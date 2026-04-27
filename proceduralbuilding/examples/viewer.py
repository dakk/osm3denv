#!/usr/bin/env python3
"""
Standalone Panda3D viewer for the procbuilding library.

Usage:
    python examples/viewer.py [--type TYPE] [--floors N] [--roof flat|gable|hip]
                              [--width W] [--depth D] [--pitch P]
"""
from procbuilding._viewer import main

if __name__ == "__main__":
    main()
