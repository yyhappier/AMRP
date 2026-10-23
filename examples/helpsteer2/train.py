#!/usr/bin/env python3
"""HelpSteer2 three/five-reward experiment entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import main

if __name__ == '__main__':
    main('helpsteer2')
