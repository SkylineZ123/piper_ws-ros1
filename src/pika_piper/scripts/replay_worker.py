#!/usr/bin/env python3
"""Internal physical replay entry; select --control-python through replay.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.replay_executor import main

if __name__ == '__main__':
    main()
