#!/usr/bin/env python3
"""Compatibility entry for doctor.py --role control (no device access)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

from utils.environment_checks import check_control


def main():
    report = check_control()
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
