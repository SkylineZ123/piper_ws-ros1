#!/usr/bin/env python3
"""Check data or control dependencies without starting ROS or opening devices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

from utils.environment_checks import check_control, check_data
from utils.reproducibility import runtime_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('data', 'control'), default='data')
    parser.add_argument('--output', help='Also write the JSON report to this file')
    args = parser.parse_args(argv)
    report = (check_data if args.role == 'data' else check_control)()
    report.update(role=args.role, runtime=runtime_snapshot(), hardware_started=False)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    print(payload, end='')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
