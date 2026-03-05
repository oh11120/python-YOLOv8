from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate YOLO label files")
    p.add_argument("--labels", default="data/yolo-flower/labels", help="labels root")
    p.add_argument("--nc", type=int, default=102)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.labels)
    bad = 0
    total = 0
    for txt in root.rglob("*.txt"):
        for line_no, line in enumerate(txt.read_text().splitlines(), 1):
            if not line.strip():
                continue
            total += 1
            parts = line.split()
            if len(parts) != 5:
                print(f"[FORMAT] {txt}:{line_no} -> {line}")
                bad += 1
                continue
            try:
                cls = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
            except ValueError:
                print(f"[PARSE] {txt}:{line_no} -> {line}")
                bad += 1
                continue
            if not (0 <= cls < args.nc):
                print(f"[CLASS] {txt}:{line_no} -> {cls}")
                bad += 1
            if any(not (0.0 <= v <= 1.0) for v in coords):
                print(f"[RANGE] {txt}:{line_no} -> {coords}")
                bad += 1
    print(f"checked_lines={total} issues={bad}")


if __name__ == "__main__":
    main()
