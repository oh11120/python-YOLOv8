from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check class balance for YOLO dataset")
    p.add_argument("--root", default="data/yolo-flower", help="dataset root containing labels/")
    p.add_argument("--out", default="", help="optional csv output path")
    return p.parse_args()


def load_labels(label_path: Path) -> List[int]:
    labels = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            cls = int(float(line.split()[0]))
        except Exception:
            continue
        labels.append(cls)
    return labels


def collect_counts(labels_root: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for txt in labels_root.rglob("*.txt"):
        for cls in load_labels(txt):
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    splits = ["train", "val", "test"]
    data = []
    all_counts: Dict[int, int] = {}

    for split in splits:
        lbl_root = root / "labels" / split
        if not lbl_root.exists():
            continue
        counts = collect_counts(lbl_root)
        total = sum(counts.values())
        for cls, cnt in counts.items():
            all_counts[cls] = all_counts.get(cls, 0) + cnt
            data.append({"split": split, "class": cls, "count": cnt, "pct": cnt / total if total else 0.0})

    df = pd.DataFrame(data)
    df_all = pd.DataFrame(
        [{"split": "all", "class": cls, "count": cnt} for cls, cnt in sorted(all_counts.items())]
    )

    if args.out:
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        df_all.to_csv(out_path.with_name(out_path.stem + "_all.csv"), index=False)

    print("Class balance summary (top 10 by count):")
    if not df_all.empty:
        print(df_all.sort_values("count", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
