from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified-ish split for YOLO datasets")
    p.add_argument("--images-root", default="data/yolo-flower/images", help="images root")
    p.add_argument("--labels-root", default="data/yolo-flower/labels", help="labels root")
    p.add_argument("--out-root", default="data/yolo-flower-split", help="output root")
    p.add_argument("--classes", default="", help="optional classes.txt path")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    p.add_argument("--skip-missing-labels", action="store_true")
    p.add_argument("--write-yaml", action="store_true", default=True)
    return p.parse_args()


def load_labels(label_path: Path) -> List[int]:
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        labels.append(cls)
    return labels


def normalize_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    s = train + val + test
    if s <= 0:
        return 0.7, 0.2, 0.1
    return train / s, val / s, test / s


def assign_split(
    img_classes: List[int],
    split_counts: Dict[str, Dict[int, int]],
    target_counts: Dict[str, Dict[int, float]],
    split_sizes: Dict[str, int],
    target_sizes: Dict[str, int],
    rng: random.Random,
) -> str:
    candidates = []
    for split in ["train", "val", "test"]:
        if split_sizes[split] >= target_sizes[split]:
            continue
        deficit = 0.0
        for c in img_classes:
            deficit += max(target_counts[split].get(c, 0.0) - split_counts[split].get(c, 0), 0.0)
        candidates.append((deficit, split))
    if not candidates:
        return min(split_sizes, key=split_sizes.get)
    max_def = max(d for d, _ in candidates)
    best = [s for d, s in candidates if d == max_def]
    return rng.choice(best)


def transfer(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    train_r, val_r, test_r = normalize_ratios(args.train, args.val, args.test)

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    out_root = Path(args.out_root)
    classes_path = Path(args.classes) if args.classes else (labels_root.parent / "classes.txt")

    images = [p for p in images_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise SystemExit("No images found")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    image_infos = []
    class_totals: Dict[int, int] = {}

    for img in images:
        rel = img.relative_to(images_root)
        label_path = (labels_root / rel).with_suffix(".txt")
        if not label_path.exists() and not args.skip_missing_labels:
            raise SystemExit(f"Missing label: {label_path}")
        if not label_path.exists():
            continue
        classes = load_labels(label_path)
        for c in classes:
            class_totals[c] = class_totals.get(c, 0) + 1
        image_infos.append((img, label_path, rel, classes))

    image_infos.sort(key=lambda x: len(x[3]), reverse=True)

    total_images = len(image_infos)
    target_sizes = {
        "train": int(total_images * train_r),
        "val": int(total_images * val_r),
        "test": total_images,
    }
    target_sizes["test"] = total_images - target_sizes["train"] - target_sizes["val"]

    target_counts = {
        "train": {c: v * train_r for c, v in class_totals.items()},
        "val": {c: v * val_r for c, v in class_totals.items()},
        "test": {c: v * test_r for c, v in class_totals.items()},
    }

    split_counts: Dict[str, Dict[int, int]] = {"train": {}, "val": {}, "test": {}}
    split_sizes = {"train": 0, "val": 0, "test": 0}
    assignments: Dict[str, List[Tuple[Path, Path, Path]]] = {"train": [], "val": [], "test": []}

    for img, label, rel, classes in image_infos:
        split = assign_split(classes, split_counts, target_counts, split_sizes, target_sizes, rng)
        assignments[split].append((img, label, rel))
        split_sizes[split] += 1
        for c in classes:
            split_counts[split][c] = split_counts[split].get(c, 0) + 1

    for split, items in assignments.items():
        for img, label, rel in items:
            dst_img = out_root / "images" / split / rel
            dst_lbl = out_root / "labels" / split / rel.with_suffix(".txt")
            transfer(img, dst_img, args.mode)
            transfer(label, dst_lbl, args.mode)

    summary = {
        "total_images": total_images,
        "split_sizes": split_sizes,
        "ratios": {"train": train_r, "val": val_r, "test": test_r},
    }
    (out_root / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.write_yaml:
        yaml_path = out_root / "data.yaml"
        lines = [
            f"path: {out_root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
        ]
        if classes_path.exists():
            names = [n.strip() for n in classes_path.read_text(encoding="utf-8").splitlines() if n.strip()]
            lines.append(f"nc: {len(names)}")
            lines.append("names:")
            for i, name in enumerate(names):
                lines.append(f"  {i}: {name}")
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Split done")
    print(summary)


if __name__ == "__main__":
    main()
