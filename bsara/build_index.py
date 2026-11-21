"""Build data/index.csv: one row per processed image with its class and *group*.

A group is the physical stain an image came from. The drip dataset ships 3-5
versions of every drop (original photo, edited, cropped, red-channel,
contrast-enhanced), so splitting by image leaks the same drop into both train
and validation. All evaluation in v2 splits by group instead.

The v1 preprocess.py flattened folders and renamed collisions as
``<stem>_<counter><ext>``, discarding which drop a file came from. We recover
the mapping by replaying that exact walk (os.walk order is deterministic for
an unchanged folder) and verify every reconstructed file exists.
"""
import os
import csv
from pathlib import Path

from bsara.paths import RAW, PROCESSED, INDEX_CSV, IMG_EXTS


def replay_preprocess(cat):
    """Yield (processed_name, raw_path) exactly as v1 preprocess.py named them."""
    src_folder = RAW / cat
    seen, count = set(), 0
    for root, _dirs, files in os.walk(src_folder):
        for f in files:
            if not f.lower().endswith(IMG_EXTS):
                continue
            dst = f
            # Windows filenames are case-insensitive: v1's os.path.exists()
            # treated '1-7.JPG' and '1-7.jpg' as a collision.
            if dst.lower() in seen:
                base, ext = os.path.splitext(f)
                dst = f"{base}_{count}{ext}"
            seen.add(dst.lower())
            count += 1
            yield dst, Path(root) / f


def variant_of(name):
    n = name.lower()
    for tag in ("redchannel", "contenh", "psedit", "ps.", "edit2", "edit", "crop"):
        if tag in n:
            return tag.strip(".")
    return "photo"


def main():
    rows = []
    # Drip and gunshot: raw files exist, recover groups from folder structure.
    for cat in ("drip", "gunshot"):
        for proc_name, raw_path in replay_preprocess(cat):
            proc = PROCESSED / cat / proc_name
            if not proc.exists():
                raise SystemExit(f"Mapping failed: {proc} missing (raw: {raw_path})")
            rel = raw_path.relative_to(RAW / cat)
            if cat == "drip":
                # paper25cm/1-3/1-3edit.jpg -> group 'paper25cm/1-3'
                group = "/".join(rel.parts[:2]) if len(rel.parts) >= 3 else rel.stem
                height = rel.parts[0]
            else:
                group, height = rel.stem, ""
            rows.append(dict(path=f"{cat}/{proc_name}", cls=cat, group=f"{cat}:{group}",
                             variant=variant_of(proc_name), drop_height=height))

    # Impact: the full-resolution raw scans are no longer on disk, only the
    # processed 640x640 copies. Each scan is its own group.
    for f in sorted(os.listdir(PROCESSED / "impact_beating")):
        if f.lower().endswith(IMG_EXTS):
            rows.append(dict(path=f"impact_beating/{f}", cls="impact_beating",
                             group=f"impact_beating:{Path(f).stem}", variant="photo",
                             drop_height=""))

    on_disk = sum(1 for c in ("drip", "gunshot", "impact_beating")
                  for f in os.listdir(PROCESSED / c) if f.lower().endswith(IMG_EXTS))
    assert len(rows) == on_disk, f"indexed {len(rows)} but {on_disk} processed images exist"
    lowered = [r["path"].lower() for r in rows]
    assert len(set(lowered)) == len(lowered), "two index rows point at the same file"

    with open(INDEX_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    imgs = Counter(r["cls"] for r in rows)
    groups = Counter(r["cls"] for r in {r["group"]: r for r in rows}.values())
    for c in imgs:
        print(f"{c:15s} images={imgs[c]:5d}  groups={groups[c]:4d}")
    print(f"wrote {INDEX_CSV}")


if __name__ == "__main__":
    main()
