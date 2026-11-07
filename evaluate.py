#!/usr/bin/env python3
"""Measure how well the matcher recognises your DATASETS (leave-one-out).

Each reference image is classified against all the OTHER images (the image
itself and its mirrored twin are excluded). Run:

    python evaluate.py                 # all 36 classes
    python evaluate.py --scope letters # letters only
    python evaluate.py --rebuild       # re-extract features first
    python evaluate.py --sample 2000   # quick check on 2000 random images

Note: if many photos come from the same person/session, leave-one-out is
optimistic because near-duplicate frames sit in the dataset. Treat it as an
upper bound; the real test is the live camera.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import numpy as np

import config as cfg
from dataset import load_references
from features import Descriptor
from matcher import ReferenceMatcher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["all", "letters", "numbers"], default="all")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--sample", type=int, default=0,
                    help="test only N random images (fast check, e.g. --sample 2000)")
    args = ap.parse_args()

    refs = load_references(use_cache=not args.rebuild,
                           progress=lambda i, n: print(f"\rFeatures {i}/{n}", end="", flush=True))
    print()
    if not len(refs):
        print(f"No usable images found in {cfg.DATASET_DIR}")
        return
    if refs.skipped:
        print(f"{len(refs.skipped)} images skipped, first few:")
        for s in refs.skipped[:8]:
            print("   ", s)

    allowed = {"all": None, "letters": cfg.LETTERS, "numbers": cfg.NUMBERS}[args.scope]
    matcher = ReferenceMatcher(refs)
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusions: Counter = Counter()
    total = correct = 0

    order = np.arange(len(refs))
    if args.sample and args.sample < len(refs):
        order = np.random.default_rng(0).choice(len(refs), args.sample, replace=False)
    t0 = time.time()

    for n_done, i in enumerate(order):
        if n_done % 200 == 0:
            eta = (time.time() - t0) / max(n_done, 1) * (len(order) - n_done)
            print(f"\rTesting {n_done}/{len(order)}  (about {eta / 60:.1f} min left)   ",
                  end="", flush=True)
        truth = str(refs.labels[i])
        if allowed is not None and truth not in allowed:
            continue
        q = Descriptor(refs.hog_gray[i], refs.hog_mask[i], refs.radial[i], refs.hu[i], refs.geom[i])
        pred = matcher.predict(q, allowed, exclude_src=int(refs.src[i]))
        total += 1
        per_class[truth][1] += 1
        if pred.label == truth:
            correct += 1
            per_class[truth][0] += 1
        else:
            confusions[(truth, pred.label)] += 1

    print(f"\r{' ' * 60}\nOverall: {correct}/{total} = {100 * correct / max(total, 1):.1f}%  (scope: {args.scope})")
    print("\nPer class (worst first):")
    for c, (ok, n) in sorted(per_class.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        print(f"  {c}: {100 * ok / max(n, 1):5.1f}%  ({ok}/{n})")
    print("\nMost common mistakes (true -> predicted):")
    for (t, p), n in confusions.most_common(12):
        print(f"  {t} -> {p}: {n}")


if __name__ == "__main__":
    main()