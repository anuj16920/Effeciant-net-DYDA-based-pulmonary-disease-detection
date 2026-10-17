"""Build the image manifest, data report and grouped splits from the configured sources.

Usage::

    python -m datasets.prepare_manifest --config configs/config.yaml

Outputs (paths from config):
    * ``manifest.csv``   - one row per usable image: image_path, label, source_dataset,
      patient_id, group_id, sha256, phash (+ raw_label, subtype, width, height)
    * ``data_report.json`` - missing sources/files, unreadable images, unmapped / excluded
      labels, class counts overall and per source, duplicate clusters, cross-source
      duplicates, label conflicts, warnings and limitations
    * ``splits.json``    - grouped+stratified held-out test + StratifiedGroupKFold folds

Grouping: images are united into one ``group_id`` if they share a SHA-256 (exact
duplicate), are within ``phash_hamming_threshold`` bits (near duplicate), or share a
parsed patient id. Splits are made on groups, so no group crosses train/val/test.

No data is downloaded; sources must be placed locally (see README).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from utils.common import save_json, setup_logging
from utils.config import get_path, load_config, project_root, resolve_path

from .preprocessing import ImageReadError, load_image
from .splits import make_splits, save_splits

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["image_path", "label", "source_dataset", "patient_id", "group_id", "sha256", "phash"]
EXTRA_COLUMNS = ["raw_label", "subtype", "width", "height"]


# --------------------------------------------------------------------------- hashing
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def phash(img: np.ndarray) -> str:
    """64-bit perceptual hash (DCT-based, same algorithm as ``imagehash.phash``) as 16 hex chars."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between uint64 hash arrays ``a`` (n,) and ``b`` (m,)."""
    x = np.ascontiguousarray(a[:, None] ^ b[None, :])
    return _POPCOUNT[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1, dtype=np.int32)


def near_duplicate_pairs(hashes_hex: Sequence[str], threshold: int, chunk: int = 256) -> List[Tuple[int, int]]:
    """All index pairs ``i < j`` whose pHash Hamming distance is ``<= threshold``."""
    if not hashes_hex:
        return []
    h = np.array([int(x, 16) for x in hashes_hex], dtype=np.uint64)
    pairs: List[Tuple[int, int]] = []
    for s in range(0, len(h), chunk):
        d = hamming_matrix(h[s:s + chunk], h)
        ii, jj = np.nonzero(d <= threshold)
        ii = ii + s
        keep = jj > ii
        pairs.extend(zip(ii[keep].tolist(), jj[keep].tolist()))
    return pairs


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# --------------------------------------------------------------------------- scanning
@dataclass
class Candidate:
    path: Path
    source: str
    raw_label: str
    label: Optional[str]
    subtype: str = ""
    patient_id: str = ""


@dataclass
class ScanResult:
    candidates: List[Candidate] = field(default_factory=list)
    missing_sources: Dict[str, str] = field(default_factory=dict)
    unmapped: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    unmapped_examples: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    excluded_explicit: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    excluded_subtype: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    unknown_subtype: Counter = field(default_factory=Counter)
    patient_id_missing: Counter = field(default_factory=Counter)
    skipped_non_image: Counter = field(default_factory=Counter)


def _raw_label(path: Path, root: Path, src: Dict[str, Any]) -> Optional[str]:
    """Raw label key from the configured rule, or None when no rule matches."""
    if src["label_from"] == "filename":
        m = re.search(src["filename_label_regex"], path.name)
        return m.group(1) if m else None
    keys = {k.lower(): k for k in src["label_map"]}
    for parent in path.relative_to(root).parents:
        if parent.name and parent.name.lower() in keys:
            return keys[parent.name.lower()]
    return None


def scan_sources(cfg: Dict[str, Any]) -> ScanResult:
    """Walk every enabled source, apply the explicit label mapping, and collect candidates."""
    res = ScanResult()
    exts = {e.lower() for e in cfg["manifest"]["image_extensions"]}
    excl = {d.lower() for d in cfg["manifest"]["exclude_dirs"]}
    for name, src in cfg["sources"].items():
        if not src["enabled"]:
            continue
        root = resolve_path(cfg, src["root"])
        if not root.is_dir():
            res.missing_sources[name] = str(root)
            logger.warning("Source '%s' not found at %s (skipped)", name, root)
            continue
        pid_rx = re.compile(src["patient_id_regex"]) if src["patient_id_regex"] else None
        subtypes = src.get("pneumonia_subtypes")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d.lower() not in excl)
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.suffix.lower() not in exts:
                    res.skipped_non_image[p.suffix.lower() or "<none>"] += 1
                    continue
                raw = _raw_label(p, root, src)
                if raw is None:
                    shown = (p.parent.relative_to(root).as_posix() if src["label_from"] == "folder"
                             else "<filename-no-match>")
                    res.unmapped[name][shown] += 1
                    if len(res.unmapped_examples[name]) < 5:
                        res.unmapped_examples[name].append(str(p))
                    continue
                label = src["label_map"][raw]
                if label is None:
                    res.excluded_explicit[name][raw] += 1
                    continue
                subtype = ""
                if subtypes and label == "Pneumonia":
                    low = fn.lower()
                    subtype = next((s for s in subtypes if s.lower() in low), "")
                    if not subtype:
                        res.unknown_subtype[name] += 1
                    elif not subtypes[subtype]:
                        res.excluded_subtype[name][subtype] += 1
                        continue
                pid = ""
                if pid_rx is not None:
                    m = pid_rx.search(fn)
                    if m:
                        pid = f"{name}:{m.group(1)}"
                    else:
                        res.patient_id_missing[name] += 1
                else:
                    res.patient_id_missing[name] += 1
                res.candidates.append(Candidate(p, name, str(raw), label, subtype, pid))
    return res


def _inspect(c: Candidate, min_side: int) -> Dict[str, Any]:
    try:
        img = load_image(c.path)
    except ImageReadError as exc:
        return {"ok": False, "reason": str(exc)}
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        return {"ok": False, "reason": f"image too small ({w}x{h} < {min_side})"}
    return {"ok": True, "sha256": sha256_file(c.path), "phash": phash(img), "width": w, "height": h}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


# --------------------------------------------------------------------------- manifest
def build_manifest(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Scan, validate, hash and group all images. Returns (manifest, report)."""
    mcfg = cfg["manifest"]
    scan = scan_sources(cfg)
    logger.info("Found %d labelled candidate images", len(scan.candidates))
    with ThreadPoolExecutor(max_workers=mcfg["hash_workers"]) as ex:
        info = list(ex.map(lambda c: _inspect(c, mcfg["min_image_side"]), scan.candidates))

    unreadable = [{"path": str(c.path), "source": c.source, "reason": i["reason"]}
                  for c, i in zip(scan.candidates, info) if not i["ok"]]
    good = [(c, i) for c, i in zip(scan.candidates, info) if i["ok"]]
    root = project_root(cfg)
    rows = [{"image_path": _rel(c.path, root), "label": c.label, "source_dataset": c.source,
             "patient_id": c.patient_id, "group_id": "", "sha256": i["sha256"], "phash": i["phash"],
             "raw_label": c.raw_label, "subtype": c.subtype, "width": i["width"], "height": i["height"]}
            for c, i in good]
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS + EXTRA_COLUMNS)
    df = df.sort_values("image_path", kind="mergesort").reset_index(drop=True)

    # ---- grouping: exact dup, near dup, same patient
    n = len(df)
    uf = UnionFind(n)
    by_sha: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(df["sha256"]):
        by_sha[s].append(i)
    for idx in by_sha.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    near = near_duplicate_pairs(df["phash"].tolist(), mcfg["phash_hamming_threshold"])
    near_only = [(i, j) for i, j in near if df.at[i, "sha256"] != df.at[j, "sha256"]]
    for i, j in near:
        uf.union(i, j)
    by_pid: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(df["patient_id"]):
        if p:
            by_pid[p].append(i)
    for idx in by_pid.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    roots = [uf.find(i) for i in range(n)]
    df["group_id"] = [f"grp_{r:06d}" for r in roots]

    # ---- duplicate clusters (exact or near) & label conflicts
    dup_members: Dict[str, set] = defaultdict(set)
    for idx in by_sha.values():
        if len(idx) > 1:
            dup_members[df.at[idx[0], "group_id"]].update(idx)
    for i, j in near:
        dup_members[df.at[i, "group_id"]].update((i, j))
    dup_clusters = []
    for gid, members in sorted(dup_members.items()):
        sub = df.loc[sorted(members)]
        dup_clusters.append({"group_id": gid, "size": len(members), "labels": sorted(sub["label"].unique()),
                             "sources": sorted(sub["source_dataset"].unique()),
                             "examples": sub["image_path"].head(4).tolist()})
    cross_source = [d for d in dup_clusters if len(d["sources"]) > 1]
    conflicts = []
    for gid, sub in df.groupby("group_id"):
        if sub["label"].nunique() > 1:
            conflicts.append({"group_id": gid, "size": len(sub), "labels": sorted(sub["label"].unique()),
                              "examples": sub["image_path"].head(4).tolist()})
    dropped_conflict = 0
    if conflicts and mcfg["drop_label_conflicts"]:
        bad = {c["group_id"] for c in conflicts}
        dropped_conflict = int(df["group_id"].isin(bad).sum())
        df = df[~df["group_id"].isin(bad)].reset_index(drop=True)

    report = make_report(cfg, df, scan, unreadable, dup_clusters, cross_source, conflicts, dropped_conflict,
                         n_exact_dup_files=sum(len(v) - 1 for v in by_sha.values() if len(v) > 1),
                         n_near_pairs=len(near_only))
    return df, report


def make_report(cfg: Dict[str, Any], df: pd.DataFrame, scan: ScanResult, unreadable: List[Dict[str, str]],
                dup_clusters: List[Dict[str, Any]], cross_source: List[Dict[str, Any]],
                conflicts: List[Dict[str, Any]], dropped_conflict: int, n_exact_dup_files: int,
                n_near_pairs: int) -> Dict[str, Any]:
    classes = cfg["classes"]
    warnings: List[str] = []
    per_source = {s: {c: int(((df["source_dataset"] == s) & (df["label"] == c)).sum()) for c in classes}
                  for s in sorted(df["source_dataset"].unique())}
    overall = {c: int((df["label"] == c).sum()) for c in classes}
    sources_per_class = {c: [s for s, cnt in per_source.items() if cnt[c] > 0] for c in classes}

    for name, root in scan.missing_sources.items():
        warnings.append(f"Source '{name}' is missing (expected at {root}); its classes are under-represented.")
    for c in classes:
        if overall[c] == 0:
            warnings.append(f"Class '{c}' has NO images. A 4-class model cannot be trained/evaluated properly.")
        elif len(sources_per_class[c]) == 1:
            warnings.append(
                f"SOURCE-SHORTCUT RISK: class '{c}' comes from a single source ({sources_per_class[c][0]}). "
                "The model may learn source/scanner artefacts instead of pathology. Inspect per-source metrics.")
    for s, cnt in per_source.items():
        present = [c for c in classes if cnt[c] > 0]
        if 0 < len(present) < len(classes):
            warnings.append(f"Source '{s}' only contains classes {present}; source identity correlates with label.")
    for s, k in scan.patient_id_missing.items():
        warnings.append(f"Source '{s}': {k} image(s) have no parseable patient id; grouping relies on "
                        "exact/near-duplicate clusters only (patients with several distinct images may leak).")
    if unreadable:
        warnings.append(f"{len(unreadable)} unreadable/too-small image(s) excluded.")
    unm = sum(sum(v.values()) for v in scan.unmapped.values())
    if unm:
        warnings.append(f"{unm} image(s) with unmapped labels excluded (see 'unmapped_labels').")
    if conflicts:
        warnings.append(f"{len(conflicts)} duplicate/patient group(s) carry conflicting labels"
                        f"{'; dropped ' + str(dropped_conflict) + ' images' if dropped_conflict else ''}.")
    if cross_source:
        warnings.append(f"{len(cross_source)} duplicate cluster(s) span multiple sources.")
    gsizes = df.groupby("group_id").size() if len(df) else pd.Series(dtype=int)
    if len(gsizes) and gsizes.max() > 100:
        warnings.append(f"Largest group has {int(gsizes.max())} images; pHash threshold may be chaining "
                        "distinct images. Consider lowering manifest.phash_hamming_threshold.")

    return {
        "n_images": int(len(df)),
        "n_groups": int(df["group_id"].nunique()) if len(df) else 0,
        "class_counts_overall": overall,
        "class_counts_per_source": per_source,
        "sources_per_class": sources_per_class,
        "missing_sources": scan.missing_sources,
        "unreadable_images": unreadable,
        "unmapped_labels": {s: dict(v) for s, v in scan.unmapped.items()},
        "unmapped_examples": dict(scan.unmapped_examples),
        "excluded_by_label_map": {s: dict(v) for s, v in scan.excluded_explicit.items()},
        "excluded_by_subtype": {s: dict(v) for s, v in scan.excluded_subtype.items()},
        "pneumonia_unknown_subtype": dict(scan.unknown_subtype),
        "skipped_non_image_files": dict(scan.skipped_non_image),
        "patient_id_missing": dict(scan.patient_id_missing),
        "duplicates": {
            "exact_duplicate_files": n_exact_dup_files,
            "near_duplicate_pairs": n_near_pairs,
            "phash_hamming_threshold": cfg["manifest"]["phash_hamming_threshold"],
            "clusters": dup_clusters,
            "cross_source_clusters": cross_source,
        },
        "label_conflicts": conflicts,
        "dropped_label_conflict_images": dropped_conflict,
        "max_group_size": int(gsizes.max()) if len(gsizes) else 0,
        "warnings": warnings,
        "limitations": [
            "Labels are taken from public dataset folder names/filenames and were not re-verified by radiologists.",
            "Classes are confounded with source datasets (scanner, hospital, preprocessing); high accuracy may "
            "reflect source shortcuts rather than pathology.",
            "Patient ids are only parseable for some sources; elsewhere grouping uses duplicate clusters only.",
            "Perceptual-hash near-duplicate detection can miss re-cropped images and can merge similar but "
            "distinct radiographs.",
        ],
    }


def run(cfg: Dict[str, Any], make_split_file: bool = True) -> int:
    """Build and write manifest, report and (optionally) splits for a loaded config."""
    df, report = build_manifest(cfg)
    out = get_path(cfg, "manifest")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    for w in report["warnings"]:
        logger.warning(w)
    if len(df) == 0:
        save_json(report, get_path(cfg, "data_report"))
        logger.error("No usable images found. Place datasets as described in README.md and re-run.")
        return 1
    if make_split_file:
        try:
            splits = make_splits(df, cfg)
            save_splits(splits, get_path(cfg, "splits"))
            report["splits"] = {"test_fraction_actual": splits["test_fraction_actual"],
                                "class_distribution": splits["class_distribution"]}
        except ValueError as exc:
            report["warnings"].append(f"Splits not created: {exc}")
            logger.error("Splits not created: %s", exc)
    save_json(report, get_path(cfg, "data_report"))
    logger.info("Manifest: %s (%d images, %d groups)", out, len(df), report["n_groups"])
    logger.info("Class counts: %s", report["class_counts_overall"])
    return 0


main_with_cfg = run


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build manifest.csv, data_report.json and splits.json")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-splits", action="store_true", help="only build manifest + report")
    args = ap.parse_args(argv)
    setup_logging()
    return run(load_config(args.config), make_split_file=not args.no_splits)


if __name__ == "__main__":
    raise SystemExit(main())
