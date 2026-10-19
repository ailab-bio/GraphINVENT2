#!/usr/bin/env python3
"""
Build pDC50 labels for one target from the TPD-DB activity table.

    python scripts/tpddb-activity-labels.py --target SMARCA2 \
        --output data/surrogates/smarca2_pdc50.tsv

This is the single source of pDC50 for this project. The raw `Activity` column
is not a number: of 7,726 PROTAC DC50 rows, 3,923 are letter grades, 2,560 are
a number with a unit, and the rest are censored values, ranges, symbols or a
number with no unit at all. Within one target the units are mixed, so a
pipeline that calls `float()` on the column is wrong by three orders of
magnitude on most of it.

What this does, in order, reporting n at every step:

  - keeps DC50 and its replicate or assay suffixes, dropping Dmax, DIC50,
    rest, because pooling degradation potency with antiproliferative potency
    gives a label that is not one endpoint
  - requires a unit token, converts to molar, and takes pDC50 = -log10(M)
  - drops censored rows (a `>` is the inactives, so dropping them truncates the
    label distribution and that is recorded rather than hidden) and ranges
  - normalizes cell-line strings case-insensitively, since LNCaP and Lncap are
    the same line and would otherwise become two features
  - aggregates replicate (compound, cell line) rows by the geometric mean of
    DC50, so one compound with eight measurements contributes one row
  - carries the cell line and the E3 ligase as columns, for the surrogate to
    one-hot rather than pool away

The letter grades are written to a second file as ordered intervals. No row in
the table carries both a grade and a number, and no (compound, cell line) pair
appears once as a grade and once as a number, so the bin edges cannot be
derived empirically from this release. They are therefore left as an ordering
and the intended consumer is an interval-censored fit (`survival:aft`), which
needs the order and not the edges.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

UNITS = {"pm": 1e-12, "nm": 1e-9, "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "mm": 1e-3}
NUMERIC = re.compile(r"^\s*(\d+\.?\d*)\s*([pnuµμm]M)\s*\*?\s*$", re.I)
CENSORED = re.compile(r"[<>≤≥]")
GRADE = re.compile(r"^\s*([A-D])\s*$")
# A < B < C < D, most potent first. The direction is a property of the source
# release and is recorded here so a reader can check it rather than infer it.
GRADE_ORDER = ["A", "B", "C", "D"]
# DC50 and its replicate or assay suffixes, but not DIC50 and not Dmax.
DC50_TYPE = re.compile(r"^DC50(?:$|[_(\s])", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", type=Path, default=Path("data/raw/tpddb"))
    ap.add_argument(
        "--target", required=True, help="exact Target Symbols value, e.g. SMARCA2"
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    main_rows = list(
        csv.DictReader((args.raw / "PROTAC_main_table.txt").open(), delimiter="\t")
    )
    act_rows = list(
        csv.DictReader((args.raw / "PROTAC_activity.txt").open(), delimiter="\t")
    )
    by_id = {(r.get("TPD ID") or "").strip(): r for r in main_rows}

    counts = defaultdict(int)
    rows = [
        r
        for r in act_rows
        if (r.get("Target Symbols") or "").strip().lower() == args.target.lower()
    ]
    counts["rows for this target"] = len(rows)
    # DC50, DC50_1, DC50_2 and DC50(SMARCA2) are all DC50 measurements,
    # suffixed by replicate or assay. DIC50 is a different endpoint and Dmax
    # a different quantity, so neither is pooled in.
    dc50 = [r for r in rows if DC50_TYPE.match((r.get("Activity Type") or "").strip())]
    counts["DC50 rows"] = len(dc50)

    numeric, grades = [], []
    for r in dc50:
        raw = (r.get("Activity") or "").strip()
        if CENSORED.search(raw):
            counts["dropped: censored"] += 1
            continue
        if GRADE.match(raw):
            grades.append((r, GRADE.match(raw).group(1).upper()))
            continue
        m = NUMERIC.match(raw)
        if not m:
            counts["dropped: unparseable or no unit"] += 1
            continue
        molar = float(m.group(1)) * UNITS[m.group(2).lower().replace("u", "µ")]
        if molar <= 0:
            counts["dropped: non-positive"] += 1
            continue
        numeric.append((r, molar))
    counts["numeric with a unit"] = len(numeric)
    counts["letter grades"] = len(grades)

    def context(record):
        tpd = (record.get("TPD ID") or "").strip()
        parent = by_id.get(tpd, {})
        smiles = (parent.get("SMILES") or "").strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        return (
            tpd,
            Chem.MolToSmiles(mol) if mol is not None else None,
            (record.get("Cell Line") or "").strip().lower() or "unspecified",
            (parent.get("Ligase") or "").strip().lower() or "unspecified",
        )

    # Replicates of one (compound, cell line) collapse to the geometric mean of
    # DC50, which is the mean of pDC50.
    grouped = defaultdict(list)
    for record, molar in numeric:
        tpd, smiles, cell, ligase = context(record)
        if not smiles:
            counts["dropped: no parseable parent SMILES"] += 1
            continue
        grouped[(smiles, cell, ligase)].append(molar)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write("smiles\tpdc50\tcell_line\tligase\tn_replicates\n")
        for (smiles, cell, ligase), values in sorted(grouped.items()):
            geo = math.exp(sum(math.log(v) for v in values) / len(values))
            handle.write(
                f"{smiles}\t{-math.log10(geo):.4f}\t{cell}\t{ligase}\t{len(values)}\n"
            )
    counts["rows written"] = len(grouped)
    counts["distinct compounds"] = len({k[0] for k in grouped})
    counts["distinct cell lines"] = len({k[1] for k in grouped})

    # The grades, as ordered intervals for an interval-censored fit.
    graded = defaultdict(set)
    for record, grade in grades:
        _, smiles, cell, ligase = context(record)
        if smiles:
            graded[(smiles, cell, ligase)].add(grade)
    grade_path = args.output.with_name(args.output.stem + "_grades.tsv")
    with grade_path.open("w") as handle:
        handle.write("smiles\tgrade\tgrade_rank\tcell_line\tligase\n")
        for (smiles, cell, ligase), gs in sorted(graded.items()):
            if len(gs) != 1:
                counts["dropped: conflicting grades"] += 1
                continue
            grade = gs.pop()
            handle.write(
                f"{smiles}\t{grade}\t{GRADE_ORDER.index(grade)}\t{cell}\t{ligase}\n"
            )
    counts["graded rows written"] = sum(1 for v in graded.values() if len(v) == 1)

    print(f"* {args.target}", flush=True)
    for key in (
        "rows for this target",
        "exact DC50 rows",
        "numeric with a unit",
        "letter grades",
        "dropped: censored",
        "dropped: unparseable or no unit",
        "dropped: non-positive",
        "dropped: no parseable parent SMILES",
        "dropped: conflicting grades",
        "rows written",
        "distinct compounds",
        "distinct cell lines",
        "graded rows written",
    ):
        if counts[key]:
            print(f"    {key:38} {counts[key]}", flush=True)
    print(f"  numeric labels -> {args.output}", flush=True)
    print(f"  ordered grades -> {grade_path}", flush=True)

    provenance = args.output.with_name(args.output.stem + "_PROVENANCE.json")
    provenance.write_text(
        json.dumps(
            {
                "target": args.target,
                "endpoint": "pDC50 = -log10(DC50 in molar)",
                "activity_type_kept": "DC50 (exact match only)",
                "replicate_aggregation": "geometric mean of DC50 per (compound, cell line)",
                "censored_rows": "dropped, which truncates the label distribution at the "
                "inactive end; the retained range is the model's "
                "applicability window",
                "grade_edges": "not derivable from this release: no row carries both a "
                "grade and a number, and no (compound, cell line) pair "
                "appears as both. Grades are an ordering only, for an "
                "interval-censored fit.",
                "grade_order_most_to_least_potent": GRADE_ORDER,
                "counts": dict(counts),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"  provenance     -> {provenance}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
