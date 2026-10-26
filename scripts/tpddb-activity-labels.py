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
  - resolves the free-text target to a canonical symbol set through
    `data/surrogates/target_symbol_map.tsv` and writes it as `poi`, with the
    raw string beside it as `poi_raw`. `--target all` drops the single-target
    filter and writes the pooled multi-target table, which is what
    POI-grouped cross-validation holds out folds from and what the pooled
    ensemble oracle is fitted on. An unmapped target string raises.

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
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
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

# Dmax and its replicate suffixes. A glue row can carry the assay concentration
# in the type, as `Dmax(10nM)`, and those are NOT pooled with a plain `Dmax`:
# maximum degradation at 10 nM and at 20 µM are different measurements of
# different things, and averaging them produces a label that is neither.
DMAX_TYPE = re.compile(r"^Dmax(?:$|[_\s])", re.I)

# Dmax is a percentage of the protein degraded, so it carries no unit token and
# a plain number is complete. Values above 100 are percent-of-control artifacts
# rather than measurements and are dropped rather than clipped, because
# clipping invents a 100% that nobody observed.
PERCENT = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*%?\s*$")
DMAX_CEILING = 100.0

DEFAULT_TARGET_MAP = Path("data/surrogates/target_symbol_map.tsv")


def load_target_map(path: Path) -> dict[str, tuple[frozenset[str], str]]:
    """
    Read the curated raw-string -> canonical symbol set map.

    `Target Symbols` is free text: the same protein arrives as `AR`,
    `Androgen receptor`, `AR L702H` and `AR-V7`, and `KRAS/CRAF` names two
    proteins at once. POI-grouped cross-validation estimates the error on an
    unseen target, and that estimate is only honest if no molecule measured
    against a protein sits in the training folds under another spelling. The
    map is data rather than code because it is a chemistry judgment a reader
    has to be able to audit; its header records the rules it was built under.
    """
    if not path.exists():
        raise SystemExit(
            f"The target-symbol map {path} does not exist.\n"
            "It carries the raw-string -> canonical-symbol mapping that makes "
            "the `poi` column meaningful. Pass --target-map to point at it, or "
            "restore it; it is committed."
        )
    table: dict[str, tuple[frozenset[str], str]] = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip() or line.startswith("#") or line.startswith("raw\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise SystemExit(
                f"{path}:{number} has {len(parts)} fields, expected at least 3 "
                f"(raw, symbols, kind): {line!r}"
            )
        raw, symbols, kind = parts[0], parts[1], parts[2]
        if kind not in ("poi", "non_human", "ambiguous", "unspecified"):
            raise SystemExit(
                f"{path}:{number} has kind {kind!r}; implemented kinds are "
                "poi, non_human, ambiguous, unspecified."
            )
        resolved = frozenset(s for s in symbols.split(";") if s and s != "-")
        table[raw] = (resolved, kind)
    return table


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", type=Path, default=Path("data/raw/tpddb"))
    ap.add_argument(
        "--target",
        required=True,
        help="exact Target Symbols value, e.g. SMARCA2, or `all` for the "
        "pooled multi-target table. The pooled table is what POI-grouped "
        "cross-validation needs: with one target per file there is no unseen "
        "target to hold out, and it is also the table the pooled-ensemble "
        "oracle design in CLAUDE.md is fitted on.",
    )
    ap.add_argument(
        "--target-map",
        type=Path,
        default=DEFAULT_TARGET_MAP,
        help="the curated raw-string -> canonical-symbol map that fills the "
        "`poi` column",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--endpoint",
        choices=("pdc50", "dmax"),
        default="pdc50",
        help=(
            "pdc50 is -log10(DC50 in molar), the potency. dmax is the maximum "
            "fraction of the protein degraded, in [0, 1]. They are different "
            "questions: a degrader can be potent and only ever remove half the "
            "protein, and an optimization that reads one as the other will "
            "chase the wrong molecule."
        ),
    )
    ap.add_argument(
        "--modality",
        choices=("protac", "glue"),
        default="protac",
        help=(
            "which pair of TPD-DB tables to read. The two are never pooled: a "
            "PROTAC presents a defined E3 ligand and a glue has none, the size "
            "regimes differ by more than a factor of two, and a pooled model "
            "would be dominated by molecular size rather than by chemistry."
        ),
    )
    args = ap.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    prefix = "PROTAC" if args.modality == "protac" else "MG"
    main_rows = list(
        csv.DictReader((args.raw / f"{prefix}_main_table.txt").open(), delimiter="\t")
    )
    act_rows = list(
        csv.DictReader((args.raw / f"{prefix}_activity.txt").open(), delimiter="\t")
    )
    by_id = {(r.get("TPD ID") or "").strip(): r for r in main_rows}

    target_map = load_target_map(args.target_map)
    pooled = args.target.lower() == "all"

    counts = defaultdict(int)
    if pooled:
        rows = list(act_rows)
    else:
        rows = [
            r
            for r in act_rows
            if (r.get("Target Symbols") or "").strip().lower() == args.target.lower()
        ]
    counts["rows for this target"] = len(rows)

    # Resolve every row's target to a canonical symbol set before anything
    # else, so an out-of-date map fails here rather than producing a `poi`
    # column that silently splits one protein across two groups.
    excluded_kinds = defaultdict(int)
    kept_rows = []
    for r in rows:
        raw_target = (r.get("Target Symbols") or "").strip()
        if raw_target not in target_map:
            raise SystemExit(
                f"The target string {raw_target!r} is not in {args.target_map}.\n"
                "An unmapped string means the map is out of date, and grouping "
                "on the raw string would leak the target across folds. Add a "
                "row for it: raw <TAB> semicolon-separated canonical symbols "
                "<TAB> kind <TAB> note. The file's header states the curation "
                "rules."
            )
        symbols, kind = target_map[raw_target]
        if kind != "poi" or not symbols:
            # `ambiguous` could not be resolved to a protein, `non_human`
            # shares no chemistry axis with the human panel, `unspecified` is
            # TPDdb's placeholder. Excluded, counted and reported -- never
            # given a group of their own, which would be a fold that holds out
            # nothing identifiable.
            excluded_kinds[f"{kind}: {raw_target}"] += 1
            continue
        # The canonical symbol set, joined, travels on the row itself: a
        # semicolon-separated sorted set, so `KRAS/CRAF` and `CRAF/KRAS` give
        # one string and the trainer's union-find sees one edge.
        r["__poi__"] = ";".join(sorted(symbols))
        r["__poi_raw__"] = raw_target
        kept_rows.append(r)
    if excluded_kinds:
        counts["dropped: target not a resolved human protein"] = sum(
            excluded_kinds.values()
        )
    rows = kept_rows
    # DC50, DC50_1, DC50_2 and DC50(SMARCA2) are all DC50 measurements,
    # suffixed by replicate or assay. DIC50 is a different endpoint and Dmax
    # a different quantity, so neither is pooled in.
    is_dmax = args.endpoint == "dmax"
    matcher = DMAX_TYPE if is_dmax else DC50_TYPE
    selected = [
        r for r in rows if matcher.match((r.get("Activity Type") or "").strip())
    ]
    counts[f"{args.endpoint} rows"] = len(selected)

    numeric, grades = [], []
    for r in selected:
        raw = (r.get("Activity") or "").strip()
        if CENSORED.search(raw):
            counts["dropped: censored"] += 1
            continue
        if GRADE.match(raw):
            grades.append((r, GRADE.match(raw).group(1).upper()))
            continue
        if is_dmax:
            m = PERCENT.match(raw)
            if not m:
                counts["dropped: unparseable"] += 1
                continue
            percent = float(m.group(1))
            if percent > DMAX_CEILING:
                counts["dropped: above 100 percent"] += 1
                continue
            if percent < 0:
                counts["dropped: negative"] += 1
                continue
            numeric.append((r, percent / 100.0))
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
    counts["numeric kept"] = len(numeric)
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
            record["__poi__"],
            record["__poi_raw__"],
        )

    # Replicates of one (compound, cell line) collapse to the geometric mean of
    # DC50, which is the mean of pDC50.
    # The POI is part of the key. One compound can carry endpoints against two
    # proteins, and a key of (compound, cell line, ligase) alone would average
    # a SMARCA2 measurement with a SMARCA4 one into a label that is neither.
    # In single-target mode the column is constant, which keeps the two modes'
    # tables the same shape.
    grouped = defaultdict(list)
    for record, value in numeric:
        tpd, smiles, cell, ligase, poi, poi_raw = context(record)
        if not smiles:
            counts["dropped: no parseable parent SMILES"] += 1
            continue
        grouped[(smiles, cell, ligase, poi, poi_raw)].append(value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write(
            f"smiles\t{args.endpoint}\tcell_line\tligase\tpoi\tpoi_raw\tn_replicates\n"
        )
        for (smiles, cell, ligase, poi, poi_raw), values in sorted(grouped.items()):
            if is_dmax:
                # Dmax is a bounded fraction, so replicates average
                # arithmetically. pDC50 is a log, so its replicates average as
                # the geometric mean of DC50, which is the mean of pDC50.
                label = sum(values) / len(values)
            else:
                geo = math.exp(sum(math.log(v) for v in values) / len(values))
                label = -math.log10(geo)
            handle.write(
                f"{smiles}\t{label:.4f}\t{cell}\t{ligase}\t{poi}\t{poi_raw}\t"
                f"{len(values)}\n"
            )
    counts["rows written"] = len(grouped)
    counts["distinct compounds"] = len({k[0] for k in grouped})
    counts["distinct cell lines"] = len({k[1] for k in grouped})
    counts["distinct canonical POIs"] = len({k[3] for k in grouped})
    counts["distinct raw target strings"] = len({k[4] for k in grouped})

    # The grades, as ordered intervals for an interval-censored fit.
    graded = defaultdict(set)
    for record, grade in grades:
        _, smiles, cell, ligase, poi, poi_raw = context(record)
        if smiles:
            graded[(smiles, cell, ligase, poi, poi_raw)].add(grade)
    grade_path = args.output.with_name(args.output.stem + "_grades.tsv")
    with grade_path.open("w") as handle:
        handle.write("smiles\tgrade\tgrade_rank\tcell_line\tligase\tpoi\tpoi_raw\n")
        for (smiles, cell, ligase, poi, poi_raw), gs in sorted(graded.items()):
            if len(gs) != 1:
                counts["dropped: conflicting grades"] += 1
                continue
            grade = gs.pop()
            handle.write(
                f"{smiles}\t{grade}\t{GRADE_ORDER.index(grade)}\t{cell}\t{ligase}\t"
                f"{poi}\t{poi_raw}\n"
            )
    counts["graded rows written"] = sum(1 for v in graded.values() if len(v) == 1)

    print(f"* {'all targets (pooled)' if pooled else args.target}", flush=True)
    for key in (
        "rows for this target",
        f"{args.endpoint} rows",
        # This used to print "exact DC50 rows" and "numeric with a unit",
        # neither of which any code path sets, so the count that survives
        # parsing was omitted from the table it belongs in.
        "numeric kept",
        "letter grades",
        "dropped: censored",
        "dropped: target not a resolved human protein",
        "dropped: unparseable or no unit",
        "dropped: non-positive",
        "dropped: no parseable parent SMILES",
        "dropped: conflicting grades",
        "rows written",
        "distinct compounds",
        "distinct cell lines",
        "distinct canonical POIs",
        "distinct raw target strings",
        "graded rows written",
    ):
        if counts[key]:
            print(f"    {key:38} {counts[key]}", flush=True)
    if excluded_kinds:
        print(
            "  targets excluded as not a resolved human protein "
            "(counted, never given a fold group):",
            flush=True,
        )
        for label, n in sorted(excluded_kinds.items(), key=lambda kv: -kv[1]):
            print(f"    {n:6d}  {label}", flush=True)
    print(f"  numeric labels -> {args.output}", flush=True)
    print(f"  ordered grades -> {grade_path}", flush=True)

    provenance = args.output.with_name(args.output.stem + "_PROVENANCE.json")
    provenance.write_text(
        json.dumps(
            {
                "target": "all (pooled)" if pooled else args.target,
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
                "target_map": str(args.target_map),
                "target_map_sha256": hashlib.sha256(
                    args.target_map.read_bytes()
                ).hexdigest(),
                "poi_column": "canonical HGNC symbols, semicolon-separated and "
                "sorted, from the target map. A composite such as KRAS/CRAF "
                "carries both symbols so that POI-grouped folds put it with "
                "each member rather than in a group of its own.",
                "poi_composites": {
                    raw: ";".join(sorted(symbols))
                    for raw, (symbols, kind) in sorted(target_map.items())
                    if kind == "poi" and len(symbols) > 1
                },
                "targets_excluded": dict(excluded_kinds),
                "rows_per_canonical_poi": {
                    poi: n
                    for poi, n in sorted(
                        Counter(k[3] for k in grouped).items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                },
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
