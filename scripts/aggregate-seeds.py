#!/usr/bin/env python3
"""
Aggregate one experiment's per-seed metrics into `results/aggregate.json`.

    python scripts/aggregate-seeds.py experiments/<experiment>

Every number this project reports is three seeds, and nothing should be
aggregated by hand in a notebook. This reads every `jobs/*/results/metrics.json`
under the experiment, groups them by job name with the seed stripped, and for
each numeric metric records the mean, the standard deviation, the per-seed
values and the seed list.

What a seed controls differs by job type, and the grouping follows that. In a
`generate` job the seed controls sampling from a fixed checkpoint, so one job
runs the seed list and its own metrics.json already covers them; in a
`goal_directed` job the seed controls the optimization run itself, so each seed
is its own job directory and those are what this combines. A job name ending in
`_seed<N>` or `_s<N>` is treated as one seed of the job before the suffix.

A metric that is null in any seed is reported with the seeds that produced a
value and a count of those that did not, because a mean over two of three seeds
is not the same quantity as a mean over three and the difference has to be
visible.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SEED_SUFFIX = re.compile(r"_(?:seed|s)(\d+)$")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    runs = sorted(args.experiment.glob("jobs/*/results/metrics.json"))
    if not runs:
        print(
            f"No metrics.json under {args.experiment}/jobs/*/results/.\n"
            "Run the experiment's jobs first; this aggregates, it does not "
            "compute.",
            file=sys.stderr,
        )
        return 1

    groups: dict = defaultdict(list)
    for path in runs:
        payload = json.loads(path.read_text())
        job = path.parts[-3]
        match = SEED_SUFFIX.search(job)
        base = job[: match.start()] if match else job
        seed = int(match.group(1)) if match else payload["run"].get("seed")
        groups[base].append((seed, job, payload))

    aggregate: dict = {}
    for base, entries in sorted(groups.items()):
        entries.sort(key=lambda e: (e[0] is None, e[0], e[1]))
        seeds = [s for s, _, _ in entries]
        duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
        if duplicates:
            raise SystemExit(
                f"The job group '{base}' has more than one run at seed(s) "
                f"{duplicates}: {[j for _, j, _ in entries]}.\n"
                "Two runs at the same seed are not two seeds, and averaging "
                "them would report a spread narrower than the truth. Remove "
                "the duplicate job directory, or give it its own seed."
            )
        keys = sorted({k for _, _, p in entries for k in p["metrics"]})
        job_block: dict = {
            "seeds": seeds,
            "n_runs": len(entries),
            "jobs": [j for _, j, _ in entries],
            "metrics": {},
        }
        for key in keys:
            pairs = [(s, p["metrics"].get(key)) for s, _, p in entries]
            numeric = [
                (s, v)
                for s, v in pairs
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            block = {
                "per_seed": {str(s): v for s, v in pairs},
                "n_with_value": len(numeric),
                "n_null": len(pairs) - len(numeric),
            }
            if numeric:
                values = [v for _, v in numeric]
                block["mean"] = statistics.fmean(values)
                # SAMPLE standard deviation, and null rather than zero for a
                # single run. This was pstdev with a 0.0 fallback, and both
                # halves misreported the intervals the paper decides on.
                #
                # A zero-width interval is not a measured spread, and the
                # stated decision rule -- a winner where intervals do not
                # overlap -- marks a winner on every comparison against one.
                # Each goal-directed experiment currently holds exactly one
                # rl_* job, so every reported optimization interval came from
                # this branch; the three generate_best_agent_seed* jobs resample
                # one checkpoint and carry no optimization variance at all.
                #
                # And the three seeds are a sample from the seed distribution,
                # not the population: pstdev understates the sample SD by 22%
                # at n = 3, which makes the non-overlap rule anti-conservative
                # (about p = 0.12 per comparison rather than 0.07).
                if len(values) > 1:
                    block["std"] = statistics.stdev(values)
                    block["std_kind"] = "sample (statistics.stdev)"
                else:
                    block["std"] = None
                    block["std_unavailable"] = (
                        f"one run in this group ({[j for _, j, _ in entries]}); "
                        "a spread needs at least two, and 0.0 would read as a "
                        "measured zero spread and win every overlap comparison"
                    )
            job_block["metrics"][key] = block
        aggregate[base] = job_block

    out = args.output or (args.experiment / "results" / "aggregate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": args.experiment.name,
                "jobs": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"* {args.experiment.name}: {len(runs)} runs in {len(aggregate)} job groups",
        flush=True,
    )
    for base, block in sorted(aggregate.items()):
        seeds = block["seeds"]
        note = (
            ""
            if len(seeds) >= 3
            else f"  <- {len(seeds)} seed(s), conventions ask for 3"
        )
        print(f"    {base:34} seeds {seeds}{note}", flush=True)
    print(f"  wrote {out}", flush=True)

    report = out.with_name("report.html")
    report.write_text(_render(args.experiment.name, aggregate))
    print(f"  wrote {report}", flush=True)
    return 0


def _render(experiment: str, aggregate: dict) -> str:
    """The experiment-level report: one table per job group, mean and spread."""
    import html as _html

    # The AIME palette, so the report and the paper figures agree.
    deep_teal, dark_slate = "#2D4F54", "#3A3D4A"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{_html.escape(experiment)}</title><style>",
        f"body{{font-family:system-ui,sans-serif;color:{dark_slate};margin:2rem;"
        "max-width:60rem}}",
        f"h1,h2{{color:{deep_teal}}} table{{border-collapse:collapse;margin:1rem 0}}",
        "th,td{border:1px solid #d8dae0;padding:.35rem .6rem;text-align:right}",
        "th:first-child,td:first-child{text-align:left}",
        "td.null{color:#8a8d98;font-style:italic}",
        ".note{color:#8a8d98;font-size:.9rem}",
        "</style></head><body>",
        f"<h1>{_html.escape(experiment)}</h1>",
        "<p class='note'>Mean and population standard deviation across the "
        "seeds of each job group, from <code>aggregate.json</code>. The spread "
        "is over the seeds that produced a value; a metric null in some seeds "
        "shows how many.</p>",
    ]
    for base, block in sorted(aggregate.items()):
        parts.append(f"<h2>{_html.escape(base)}</h2>")
        warn = (
            ""
            if len(block["seeds"]) >= 3
            else f" <span class='note'>only {len(block['seeds'])} seed(s); the "
            "reporting conventions ask for three</span>"
        )
        parts.append(f"<p class='note'>seeds {block['seeds']}{warn}</p>")
        parts.append(
            "<table><tr><th>metric</th><th>mean</th><th>std</th>"
            "<th>seeds with a value</th></tr>"
        )
        for key, cell in sorted(block["metrics"].items()):
            if "mean" not in cell:
                parts.append(
                    f"<tr><td>{_html.escape(key)}</td>"
                    "<td class='null' colspan='3'>no numeric value in "
                    "any seed</td></tr>"
                )
                continue
            n = f"{cell['n_with_value']}"
            if cell["n_null"]:
                n += f" <span class='note'>({cell['n_null']} null)</span>"
            # A single-run group has no spread, and the cell says so rather
            # than printing 0.0000, which reads as a measured zero.
            std = (
                f"{cell['std']:.4f}"
                if cell["std"] is not None
                else "<span class='null'>one run</span>"
            )
            parts.append(
                f"<tr><td>{_html.escape(key)}</td><td>{cell['mean']:.4f}</td>"
                f"<td>{std}</td><td>{n}</td></tr>"
            )
        parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    sys.exit(main())
