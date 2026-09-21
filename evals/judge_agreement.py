"""Measure how often the LLM judge agrees with a human grader.

    python -m evals.judge_agreement sample   # writes evals/results/labels.csv
    # fill in the human_grade column (correct | partial | wrong), then:
    python -m evals.judge_agreement score

`sample` picks 10 answers spread across question types from the latest result
file of each retrieval split and writes them WITHOUT the judge's grade, so the
human label is blind. The judge's grades go to labels_key.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter

import yaml

from evals.run_evals import EVALS_DIR, RESULTS_DIR

SHEET = RESULTS_DIR / "labels.csv"
KEY = RESULTS_DIR / "labels_key.json"
GRADES = ("correct", "partial", "wrong")
N_SAMPLES = 10
SEED = 7


def _cases() -> dict[str, dict]:
    cases = yaml.safe_load((EVALS_DIR / "retrieval_cases.yaml").read_text(encoding="utf-8"))
    return {c["id"]: c for c in cases}


def _latest_records() -> list[dict]:
    """Records from the newest result file of each retrieval split."""
    records: list[dict] = []
    for split in ("report", "tune"):
        files = sorted(RESULTS_DIR.glob(f"2*_retrieval_*_{split}.json"))
        if files:
            records += json.loads(files[-1].read_text(encoding="utf-8"))
    return records


def sample() -> None:
    cases = _cases()
    by_type: dict[str, list[dict]] = {}
    for r in _latest_records():
        if r["grade"] != "unparsed":
            by_type.setdefault(r["type"], []).append(r)
    if not by_type:
        raise SystemExit("no retrieval results found; run the retrieval suite first")
    rng = random.Random(SEED)
    picked: list[dict] = []
    pools = {t: rs[:] for t, rs in sorted(by_type.items())}
    for pool in pools.values():
        rng.shuffle(pool)
    while len(picked) < N_SAMPLES and any(pools.values()):    # round-robin across types
        for pool in pools.values():
            if pool and len(picked) < N_SAMPLES:
                picked.append(pool.pop())
    rng.shuffle(picked)

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "run", "question", "expected_answer", "actual_answer", "human_grade"])
        for r in picked:
            c = cases[r["id"]]
            w.writerow([r["id"], r["run"], c["question"], c["expected_answer"], r["answer"], ""])
    KEY.write_text(json.dumps({f"{r['id']}:{r['run']}": r["grade"] for r in picked}, indent=2), encoding="utf-8")
    print(f"wrote {SHEET} ({len(picked)} rows). Fill in human_grade with one of {GRADES}, then run `score`.")


def score() -> None:
    if not SHEET.exists() or not KEY.exists():
        raise SystemExit("run `sample` first")
    key = json.loads(KEY.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    with SHEET.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human = row["human_grade"].strip().lower()
            if not human:
                raise SystemExit(f"{row['id']} run {row['run']} has no human_grade yet")
            if human not in GRADES:
                raise SystemExit(f"{row['id']}: human_grade must be one of {GRADES}, got {human!r}")
            pairs.append((human, key[f"{row['id']}:{row['run']}"]))
    agree = sum(h == j for h, j in pairs)
    print(f"judge agrees with the human on {agree}/{len(pairs)} ({agree / len(pairs):.0%})")
    print("\nhuman -> judge")
    for (h, j), n in sorted(Counter(pairs).items()):
        print(f"  {h:8} -> {j:8} {n}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sample", "score"])
    {"sample": sample, "score": score}[p.parse_args().command]()


if __name__ == "__main__":
    main()
