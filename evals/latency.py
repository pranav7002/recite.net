"""Voice latency runner: the same recorded questions through POST /voice under
each config, so the only thing that changes between rows is the pipeline.

    python -m evals.latency record                 # once: 20 questions -> evals/audio/*.wav (macOS `say`)
    python -m evals.latency run --runs 5           # needs `uvicorn backend.api:app --port 8000` running
    python -m evals.latency summarize              # table -> evals/results/<date>_latency.md

`run` appends one JSON line per turn to evals/results/latency_runs.jsonl and
skips (question, config, run) triples already there, so a run stopped by the
daily quota resumes where it left off the next day.

Numbers per turn:
  client_ttfa_ms  request sent -> first audio chunk received, measured here.
                  The browser's number adds playback start on top (~tens of ms).
  server marks    stt_ms, answer_ready_ms, first_audio_ms (from request arrival).
  wait/backoff    free-tier limiter waits and 429 retry sleeps inside the turn.
  system_ttfa_ms  first_audio_ms minus wait and backoff: what the pipeline costs
                  without the free tier's queueing. Report both columns.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from backend.speakable import speakable

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
RESULTS = ROOT / "results"
RUNS = RESULTS / "latency_runs.jsonl"
CONFIGS = ("sequential", "stream", "stream_nocheck")


def questions(n: int = 20) -> list[dict]:
    """First n tune/report questions whose answers are in the text layer.
    Held-out stays untouched; image-only questions are skipped because their
    answer is a refusal, which would make those turns unrepresentatively short."""
    cases = yaml.safe_load((ROOT / "retrieval_cases.yaml").read_text())
    keep = [c for c in cases if c.get("split") in ("tune", "report") and not c.get("note")]
    return keep[:n]


def cmd_record(args) -> None:
    AUDIO.mkdir(exist_ok=True)
    for c in questions(args.n):
        out = AUDIO / f"{c['id']}.wav"
        if out.exists() and not args.force:
            continue
        spoken = speakable(c["question"])          # "$\sigma^2$" -> "sigma squared"
        subprocess.run(["say", "-v", args.voice, "-o", str(out),
                        "--file-format=WAVE", "--data-format=LEI16@16000", spoken], check=True)
        print(f"{out.name}: {spoken}")


def done_keys() -> set[tuple]:
    if not RUNS.exists():
        return set()
    return {(r["id"], r["config"], r["run"]) for r in map(json.loads, RUNS.open()) if not r.get("error")}


def one_turn(url: str, wav: Path, config: str) -> dict:
    t_send = time.monotonic()
    with wav.open("rb") as f:
        resp = requests.post(f"{url}/voice", stream=True, timeout=600,
                             files={"file": (wav.name, f, "audio/wav")},
                             data={"session_id": f"latency-{uuid.uuid4().hex[:8]}",   # fresh: no history
                                   "t0": str(time.time() * 1000), "config": config})
        resp.raise_for_status()
        out: dict = {"chunks": 0}
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            item = json.loads(line[6:])
            if "transcript" in item:
                out["transcript"] = item["transcript"]
            if item.get("audio_b64"):
                out["chunks"] += 1
                out.setdefault("client_ttfa_ms", round((time.monotonic() - t_send) * 1000, 1))
            if item.get("done"):
                out["client_total_ms"] = round((time.monotonic() - t_send) * 1000, 1)
                out["marks"] = item.get("marks", {})
                out["trace"] = {k: v for k, v in (item.get("trace") or {}).items() if k != "pending_action"}
    return out


def cmd_run(args) -> None:
    RESULTS.mkdir(exist_ok=True)
    configs = args.configs.split(",")
    done = done_keys()
    qs = questions(args.n)
    missing = [q for q in qs if not (AUDIO / f"{q['id']}.wav").exists()]
    if missing:
        raise SystemExit(f"no audio for {[q['id'] for q in missing]}; run `record` first")
    requests.get(f"{args.url}/health", timeout=5).raise_for_status()
    todo = [(run, q, cfg) for run in range(1, args.runs + 1) for q in qs for cfg in configs
            if (q["id"], cfg, run) not in done]
    print(f"{len(todo)} turns to go ({len(done)} already recorded)")
    # Configs interleaved per question, so quota pressure and server warmth hit all three alike.
    for i, (run, q, cfg) in enumerate(todo, 1):
        row = {"id": q["id"], "config": cfg, "run": run, "ts": time.time()}
        try:
            row |= one_turn(args.url, AUDIO / f"{q['id']}.wav", cfg)
        except Exception as e:  # noqa: BLE001 — record and move on; a rerun retries it
            row["error"] = f"{type(e).__name__}: {e}"
        with RUNS.open("a") as f:
            f.write(json.dumps(row) + "\n")
        m = row.get("marks", {})
        print(f"[{i}/{len(todo)}] {q['id']:<14} {cfg:<15} run {run}  "
              f"client {row.get('client_ttfa_ms', '-')} ms  wait {m.get('wait_ms', '-')} ms"
              + (f"  ERROR {row['error']}" if "error" in row else ""))
        if "QuotaExhausted" in row.get("error", "") or "429" in row.get("error", ""):
            raise SystemExit("daily quota reached; rerun tomorrow to resume")
        if row.get("error", "").startswith("ConnectionError"):
            raise SystemExit("server not reachable (restarted or stopped?); start it and rerun to resume")


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def cmd_summarize(args) -> None:
    rows = [r for r in map(json.loads, RUNS.open()) if not r.get("error") and r.get("marks")]
    cols = [("client_ttfa_ms", "TTFA, client"), ("system_ttfa_ms", "TTFA minus quota waits"),
            ("stt_ms", "STT"), ("answer_ready_ms", "Answer ready"), ("wait_ms", "Limiter + backoff")]
    today = datetime.now(tz=timezone.utc).date()
    lines = [f"# Voice latency, {today}", "",
             (f"{len(rows)} turns from `{RUNS.name}`. p50 / p95 in ms. "
              "Audio: macOS `say`, identical files for every config."), "",
             "| Config | n | " + " | ".join(c[1] for c in cols) + " |",
             "| --- | --- | " + " | ".join("---" for _ in cols) + " |"]
    for cfg in CONFIGS:
        rs = [r for r in rows if r["config"] == cfg]
        if not rs:
            continue
        vals = {k: [] for k, _ in cols}
        for r in rs:
            m = r["marks"]
            quota = m.get("wait_ms", 0) + m.get("backoff_ms", 0)
            vals["client_ttfa_ms"].append(r["client_ttfa_ms"])
            vals["system_ttfa_ms"].append(m["first_audio_ms"] - quota)
            vals["stt_ms"].append(m["stt_ms"])
            vals["answer_ready_ms"].append(m["answer_ready_ms"])
            vals["wait_ms"].append(quota)
        cells = [f"{pct(v, 50):,.0f} / {pct(v, 95):,.0f}" for v in vals.values()]
        lines.append(f"| `{cfg}` | {len(rs)} | " + " | ".join(cells) + " |")
    heard = [r for r in rows if r.get("transcript")]
    lines += ["", (f"Transcripts checked: {len(heard)}. Spot-check a few against the questions "
                   "before trusting the numbers (a misheard question changes the answer, and its length).")]
    out = RESULTS / f"{today:%Y%m%d}_latency.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("--n", type=int, default=20)
    r.add_argument("--voice", default="Samantha"); r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_record)
    u = sub.add_parser("run"); u.add_argument("--n", type=int, default=20)
    u.add_argument("--runs", type=int, default=5); u.add_argument("--configs", default=",".join(CONFIGS))
    u.add_argument("--url", default="http://localhost:8000"); u.set_defaults(fn=cmd_run)
    s = sub.add_parser("summarize"); s.set_defaults(fn=cmd_summarize)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
