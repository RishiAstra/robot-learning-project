#!/usr/bin/env python3
"""
eval_bc_all.py — find all model_epoch_2000* checkpoints, evaluate them.

Each result is saved as <out-dir>/<task>.json (source of truth for resuming).
A CSV summary is (re)generated from all JSONs at the end for convenience.

Usage:
    python eval_bc_all.py TASK [TASK ...] [--rollouts N] [--horizon H] [--out-dir PATH]

Examples:
    python eval_bc_all.py coffee_d1 coffee_d2
    python eval_bc_all.py coffee_d1 --rollouts 100 --horizon 500
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent))
PYTHON_BIN   = os.environ.get("PYTHON_BIN",
                   str(Path.home() / "miniconda3/envs/mimicgen/bin/python"))

DEFAULT_TRAINING_ROOT = PROJECT_ROOT / "mimicgen/training_results/core"
DEFAULT_OUT_DIR       = PROJECT_ROOT / "tmp_runs/bc_eval"
DEFAULT_ROLLOUTS      = 50
DEFAULT_HORIZON       = 400

CSV_FIELDS = ["task", "Success_Rate", "Num_Success", "Return",
              "Horizon", "n_rollouts", "env_name", "checkpoint"]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        d = d[0]
    elif isinstance(d, dict) and len(d) == 1:
        inner = next(iter(d.values()))
        if isinstance(inner, dict):
            d = inner
    return d


def rebuild_csv(out_dir: Path):
    """Regenerate the CSV from all *.json files in out_dir."""
    jsons = sorted(out_dir.glob("*.json"))
    if not jsons:
        return
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for jpath in jsons:
            try:
                d = load_json(jpath)
                writer.writerow({k: d.get(k, "") for k in CSV_FIELDS})
            except Exception as e:
                print(f"[WARN] Could not read {jpath.name} for CSV: {e}")
    print(f"[INFO] CSV written: {csv_path}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="+", metavar="TASK",
                        help="Tasks to evaluate (e.g. coffee_d1 coffee_d2).")
    parser.add_argument("--rollouts",      type=int,  default=DEFAULT_ROLLOUTS)
    parser.add_argument("--horizon",       type=int,  default=DEFAULT_HORIZON)
    parser.add_argument("--out-dir",       type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--python",        default=PYTHON_BIN)
    args = parser.parse_args()

    training_root = args.training_root
    out_dir       = args.out_dir
    eval_script   = PROJECT_ROOT / "rl_finetune/evaluate_checkpoints.py"

    if not training_root.exists():
        sys.exit(f"[ERR] Training root not found: {training_root}")
    if not eval_script.exists():
        sys.exit(f"[ERR] Eval script not found: {eval_script}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # discover checkpoints
    checkpoints = sorted(training_root.glob(
        "*/low_dim/trained_models/*/*/models/model_epoch_2000*.pth"
    ))

    wanted = set(args.tasks)
    checkpoints = [c for c in checkpoints
                   if c.relative_to(training_root).parts[0] in wanted]
    missing = wanted - {c.relative_to(training_root).parts[0] for c in checkpoints}
    if missing:
        print(f"[WARN] No checkpoints found for: {', '.join(sorted(missing))}")
    if not checkpoints:
        sys.exit("[INFO] Nothing to evaluate.")

    print(f"[INFO] {len(checkpoints)} checkpoint(s) — rollouts={args.rollouts} horizon={args.horizon}")
    print(f"[INFO] Output dir: {out_dir}")
    print("─" * 70)

    done = skipped = failed = 0

    for ckpt in checkpoints:
        task     = ckpt.relative_to(training_root).parts[0]
        out_json = out_dir / f"{task}.json"

        if out_json.exists():
            try:
                d = load_json(out_json)
                rate = d.get("Success_Rate", "?")
                print(f"[SKIP] {task}  ->  already done (Success_Rate={rate})")
                skipped += 1
                continue
            except Exception:
                print(f"[WARN] {task}  ->  existing JSON unreadable, re-running")

        print(f"[RUN]  {task}  ->  {ckpt.name}")

        result = subprocess.run(
            [args.python, str(eval_script),
             "--agent",       str(ckpt),
             "--n-rollouts",  str(args.rollouts),
             "--horizon",     str(args.horizon),
             "--output-json", str(out_json)],
            check=False,
        )

        if result.returncode != 0:
            print(f"[ERR]  {task} failed (exit {result.returncode})")
            out_json.unlink(missing_ok=True)  # don't leave a partial file
            failed += 1
            continue

        try:
            d    = load_json(out_json)
            rate = d.get("Success_Rate", "?")
            num  = d.get("Num_Success",  "?")
            print(f"[DONE] {task}  ->  Success_Rate={rate}  ({num} / {args.rollouts})")
            done += 1
        except Exception as e:
            print(f"[WARN] {task}  ->  eval finished but JSON unreadable: {e}")
            failed += 1

    print("─" * 70)
    print(f"[INFO] Evaluated: {done}  |  Skipped: {skipped}  |  Failed: {failed}")

    rebuild_csv(out_dir)


if __name__ == "__main__":
    main()