#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Office-Home equal-task-pass control for the two-stage DAC handoff method.
#
# Usage:
#   all 12 tasks: bash tools/run_office_home_single_stage_refinement35_all.sh 2020
#   one task:     bash tools/run_office_home_single_stage_refinement35_all.sh 2020 AC
#
# Protocol:
#   source F/B/C -> single-stage cyclic refinement
#   7 cycles x 5 target epochs = 35 task-model target passes
#
# This control matches the two-stage method's 35 task-model passes. It does not
# claim to match VLM/full-scan counts; those are reported separately.

experiment_seed="${1:-2020}"
task_filter="${2:-all}"
cycles=7
epochs_per_cycle=5
expected_passes=$((cycles * epochs_per_cycle))
domain_keys=(A C P R)
method="plmatch_single_stage_refinement35_office_home_seed${experiment_seed}"
result_dir="output/uda/benchmark_tables"
console_dir="${result_dir}/console_single_stage_refinement35_seed${experiment_seed}"
raw_accuracy_path="${result_dir}/office_home_single_stage_refinement35_seed${experiment_seed}_accuracy_raw.csv"
summary_path="${result_dir}/office_home_single_stage_refinement35_seed${experiment_seed}.csv"

if ! [[ "$experiment_seed" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer: ${experiment_seed}" >&2
  exit 1
fi

if [ "$task_filter" != "all" ]; then
  case "$task_filter" in
    AC|AP|AR|CA|CP|CR|PA|PC|PR|RA|RC|RP) ;;
    *)
      echo "Task must be all or one of: AC AP AR CA CP CR PA PC PR RA RC RP" >&2
      exit 1
      ;;
  esac
fi

if [ ! -f "data/office-home/classname.txt" ]; then
  echo "Missing Office-Home input: data/office-home/classname.txt" >&2
  exit 1
fi

for key in "${domain_keys[@]}"; do
  for part in F B C; do
    required_path="source/uda/office-home/${key}/source_${part}.pt"
    if [ ! -f "$required_path" ]; then
      echo "Missing Office-Home source weight: $required_path" >&2
      echo "Train sources first: bash tools/train_office_home_sources.sh" >&2
      exit 1
    fi
  done
done

mkdir -p "$result_dir" "$console_dir"

completed_tasks=0
for s in 0 1 2 3; do
  for t in 0 1 2 3; do
    if [ "$s" -eq "$t" ]; then
      continue
    fi

    task="${domain_keys[$s]}${domain_keys[$t]}"
    if [ "$task_filter" != "all" ] && [ "$task" != "$task_filter" ]; then
      continue
    fi

    target_domain=(Art Clipart Product RealWorld)
    target_list="data/office-home/${target_domain[$t]}_list.txt"
    if [ ! -f "$target_list" ]; then
      echo "Missing Office-Home target list: $target_list" >&2
      exit 1
    fi

    task_dir="output/uda/office-home/${task}/${method}"
    console_log="${console_dir}/${task}.log"
    framework_logs=("$task_dir"/*.txt)

    if [ "${#framework_logs[@]}" -gt 1 ]; then
      echo "Expected at most one ${task} framework log, found ${#framework_logs[@]}" >&2
      exit 1
    fi

    if [ "${#framework_logs[@]}" -eq 1 ]; then
      logged_passes=$(grep -c "Task: ${task}" "${framework_logs[0]}" || true)
      if [ "$logged_passes" -eq "$expected_passes" ] \
        && grep -q "Cycle: ${cycles}/${cycles}" "${framework_logs[0]}" \
        && grep -q "DUET first-cycle prior: enabled=False; power=0.000" "${framework_logs[0]}"; then
        if [ ! -f "$console_log" ] \
          || ! grep -q "Running time: .* Seconds" "$console_log"; then
          echo "Completed ${task} result exists but its runtime log is missing: $console_log" >&2
          echo "Move the existing task directory aside, then rerun this task." >&2
          exit 1
        fi
        echo "==> [${task}] Reusing completed 35-pass single-stage control"
        completed_tasks=$((completed_tasks + 1))
        continue
      fi

      echo "Partial or incompatible result exists for ${task}: ${task_dir}" >&2
      echo "Move it aside before rerunning this task; the script will not delete it." >&2
      exit 1
    fi

    if [ -d "$task_dir" ] \
      && [ -n "$(find "$task_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
      echo "Non-empty result directory exists without a framework log: ${task_dir}" >&2
      echo "Move it aside before rerunning this task; the script will not delete it." >&2
      exit 1
    fi
    if [ -e "$console_log" ]; then
      echo "Console log already exists without a completed result: $console_log" >&2
      echo "Move it aside before rerunning this task; the script will not overwrite it." >&2
      exit 1
    fi

    echo "==> [${task}] Single-stage cyclic refinement: ${cycles} cycles x ${epochs_per_cycle} epochs = ${expected_passes} passes"
    python image_target_of_oh_vs.py \
      --cfg cfgs/office-home/plmatch.yaml \
      CKPT_DIR . SETTING.OUTPUT_SRC source \
      MODEL.METHOD "$method" \
      SETTING.S "$s" SETTING.T "$t" SETTING.SEED "$experiment_seed" \
      TEST.MAX_EPOCH "$epochs_per_cycle" TEST.INTERVAL "$epochs_per_cycle" \
      ACTIVE.CYCLE "$cycles" ACTIVE.ADAPTATION_LIST "" \
      DUET_HANDOFF.FINAL_EXTRA_EPOCHS 0 \
      2>&1 | tee "$console_log"

    framework_logs=("$task_dir"/*.txt)
    if [ "${#framework_logs[@]}" -ne 1 ]; then
      echo "Expected one ${task} framework log, found ${#framework_logs[@]}" >&2
      exit 1
    fi
    logged_passes=$(grep -c "Task: ${task}" "${framework_logs[0]}" || true)
    if [ "$logged_passes" -ne "$expected_passes" ]; then
      echo "${task} logged ${logged_passes} task passes, expected ${expected_passes}" >&2
      exit 1
    fi
    if ! grep -q "Cycle: ${cycles}/${cycles}" "${framework_logs[0]}"; then
      echo "${task} did not reach Cycle ${cycles}/${cycles}" >&2
      exit 1
    fi
    if ! grep -q "DUET first-cycle prior: enabled=False; power=0.000" "${framework_logs[0]}"; then
      echo "${task} did not use the unmodified single-stage refinement path" >&2
      exit 1
    fi
    if ! grep -q "Running time: .* Seconds" "$console_log"; then
      echo "${task} completed but runtime was not recorded" >&2
      exit 1
    fi
    completed_tasks=$((completed_tasks + 1))
  done
done

if [ "$completed_tasks" -eq 0 ]; then
  echo "No Office-Home tasks matched: $task_filter" >&2
  exit 1
fi

python tools/extract_final_accuracy.py \
  --glob "output/uda/office-home/*/${method}/*.txt" \
  --selection final \
  > "$raw_accuracy_path"

python - \
  "$raw_accuracy_path" "$summary_path" "$console_dir" \
  "$experiment_seed" "$cycles" "$epochs_per_cycle" "$task_filter" <<'PY'
import csv
import re
import statistics
import sys
from pathlib import Path

raw_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
console_dir = Path(sys.argv[3])
seed = int(sys.argv[4])
cycles = int(sys.argv[5])
epochs_per_cycle = int(sys.argv[6])
task_filter = sys.argv[7]
expected_passes = cycles * epochs_per_cycle
expected_rows = 12 if task_filter == "all" else 1
domain_names = {
    "A": "Art",
    "C": "Clipart",
    "P": "Product",
    "R": "RealWorld",
}

with raw_path.open(newline="", encoding="utf-8") as handle:
    raw_rows = list(csv.DictReader(handle))

if task_filter != "all":
    raw_rows = [row for row in raw_rows if row["task"] == task_filter]

if len(raw_rows) != expected_rows:
    raise SystemExit(
        f"Expected {expected_rows} completed result rows, found {len(raw_rows)}"
    )

fieldnames = [
    "method",
    "implementation",
    "task",
    "source",
    "target",
    "seed",
    "cycles",
    "epochs_per_cycle",
    "task_model_passes",
    "final_accuracy",
    "peak_accuracy",
    "peak_minus_final",
    "train_seconds",
    "train_time",
    "time_scope",
    "status",
    "console_log",
    "framework_log",
]
summary_rows = []
for row in sorted(raw_rows, key=lambda item: item["task"]):
    task = row["task"]
    if int(row["final_cycle"]) != cycles:
        raise SystemExit(
            f"{task}: final_cycle={row['final_cycle']}, expected {cycles}"
        )
    console_log = console_dir / f"{task}.log"
    console_text = console_log.read_text(errors="ignore")
    runtime_match = re.search(
        r"Running time:\s*([0-9.]+)\s*Seconds", console_text
    )
    if runtime_match is None:
        raise SystemExit(f"{task}: runtime missing from {console_log}")
    train_seconds = float(runtime_match.group(1))
    rounded_seconds = int(round(train_seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    summary_rows.append(
        {
            "method": "single_stage_refinement_35pass",
            "implementation": "plmatch",
            "task": task,
            "source": domain_names[task[0]],
            "target": domain_names[task[1]],
            "seed": seed,
            "cycles": cycles,
            "epochs_per_cycle": epochs_per_cycle,
            "task_model_passes": expected_passes,
            "final_accuracy": row["final_accuracy"],
            "peak_accuracy": row["peak_accuracy"],
            "peak_minus_final": row["peak_minus_final"],
            "train_seconds": f"{train_seconds:.2f}",
            "train_time": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "time_scope": "task",
            "status": "completed",
            "console_log": str(console_log),
            "framework_log": row["log"],
        }
    )

with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

mean_accuracy = statistics.mean(
    float(row["final_accuracy"]) for row in summary_rows
)
total_seconds = sum(float(row["train_seconds"]) for row in summary_rows)
print(f"==> Completed rows: {len(summary_rows)}")
print(f"==> Fixed-final mean accuracy: {mean_accuracy:.2f}%")
print(f"==> Total measured task time: {total_seconds:.2f} seconds")
print(f"==> Verified summary: {summary_path}")
PY

echo "==> Completed Office-Home tasks in this invocation: ${completed_tasks}"
echo "==> Protocol: ${cycles} cycles x ${epochs_per_cycle} epochs = ${expected_passes} task-model passes"
echo "==> Final summary: ${summary_path}"
