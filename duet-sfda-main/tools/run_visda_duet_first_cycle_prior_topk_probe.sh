#!/usr/bin/env bash
set -euo pipefail

seed="${SEED:-2020}"
probe_mode="${PROBE_MODE:-cycle0}"
case "$probe_mode" in
  cycle0)
    method="duet_first_cycle_prior_topk_probe_cycle0_preflight_seed${seed}"
    mode_args=(
      ACTIVE.CYCLE 1
      FAILURE_AUDIT.ENABLED True
      FAILURE_AUDIT.STOP_AFTER_PRE_CYCLE 1
    )
    ;;
  full)
    method="duet_first_cycle_prior_topk_probe_visda_seed${seed}"
    mode_args=()
    ;;
  *)
    echo "PROBE_MODE must be cycle0 or full, found: $probe_mode" >&2
    exit 1
    ;;
esac
run_dir="output/uda/VISDA-C/TV/${method}"
cycle0_summary="${run_dir}/conflict_probe/task_TV_seed_${seed}/cycle_000/conflict_summary.json"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing VisDA-C probe input: $path" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ]; then
  echo "Refusing to overwrite existing probe run: $run_dir" >&2
  exit 1
fi

echo "==> Clean DUET first-cycle prior with per-cycle Top-k conflict probe"
echo "==> Probe uses detached pre-prior probabilities; labels are oracle diagnostic only"
if [ "$probe_mode" = "cycle0" ]; then
  echo "==> Cycle-0 preflight only; stops before the first optimizer step"
else
  echo "==> Full 8-cycle DUET-FCP probe run"
fi
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_topk_probe.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.ADAPTATION_LIST "" \
  "${mode_args[@]}"

python - "$cycle0_summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing cycle-0 probe summary: {path}")
summary = json.loads(path.read_text())
regression = summary["known_visda_cycle0_regression"]
if not regression["applicable"] or not regression["passed"]:
    failed = [name for name, passed in regression["checks"].items() if not passed]
    raise SystemExit(f"Known VisDA cycle-0 count regression failed: {failed}")
print(json.dumps(summary["overall"], indent=2))
PY

echo "==> Probe complete: ${run_dir}/conflict_probe"
