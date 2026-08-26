#!/usr/bin/env bash
set -euo pipefail

# DUET-FCP 全量 softmax 导出（DCPL 噪声转移矩阵离线诊断用）。
# 数据是 probe 每次循环本来就在内存里算的完整 12 类概率，这里只是多存一份 npz，
# 不改变任何训练/诊断语义；GT 只用于离线评估。
#
# 用法：
#   PROBE_MODE=cycle0 bash tools/run_visda_duet_softmax_dump.sh   # 只跑 cycle 0（不训练，~半小时）
#   PROBE_MODE=full  bash tools/run_visda_duet_softmax_dump.sh    # 8 个 cycle 都导出（需一次完整训练）

seed="${SEED:-2020}"
probe_mode="${PROBE_MODE:-cycle0}"
case "$probe_mode" in
  cycle0)
    method="duet_first_cycle_prior_topk_probe_softmax_dump_cycle0_preflight_seed${seed}"
    mode_args=(
      ACTIVE.CYCLE 1
      FAILURE_AUDIT.ENABLED True
      FAILURE_AUDIT.STOP_AFTER_PRE_CYCLE 1
    )
    ;;
  full)
    method="duet_first_cycle_prior_topk_probe_softmax_dump_visda_seed${seed}"
    mode_args=()
    ;;
  *)
    echo "PROBE_MODE must be cycle0 or full, found: $probe_mode" >&2
    exit 1
    ;;
esac

run_dir="output/uda/VISDA-C/TV/${method}"
dump_dir="${run_dir}/softmax_dump"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing VisDA-C softmax-dump input: $path" >&2
    exit 1
  fi
done

if [ -e "$run_dir" ]; then
  echo "Refusing to overwrite existing softmax-dump run: $run_dir" >&2
  exit 1
fi

echo "==> DUET-FCP full-softmax dump (DCPL offline diagnostic), VisDA-C TV, seed=${seed}"
if [ "$probe_mode" = "cycle0" ]; then
  echo "==> Cycle-0 preflight only: one forward pass over the target pool, no adaptation"
else
  echo "==> Full 8-cycle run: one npz per cycle under ${dump_dir}"
fi
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_softmax_dump.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.ADAPTATION_LIST "" \
  CONFLICT_PROBE.DUMP_DIR "$dump_dir" \
  "${mode_args[@]}"

if [ "$probe_mode" = "cycle0" ]; then
  python - "$dump_dir" "$run_dir" <<'PY'
import sys
from pathlib import Path

import numpy as np

dump_dir = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
npz_path = dump_dir / "cycle_000.npz"
if not npz_path.is_file():
    raise SystemExit(f"Missing cycle-0 softmax dump: {npz_path}")

summary_path = run_dir / "conflict_probe/task_TV_seed_2020/cycle_000/conflict_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"Missing cycle-0 probe summary: {summary_path}")
import json

summary = json.loads(summary_path.read_text())
total = int(summary["overall"]["total_samples"])

with np.load(npz_path, allow_pickle=True) as data:
    task = data["task_probability"]
    clip = data["clip_probability"]
    labels = data["labels"]
    indices = data["sample_indices"]
    if task.shape != (total, 12) or clip.shape != (total, 12):
        raise SystemExit(
            f"Unexpected softmax shape: task={task.shape} clip={clip.shape}, expected ({total}, 12)"
        )
    if not np.allclose(task.sum(axis=1), 1.0, atol=1e-4) or not np.allclose(
        clip.sum(axis=1), 1.0, atol=1e-4
    ):
        raise SystemExit("Softmax rows do not sum to one")
    if np.unique(indices).size != total:
        raise SystemExit("sample_indices are not a unique permutation of the target pool")
    if labels.shape != (total,):
        raise SystemExit(f"labels shape {labels.shape}, expected ({total},)")

print(f"==> cycle_000 dump OK: {total} samples x 12 classes (task + CLIP), GT present")
print(f"    conflicts={summary['overall']['conflict_samples']} "
      f"(known deviation: repo contract expects 28255; actual CSV count is authoritative)")
PY
else
  python - "$dump_dir" <<'PY'
import sys
from pathlib import Path

dump_dir = Path(sys.argv[1])
expected = [dump_dir / f"cycle_{cycle:03d}.npz" for cycle in range(8)]
missing = [str(path) for path in expected if not path.is_file()]
if missing:
    raise SystemExit("Missing softmax dump files: " + ", ".join(missing))
print("==> 8-cycle dump OK: " + ", ".join(path.name for path in expected))
PY
fi

echo "==> Done: ${dump_dir} (npz 共约 11 MB，拷回本地即可离线诊断)"
