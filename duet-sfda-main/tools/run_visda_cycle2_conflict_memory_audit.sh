#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="duet_support_conditioned_clip_cycle2_memory_audit_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
output_dir="output/uda/VISDA-C/TV/${method}"
snapshot_dir="${output_dir}/cycle2_conflict_memory_snapshots"
audit_dir="${output_dir}/cycle2_conflict_memory_audit"
result_dir="output/uda/VISDA-C"
previous_method="duet_support_conditioned_clip_visda_proxy25_seed2020"
previous_dir="${result_dir}/TV/${previous_method}"
previous_gate="${result_dir}/duet_support_conditioned_clip_visda_proxy25_seed2020_gate.json"
previous_source_hash="${result_dir}/duet_support_conditioned_clip_visda_proxy25_seed2020_source_sha256.txt"
previous_proxy_hash="${result_dir}/duet_support_conditioned_clip_visda_proxy25_seed2020_proxy_sha256.txt"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt \
  "$previous_gate" \
  "$previous_source_hash" \
  "$previous_proxy_hash"; do
  if [ ! -f "$path" ]; then
    echo "Missing cycle-2 audit input: $path" >&2
    exit 1
  fi
done

previous_logs=("$previous_dir"/*.txt)
if [ "${#previous_logs[@]}" -ne 1 ]; then
  echo "Expected one completed first-cycle candidate log" >&2
  exit 1
fi
python - "$previous_gate" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1]))
if not (
    gate.get("decision") == "REJECT_SUPPORT_CONDITIONED_CLIP_PROXY"
    and gate.get("control_final") == 87.93
    and gate.get("candidate_final") == 87.94
    and gate.get("final_delta") == 0.01
):
    raise SystemExit("Previous matched-proxy result is not the locked 87.94 run")
print("==> Previous matched proxy locked: 87.93 -> 87.94")
PY

expected_proxy=$(mktemp)
observed_source_hash=$(mktemp)
observed_proxy_hash=$(mktemp)
trap 'rm -f "$expected_proxy" "$observed_source_hash" "$observed_proxy_hash"' EXIT
python tools/prepare_visda_proxy_subset.py \
  --input data/VISDA-C/validation_list.txt \
  --output "$expected_proxy" \
  --ratio 0.25 \
  --seed "$seed" \
  --force > /dev/null
if [ ! -f "$proxy_list" ]; then
  python tools/prepare_visda_proxy_subset.py \
    --input data/VISDA-C/validation_list.txt \
    --output "$proxy_list" \
    --ratio 0.25 \
    --seed "$seed"
fi
if ! cmp -s "$expected_proxy" "$proxy_list"; then
  echo "Proxy list is not the deterministic ratio=0.25 seed=2020 subset" >&2
  exit 1
fi
sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt > "$observed_source_hash"
sha256sum "$proxy_list" > "$observed_proxy_hash"
if ! cmp -s "$observed_source_hash" "$previous_source_hash"; then
  echo "Source checkpoints differ from the completed candidate" >&2
  exit 1
fi
if ! cmp -s "$observed_proxy_hash" "$previous_proxy_hash"; then
  echo "Proxy list differs from the completed candidate" >&2
  exit 1
fi
if [ -d "$output_dir" ]; then
  echo "Existing audit directory found; refusing to overwrite: $output_dir" >&2
  exit 1
fi

echo "==> Targeted cycle-2 boundary audit; not another four-cycle proxy"
echo "==> Runs cycle 1, writes pre-cycle-2 probabilities, then stops before optimization"
echo "==> Expected GPU time is about 11 minutes; no model checkpoint is saved"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_support_conditioned_clip.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 2 \
  ACTIVE.ADAPTATION_LIST "$proxy_list" \
  FAILURE_AUDIT.ENABLED True \
  FAILURE_AUDIT.DIR cycle2_conflict_memory_snapshots \
  FAILURE_AUDIT.FEATURE_DTYPE float16 \
  FAILURE_AUDIT.STOP_AFTER_PRE_CYCLE 2

logs=("$output_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one audit log" >&2
  exit 1
fi
if [ "$(grep -c 'Task: TV' "${logs[0]}")" -ne 4 ]; then
  echo "Audit must perform exactly one cycle / four evaluation checkpoints" >&2
  exit 1
fi
if ! grep -q \
  'Failure audit stop: after_pre_cycle=2; optimizer_steps_in_cycle=0' \
  "${logs[0]}"; then
  echo "Audit did not stop before cycle-2 optimization" >&2
  exit 1
fi
for snapshot in pre_cycle01.npz pre_cycle02.npz; do
  if [ ! -f "${snapshot_dir}/${snapshot}" ]; then
    echo "Missing expected audit snapshot: ${snapshot_dir}/${snapshot}" >&2
    exit 1
  fi
done

python - "${previous_logs[0]}" "${logs[0]}" <<'PY'
import re
import sys

pattern = re.compile(
    r"Task:\s*TV,\s*Iter:(\d+)/(\d+);\s*Cycle:\s*(\d+)/(\d+);\s*"
    r"Accuracy\s*=\s*([0-9.]+)%"
)
def cycle1(path):
    records = []
    for iteration, maximum, cycle, cycles, accuracy in pattern.findall(
        open(path, errors="ignore").read()
    ):
        if int(cycle) == 1:
            records.append((int(iteration), float(accuracy)))
    return records

previous = cycle1(sys.argv[1])
audit = cycle1(sys.argv[2])
if len(previous) != 4 or len(audit) != 4:
    raise SystemExit("Cycle-1 replay needs four matched checkpoints")
if [item[0] for item in previous] != [item[0] for item in audit]:
    raise SystemExit("Cycle-1 replay checkpoint iterations changed")
maximum_error = max(abs(a[1] - b[1]) for a, b in zip(previous, audit))
if maximum_error > 0.05:
    raise SystemExit(f"Cycle-1 replay drifted by {maximum_error:.4f} pp")
print(f"==> Cycle-1 replay contract passed; max_accuracy_error={maximum_error:.4f}pp")
PY

python tools/audit_visda_cycle2_conflict_memory.py \
  --pre-cycle1 "${snapshot_dir}/pre_cycle01.npz" \
  --pre-cycle2 "${snapshot_dir}/pre_cycle02.npz" \
  --output-dir "$audit_dir"

echo "==> Audit complete: ${audit_dir}/visda_cycle2_conflict_memory_summary.json"
echo "==> PASS authorizes method design only; no proxy or full training was started"
