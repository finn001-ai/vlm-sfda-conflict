#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="duet_clip_confidence_delay_visda_proxy25_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
candidate_dir="output/uda/VISDA-C/TV/${method}"
result_dir="output/uda/VISDA-C"
evidence_base="output/uda/VISDA-C/TV/duet_support_conditioned_clip_cycle2_memory_audit_seed2020"
snapshot="${evidence_base}/cycle2_conflict_memory_snapshots/pre_cycle01.npz"
audit_dir="${evidence_base}/agreement_rank_residual_audit"
audit_summary="${audit_dir}/visda_agreement_rank_residual_summary.json"
audit_lock="${audit_dir}/visda_agreement_rank_residual_signal_lock.json"
audit_signal="${audit_dir}/visda_agreement_rank_residual_label_free.npz"
control_summary="${result_dir}/plmatch_visda_proxy25_seed2020_summary.json"
control_source_hash="${result_dir}/plmatch_visda_proxy25_seed2020_source_sha256.txt"
control_proxy_hash="${result_dir}/plmatch_visda_proxy25_seed2020_proxy_sha256.txt"
candidate_summary="${result_dir}/${method}_summary.json"
candidate_per_class="${result_dir}/${method}_per_class.csv"
gate="${result_dir}/${method}_gate.json"
candidate_source_hash="${result_dir}/${method}_source_sha256.txt"
candidate_proxy_hash="${result_dir}/${method}_proxy_sha256.txt"
candidate_contract_hash="${result_dir}/${method}_contract_sha256.txt"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt \
  "$snapshot" \
  "$audit_summary" \
  "$audit_lock" \
  "$audit_signal" \
  "$control_summary" \
  "$control_source_hash" \
  "$control_proxy_hash"; do
  if [ ! -f "$path" ]; then
    echo "Missing CLIP-confidence delay proxy input: $path" >&2
    echo "No control rerun or candidate training was started." >&2
    exit 1
  fi
done

python - "$audit_summary" "$audit_lock" "$audit_signal" "$snapshot" <<'PY'
import hashlib
import json
import sys

import numpy as np
import torch

from src.utils.clip_confidence_delay import class_balanced_clip_confidence_delay

summary_path, lock_path, signal_path, snapshot_path = sys.argv[1:]
summary = json.load(open(summary_path))
lock = json.load(open(lock_path))
lock_hash = hashlib.sha256(open(lock_path, "rb").read()).hexdigest()
signal_hash = hashlib.sha256(open(signal_path, "rb").read()).hexdigest()
snapshot_hash = hashlib.sha256(open(snapshot_path, "rb").read()).hexdigest()
oracle = summary.get("oracle_metrics", {})
clip = oracle.get("baselines", {}).get("clip_confidence", {})
comparison = oracle.get("comparisons", {}).get("clip_confidence", {})
group = oracle.get("wrong_captures_by_pseudo_class_group", {})
checks = {
    "audit_rejects_rank_residual_not_clip_baseline": summary.get("decision") == "REJECT",
    "input_contract": summary.get("input_contract", {}).get("passed") is True,
    "labels_locked": summary.get("labels_used_only_after_signal_lock") is True,
    "lock_hash": lock_hash == summary.get("signal_lock_sha256"),
    "signal_hash": signal_hash == lock.get("signal_npz", {}).get("sha256"),
    "snapshot_hash": (
        snapshot_hash
        == lock.get("inputs", {}).get("pre_cycle1_snapshot", {}).get("sha256")
    ),
    "label_free_comparator": (
        lock.get("contains_target_labels") is False
        and lock.get("candidate_contract", {}).get("target_label_thresholds") is False
        and lock.get("candidate_contract", {}).get("fitted_parameters") is False
    ),
    "fixed_class_balanced_coverage": (
        lock.get("candidate_contract", {}).get("selection")
        == "largest 10 percent independently within each common pseudo class"
        and lock.get("label_free_metrics", {}).get("candidate_selected") == 683
    ),
    "clip_confidence_is_clear_winner": (
        clip.get("wrong_captured") == 197
        and clip.get("selected") == 683
        and clip.get("selection_error_precision_pct", 0.0) > 28.84
        and comparison.get("captured_error_gain") == -64
        and comparison.get("paired_bootstrap_95_ci_pp", [0.0, 0.0])[1] < 0.0
    ),
    "clip_confidence_wins_car_truck_noncar": (
        group.get("car", {}).get("clip_confidence") == 53
        and group.get("truck", {}).get("clip_confidence") == 7
        and group.get("noncar", {}).get("clip_confidence") == 137
    ),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"CLIP-confidence evidence contract failed: {failed}")

# Recompute the production selector from label-free snapshot arrays and require
# byte-identical sample membership with the mask locked before oracle labels.
with np.load(snapshot_path, allow_pickle=False) as snapshot:
    task_prediction = np.asarray(snapshot["source_label"], dtype=np.int64)
    clip_prediction = np.asarray(snapshot["clip_label"], dtype=np.int64)
    clip_probability = np.asarray(snapshot["clip_prob"], dtype=np.float32)
with np.load(signal_path, allow_pickle=False) as signal:
    expected = np.asarray(signal["clip_confidence_selected"], dtype=bool)
matching = task_prediction == clip_prediction
actual = class_balanced_clip_confidence_delay(
    torch.from_numpy(matching),
    torch.from_numpy(task_prediction),
    torch.from_numpy(clip_probability),
)["delayed"].numpy()
if not np.array_equal(actual, expected) or int(actual.sum()) != 683:
    raise SystemExit("Production delay selector differs from locked label-free mask")
print("==> Offline evidence authorizes exactly one matched proxy25 candidate")
print("    Cycle-1 agreements=6777; delayed=683; wrong captured=197/399")
print("    Delayed-set oracle error precision=28.8433%; retained accuracy=96.6853%")
print("    Production selector exactly matches the pre-oracle locked mask")
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
if ! cmp -s "$observed_source_hash" "$control_source_hash"; then
  echo "Source checkpoint hashes differ from the matched DUET control" >&2
  exit 1
fi
if ! cmp -s "$observed_proxy_hash" "$control_proxy_hash"; then
  echo "Proxy-list hash differs from the matched DUET control" >&2
  exit 1
fi
python - "$control_summary" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
final = summary.get("final", {})
if not (
    summary.get("num_checkpoints") == 16
    and final.get("cycle") == 4
    and final.get("accuracy") == 87.93
):
    raise SystemExit("Matched arithmetic-DUET control is not the locked 87.93 run")
print("==> Reusing matched arithmetic-DUET proxy control: final=87.93")
PY

if [ -d "$candidate_dir" ]; then
  echo "Existing candidate directory found; refusing to overwrite: $candidate_dir" >&2
  exit 1
fi

cp "$observed_source_hash" "$candidate_source_hash"
cp "$observed_proxy_hash" "$candidate_proxy_hash"
sha256sum \
  conf.py \
  cfgs/visda/duet_clip_confidence_delay.yaml \
  src/methods/oh/plmatch.py \
  src/methods/oh/duet_clip_confidence_delay.py \
  src/utils/clip_confidence_delay.py \
  "$audit_lock" \
  "$audit_signal" \
  > "$candidate_contract_hash"

echo "==> One matched class-balanced CLIP-confidence delay proxy25"
echo "==> Only cycle-1 hard pseudo-label admission changes; CLIP KL is untouched"
echo "==> Delayed agreements can enter naturally from cycle 2 onward"
echo "==> No control rerun; expected candidate GPU time is about 40 minutes"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_clip_confidence_delay.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD "$method" \
  SETTING.SEED "$seed" SETTING.S 0 SETTING.T 1 \
  ACTIVE.CYCLE 4 \
  ACTIVE.ADAPTATION_LIST "$proxy_list"

logs=("$candidate_dir"/*.txt)
if [ "${#logs[@]}" -ne 1 ]; then
  echo "Expected exactly one candidate log, found ${#logs[@]}" >&2
  exit 1
fi
startup_contract="DUET CLIP-confidence delay: enabled=True; first_cycle_only=True; per_pseudo_class_fraction=0.100"
if ! grep -Fq "$startup_contract" \
  "${logs[0]}"; then
  echo "Candidate did not enable the locked CLIP-confidence delay" >&2
  exit 1
fi
cycle1_contract="DUET CLIP-confidence delay applied: cycle=1; original_agreements=6777; delayed=683; retained=6094;"
if ! grep -Fq "$cycle1_contract" \
  "${logs[0]}"; then
  echo "Cycle-1 selector did not reproduce the locked 6777/683/6094 contract" >&2
  exit 1
fi
inactive_pattern='DUET CLIP-confidence delay applied: cycle=[234]; delayed=0; first_cycle_only=True'
if [ "$(grep -Ec "$inactive_pattern" "${logs[0]}")" -ne 3 ]; then
  echo "Delay must be inactive after cycle 1" >&2
  exit 1
fi
if ! grep -Fq \
  "Number of valid pseudo-labeled samples: 6094/13847; Accuracy = 96.69%" \
  "${logs[0]}"; then
  echo "Cycle-1 oracle replay did not reproduce the audited retained precision" >&2
  exit 1
fi
if [ "$(grep -c 'Task: TV' "${logs[0]}")" -ne 16 ]; then
  echo "Candidate did not finish the four-cycle proxy contract" >&2
  exit 1
fi

python tools/summarize_visda_run.py \
  --glob "$candidate_dir/*.txt" \
  --out "$candidate_summary" \
  --csv-out "$candidate_per_class" \
  --class-names data/VISDA-C/classname.txt

python tools/analyze_duet_clip_confidence_delay_proxy.py \
  --control-summary "$control_summary" \
  --candidate-summary "$candidate_summary" \
  --out "$gate"

echo "==> Gate: $gate"
echo "==> Even PASS does not authorize or start a full VisDA run"
