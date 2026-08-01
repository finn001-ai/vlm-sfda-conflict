#!/usr/bin/env bash
set -euo pipefail

seed=2020
method="duet_boundary_router_visda_proxy25_seed${seed}"
proxy_list="data/VISDA-C/validation_proxy25_seed2020_list.txt"
candidate_dir="output/uda/VISDA-C/TV/${method}"
result_dir="output/uda/VISDA-C"
control_summary="${CONTROL_SUMMARY:-${result_dir}/plmatch_visda_proxy25_seed2020_summary.json}"
archive_control_dir="${ARCHIVED_CONTROL_DIR:-../archive/sfda_conflict_visda_proxy_loss_audit_2026-07-23}"
archive_control_summary="${archive_control_dir}/plmatch_visda_proxy25_seed2020_summary.json"
archive_control_log="${archive_control_dir}/plmatch_proxy25_control_terminal_record.txt"
archive_checksums="${archive_control_dir}/SHA256SUMS"
historical_control_commit="91ef7df"
candidate_summary="${result_dir}/duet_boundary_router_visda_proxy25_seed2020_summary.json"
gate="${result_dir}/duet_boundary_router_visda_proxy25_seed2020_gate.json"
control_source_hash="${result_dir}/plmatch_visda_proxy25_seed2020_source_sha256.txt"
control_proxy_hash="${result_dir}/plmatch_visda_proxy25_seed2020_proxy_sha256.txt"
control_contract_hash="${result_dir}/plmatch_visda_proxy25_seed2020_contract_sha256.txt"
candidate_source_hash="${result_dir}/duet_boundary_router_visda_proxy25_seed2020_source_sha256.txt"
candidate_proxy_hash="${result_dir}/duet_boundary_router_visda_proxy25_seed2020_proxy_sha256.txt"
candidate_contract_hash="${result_dir}/duet_boundary_router_visda_proxy25_seed2020_contract_sha256.txt"
control_provenance="matched_current_output_hashes"

for path in \
  data/VISDA-C/validation_list.txt \
  data/VISDA-C/classname.txt \
  source/uda/VISDA-C/T/source_F.pt \
  source/uda/VISDA-C/T/source_B.pt \
  source/uda/VISDA-C/T/source_C.pt; do
  if [ ! -f "$path" ]; then
    echo "Missing VisDA-C input: $path" >&2
    exit 1
  fi
done

expected_proxy=$(mktemp)
current_source_hash=$(mktemp)
current_proxy_hash=$(mktemp)
current_contract_hash=$(mktemp)
historical_plmatch=$(mktemp)
trap 'rm -f "$expected_proxy" "$current_source_hash" "$current_proxy_hash" "$current_contract_hash" "$historical_plmatch"' EXIT

current_control_complete=true
for artifact in \
  "$control_summary" \
  "$control_source_hash" \
  "$control_proxy_hash" \
  "$control_contract_hash"; do
  if [ ! -f "$artifact" ]; then
    current_control_complete=false
  fi
done

if [ "$current_control_complete" = false ]; then
  for artifact in \
    "$archive_control_summary" \
    "$archive_control_log" \
    "$archive_checksums"; do
    if [ ! -f "$artifact" ]; then
      echo "Missing archived DUET control evidence: $artifact" >&2
      exit 1
    fi
  done

  verify_archive_file() {
    local filename="$1"
    local expected_sha
    local actual_sha
    expected_sha=$(awk -v filename="$filename" '$2 == filename {print $1}' "$archive_checksums")
    actual_sha=$(sha256sum "${archive_control_dir}/${filename}" | awk '{print $1}')
    if [ -z "$expected_sha" ] || [ "$actual_sha" != "$expected_sha" ]; then
      echo "Archived control checksum failed: $filename" >&2
      exit 1
    fi
  }
  verify_archive_file "plmatch_visda_proxy25_seed2020_summary.json"
  verify_archive_file "plmatch_proxy25_control_terminal_record.txt"

  python - "$archive_control_summary" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
final = summary.get("final", {})
if not (
    summary.get("num_checkpoints") == 16
    and final.get("cycle") == 4
    and final.get("accuracy") == 87.93
):
    raise SystemExit("Archived DUET control contract is not the predeclared 87.93 run")
PY
  if [ "$(grep -c "Task: TV" "$archive_control_log")" -ne 16 ]; then
    echo "Archived DUET control log does not contain 16 checkpoints" >&2
    exit 1
  fi
  if ! grep -q \
    "PLMatch adaptation proxy list: data/VISDA-C/validation_proxy25_seed2020_list.txt; adaptation_samples=13847; full_evaluation_samples=55388" \
    "$archive_control_log"; then
    echo "Archived DUET control does not match the 25% proxy/full-evaluation contract" >&2
    exit 1
  fi
  if ! git show \
    "${historical_control_commit}:duet-sfda-main/src/methods/oh/plmatch.py" \
    > "$historical_plmatch"; then
    echo "Cannot audit historical DUET control commit: $historical_control_commit" >&2
    exit 1
  fi
  if ! grep -Fq \
    "all_mix_output = (all_output + clip_all_output) / 2" \
    "$historical_plmatch"; then
    echo "Historical DUET control was not arithmetic probability fusion" >&2
    exit 1
  fi

  control_summary="$archive_control_summary"
  control_provenance="archived_2026-07-23_source_and_list_hashes_unavailable"
  echo "==> Reusing archived arithmetic-DUET proxy control: final=87.93"
  echo "    Provenance note: historical source/list byte hashes were not archived"
fi

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

sha256sum source/uda/VISDA-C/T/source_{F,B,C}.pt > "$current_source_hash"
sha256sum "$proxy_list" > "$current_proxy_hash"
sha256sum \
  conf.py \
  cfgs/visda/plmatch.yaml \
  src/methods/oh/plmatch.py \
  src/utils/conflict_boundary.py \
  > "$current_contract_hash"
if [ "$current_control_complete" = true ]; then
  if ! cmp -s "$current_source_hash" "$control_source_hash"; then
    echo "Source checkpoint hashes differ from the matched control" >&2
    exit 1
  fi
  if ! cmp -s "$current_proxy_hash" "$control_proxy_hash"; then
    echo "Proxy-list hash differs from the matched control" >&2
    exit 1
  fi
  if ! cmp -s "$current_contract_hash" "$control_contract_hash"; then
    echo "DUET code/config hashes differ from the matched control" >&2
    exit 1
  fi
fi

if [ -d "$candidate_dir" ]; then
  echo "Existing boundary-router candidate directory found; refusing to overwrite: $candidate_dir" >&2
  exit 1
fi

mkdir -p "$result_dir"
cp "$current_source_hash" "$candidate_source_hash"
cp "$current_proxy_hash" "$candidate_proxy_hash"
cp "$current_contract_hash" "$candidate_contract_hash"

echo "==> DUET boundary-router proxy: cycle-1 top-20% conflict soft routing only"
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_boundary_router.yaml \
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
if ! grep -q \
  "DUET boundary router: enabled=True; first_cycle_only=True; top_fraction=0.200" \
  "${logs[0]}"; then
  echo "Candidate did not enable the predeclared boundary router" >&2
  exit 1
fi
if [ "$(grep -c "DUET boundary routing: cycle=1;" "${logs[0]}")" -ne 1 ]; then
  echo "Cycle-1 boundary-routing contract failed" >&2
  exit 1
fi
if ! grep -Eq "DUET boundary routing: cycle=1; active_conflicts=[1-9][0-9]*; selected=[1-9][0-9]*;" "${logs[0]}"; then
  echo "Boundary router selected no cycle-1 conflicts" >&2
  exit 1
fi
if [ "$(grep -Ec "DUET boundary routing: cycle=[234]; active_conflicts=0; selected=0;" "${logs[0]}")" -ne 3 ]; then
  echo "Boundary router was not restricted to cycle 1" >&2
  exit 1
fi
if [ "$(grep -c "Task: TV" "${logs[0]}")" -ne 16 ]; then
  echo "Candidate did not finish the 4-cycle proxy contract" >&2
  exit 1
fi

python tools/summarize_visda_run.py \
  --glob "$candidate_dir/*.txt" \
  --out "$candidate_summary" \
  --csv-out "$result_dir/duet_boundary_router_visda_proxy25_seed2020_per_class.csv" \
  --class-names data/VISDA-C/classname.txt

python tools/analyze_duet_boundary_router_visda_proxy.py \
  --control-summary "$control_summary" \
  --candidate-summary "$candidate_summary" \
  --control-provenance "$control_provenance" \
  --out "$gate"

echo "==> Gate: $gate"
