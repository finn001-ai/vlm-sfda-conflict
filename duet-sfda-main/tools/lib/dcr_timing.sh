#!/usr/bin/env bash

# Small, dependency-free wall-clock recorder shared by the formal DCR runners.
# The CSV is intentionally kept outside the Python logs so Stage 1 and Stage 2
# remain independently measurable even though they are separate processes.

dcr_timing_init() {
  local timing_file="$1"
  mkdir -p "$(dirname "$timing_file")"
  if [ ! -f "$timing_file" ]; then
    printf 'stage,seconds,hms,reused,start_time,end_time\n' > "$timing_file"
  fi
}

dcr_timing_has_stage() {
  local timing_file="$1"
  local stage="$2"
  [ -f "$timing_file" ] \
    && awk -F, -v stage="$stage" '$1 == stage {found = 1} END {exit !found}' \
      "$timing_file"
}

dcr_timing_format_seconds() {
  local total_seconds="$1"
  if [[ ! "$total_seconds" =~ ^[0-9]+$ ]]; then
    printf 'NA'
    return
  fi
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

dcr_timing_record() {
  local timing_file="$1"
  local stage="$2"
  local seconds="$3"
  local reused="$4"
  local start_time="$5"
  local end_time="$6"
  local formatted
  local temporary_file

  dcr_timing_init "$timing_file"
  formatted="$(dcr_timing_format_seconds "$seconds")"
  temporary_file="${timing_file}.tmp.$$"
  awk -F, -v stage="$stage" 'NR == 1 || $1 != stage' \
    "$timing_file" > "$temporary_file"
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$stage" "$seconds" "$formatted" "$reused" "$start_time" "$end_time" \
    >> "$temporary_file"
  mv "$temporary_file" "$timing_file"
}

dcr_timing_record_total() {
  local timing_file="$1"
  local stage1_seconds stage2_seconds total_seconds
  local stage1_start stage2_end

  stage1_seconds="$(awk -F, '$1 == "stage1" {print $2}' "$timing_file")"
  stage2_seconds="$(awk -F, '$1 == "stage2" {print $2}' "$timing_file")"
  stage1_start="$(awk -F, '$1 == "stage1" {print $5}' "$timing_file")"
  stage2_end="$(awk -F, '$1 == "stage2" {print $6}' "$timing_file")"
  if [[ "$stage1_seconds" =~ ^[0-9]+$ ]] \
    && [[ "$stage2_seconds" =~ ^[0-9]+$ ]]; then
    total_seconds="$((stage1_seconds + stage2_seconds))"
    dcr_timing_record \
      "$timing_file" total "$total_seconds" false "$stage1_start" "$stage2_end"
  else
    dcr_timing_record "$timing_file" total NA partial "$stage1_start" "$stage2_end"
  fi
}
