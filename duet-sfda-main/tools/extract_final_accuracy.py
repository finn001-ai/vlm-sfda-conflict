#!/usr/bin/env python
"""Extract final accuracy and oracle peak diagnostics from training logs."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path


ACCURACY_PATTERN = re.compile(
    r"(?<!Trajectory Ensemble )Task:\s*([A-Z]{2}),\s*Iter:\s*(\d+)/(\d+);\s*"
    r"Cycle:\s*(\d+)/(\d+);\s*Accuracy\s*=\s*([0-9.]+)%"
)
TRAJECTORY_ACCURACY_PATTERN = re.compile(
    r"Trajectory Ensemble Task:\s*([A-Z]{2}),\s*Iter:\s*(\d+)/(\d+);\s*"
    r"Cycle:\s*(\d+)/(\d+);\s*Accuracy\s*=\s*([0-9.]+)%"
)


def select_final_and_peak(text: str, pattern=ACCURACY_PATTERN):
    matches = pattern.findall(text)
    if not matches:
        return None, None
    return matches[-1], max(matches, key=lambda item: float(item[5]))


def select_primary(final, peak, selection: str):
    if selection == "peak":
        return peak
    if selection == "final":
        return final
    raise ValueError("selection must be final or peak")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="output/uda/office-home/*/*/*.txt",
        help="Glob for log txt files.",
    )
    parser.add_argument(
        "--selection",
        choices=("final", "peak"),
        default="final",
        help="Which logged checkpoint populates the primary accuracy column.",
    )
    parser.add_argument(
        "--record-type",
        choices=("standard", "trajectory"),
        default="standard",
        help="Select standard checkpoints or trajectory-ensemble records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.glob))
    cfg_patterns = {
        "cand_par": re.compile(r"^\s*CAND_PAR:\s*([^\s]+)", re.MULTILINE),
        "cand_start_cycle": re.compile(r"^\s*CAND_START_CYCLE:\s*([^\s]+)", re.MULTILINE),
        "cand_tau": re.compile(r"^\s*CAND_TAU:\s*([^\s]+)", re.MULTILINE),
        "cand_weight": re.compile(r"^\s*CAND_WEIGHT:\s*([^\s]+)", re.MULTILINE),
        "kl_mode": re.compile(r"^\s*KL_MODE:\s*([^\s]+)", re.MULTILINE),
        "kl_candidate": re.compile(r"^\s*KL_CANDIDATE:\s*([^\s]+)", re.MULTILINE),
        "calib_mode": re.compile(r"^\s*CALIB_MODE:\s*([^\s]+)", re.MULTILINE),
        "calib_power": re.compile(r"^\s*CALIB_POWER:\s*([^\s]+)", re.MULTILINE),
        "calib_auto_lambda": re.compile(r"^\s*CALIB_AUTO_LAMBDA:\s*([^\s]+)", re.MULTILINE),
        "topo_graph_k": re.compile(r"^\s*TOPO_GRAPH_K:\s*([^\s]+)", re.MULTILINE),
        "topo_alpha": re.compile(r"^\s*TOPO_ALPHA:\s*([^\s]+)", re.MULTILINE),
        "topo_steps": re.compile(r"^\s*TOPO_STEPS:\s*([^\s]+)", re.MULTILINE),
        "topo_anchor_ratio": re.compile(r"^\s*TOPO_ANCHOR_RATIO:\s*([^\s]+)", re.MULTILINE),
        "topo_target_mix": re.compile(r"^\s*TOPO_TARGET_MIX:\s*([^\s]+)", re.MULTILINE),
        "graph_teacher_fusion": re.compile(r"^\s*GRAPH_TEACHER_FUSION:\s*([^\s]+)", re.MULTILINE),
        "gtf_apply_to": re.compile(r"^\s*GTF_APPLY_TO:\s*([^\s]+)", re.MULTILINE),
        "gtf_strength": re.compile(r"^\s*GTF_STRENGTH:\s*([^\s]+)", re.MULTILINE),
        "gtr_par": re.compile(r"^\s*GTR_PAR:\s*([^\s]+)", re.MULTILINE),
        "gtr_stable_cycles": re.compile(r"^\s*GTR_STABLE_CYCLES:\s*([^\s]+)", re.MULTILINE),
        "gtr_memory": re.compile(r"^\s*GTR_MEMORY:\s*([^\s]+)", re.MULTILINE),
        "gtr_min_graph_conf": re.compile(r"^\s*GTR_MIN_GRAPH_CONF:\s*([^\s]+)", re.MULTILINE),
        "gtr_min_disagreement": re.compile(r"^\s*GTR_MIN_DISAGREEMENT:\s*([^\s]+)", re.MULTILINE),
        "temporal_diag": re.compile(r"^\s*TEMPORAL_DIAG:\s*([^\s]+)", re.MULTILINE),
        "pl_expand": re.compile(r"^\s*PL_EXPAND:\s*([^\s]+)", re.MULTILINE),
        "pl_topk_per_class": re.compile(r"^\s*PL_TOPK_PER_CLASS:\s*([^\s]+)", re.MULTILINE),
        "pl_min_conf": re.compile(r"^\s*PL_MIN_CONF:\s*([^\s]+)", re.MULTILINE),
        "pl_memory": re.compile(r"^\s*PL_MEMORY:\s*([^\s]+)", re.MULTILINE),
        "pl_stable_cycles": re.compile(r"^\s*PL_STABLE_CYCLES:\s*([^\s]+)", re.MULTILINE),
        "pl_stable_memory": re.compile(r"^\s*PL_STABLE_MEMORY:\s*([^\s]+)", re.MULTILINE),
        "pl_memory_warmup_cycles": re.compile(r"^\s*PL_MEMORY_WARMUP_CYCLES:\s*([^\s]+)", re.MULTILINE),
        "pl_memory_min_conf": re.compile(r"^\s*PL_MEMORY_MIN_CONF:\s*([^\s]+)", re.MULTILINE),
        "pl_class_balance": re.compile(r"^\s*PL_CLASS_BALANCE:\s*([^\s]+)", re.MULTILINE),
        "pl_balance_coverage": re.compile(r"^\s*PL_BALANCE_COVERAGE:\s*([^\s]+)", re.MULTILINE),
        "pl_balance_min_per_class": re.compile(r"^\s*PL_BALANCE_MIN_PER_CLASS:\s*([^\s]+)", re.MULTILINE),
        "proto_adapt": re.compile(r"^\s*PROTO_ADAPT:\s*([^\s]+)", re.MULTILINE),
        "proto_mix": re.compile(r"^\s*PROTO_MIX:\s*([^\s]+)", re.MULTILINE),
        "proto_temperature": re.compile(r"^\s*PROTO_TEMPERATURE:\s*([^\s]+)", re.MULTILINE),
        "proto_min_per_class": re.compile(r"^\s*PROTO_MIN_PER_CLASS:\s*([^\s]+)", re.MULTILINE),
        "proto_momentum": re.compile(r"^\s*PROTO_MOMENTUM:\s*([^\s]+)", re.MULTILINE),
        "target_head_adapt": re.compile(r"^\s*TARGET_HEAD_ADAPT:\s*([^\s]+)", re.MULTILINE),
        "target_head_variant": re.compile(r"^\s*TARGET_HEAD_VARIANT:\s*([^\s]+)", re.MULTILINE),
        "target_head_mix": re.compile(r"^\s*TARGET_HEAD_MIX:\s*([^\s]+)", re.MULTILINE),
        "target_head_start_cycle": re.compile(r"^\s*TARGET_HEAD_START_CYCLE:\s*([^\s]+)", re.MULTILINE),
        "target_head_lr_mult": re.compile(r"^\s*TARGET_HEAD_LR_MULT:\s*([^\s]+)", re.MULTILINE),
        "target_head_ema": re.compile(r"^\s*TARGET_HEAD_EMA:\s*([^\s]+)", re.MULTILINE),
        "target_head_ema_momentum": re.compile(r"^\s*TARGET_HEAD_EMA_MOMENTUM:\s*([^\s]+)", re.MULTILINE),
        "target_residual_max_gate": re.compile(r"^\s*TARGET_RESIDUAL_MAX_GATE:\s*([^\s]+)", re.MULTILINE),
        "target_residual_gate_init": re.compile(r"^\s*TARGET_RESIDUAL_GATE_INIT:\s*([^\s]+)", re.MULTILINE),
        "trajectory_ensemble": re.compile(r"^\s*TRAJECTORY_ENSEMBLE:\s*([^\s]+)", re.MULTILINE),
        "trajectory_snapshot_intervals": re.compile(r"^\s*TRAJECTORY_SNAPSHOT_INTERVALS:\s*(\[[^\n]+\])", re.MULTILINE),
        "pair_flow_rank": re.compile(r"^\s*PAIR_FLOW_RANK:\s*([^\s]+)", re.MULTILINE),
        "pair_flow_min_count": re.compile(r"^\s*PAIR_FLOW_MIN_COUNT:\s*([^\s]+)", re.MULTILINE),
        "pair_flow_min_cycles": re.compile(r"^\s*PAIR_FLOW_MIN_CYCLES:\s*([^\s]+)", re.MULTILINE),
        "pair_flow_max_gate": re.compile(r"^\s*PAIR_FLOW_MAX_GATE:\s*([^\s]+)", re.MULTILINE),
        "pair_flow_gate_init": re.compile(r"^\s*PAIR_FLOW_GATE_INIT:\s*([^\s]+)", re.MULTILINE),
        "pair_feature_adapt": re.compile(r"^\s*PAIR_FEATURE_ADAPT:\s*([^\s]+)", re.MULTILINE),
        "pair_feature_start_cycle": re.compile(r"^\s*PAIR_FEATURE_START_CYCLE:\s*([^\s]+)", re.MULTILINE),
        "pair_feature_lr_mult": re.compile(r"^\s*PAIR_FEATURE_LR_MULT:\s*([^\s]+)", re.MULTILINE),
        "pair_feature_min_active_rank": re.compile(r"^\s*PAIR_FEATURE_MIN_ACTIVE_RANK:\s*([^\s]+)", re.MULTILINE),
        "pair_feature_max_gate": re.compile(r"^\s*PAIR_FEATURE_MAX_GATE:\s*([^\s]+)", re.MULTILINE),
        "pair_feature_gate_init": re.compile(r"^\s*PAIR_FEATURE_GATE_INIT:\s*([^\s]+)", re.MULTILINE),
        "cov_transport_adapt": re.compile(r"^\s*COV_TRANSPORT_ADAPT:\s*([^\s]+)", re.MULTILINE),
        "cov_transport_start_cycle": re.compile(r"^\s*COV_TRANSPORT_START_CYCLE:\s*([^\s]+)", re.MULTILINE),
        "cov_transport_min_anchors": re.compile(r"^\s*COV_TRANSPORT_MIN_ANCHORS:\s*([^\s]+)", re.MULTILINE),
        "cov_transport_rank": re.compile(r"^\s*COV_TRANSPORT_RANK:\s*([^\s]+)", re.MULTILINE),
        "cov_transport_max_gate": re.compile(r"^\s*COV_TRANSPORT_MAX_GATE:\s*([^\s]+)", re.MULTILINE),
        "tau_low": re.compile(r"^\s*TAU_LOW:\s*([^\s]+)", re.MULTILINE),
        "promote_k": re.compile(r"^\s*PROMOTE_K:\s*([^\s]+)", re.MULTILINE),
        "accd_enabled": re.compile(r"^\s*ENABLED:\s*([^\s]+)", re.MULTILINE),
        "accd_graph_k": re.compile(r"^\s*GRAPH_K:\s*([^\s]+)", re.MULTILINE),
        "accd_anchor_ratio": re.compile(r"^\s*ANCHOR_RATIO:\s*([^\s]+)", re.MULTILINE),
        "accd_anchor_memory": re.compile(r"^\s*ANCHOR_MEMORY:\s*([^\s]+)", re.MULTILINE),
        "accd_candidate_mass": re.compile(r"^\s*CANDIDATE_MASS:\s*([^\s]+)", re.MULTILINE),
        "accd_candidate_margin": re.compile(r"^\s*CANDIDATE_MARGIN:\s*([^\s]+)", re.MULTILINE),
        "accd_stable_cycles": re.compile(r"^\s*STABLE_CYCLES:\s*([^\s]+)", re.MULTILINE),
        "accd_resolution_memory": re.compile(r"^\s*RESOLUTION_MEMORY:\s*([^\s]+)", re.MULTILINE),
        "accd_resolution_target": re.compile(r"^\s*RESOLUTION_TARGET:\s*([^\s]+)", re.MULTILINE),
        "accd_resolution_action": re.compile(r"^\s*RESOLUTION_ACTION:\s*([^\s]+)", re.MULTILINE),
    }
    print("method,task,record_type,selection,cycle,iter,accuracy,final_accuracy,final_cycle,final_iter,peak_accuracy,peak_cycle,peak_iter,peak_minus_final,cand_par,cand_start_cycle,cand_tau,cand_weight,kl_mode,kl_candidate,calib_mode,calib_power,calib_auto_lambda,topo_graph_k,topo_alpha,topo_steps,topo_anchor_ratio,topo_target_mix,graph_teacher_fusion,gtf_apply_to,gtf_strength,gtr_par,gtr_stable_cycles,gtr_memory,gtr_min_graph_conf,gtr_min_disagreement,temporal_diag,pl_expand,pl_topk_per_class,pl_min_conf,pl_memory,pl_stable_cycles,pl_stable_memory,pl_memory_warmup_cycles,pl_memory_min_conf,pl_class_balance,pl_balance_coverage,pl_balance_min_per_class,proto_adapt,proto_mix,proto_temperature,proto_min_per_class,proto_momentum,target_head_adapt,target_head_variant,target_head_mix,target_head_start_cycle,target_head_lr_mult,target_head_ema,target_head_ema_momentum,target_residual_max_gate,target_residual_gate_init,residual_gate_final,trajectory_ensemble,trajectory_snapshot_intervals,pair_flow_rank,pair_flow_min_count,pair_flow_min_cycles,pair_flow_max_gate,pair_flow_gate_init,pair_flow_gate_final,pair_flow_active_rank,pair_feature_adapt,pair_feature_start_cycle,pair_feature_lr_mult,pair_feature_min_active_rank,pair_feature_max_gate,pair_feature_gate_init,pair_feature_gate_final,pair_feature_router_norm,pair_feature_effective,cov_transport_adapt,cov_transport_start_cycle,cov_transport_min_anchors,cov_transport_rank,cov_transport_max_gate,cov_transport_active_classes,cov_transport_coverage,cov_transport_mean_shift,tau_low,promote_k,accd_enabled,accd_graph_k,accd_anchor_ratio,accd_anchor_memory,accd_candidate_mass,accd_candidate_margin,accd_stable_cycles,accd_resolution_memory,accd_resolution_target,accd_resolution_action,log")
    accuracy_pattern = (
        TRAJECTORY_ACCURACY_PATTERN
        if args.record_type == "trajectory"
        else ACCURACY_PATTERN
    )
    for path in paths:
        text = path.read_text(errors="ignore")
        final, peak = select_final_and_peak(text, accuracy_pattern)
        if final is None:
            continue
        task, iter_num, max_iter, cycle, max_cycle, acc = final
        _, peak_iter_num, peak_max_iter, peak_cycle, peak_max_cycle, peak_acc = peak
        peak_gap = float(peak_acc) - float(acc)
        selected = select_primary(final, peak, args.selection)
        (
            _,
            selected_iter,
            selected_max_iter,
            selected_cycle,
            selected_max_cycle,
            selected_acc,
        ) = selected
        parts = path.parts
        method = parts[-2] if len(parts) >= 2 else "unknown"
        cfg_values = {}
        for key, cfg_pattern in cfg_patterns.items():
            match = cfg_pattern.search(text)
            cfg_values[key] = match.group(1) if match else ""
        residual_gate_matches = re.findall(r"residual_gate=([0-9.]+)", text)
        residual_gate_final = residual_gate_matches[-1] if residual_gate_matches else ""
        pair_gate_matches = re.findall(r"pair_flow_gate=([0-9.]+)", text)
        pair_rank_matches = re.findall(r"pair_flow_active_rank=(\d+)", text)
        pair_flow_gate_final = pair_gate_matches[-1] if pair_gate_matches else ""
        pair_flow_active_rank = pair_rank_matches[-1] if pair_rank_matches else ""
        pair_feature_gate_matches = re.findall(r"pair_feature_gate=([0-9.]+)", text)
        pair_feature_gate_final = (
            pair_feature_gate_matches[-1] if pair_feature_gate_matches else ""
        )
        pair_feature_norm_matches = re.findall(
            r"pair_feature_router_norm=([0-9.]+)", text
        )
        pair_feature_router_norm = (
            pair_feature_norm_matches[-1] if pair_feature_norm_matches else ""
        )
        pair_feature_effective_matches = re.findall(
            r"pair_feature_effective=(True|False)", text
        )
        pair_feature_effective = (
            pair_feature_effective_matches[-1]
            if pair_feature_effective_matches
            else ""
        )
        cov_active_matches = re.findall(
            r"cov_transport_active_classes=(\d+)", text
        )
        cov_coverage_matches = re.findall(
            r"cov_transport_coverage=([0-9.]+)", text
        )
        cov_shift_matches = re.findall(
            r"cov_transport_mean_shift=([0-9.]+)", text
        )
        cov_transport_active_classes = (
            cov_active_matches[-1] if cov_active_matches else ""
        )
        cov_transport_coverage = (
            cov_coverage_matches[-1] if cov_coverage_matches else ""
        )
        cov_transport_mean_shift = (
            cov_shift_matches[-1] if cov_shift_matches else ""
        )
        trajectory_snapshot_intervals = cfg_values[
            'trajectory_snapshot_intervals'
        ].replace(",", "|")
        print(
            f"{method},{task},{args.record_type},{args.selection},"
            f"{selected_cycle}/{selected_max_cycle},"
            f"{selected_iter}/{selected_max_iter},{selected_acc},"
            f"{acc},{cycle}/{max_cycle},{iter_num}/{max_iter},"
            f"{peak_acc},{peak_cycle}/{peak_max_cycle},"
            f"{peak_iter_num}/{peak_max_iter},{peak_gap:.2f},"
            f"{cfg_values['cand_par']},{cfg_values['cand_start_cycle']},"
            f"{cfg_values['cand_tau']},{cfg_values['cand_weight']},"
            f"{cfg_values['kl_mode']},{cfg_values['kl_candidate']},"
            f"{cfg_values['calib_mode']},{cfg_values['calib_power']},{cfg_values['calib_auto_lambda']},"
            f"{cfg_values['topo_graph_k']},{cfg_values['topo_alpha']},"
            f"{cfg_values['topo_steps']},{cfg_values['topo_anchor_ratio']},"
            f"{cfg_values['topo_target_mix']},"
            f"{cfg_values['graph_teacher_fusion']},{cfg_values['gtf_apply_to']},"
            f"{cfg_values['gtf_strength']},"
            f"{cfg_values['gtr_par']},{cfg_values['gtr_stable_cycles']},"
            f"{cfg_values['gtr_memory']},{cfg_values['gtr_min_graph_conf']},"
            f"{cfg_values['gtr_min_disagreement']},"
            f"{cfg_values['temporal_diag']},"
            f"{cfg_values['pl_expand']},{cfg_values['pl_topk_per_class']},{cfg_values['pl_min_conf']},"
            f"{cfg_values['pl_memory']},{cfg_values['pl_stable_cycles']},"
            f"{cfg_values['pl_stable_memory']},{cfg_values['pl_memory_warmup_cycles']},"
            f"{cfg_values['pl_memory_min_conf']},"
            f"{cfg_values['pl_class_balance']},{cfg_values['pl_balance_coverage']},"
            f"{cfg_values['pl_balance_min_per_class']},"
            f"{cfg_values['proto_adapt']},{cfg_values['proto_mix']},"
            f"{cfg_values['proto_temperature']},{cfg_values['proto_min_per_class']},"
            f"{cfg_values['proto_momentum']},"
            f"{cfg_values['target_head_adapt']},{cfg_values['target_head_variant']},"
            f"{cfg_values['target_head_mix']},"
            f"{cfg_values['target_head_start_cycle']},{cfg_values['target_head_lr_mult']},"
            f"{cfg_values['target_head_ema']},{cfg_values['target_head_ema_momentum']},"
            f"{cfg_values['target_residual_max_gate']},"
            f"{cfg_values['target_residual_gate_init']},{residual_gate_final},"
            f"{cfg_values['trajectory_ensemble']},"
            f"{trajectory_snapshot_intervals},"
            f"{cfg_values['pair_flow_rank']},{cfg_values['pair_flow_min_count']},"
            f"{cfg_values['pair_flow_min_cycles']},{cfg_values['pair_flow_max_gate']},"
            f"{cfg_values['pair_flow_gate_init']},{pair_flow_gate_final},"
            f"{pair_flow_active_rank},"
            f"{cfg_values['pair_feature_adapt']},"
            f"{cfg_values['pair_feature_start_cycle']},"
            f"{cfg_values['pair_feature_lr_mult']},"
            f"{cfg_values['pair_feature_min_active_rank']},"
            f"{cfg_values['pair_feature_max_gate']},"
            f"{cfg_values['pair_feature_gate_init']},{pair_feature_gate_final},"
            f"{pair_feature_router_norm},{pair_feature_effective},"
            f"{cfg_values['cov_transport_adapt']},"
            f"{cfg_values['cov_transport_start_cycle']},"
            f"{cfg_values['cov_transport_min_anchors']},"
            f"{cfg_values['cov_transport_rank']},"
            f"{cfg_values['cov_transport_max_gate']},"
            f"{cov_transport_active_classes},{cov_transport_coverage},"
            f"{cov_transport_mean_shift},"
            f"{cfg_values['tau_low']},{cfg_values['promote_k']},"
            f"{cfg_values['accd_enabled']},{cfg_values['accd_graph_k']},{cfg_values['accd_anchor_ratio']},"
            f"{cfg_values['accd_anchor_memory']},"
            f"{cfg_values['accd_candidate_mass']},{cfg_values['accd_candidate_margin']},"
            f"{cfg_values['accd_stable_cycles']},{cfg_values['accd_resolution_memory']},"
            f"{cfg_values['accd_resolution_target']},{cfg_values['accd_resolution_action']},{path}"
        )


if __name__ == "__main__":
    main()
