import unittest

from tools.summarize_reciprocal_boundary_preflight import (
    mechanism_summary,
    parse_office_records,
    summarize_office,
    summarize_visda,
    summarize_visda_full,
)


CONFIG = """
  RECIPROCAL_BOUNDARY: True
  TARGET_HEAD_ADAPT: False
  PAIR_FEATURE_ADAPT: False
  COV_TRANSPORT_ADAPT: False
  GRAPH_TEACHER_FUSION: False
  CALIB_MODE: none
  PL_MEMORY: monotonic
  CAND_PAR: 0.0
  GTR_PAR: 0.0
  BOUNDARY_MARGIN_PAR: 0.1
  BOUNDARY_CONSISTENCY_PAR: {consistency}
  BOUNDARY_KEEP_PAR: {keep}
"""


def mechanism_text(consistency=0.05, keep=0.05, cycles=4):
    lines = [CONFIG.format(consistency=consistency, keep=keep)]
    for cycle in range(1, cycles + 1):
        active = 0 if cycle == 1 else 2
        frozen = cycle >= 2
        lines.append(
            "DCCL reciprocal boundary state: "
            f"cycle={cycle}; conflicts=100; stable_anchors=80; "
            f"eligible_pairs={active}; active_pairs={active}; "
            f"active_conflicts={active * 10}; frozen={frozen}; pairs=[]"
        )
        lines.append(
            "DCCL loss diagnostics: "
            f"cycle={cycle}; boundary_margin_raw={0.3 if active else 0.0}; "
            f"boundary_consistency_raw={0.2 if active else 0.0}; "
            f"boundary_keep_raw={0.1 if active else 0.0}"
        )
        lines.append(
            "DCCL reciprocal boundary action: "
            f"cycle={cycle}; changed_top1={active}; "
            f"mean_probability_l1={0.01 if active else 0.0:.8f}; "
            f"max_probability_l1={0.02 if active else 0.0:.8f}"
        )
    return "\n".join(lines)


def visda_log(accuracy, classes, consistency=0.05, keep=0.05, cycles=4):
    lines = [mechanism_text(consistency, keep, cycles)]
    for cycle in range(1, cycles + 1):
        for interval in range(1, 5):
            lines.append(
                "DCCL reciprocal boundary checkpoint: "
                f"task=TV; cycle={cycle}; iteration={interval}; "
                "boundary_active_pairs=2; boundary_active_conflicts=20; "
                "boundary_coefficient_norm=0.100000; boundary_frozen=True"
            )
            lines.append(
                "Task: TV, Iter:{}/4; Cycle: {}/{}; Accuracy = {:.2f}%; "
                "classifier_loss = 1".format(
                    interval, cycle, cycles, accuracy
                )
            )
            lines.append(" ".join(str(value) for value in classes))
    return "\n".join(lines)


def office_log(task, accuracy, candidate):
    lines = [mechanism_text()] if candidate else []
    for cycle in range(1, 5):
        for interval in range(1, 5):
            if candidate:
                lines.append(
                    "DCCL reciprocal boundary checkpoint: "
                    f"task={task}; cycle={cycle}; iteration={interval}; "
                    "boundary_active_pairs=2; boundary_active_conflicts=20; "
                    "boundary_coefficient_norm=0.100000; boundary_frozen=True"
                )
            lines.append(
                f"Task: {task}, Iter:{interval}/4; Cycle: {cycle}/4; "
                f"Accuracy = {accuracy:.2f}%; classifier_loss = 1"
            )
    return "\n".join(lines)


def host_log(text):
    return (
        CONFIG.format(consistency=0.0, keep=0.0).replace(
            "RECIPROCAL_BOUNDARY: True",
            "RECIPROCAL_BOUNDARY: False",
        )
        + "\n"
        + text
    )


class ReciprocalBoundaryPreflightTest(unittest.TestCase):
    def test_mechanism_requires_head_update_and_active_losses(self):
        summary = mechanism_summary(
            mechanism_text()
            + "\n"
            + (
                "boundary_active_pairs=2; boundary_active_conflicts=20; "
                "boundary_coefficient_norm=0.100000; boundary_frozen=True"
            ),
            require_consistency=True,
            require_keep=True,
        )
        self.assertTrue(summary["valid"])

    def test_visda_gate_uses_final_hard_classes_and_mechanism(self):
        control_classes = [80.0] * 12
        full_classes = [80.0] * 12
        for index in (3, 7, 11):
            full_classes[index] = 81.0
        variants = {
            "margin_only": visda_log(
                80.1, full_classes, consistency=0.0, keep=0.0
            ),
            "margin_consistency": visda_log(
                80.2, full_classes, consistency=0.05, keep=0.0
            ),
            "full": visda_log(80.5, full_classes),
        }
        summary = summarize_visda(
            visda_log(80.0, control_classes),
            host_log(visda_log(80.0, control_classes)),
            variants,
            max_host_gap=0.1,
            min_final_improvement=0.2,
            min_hard_improvement=0.2,
            max_other_regression=0.1,
            min_hard_class_delta=0.0,
        )

        self.assertEqual(summary["decision"], "pass_visda_proxy_gate")
        self.assertTrue(summary["variants"]["full"]["mechanism"]["valid"])

    def test_office_gate_uses_matched_final_checkpoints(self):
        controls = {
            task: (None, office_log(task, 80.0, candidate=False))
            for task in ("AC", "PC", "RC")
        }
        hosts = {
            task: (None, host_log(office_log(task, 80.0, candidate=False)))
            for task in ("AC", "PC", "RC")
        }
        candidates = {
            "AC": (None, office_log("AC", 80.5, candidate=True)),
            "PC": (None, office_log("PC", 80.4, candidate=True)),
            "RC": (None, office_log("RC", 79.8, candidate=True)),
        }
        summary = summarize_office(
            controls,
            hosts,
            candidates,
            min_mean_improvement=0.2,
            min_task_delta=-0.3,
            min_task_wins=2,
        )

        self.assertEqual(summary["decision"], "pass_office_home_preflight_gate")
        self.assertEqual(len(parse_office_records(candidates["AC"][1])), 16)

    def test_visda_full_gate_requires_eight_cycles(self):
        control_classes = [80.0] * 12
        candidate_classes = [80.0] * 12
        for index in (3, 7, 11):
            candidate_classes[index] = 81.0
        summary = summarize_visda_full(
            visda_log(80.0, control_classes, cycles=8),
            visda_log(80.5, candidate_classes, cycles=8),
            min_final_improvement=0.2,
            min_hard_improvement=0.2,
            max_other_regression=0.1,
            min_hard_class_delta=0.0,
        )

        self.assertEqual(summary["decision"], "pass_visda_full_seed2020_gate")


if __name__ == "__main__":
    unittest.main()
