#!/usr/bin/env bash
set -euo pipefail

# Fast local debug / test commands for duet_first_cycle_prior_context_transformer.
# Does not need target data, source weights or a GPU.

echo "==> 1) syntax check (py_compile)"
python -m py_compile \
  src/utils/duet_context.py \
  src/methods/oh/duet_first_cycle_prior_context_transformer.py \
  image_target_of_oh_vs.py

echo "==> 2) unit tests (CPU, synthetic tensors)"
python -m pytest tests/test_duet_context_transformer.py -q

echo "==> 3) synthetic end-to-end smoke of the refinement pipeline"
python - <<'PY'
import torch
from src.utils.duet_context import (
    DuetContextConflictTransformer,
    run_context_refinement,
)

torch.manual_seed(0)
n, c, d = 512, 8, 32
feat = torch.randn(n, d)
task_prob = torch.softmax(torch.randn(n, c) + feat[:, :c] * 0.3, dim=1)
clip_prob = torch.softmax(torch.randn(n, c) + feat[:, :c] * 0.2, dim=1)
# 提高可分离性，保证出现高置信 anchor
task_prob = torch.softmax(feat[:, :c] * 4.0, dim=1)
clip_prob = torch.softmax(feat[:, :c] * 3.5 + torch.randn(n, c) * 0.2, dim=1)
gt = task_prob.argmax(dim=1)
transformer = DuetContextConflictTransformer(
    feature_dim=d, num_classes=c, model_dim=32, num_heads=4, ffn_dim=64
)
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
logs = []
result = run_context_refinement(
    task_prob, clip_prob, feat, num_classes=c,
    pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
    labels=gt, sample_indices=torch.arange(n),
    anchors_per_class=8, transformer=transformer, optimizer=optimizer,
    train_steps_per_cycle=20, train_batch_size=32, seed=2020, cycle=2,
    log_fn=logs.append,
)
for line in logs:
    print(line)
assert result["refined_targets"].shape == (n, c)
assert result["resolved_mask"].dtype == torch.bool
print("SMOKE_OK resolved={} weak_rejected={}".format(
    int(result["resolved_mask"].sum().item()),
    int(result["weak_rejected_mask"].sum().item()),
))
PY

echo "==> 4) offline Phase-1 diagnostics (expects an npz dump)"
echo "python tools/run_duet_context_diagnostics.py --npz <dump.npz> --num-classes 12"
