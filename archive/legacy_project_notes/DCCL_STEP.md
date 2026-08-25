# Step 4: DCCL Training Smoke Test

This step implements the first training version of:

```text
DCCL: Dynamic Conflict Candidate Learning
```

It is intentionally conservative: it copies the DUET/PLMatch training path and
adds conflict candidate learning plus simple promotion/rejection states.

## 2026-07-17 Update

Early cloud results show that aggressive promotion is not reliable:

| Task | DCCL default | Conservative DCCL |
|---|---:|---:|
| A->C | 71.34 | 71.87 |
| A->P | 90.61 | 90.67 |
| A->R | 90.36 | 90.87 |

Default DCCL uses:

```text
CAND_PAR=0.05
PROMOTE_K=2
```

Conservative DCCL uses:

```text
CAND_PAR=0.01
PROMOTE_K=999
```

The current recommended path is therefore:

```text
conservative candidate-set learning first,
promotion only as an ablation or future module.
```

## Files Added

```text
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/cfgs/office-home/dccl.yaml
duet-sfda-main/tools/run_office_home_dccl_smoke.sh
duet-sfda-main/tools/run_office_home_dccl_conservative_smoke.sh
duet-sfda-main/tools/run_office_home_dccl_ac_sweep.sh
duet-sfda-main/tools/run_office_home_dccl_conservative_all.sh
duet-sfda-main/tools/run_office_home_plmatch_all.sh
```

The method is registered in:

```text
duet-sfda-main/image_target_of_oh_vs.py
duet-sfda-main/conf.py
```

## Cloud Commands

```bash
cd /openbayes/home/vlm-sfda-conflict
git pull
cd duet-sfda-main
```

Run the conservative three-task smoke test:

```bash
bash tools/run_office_home_dccl_conservative_smoke.sh
```

This runs:

```text
Art -> Clipart
Art -> Product
Art -> RealWorld
```

## Single Task Command

```bash
python image_target_of_oh_vs.py \
  --cfg cfgs/office-home/dccl.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  SETTING.S 0 SETTING.T 1 \
  DCCL.CAND_PAR 0.01 \
  DCCL.PROMOTE_K 999
```

Change `SETTING.T` to:

```text
1 = Clipart
2 = Product
3 = RealWorld
```

## What DCCL-lite Does

Agreement samples:

```text
source_pred == clip_pred
use hard pseudo-label
```

Conflict samples:

```text
source_pred != clip_pred
use candidate-set loss:
L_candidate = -log(p(source_pred) + p(clip_pred))
```

Dynamic state:

```text
candidate_mass = p(source_pred) + p(clip_pred)
candidate_gap = abs(p(source_pred) - p(clip_pred))

if mass >= TAU_HIGH and gap >= GAP_PROMOTE for PROMOTE_K cycles:
    promote to hard pseudo-label

if mass < TAU_LOW:
    reject for current cycle
```

Default hyperparameters:

```text
CAND_PAR = 0.05
TAU_LOW = 0.4
TAU_HIGH = 0.7
GAP_PROMOTE = 0.3
PROMOTE_K = 2
```

Additional candidate controls:

```text
CAND_TAU
only apply candidate loss if p(source_pred) + p(clip_pred) >= CAND_TAU

CAND_WEIGHT
none: all selected candidates have equal weight
mass: weight by p(source_pred) + p(clip_pred)
ramp: linearly ramp weight above CAND_TAU
```

## A->C Sweep Before Full Runs

```bash
bash tools/run_office_home_dccl_ac_sweep.sh
python tools/extract_final_accuracy.py --glob 'output/uda/office-home/*/*/*.txt' \
  > output/uda/office-home/stage3_accuracy.csv
```

Then choose the best A->C setting and run all 12 tasks:

```bash
bash tools/run_office_home_dccl_conservative_all.sh CAND_PAR CAND_TAU CAND_WEIGHT TAU_LOW
```

## First Decision Rule

Compare against DUET/PLMatch logs for the same tasks.

If at least two of:

```text
A->C
A->P
A->R
```

improve over PLMatch, expand DCCL to all 12 Office-Home tasks.

If not, do not keep adding complexity blindly. First test whether static
candidate loss or rejection is the failure point.
