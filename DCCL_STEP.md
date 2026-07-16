# Step 4: DCCL Training Smoke Test

This step implements the first training version of:

```text
DCCL: Dynamic Conflict Candidate Learning
```

It is intentionally conservative: it copies the DUET/PLMatch training path and
adds conflict candidate learning plus simple promotion/rejection states.

## Files Added

```text
duet-sfda-main/src/methods/oh/dccl.py
duet-sfda-main/cfgs/office-home/dccl.yaml
duet-sfda-main/tools/run_office_home_dccl_smoke.sh
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

Run the three-task smoke test:

```bash
bash tools/run_office_home_dccl_smoke.sh
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
  SETTING.S 0 SETTING.T 1
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
