# Swap 冲突选边方案（DUET-FCP + swap 硬伪标签）

状态：**已实现并接入训练路径，默认关闭。** 规则与归档分析
`archive/sfda_conflict_visda_topk_swap_analysis_2026-08-04` 完全一致
（回归测试逐 D 复现归档 `data/selection_curve.csv`）。推荐配置为
**D=2.0 + 方向门槛 0.8 + 前 6 个 cycle 激活**（见第 3/6 节）。

## 1. 规则是什么

训练时 task 模型和 CLIP 各给出一个 12 类的 softmax 分布。当两个模型"互相指认
对方的第一名"时，样本被称为 **bidirectional_cross_support（纯 swap）冲突**：

```text
A = task top1（同时是 clip top2）
B = clip top1（同时是 task top2）
A ≠ B
```

普通 DUET 对这类样本不给硬标签（task top1 与 clip top1 不一致，进不了
agreement mask）。本方案只对这类样本补充硬伪标签，其他样本完全走原逻辑。

## 2. 选边公式

记 pA/pB 为 task 的 top1/top2 概率，qA/qB 为 clip 的 top2/top1 分数
（都是 softmax 概率，口径为 **prior 校准前**，与 Top-k probe 导出一致）：

```text
eA = pA * qA
eB = pB * qB

cycle 0：直接取 CLIP top1（B），不设门槛。
cycle >= 1：
    log(eA) - log(eB) >= D   -> 标签 = A
    log(eB) - log(eA) >= D   -> 标签 = B
    否则 abstain（不产生伪标签，不进训练损失）
```

对数计算统一加 `eps = 1e-9` 防零（`log(max(x, 1e-9))`），与归档脚本一致。
log 差相等（`log(eA) == log(eB)`）时回退到 B，与归档脚本一致。

**方向门槛（v2，推荐开启）**：除此之外，可以用 `MIN_DIRECTION_ACCURACY`
（默认 0 = 关闭）按"方向"再过滤一层。每个方向 (A=task top1, B=clip top1)
有一份**离线锁定的 cycle-0 CLIP 精度表**（`CYCLE0_DIRECTION_ACCURACY`，
训练前用归档 cycle-0 数据审计一次后固化，不需要当次 GT）。方向精度低于阈值
的样本直接 abstain，所有 cycle（含 cycle 0）都适用。这是为了堵住实测发现的
伤害源：car→truck 方向 CLIP 只有 69% 可靠、car→motorcycle 只有 46%（比
随机差），不加方向门槛时这些错误标签会把车教成卡车/公交，导致 final 的 car
精度掉 ~2pp。

## 3. 超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DUET_SWAP.ENABLED` | `False` | 独立配置开关，必须显式置 True 才启用 |
| `DUET_SWAP.GATE_D` | `4.0` | 决策强度门槛 D（yaml 推荐 2.0，可覆盖） |
| `DUET_SWAP.MIN_DIRECTION_ACCURACY` | `0.0` | 方向门槛，0=关闭；0.8 为推荐值 |
| `DUET_SWAP.LAST_ACTIVE_CYCLE` | `8` | 最后一个激活的 cycle（1-based），之后不再产生新标签；推荐 6 |
| `eps` | `1e-9` | 对数防零，固定，不对外暴露 |

**推荐配置（yaml 已内置）**：`GATE_D=2.0` + `MIN_DIRECTION_ACCURACY=0.8` +
`LAST_ACTIVE_CYCLE=6`。
理由见第 4 节：在归档证据上它比 D=4.0 全集的净正确标签更多，错误标签少 2/3，
且不再向 car/truck 等脆弱方向注入错误。

## 4. 预期指标（TV / VISDA-C / seed 2020，8 cycle 汇总）

cycle 0 走 CLIP、cycle 1–7 走"偏好比 + 门槛"，共 19,398 个 swap 观测：

| D≥ | 决策数 | 覆盖 | 正确 | 错误 | 精度 |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 19,398 | 100.0% | 11,279 | 8,119 | 58.1% |
| 0.5 | 16,062 | 82.8% | 9,651 | 6,411 | 60.1% |
| 1.0 | 13,248 | 68.3% | 8,203 | 5,045 | 61.9% |
| 1.5 | 11,013 | 56.8% | 6,997 | 4,016 | 63.5% |
| 2.0 | 9,196 | 47.4% | 6,013 | 3,183 | 65.4% |
| 2.5 | 7,742 | 39.9% | 5,182 | 2,560 | 66.9% |
| 3.0 | 6,577 | 33.9% | 4,494 | 2,083 | 68.3% |
| 3.5 | 5,633 | 29.0% | 3,921 | 1,712 | 69.6% |
| 4.0 | 4,854 | 25.0% | 3,459 | 1,395 | 71.3% |

cycle 0 单独：直接信 CLIP 的精度 **77.0%**（全流程最高点）。推荐操作点：
追求正确标签总数用 D≈0–0.5；追求标签可信度用 D≥2.0（默认）；最保守 D≥4.0。

**方向门槛变体（cycle-0 锁定方向精度表，离线审计、部署无需 GT）**：

| 配置 | 决策数 | 精度 | 净正确（正确−错误） |
|---|---:|---:|---:|
| D=4.0 全集（现状） | 4,854 | 71.3% | +2,064 |
| D=4.0 + 方向≥0.80 | 2,325 | 81.2% | +1,451 |
| **D=2.0 + 方向≥0.80（推荐）** | **4,220** | **75.1%** | **+2,116** |

推荐组合的决策数比 D=4.0 全集少一点，但错误标签从 3,183 降到 1,052，且
car→truck/car→bus/car→motorcycle 等低质量方向全部 abstain——实测中 car
掉 ~2pp 正是这些方向的错误标签造成的。

**后期停用（LAST_ACTIVE_CYCLE=6）**：cycle 7–8 的 swap 标签精度只有
60–65%，且逐样本追踪显示那 883 次"决策"里真正新增的只有 381 个（其余 502
个是之前 cycle 已给过标签的重复样本），其中 148 个是错的、净正确只有 +85。
后期 abstain 用 85 个净正确标签换掉 148 个错误标签不进入训练，几乎没有
信号损失。逐 cycle 精度：cycle 1–3 ≈76%，cycle 4–6 ≈66–69%，cycle 7–8
62.1%/58.8%；cycle 8 时 label_mask 已覆盖 99.23%，可挖的新样本空间也
极小。

## 5. 实现位置与训练路径接入

- 规则模块（纯函数，检测 + 决策两层）：`src/utils/swap_conflict_selection.py`
  - `swap_evidence`：检测 swap 并提取 pA/pB/qA/qB；
  - `decide_swap_evidence`：决策层，与归档脚本逐行同数学；
  - `select_swap_labels`：整批返回 `(labels, selected)`，abstain = -1；
  - `summarize_swap_decisions`：oracle-diagnostic 汇总（GT 只用于评估）。
- 训练路径：`src/methods/oh/plmatch.py`
  - `train_target(..., swap_conflict_selection=False)`；
  - `obtain_label` 在 prior 校准**前**计算 swap 决策（与 Top-k probe 同一
    概率口径），把 gate 通过的样本并入 `label_mask`，并把 `mem_label` 覆写为
    选定的 A/B；abstain 样本不进 `label_mask` → 不进硬标签 CE 损失；
  - 非 swap 冲突、非冲突样本不进入该规则，原 DUET 策略不变；
  - `src/utils/topk_conflict_probe.py` 未改动，diagnostic 语义不变。
- 方法入口：`src/methods/oh/duet_first_cycle_prior_swap_selection.py`
  （基于原始 DUET + stage-14 first-cycle prior，即 DUET-FCP）。
- 配置：`conf.py` 新增 `DUET_SWAP.ENABLED/GATE_D`（默认关闭）；
  `cfgs/visda/duet_first_cycle_prior_swap_selection.yaml` 显式开启。
- 测试：`tests/test_swap_conflict_selection.py`
  （swap 判定 / A/B 选择 / D 门槛 / abstain / cycle 0 特例 /
  pB、qA≈0 边界 / 回归复现归档曲线）。

## 6. 使用方式

```bash
# 推荐：D=2.0 + 方向门槛 0.8 + 前 6 cycle（runner/yaml 默认即此配置）
bash tools/run_visda_duet_first_cycle_prior_swap_selection.sh

# 显式覆盖：复现原 D=4.0 全集（不开方向过滤、全 8 cycle）
SWAP_GATE_D=4.0 SWAP_DIRECTION_ACCURACY=0.0 SWAP_LAST_CYCLE=8 \
  bash tools/run_visda_duet_first_cycle_prior_swap_selection.sh

# 手动跑（D 通过命令行覆盖）
python image_target_of_oh_vs.py \
  --cfg cfgs/visda/duet_first_cycle_prior_swap_selection.yaml \
  CKPT_DIR . SETTING.OUTPUT_SRC source \
  MODEL.METHOD duet_first_cycle_prior_swap_selection_visda_seed2020 \
  SETTING.SEED 2020 SETTING.S 0 SETTING.T 1 ACTIVE.CYCLE 8 \
  DUET_SWAP.GATE_D 2.0 DUET_SWAP.MIN_DIRECTION_ACCURACY 0.8 \
  DUET_SWAP.LAST_ACTIVE_CYCLE 6
```

每次训练循环的日志会输出
`DUET swap-conflict selection: cycle=...; swap_conflicts=...; decisions=...;
abstain=...; correct=...; precision=...`（correct/precision 用 GT 评估，
仅作诊断，不参与标签生成）。运行脚本拒绝覆盖已有输出目录。

## 7. 已知限制

- **cycle 0 契约偏差未解决**：现有 CSV 的 cycle_000 冲突数 28,223，与仓库契约
  `VISDA_CYCLE0_REGRESSION`（28,255）不一致（差 32）。本实现以实际 CSV 为准，
  未改契约常量；`tests/test_topk_conflict_probe.py` 仍按旧契约校验。
- swap 仅占全部观测的 4.4%，对总精度直接贡献有限；价值在于补齐候选集监督的
  盲区（swap 的 top2 union 只有 2 个类，无法靠候选集恢复）。
- 全部统计为 oracle-diagnostic：GT 只用于评估，不进入标签生成。
- 本训练路径锁定 VISDA-C + CLIP ViT-B/32（与归档验证范围一致）；
  规则纯函数本身不绑定数据集。
- 决策使用 prior 校准前的概率口径；cycle 0 的标签即"校准前 CLIP top1"。
- 方向精度表来自归档 cycle-0 数据（65 个方向），小样本方向统计噪声大但样本
  极少；阈值 0.8 下保留的方向以 motorcycle→bicycle、bus↔train、truck→car 等
  高质量方向为主。
- 方向表与"前 6 cycle 停用"的预期数字均来自归档 CSV 的伪标签层面评估；
  端到端训练效果仍需完整 run 验证。
- 相等 tie 固定回退 B；`log(eA)-log(eB)` 恰为 0 且 D=0 时按 B 处理。
- 该规则只在标签准入处新增样本，不改变 consistency/KL 损失、优化器与
  CLIP 微调目标。
