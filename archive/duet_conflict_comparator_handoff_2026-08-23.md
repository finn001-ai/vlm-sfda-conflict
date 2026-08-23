# DUET Task–CLIP Conflict 路线存档

更新时间：2026-08-23  
状态：暂停研究，保留代码和证据，未来可继续  
原则：目标域 GT 只允许用于离线评估，不能决定训练、阈值或样本选择。

## 1. 最初目标

DUET 中 Task 与 CLIP 的 Top-1 不一致时，构成真实冲突：

```text
A = Task Top-1
B = CLIP Top-1
A != B
```

我们的目标是利用概率、置信度、特征锚点、增强稳定性和历史轨迹，判断 A/B 中谁更可信，使冲突处理优于 DUET 原本的 fallback，并最终提高目标域 ACC。

论文级成功标准一直是：

- 在同一批真实冲突上，方法优于实际 DUET fallback，而不只是优于 Task 或 CLIP。
- 有效覆盖至少达到约 40%–60% 的冲突，而不是只解决最容易的前 20%。
- `coverage × accuracy gain` 足够大，能传导为最终 ACC 提升。
- 多个 seed 或任务稳定提高，理想幅度约 0.5–1.0 个百分点。
- 全过程不使用目标 GT 选择阈值、模型或 checkpoint。

## 2. 已实现的主要版本

### Pairwise Comparator

- 用高置信 agreement 样本建立按类别平衡的 anchor bank。
- 通过 strong augmentation 下 Task/CLIP 单侧翻转生成 synthetic conflicts。
- 输入包括候选 A/B 概率、entropy、margin、anchor similarity 等静态证据。
- 做过 synthetic/real distribution matching、方向平衡、persistent comparator 和 replay。
- 做过 hard admission、soft target、residual、固定覆盖率、绝对阈值等干预方式。

### 真实冲突历史监督

- 保存跨 cycle 的冲突候选和预测轨迹。
- 使用 exact stable pair、single-candidate overlap、multi-view/temporal stability 产生 GT-free 伪监督。
- synthetic supervision 用于预训练，历史成熟冲突用于真实冲突微调。
- 后期把目标改为“DUET fallback 与 challenger 二选一”，直接优化是否应推翻 fallback。

### 其他诊断与尝试

- same-subset evaluation：
  - `resolved_subset_task_acc`
  - `resolved_subset_clip_acc`
  - `resolved_comparator_acc`
  - `resolved_candidate_oracle_acc`
  - `conditional_arbitration_acc`
- 固定冲突 cohort 的训练轨迹与 fixed-coverage trajectory。
- synthetic validation early stopping 与 best-checkpoint restore。
- GT-supervised 16D/20D feature probe，仅作离线诊断。
- agreement/shared Top-2 ambiguity diagnostic，仅作离线诊断。
- 自适应增加 anchors、不同 K、邻域/原型/时间特征、reliability weighting 等。

## 3. 关键实验结果

### VisDA 基准

```text
官方全量 DUET final ACC：91.50
官方全量 DUET process peak：91.52
proxy25 matched DUET control：87.93
```

注意：proxy25 的 13,847 个 adaptation 样本由 GT 做类别分层抽样，但评估仍在完整 55,388 个样本上。因此 proxy25 只能用于快速内部筛选，不能作为论文正式结果，也不能与全量训练结果混为一谈。

### 早期 DCCL / Stage14 全量结果

```text
final ACC：91.04
process peak：91.07
```

Cycle 4 的典型现象：

```text
selected mixed-label precision：93.46（DUET：90.42）
coverage：85.57（DUET：96.36）
global mixed accuracy：87.98（DUET：88.94）
```

结论：局部标签精度提高，但丢失的覆盖更多，最终低于 DUET。

### Office-Home A→C Pairwise Comparator

固定真实冲突 cohort 上，Comparator 明显学到了一部分选择能力：

| Cycle | Task | CLIP | Comparator peak | Comparator@200 |
|---|---:|---:|---:|---:|
| C2 | 23.66 | 32.09 | 36.19 | 35.46 |
| C3 | 27.94 | 27.58 | 35.13 | 35.01 |
| C4 | 30.66 | 26.96 | 35.90 | 35.59 |

固定 rank-20% 子集：

| Cycle | Task | CLIP | Comparator | Candidate oracle | Conditional arbitration |
|---|---:|---:|---:|---:|---:|
| C2 | 15.02 | 46.89 | 51.28 | 61.90 | 82.84 |
| C3 | 22.42 | 31.52 | 40.61 | 53.94 | 75.28 |
| C4 | 30.95 | 30.95 | 46.03 | 61.90 | 74.36 |

这证明 Comparator 在筛出的同一子集上可以优于 Task 和 CLIP。但最终 ACC 只提高约 0.19 个百分点，且不够稳定，不能构成论文结论。

### 最新 VisDA proxy25：transition-prior residual

对应日志：`/Users/stranger/Downloads/duet_first_cycle_prior_context_transformer_260822_142230.txt`

```text
adaptation samples：13,847
Cycle 2 strict conflicts：2,375
历史 matured conflicts：3,087 / 6,975 = 44.26%
历史伪标签离线 precision：94.62%

选中冲突：1,425 = 当前冲突的 60%
同子集 DUET fallback acc：55.30%
同子集 gated comparator acc：57.19%
同子集 gain：+1.89 pp

实际发生 switch：481
switch precision：44.07%
beneficial switches：212
harmful switches：185
net corrections：+27

Task teacher samples：337
reliability effective sample equivalent：98.31
最终 Task ACC：85.54%
```

最关键的事实：表面覆盖了 60% 冲突，但真正发生改变的只有 481 个样本；折算到全部 adaptation 数据只占 3.47%。带可靠性权重后的有效监督量约 98 个样本，只占 0.71%。同子集增益因此无法传导成明显最终 ACC。

前一版日志 `/Users/stranger/Downloads/duet_first_cycle_prior_context_transformer_260822_135400.txt` 也出现相同问题：60%–80% 的选择覆盖最后只产生很少净修正。

## 4. 已经确认的结论

1. 冲突中确实存在互补信息，不是完全没有研究价值。
2. Candidate oracle 与 same-subset 结果说明，理论上存在比单独 Task、CLIP 和 DUET fallback 更好的选择空间。
3. Comparator 能在容易、可分的冲突子集上学到信号；历史成熟标签也可以有很高离线精度。
4. 目前所有版本都没有把局部优势稳定转化为最终 ACC 优势。
5. 主要瓶颈不是简单的 K 太小、step 不够或 gate 阈值不合适，而是“有效干预规模”过小以及监督与当前困难冲突错位。

## 5. 当前失败的本质原因

- **Synthetic → real mismatch**：增强翻转产生的是容易样本，真实冲突更靠近决策边界。
- **历史监督 survivor bias**：能稳定成熟的冲突通常本来就更容易；当前最难冲突没有可靠标签。
- **选择覆盖不等于有效覆盖**：60% 被选中，不代表 60% 真正改变标签或获得足够梯度。
- **离散 A/B 决策丢失信息**：完整 K 类分布被压成二选一，忽略其他类别及概率结构。
- **局部正确不等于训练收益**：少量 conflict 修正会被大量 agreement 标签、CE/KL 和后续共同适应稀释。
- **置信度膨胀**：训练越久 margin 越大，但真实仲裁未同步改善；synthetic val loss 也不能可靠代表 real-conflict quality。
- **顺序更新与教师漂移**：Task/CLIP 在不同阶段更新后，先前学到的 comparator 证据会过时或高度相关。

## 6. 已尝试但不值得原样重复的方向

- 单独增大 anchor K 或扫 K。
- 只把 fixed coverage 从 20% 调到 40%/60%/80%。
- 继续扫绝对 margin threshold。
- 单纯增加 Comparator steps，或依赖 synthetic validation early stopping。
- hard admission、soft-only、residual、只在 C2 使用等简单切换。
- synthetic distribution matching、same-view、memory replay。
- 只增加同类静态概率、entropy、margin、anchor similarity。
- 简单 confidence、neighbor、prototype、graph、PCGrad、attribute/patch selector。
- 直接把 agreement shared Top-2 写回 pseudo-label。

这些尝试不是完全无效，而是已有证据表明它们单独不足以产生论文级最终 ACC 增益。

## 7. 尚未被否定、以后可继续的入口

若重新启动这条路线，优先考虑改变问题形式，而不是继续调现有 gate：

1. 使用完整 K 类分布和连续共识，不再只做静态 A/B 裁判。
2. 把历史轨迹作为每个样本的连续锚点/先验，而不是只给少量 matured conflict 一个离散标签。
3. Task 与 CLIP 基于同一份 detached consensus 同步更新，减少教师漂移和顺序偏差。
4. 让所有 conflict 获得可控的训练信号，同时保留 DUET 高质量 agreement 标签；重点监控真正改变梯度的 effective coverage。
5. 任何新方法都必须在同一子集比较：Comparator、新标签、实际 DUET fallback、candidate oracle。

重新立项的最低验收线：

```text
有效冲突覆盖 >= 40%–60%
同子集方法明显优于实际 DUET fallback
coverage × gain 足以预测最终收益
最终 ACC 多 seed/任务稳定 +0.5～1.0 pp
无目标 GT 参与训练决策
```

## 8. 代码与复现实验入口

当前仓库：`/Users/stranger/Documents/领域迁移/duet-sfda-main`

暂停时版本：

```text
branch：main
commit：b6b199ab4462b538c60c7a4d84dd75f12f00c078
```

主要文件：

- `src/methods/oh/duet_first_cycle_prior_context_transformer.py`
- `src/utils/duet_context.py`
- `cfgs/visda/duet_first_cycle_prior_context_transformer.yaml`
- `cfgs/office-home/duet_first_cycle_prior_context_transformer.yaml`

文件校验值：

```text
5ece67171b8500014ebf6de8ffc8b2db078130130ff358486772af8931e75774  src/methods/oh/duet_first_cycle_prior_context_transformer.py
b2b8b3632f69232ae92b7a3c1025d7998b4e342bbb18b110e43076576a36a412  src/utils/duet_context.py
deddca7200b58775ddd5714717b5c35e539256b9641bd98d5a79ab0279cc0a5b  cfgs/visda/duet_first_cycle_prior_context_transformer.yaml
3f3cefb998082e2e5bcfc37ae51c1b42a7ea0ee43beacbc9ec8e5dc34f93d46a  cfgs/office-home/duet_first_cycle_prior_context_transformer.yaml
```

高级实验主要由 `tools/run_visda_*comparator*_proxy25.sh`、`tools/run_visda_*temporal*_proxy25.sh` 和 `tools/run_visda_*transition*_proxy25.sh` 覆盖配置。默认 YAML 本身不等于最新日志的配置，复现时必须同时检查对应运行脚本。

Cycle 1 checkpoint 复用机制已经存在，历史实验使用过：

```text
output/checkpoints/duet_fcp_context_visda_proxy25_seed2020_cycle1.pt
```

## 9. 相关旧档案索引

- `archive/SFDA_CONFLICT_CURRENT_HANDOFF.md`
- `archive/sfda_duet_context_transformer_runs_2026-08-08.md`
- `archive/sfda_conflict_visda_audit_campaign_2026-08-03/README_CN.md`
- `archive/sfda_conflict_visda_full_duet_control_2026-07-24/README.md`
- `archive/sfda_conflict_visda_proxy_loss_audit_2026-07-23/README.md`
- `duet-sfda-main/docs/experiment_logs/SFDA-B-260812-001.md`
- `archive/sfda_conflict_stage1_2026-07-15` 至 `archive/sfda_conflict_stage24_boundary_flip_duet_2026-07-28`

## 10. 一句话交接

这条路线不是被证明“冲突无法解决”，而是已经证明：现有 Comparator 能局部选得更准，却因为 synthetic/历史监督偏向容易样本、有效梯度覆盖极小和 A/B 离散形式的信息损失，无法稳定提升最终 ACC。未来如果重启，应从“全量、连续、同步的共识学习”或新的独立证据入手，不应继续原样调 K、gate 和训练步数。
