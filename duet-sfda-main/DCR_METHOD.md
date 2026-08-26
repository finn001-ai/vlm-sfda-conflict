# DCR 方法说明

DCR（Delayed Credit Refinement）是当前唯一保留的正式方法。它分两个训练阶段，内部由 DCM、CLM 和 ARG 三个模块组成。全过程不使用目标域真实标签做训练决策。

## 1. DCM：Delayed Credit Memory

DCM 的作用是给每个样本建立一份独立的历史记录。

对同一个样本，DCM 保存上一轮 Task 和 VLM 的完整类别概率。到下一轮时，用当前 Task/VLM 的平均分布当作新观察，比较上一轮哪个分支更接近它。过去连续预测得更好的分支，在该样本上获得更高权重。

关键点：

- 每个样本只用自己的历史，不用容易样本训一个裁判器去猜困难样本。
- 保存的是完整概率分布，不只是 top-1 标签。
- Task/VLM 当前越一致、与过去越稳定，本次记忆更新越快。
- DCM 阶段同时训练 Task 模型和 prompt。一致样本的硬标签权重较大，冲突样本只保留较小权重；软目标覆盖所有样本。

实现：`src/methods/oh/dcr_memory.py` 和 `src/utils/dcr_credit_memory.py`。

## 2. CLM：Conflict-Locked Memory

CLM 的作用是防止已经积累的历史证据被当前冲突覆盖。

进入第二阶段后：

- Task 和 VLM top-1 一致：允许 DCM 记忆继续更新。
- Task 和 VLM top-1 冲突：锁住该样本之前的记忆，不让当前任一分支直接把它改写。

这个模块不会把冲突样本强行加入硬 CE。正式设置中 `CONFLICT_HARD_FRACTION=0.0`。

实现：`src/utils/dcr_refinement.py` 中的 `memory_write_mode=locked`。

## 3. ARG：Asymmetric Residual Guidance

ARG 的作用是只修正有明确历史支持的冲突，而不是对两个分支做全局加权平均。

对当前仍未解决的 Task/VLM 冲突：

- 如果锁定记忆中“Task 当前候选类”的概率高于“VLM 当前候选类”，就用 DCM 记忆替换这个样本的 KL 软目标。
- 如果历史仍支持 VLM，就保留原 VLM 软目标。
- 已进入硬伪标签集的样本不再做 ARG 替换，避免硬 CE 和软目标互相打架。

所以 ARG 是单向、残差式的修正：只在原 VLM-KL 可能抹掉 Task 历史优势时介入。

实现：`src/utils/dcr_refinement.py` 中的 `soft_replacement_mode=task_supported`，以及 `src/methods/oh/dcr.py` 中的 KL 目标替换。

## 整体流程

```text
Task/VLM 多轮预测
        ↓
DCM：按样本累积历史信用，生成完整分布记忆
        ↓
第二阶段每个 cycle 重新判断 Task/VLM 是否冲突
        ↓
CLM：一致样本可更新，冲突样本锁住历史
        ↓
ARG：只对“历史支持 Task”的未解决冲突替换 KL 软目标
        ↓
更新 Task 模型和 VLM prompt
```

## 当前代码入口

- DCR 第一阶段：`src/methods/oh/dcr_memory.py`
- DCR 第二阶段：`src/methods/oh/dcr.py`
- 纯 PLMatch 对照：`src/methods/oh/plmatch.py`
- Office-Home：`bash tools/run_office_home_dcr_all.sh 2020`
- VisDA-C：`bash tools/run_visda_dcr.sh 2020`
- DomainNet-126：`DATA_DIR=/path/to/data bash tools/run_domainnet126_dcr_all.sh 2020`

旧实验文件只作历史记录，位于 `../archive/duet_development_code_2026-08-26/`，不再属于当前运行入口。
