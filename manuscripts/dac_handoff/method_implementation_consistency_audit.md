# 方法描述与当前实现一致性审计

审计日期：2026-08-26

## 结论

当前中文稿第3节已经与Office-Home和VisDA-C的正式执行路径对齐。论文主方法固定为“DAC状态保留残差精炼”，旧的状态重置5×4流程只作为历史对照，不再作为主方法。两个数据集使用相同方法逻辑，但采用各自固定并公开的第二阶段训练长度。

## 逐项核对

| 核对项 | 当前实现 | 论文描述 | 状态 |
|---|---|---|---|
| 第一阶段长度 | DAC 15 epochs | DAC 15 epochs | 一致 |
| 初始记忆 | Task与VLM完整概率的算术平均 | 相同 | 一致 |
| 延迟信用 | 用当前联合观测回看上一轮Task/VLM分布 | 相同 | 一致 |
| DAC状态 | memory、previous task/clip、两路折扣损失、feedback mass、两路权重 | 相同 | 一致 |
| 交接内容 | F/B/C及完整DAC状态 | 相同 | 一致 |
| 不交接内容 | 第一阶段prompt与优化器状态 | 相同 | 一致 |
| 第二阶段任务模型 | 冻结C，更新F/B | 相同 | 一致 |
| 第二阶段VLM | 从预训练权重重新建立；文本侧冻结，视觉侧更新 | 相同 | 一致 |
| Office-Home第二阶段长度 | 4 cycles × 4 epochs | 相同 | 一致 |
| Office-Home总任务模型遍历 | 15+16=31 passes | 相同 | 一致 |
| VisDA-C第二阶段长度 | 8 cycles × 4 epochs | 相同 | 一致 |
| VisDA-C总任务模型遍历 | 15+32=47 passes | 相同 | 一致 |
| 冲突硬准入 | 0 | 不给冲突新增硬标签 | 一致 |
| 冲突记忆 | 只冻结memory写入，其余信用历史继续更新 | 相同 | 一致 |
| 一致样本准入 | 跨周期累积 | 相同 | 一致 |
| 残差软替换 | 只处理未准入冲突，且DAC记忆更支持Task当前候选 | 相同 | 一致 |
| Task损失权重 | consistency 0.2、hard CE 0.4、KL 0.4 | 相同 | 一致 |
| 周期内更新顺序 | 全量扫描并固定监督信号；VLM视觉更新1遍；F/B训练4 epochs | 相同 | 一致 |
| VLM更新目标 | 使用当前Task/VLM算术融合，不使用残差教师 | 相同 | 一致 |
| Office-Home汇报 | 12任务统一取固定16点轨迹中的best acc；同步保留final acc | 相同 | 一致 |
| VisDA-C汇报 | 取固定32点轨迹中的best宏平均；同步保留final宏平均 | 相同 | 一致 |

## 需要特别披露的边界

1. Office-Home日志只保存固定最终F/B/C，没有保存每个任务的best检查点。85.36%是从16个评估点中事后提取的峰值平均，不是可直接部署的无标签检查点选择规则；固定最终平均为85.25%。
2. VisDA-C正式脚本、论文和测试统一为8 cycles × 4 epochs、冲突硬准入0、task-supported残差、累积一致准入和VLM视觉更新。2026-08-26运行共完成47 passes，峰值为92.05%，固定最终为92.03%；前4周期轨迹仅作为训练动态记录。完整审计见`visda_credit_residual_log_audit_20260826.md`。
3. DomainNet-126现有入口只支持普通plmatch路径，尚没有与Office-Home主方法完全一致的DAC状态生成及残差精炼运行链路。它不只是“结果没跑”，还需要先补齐正式配置和运行脚本。
4. 当前结果支持Office-Home和VisDA-C完整方案在seed 2020下的准确率优势。由于VisDA-C本文方法为47 passes、本地单阶段基线为32 passes，该比较不支持同算力或效率优势；多种子和核心机制消融完成前，也不支持统计显著或跨数据集普遍有效。

## 当前正式入口

- Office-Home：`tools/run_office_home_dac_credit_residual_refinement.sh`
- VisDA-C：`tools/run_visda_dac_credit_residual_refinement_full.sh`
- DomainNet-126：[待补充：当前方法正式入口]
