# DCR-SFDA论文材料

- `manuscript_zh.md`：当前中文证据版稿件。
- `references.bib`：正文引用库。
- `citation_audit.md`：引用核验记录。
- `data/dcr_sfda_ablation_summary_seed2020.csv`：Office-Home四任务交互消融原始汇总。
- `dcr_sfda_ablation_audit_seed2020.md`：消融语义和可支持结论审计。
- `method_implementation_consistency_audit.md`：当前方法与代码路径核对。

当前统一候选配置为：

```text
uniform + locked + task_supported ARG + rank_adaptive
```

它对应两阶段DCR-SFDA：第一阶段维护Task/VLM等权全分布共识记忆，并用一致性和时间稳定性调节写入速度；第二阶段保留F/B/C与记忆，对冲突锁定记忆，仅在历史支持Task时替换未准入冲突的软目标。

当前证据状态：

- Office-Home预设四任务环完成，固定最终平均85.43%，峰值平均85.47%。
- VisDA-C完成，峰值91.97%，固定最终91.94%。
- Office-Home全12任务和DomainNet-126当前配置全量结果待补充。
- Office-Home 85.36%/85.25%、VisDA-C 92.05%/92.03%属于历史延迟信用变体，不是当前统一配置结果。

正式全量入口必须显式传入 `uniform_locked_arg`，因为脚本默认参数仍为历史 `delayed`：

```bash
bash tools/run_office_home_dcr_all.sh 2020 uniform_locked_arg
bash tools/run_visda_dcr.sh 2020 uniform_locked_arg
bash tools/run_domainnet126_dcr_all.sh 2020 uniform_locked_arg
```

正文使用Pandoc/Citeproc引用键，例如 `[@liang2020shot]`。原始日志可以保留开发期文件名，但论文正文和图表统一使用功能名称。
