# DAC交接框架论文材料

- `manuscript_zh.md`：中文完整审阅稿；
- `references.bib`：正文引用库，优先采用出版社、会议论文集、PMLR、CVF和DOI元数据；
- `citation_audit.md`：引用核验范围和仍需人工复查的项目。

正文使用 Pandoc/Citeproc 形式的引用键，例如 `[@liang2020shot]`。英文投稿时可继续沿用同一份 BibTeX，再按目标期刊模板转换样式。

当前方法在主文中统一写作“本文两阶段方法”或“DAC交接框架”。DUET只作为相关工作、直接基线及第二阶段思想来源出现；`plmatch.py`仅在复现说明中作为代码文件名出现。

## Office-Home固定协议

- 本文完整方法：DAC 15 epochs + 周期式精炼5 cycles × 4 epochs，共35次任务模型目标集遍历；
- 12个迁移任务使用同一设置，作者已确认；正式脚本会自动检查`Cycle 5/5`和`handoff_target_passes=20`；
- 等passes单阶段对照：从同一源F/B/C直接运行7 cycles × 5 epochs，共35 passes；
- 单阶段对照脚本：`duet-sfda-main/tools/run_office_home_single_stage_refinement35_all.sh`；
- 所有主结果均使用预先确定的最终检查点，不使用目标标签选择中间峰值。

原始CSV与日志可以保留内部实现名称；论文正文、图表和对外结果表统一使用功能名称。
