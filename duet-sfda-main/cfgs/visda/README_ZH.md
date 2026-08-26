# VisDA-C 当前入口

当前只保留两份正式配置：

- `dcr.yaml`：DCR（DCM + CLM + ARG）。
- `plmatch.yaml`：干净的 PLMatch 对照。

运行完整 VisDA-C DCR：

```bash
bash tools/run_visda_dcr.sh 2020
```

运行相同 8-cycle 预算的纯 PLMatch 对照：

```bash
bash tools/run_visda_plmatch.sh 2020
```

旧 DUET/DAC、proxy、Comparator 和诊断脚本已移至
`../archive/duet_development_code_2026-08-26/`，不再是当前运行入口。
