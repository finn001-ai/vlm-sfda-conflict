# 本地缺失产物说明

## 原则

本归档只保存 2026-08-03 归档时仍可从本机文件系统读取的原始产物。不会根据
聊天摘要重新伪造 JSON、CSV、NPZ、gate 或训练日志。

## 已完整保存

- pair-neutralization 的 label-free NPZ、signal lock、oracle CSV、classwise
  oracle CSV、JSON summary 和 Markdown summary；
- 9 份仍存在于 Codex attachments 的原始终端记录；
- `748e716..f661d90` 的完整累计实现补丁、提交清单和变更路径清单；
- 已有历史 archive 中的原始 DUET、CT-DUET、Stage14、proxy loss 和
  structural ablation 证据（通过 README 引用，不重复复制）。

## 已知但归档时不在本机的产物

以下实验的 summary/gate/CSV/NPZ 曾由用户从云端提供，但归档时已不在
Downloads、workspace 或 attachments 中：

- boundary distance 和 boundary router；
- feature-gravity/spatial-causal 的结构化 CSV/NPZ（终端日志仍在）；
- pairwise attributes、attribute mass、attribute reliability 的结构化产物；
- candidate set、candidate-set gradient 的结构化产物；
- cycle-2 conflict memory 和 support-conditioned CLIP memory 的结构化产物；
- agreement rank/weight/shared-runner-up/revocation；
- DVO/TMI、agreement-neighbor CLIP；
- PCGrad output/feature/parameter/compatibility 的结构化产物；
- prototype transport、VSFOT、agreement transport、GMM、temporal mutual rise；
- agreement label impact；
- patch contribution、risk-control、KL suppression、temporal persistence。

这些实验的代码和提交身份仍可从 `code/COMMITS.txt`、累计补丁和 Git 历史
恢复；`EXPERIMENT_INDEX.csv` 只记录已经观察到的判定，不冒充缺失的原始
文件。

## 明确不纳入

- VisDA-C 原始图片和列表：属于数据集本体，不是本轮实验生成物；
- source checkpoint、CLIP 权重和下载缓存：体积大且不是本轮结果证据；
- Python `__pycache__`、临时下载和不完整 output 目录；
- 目标标签派生规则：本轮不存在，也不得在恢复时新增。

如云端仍保留上述缺失目录，应按原目录复制回本归档的 `recovered/` 子目录，
重新生成 `SHA256SUMS`，并在新的 Git 提交中注明来源和恢复日期；不要覆盖当前
已锁定文件。
