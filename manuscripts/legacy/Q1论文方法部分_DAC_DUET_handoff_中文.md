# 3 方法

## 3.1 问题定义

设源域数据不可访问，但可以获得在源域上训练完成的任务模型

\[
f_{\theta}=g_{\theta}\circ b_{\theta}\circ h_{\theta},
\]

其中，\(h_{\theta}\)、\(b_{\theta}\) 和 \(g_{\theta}\) 分别表示特征提取器、瓶颈层和分类器。目标域仅提供无标签数据集

\[
\mathcal D_t=\{x_i\}_{i=1}^{N},
\]

其类别空间包含 \(C\) 个类别，并与源域类别空间一致。除任务模型外，我们还使用一个预训练视觉—语言模型（vision–language model, VLM）作为第二个预测视角。任务模型和 VLM 对目标样本 \(x_i\) 输出完整类别概率分布

\[
\mathbf p_{i}^{T}=f_{\theta}(x_i),\qquad
\mathbf p_{i}^{V}=f_{\phi}(x_i),
\]

其中，上标 \(T\) 和 \(V\) 分别表示任务模型与 VLM。本文的目标是在不访问源域样本、不使用目标域标签进行训练或模型选择的条件下，得到适配后的目标任务模型 \(f_{\theta^*}\)。VLM及其适配状态仅在训练阶段提供辅助监督，最终推理只保留任务模型。

## 3.2 DAC→DUET总体流程

直接在整个训练过程中采用同一种双视角融合规则存在两个问题。首先，训练初期的任务模型和 VLM 经常产生冲突，若立即依赖当前置信度选边，容易把瞬时偏差写入伪标签。其次，随着两个分支共同适配，其预测会逐渐趋同，后期仅依赖二者当前差异难以继续提供有效的互补信息。基于这一观察，本文将目标域适配划分为两个功能不同的阶段。

第一阶段采用延迟一致性信用分配（Delayed Agreement Credit, DAC）。DAC为每个目标样本维护独立的跨轮预测历史，依据任务模型与VLM在上一轮的预测对下一轮联合观测的解释能力，为两个分支分配连续权重，并据此更新全类别分布记忆。该阶段不学习跨样本路由函数，不设置冲突筛选阈值，也不丢弃任何目标样本。

第二阶段将DAC完成适配的任务模型完整交接给原版DUET。具体而言，特征提取器、瓶颈层和分类器的最终参数全部作为DUET的初始化，而DAC的分布记忆、历史信用和提示参数均不传递。随后，DUET重新建立任务模型与VLM之间的周期式伪标签更新，使DAC获得的目标域初始化继续受益于双视角互补、增广一致性和VLM软监督。[待补充引用：DUET原论文及其官方实现]

> **[待补充图1：DAC→DUET总体框架图]**  
> 左侧展示源模型与预训练VLM；中间上方展示DAC的“当前双视角预测—延迟信用—全分布记忆—目标模型更新”；中间下方明确标出只交接任务模型的特征提取器、瓶颈层和分类器，不交接DAC memory或prompt；右侧展示原版DUET的周期式预测刷新、累积一致伪标签、强弱增广一致性和VLM软监督。图中应标注“target labels are used for evaluation only”。

## 3.3 第一阶段：延迟一致性信用分配

### 3.3.1 全分布状态初始化

DAC首先在完整目标集上扫描任务模型和VLM，得到第0轮概率分布 \(\mathbf p_i^{T,0}\) 和 \(\mathbf p_i^{V,0}\)。由于此时尚不存在可供回看的未来观测，两个分支获得相同的初始权重：

\[
w_i^{T,0}=w_i^{V,0}=\frac{1}{2}.
\]

每个样本的全分布记忆初始化为两路预测的算术平均：

\[
\mathbf m_i^0=\frac{1}{2}
\left(\mathbf p_i^{T,0}+\mathbf p_i^{V,0}\right).
\]

这里的 \(\mathbf m_i^0\in\mathbb R^C\) 保留全部类别概率，而不是只保存Top-1伪标签。因此，即使两个分支的首选类别不同，次优类别及其相对概率仍然能够进入后续监督。

### 3.3.2 延迟反馈可靠性

在第 \(e\) 个DAC轮次，当前任务模型和VLM产生 \(\mathbf p_i^{T,e}\) 与 \(\mathbf p_i^{V,e}\)。我们将二者的算术平均视为当前联合观测：

\[
\mathbf o_i^e=\frac{1}{2}
\left(\mathbf p_i^{T,e}+\mathbf p_i^{V,e}\right).
\]

为了统一衡量两个概率分布之间的差异，本文使用归一化Jensen–Shannon散度：

\[
d_{\mathrm{JS}}(\mathbf p,\mathbf q)
=\frac{
D_{\mathrm{KL}}(\mathbf p\|\mathbf r)
+D_{\mathrm{KL}}(\mathbf q\|\mathbf r)
}{2\log 2},
\qquad
\mathbf r=\frac{\mathbf p+\mathbf q}{2},
\]

其取值位于 \([0,1]\)。[待补充引用：Jensen–Shannon divergence]

当前联合观测的可靠程度由双视角一致性和跨轮稳定性共同决定：

\[
a_i^e=1-d_{\mathrm{JS}}
\left(\mathbf p_i^{T,e},\mathbf p_i^{V,e}\right),
\]

\[
s_i^e=1-d_{\mathrm{JS}}
\left(\mathbf o_i^e,\mathbf m_i^{e-1}\right),
\]

\[
\varphi_i^e=a_i^e s_i^e.
\]

其中，\(a_i^e\) 衡量两个分支当前是否给出相近的完整类别分布，\(s_i^e\) 衡量当前联合观测是否与历史记忆保持一致。只有当两者同时较高时，当前观测才会产生较大的信用反馈；但 \(\varphi_i^e\) 仅连续调节记账强度，不作为保留或删除样本的硬门槛。

### 3.3.3 基于下一轮观测的专家记账

当 \(e\geq1\) 时，DAC不根据同一时刻的softmax置信度直接决定应信任哪一侧，而是回看上一轮预测对当前联合观测的预测能力。任务模型和VLM的延迟损失分别定义为

\[
\ell_{i,T}^{e}=d_{\mathrm{JS}}
\left(\mathbf p_i^{T,e-1},\mathbf o_i^e\right),
\qquad
\ell_{i,V}^{e}=d_{\mathrm{JS}}
\left(\mathbf p_i^{V,e-1},\mathbf o_i^e\right).
\]

随后，以折扣因子 \(\gamma\) 累积每个样本自身的历史损失：

\[
S_{i,k}^{e}=\gamma S_{i,k}^{e-1}
+\varphi_i^e\ell_{i,k}^{e},
\qquad k\in\{T,V\},
\]

\[
Z_i^e=\gamma Z_i^{e-1}+\varphi_i^e,
\qquad
\bar\ell_{i,k}^{e}=\frac{S_{i,k}^{e}}{Z_i^e+\epsilon}.
\]

两个分支的连续信用权重由历史平均损失的负指数归一化得到：

\[
w_{i,k}^{e}=
\frac{\exp(-\eta\bar\ell_{i,k}^{e})}
{\sum_{j\in\{T,V\}}\exp(-\eta\bar\ell_{i,j}^{e})}.
\]

历史上更接近后续联合观测的分支具有更小的 \(\bar\ell\)，因而获得更高权重。该记账过程严格在同一样本内部进行：样本 \(x_i\) 的历史只决定 \(x_i\) 的融合权重，不被用于训练一个需要泛化到其他样本的路由器。

### 3.3.4 全覆盖分布记忆更新

根据当前信用权重，构造即时融合分布

\[
\mathbf u_i^e=w_{i,T}^{e}\mathbf p_i^{T,e}
+w_{i,V}^{e}\mathbf p_i^{V,e}.
\]

为避免记忆在不稳定轮次中发生剧烈变化，使用反馈可靠性连续调节更新率：

\[
r_i^e=\mu\left(\frac{1+\varphi_i^e}{2}\right),
\]

\[
\mathbf m_i^e=
(1-r_i^e)\mathbf m_i^{e-1}
+r_i^e\mathbf u_i^e.
\]

更新后对 \(\mathbf m_i^e\) 重新归一化，使其保持为合法概率分布。由于 \(r_i^e\geq\mu/2\)，所有目标样本的记忆都会被更新；稳定一致样本更新得更快，不稳定冲突样本更新得更保守。该设计实现了100%的软监督覆盖，同时避免使用置信度阈值、固定锚点或拒绝机制。

当前实现设置折扣因子 \(\gamma=0.9\)、信用敏感度 \(\eta=4.0\)、基础记忆更新率 \(\mu=0.5\) 以及数值稳定项 \(\epsilon=10^{-5}\)。

## 3.4 DAC目标域优化

DAC使用分布记忆 \(\mathbf m_i^e\) 同时监督任务模型和VLM提示分支。VLM的视觉编码器与文本编码器在该阶段保持冻结，仅更新共享提示上下文；任务模型的特征提取器、瓶颈层和分类器共同更新。

### 3.4.1 一致样本与冲突样本的硬监督

为保留DUET中高精度的一致伪标签，DAC在训练初期对两路预测分别进行类别先验校准。对任一预测矩阵 \(\mathbf P\in\mathbb R^{N\times C}\)，经验类别先验和校准概率为

\[
\pi_c=\frac{1}{N}\sum_{i=1}^{N}P_{i,c},
\]

\[
\widetilde P_{i,c}=
\frac{P_{i,c}/(\pi_c+\epsilon)^{\alpha_p}}
{\sum_{j=1}^{C}P_{i,j}/(\pi_j+\epsilon)^{\alpha_p}}.
\]

需要强调的是，该校准只用于构造DAC阶段的一致硬伪标签，不参与延迟信用和分布记忆更新。校准阶段结束后直接令 \(\widetilde{\mathbf p}=\mathbf p\)。记 \(\widetilde{\mathbf p}_i^{T,e}\) 与 \(\widetilde{\mathbf p}_i^{V,e}\) 为用于硬标签判断的概率，并定义

\[
A_i^e=\mathbb I\left[
\arg\max_c\widetilde p_{i,c}^{T,e}
=\arg\max_c\widetilde p_{i,c}^{V,e}
\right].
\]

硬伪标签为

\[
\widehat y_i^e=
\begin{cases}
\displaystyle
\arg\max_c
\left(\widetilde p_{i,c}^{T,e}
+\widetilde p_{i,c}^{V,e}\right), & A_i^e=1,\\[6pt]
\displaystyle
\arg\max_c m_{i,c}^e, & A_i^e=0.
\end{cases}
\]

因此，一致样本保留原版DUET的共同预测，冲突样本则由DAC全分布记忆提供硬标签。两类样本均参与训练，但使用不同权重：

\[
\beta_i=
\begin{cases}
\beta_A,&A_i^e=1,\\
\beta_C,&A_i^e=0.
\end{cases}
\]

当前Office-Home设置在前4个DAC轮次启用先验校准，\(\alpha_p=0.8\)、\(\beta_A=0.4\)、\(\beta_C=0.05\)。较小的 \(\beta_C\) 用于降低无标签冲突硬决策可能带来的错误放大。

### 3.4.2 全分布对齐与总目标

除硬伪标签外，DAC还使用完整记忆分布进行互信息对齐。记 \(\mathcal L_{\mathrm{IIC}}(\mathbf P,\mathbf M)\) 为预测矩阵与记忆矩阵之间的负批次互信息：[待补充引用：IIC]

\[
\mathbf J=
\frac{\mathbf P^{\mathsf T}\mathbf M}
{\sum_{a=1}^{C}\sum_{b=1}^{C}
(\mathbf P^{\mathsf T}\mathbf M)_{a,b}},
\]

\[
\mathcal L_{\mathrm{IIC}}(\mathbf P,\mathbf M)
=-\sum_{a=1}^{C}\sum_{b=1}^{C}J_{a,b}
\log\frac{J_{a,b}}
{J_{a,\cdot}J_{\cdot,b}}.
\]

任务模型的硬监督损失为

\[
\mathcal L_{\mathrm{hard}}
=\frac{1}{|\mathcal B|}
\sum_{i\in\mathcal B}
\beta_i\,
\operatorname{CE}
\left(f_{\theta}(x_i),\widehat y_i^e\right).
\]

为降低预测向少数类别坍缩的风险，进一步最大化批次平均预测的类别熵：

\[
\mathcal H_{\mathrm{div}}
=H\left(
\frac{1}{|\mathcal B|}
\sum_{i\in\mathcal B}\mathbf p_i^{T,e}
\right).
\]

最终，DAC阶段的任务模型损失、VLM提示损失和总损失分别写为

\[
\mathcal L_T^{\mathrm{DAC}}
=\lambda_I\mathcal L_{\mathrm{IIC}}(\mathbf P^T,\mathbf M)
+\mathcal L_{\mathrm{hard}}
-\lambda_D\mathcal H_{\mathrm{div}},
\]

\[
\mathcal L_V^{\mathrm{DAC}}
=\mathcal L_{\mathrm{IIC}}(\mathbf P^V,\mathbf M),
\]

\[
\mathcal L_{\mathrm{DAC}}
=\mathcal L_T^{\mathrm{DAC}}
+\mathcal L_V^{\mathrm{DAC}}.
\]

Office-Home实验采用 \(\lambda_I=1.0\) 和 \(\lambda_D=0.1\)。所有标签和分布均由目标样本预测产生，目标真实标签不参与记忆更新、损失计算、轮数确定或检查点选择。

## 3.5 完整任务模型交接

DAC的作用不是替代DUET，而是为DUET提供一个已经适应目标域的任务模型起点。完成 \(E_D\) 个DAC轮次后，得到

\[
\theta_{\mathrm{DAC}}^{*}
=\{\theta_h^{*},\theta_b^{*},\theta_g^{*}\}.
\]

我们将三个任务模型组件完整交接给DUET：

\[
\theta_{\mathrm{DUET}}^{0}
\leftarrow\theta_{\mathrm{DAC}}^{*}.
\]

与之相对，DAC维护的 \(\mathbf M\)、折扣历史损失、专家权重和提示上下文均在阶段边界处终止，不输入DUET。DUET阶段重新载入其原始VLM分支，并完全沿用发布版DUET的伪标签更新和优化规则。该隔离设计使最终改进能够被解释为“DAC任务模型初始化与DUET后续适配的互补”，而不是两个阶段共享隐含教师状态或额外路由器带来的结果。

完整交接还保留DAC适配后的分类器。进入DUET后，分类器按原版设置冻结，只继续更新任务模型的特征提取器和瓶颈层。最终推理使用DUET输出的任务模型，不需要DAC分布记忆、VLM或任何额外预测头。

## 3.6 第二阶段：原版DUET周期式适配

设第 \(r\) 个DUET周期开始时，任务模型和VLM在完整目标集上的预测分别为 \(\mathbf q_i^{T,r}\) 和 \(\mathbf q_i^{V,r}\)。当前一致集合定义为

\[
\mathcal A_r=
\left\{i\mid
\arg\max_c q_{i,c}^{T,r}
=\arg\max_c q_{i,c}^{V,r}
\right\}.
\]

DUET使用单调累积的准入掩码：

\[
\mathcal M_r=\mathcal M_{r-1}\cup\mathcal A_r,
\qquad \mathcal M_0=\mathcal A_0.
\]

被准入样本的硬伪标签由两路完整分布的算术平均确定：

\[
\widehat y_i^r=
\arg\max_c
\frac{q_{i,c}^{T,r}+q_{i,c}^{V,r}}{2},
\qquad i\in\mathcal M_r.
\]

对任务模型，原版DUET联合使用三类损失。第一，强弱增广一致性约束

\[
\mathcal L_{\mathrm{con}}
=D_{\mathrm{KL}}
\left(\mathbf q_i^{T,w}\|\mathbf q_i^{T,s}\right);
\]

第二，在累积准入集合上的交叉熵

\[
\mathcal L_{\mathrm{cls}}
=\frac{1}{|\mathcal B\cap\mathcal M_r|}
\sum_{i\in\mathcal B\cap\mathcal M_r}
\operatorname{CE}
\left(f_{\theta}(x_i^w),\widehat y_i^r\right);
\]

第三，以VLM完整类别分布作为软目标的蒸馏损失

\[
\mathcal L_{\mathrm{kl}}
=D_{\mathrm{KL}}
\left(\mathbf q_i^{V,r}\|\mathbf q_i^{T,w}\right).
\]

任务模型的DUET目标为

\[
\mathcal L_{\mathrm{DUET}}
=\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}
+\lambda_{\mathrm{cls}}\mathcal L_{\mathrm{cls}}
+\lambda_{\mathrm{kl}}\mathcal L_{\mathrm{kl}}.
\]

Office-Home实验中，\(\lambda_{\mathrm{con}}=0.2\)、\(\lambda_{\mathrm{cls}}=0.4\)、\(\lambda_{\mathrm{kl}}=0.4\)。每个周期还按照原版DUET的Tsallis互信息目标更新VLM视觉分支，文本侧保持冻结。[待补充：DUET的Tsallis互信息公式、参数定义及原论文引用] 本文没有修改DUET阶段的伪标签准入、损失函数或优化器；方法差异仅来自前置DAC及完整任务模型交接。

## 3.7 训练协议与算法流程

DAC和DUET均在完整无标签目标集上训练，并使用固定最终检查点作为报告结果，不根据目标域准确率选择中间轮次。DAC阶段在Office-Home上运行15个轮次，批大小为64。任务模型采用带0.9动量、\(10^{-3}\)权重衰减和Nesterov加速的SGD；特征提取器、瓶颈层和分类器的初始学习率分别为 \(5\times10^{-4}\)、\(5\times10^{-3}\) 和 \(5\times10^{-4}\)。提示上下文同样使用SGD，学习率为 \(5\times10^{-4}\)。两类优化器均采用 \((1+10t/T)^{-0.75}\) 的多项式衰减。

DUET阶段采用ResNet-50任务模型和CLIP ViT-B/32。分类器冻结，特征提取器和瓶颈层的初始学习率分别为 \(10^{-3}\) 和 \(10^{-2}\)，并在每个周期内使用余弦学习率衰减。VLM视觉分支采用Adam，学习率为 \(10^{-6}\)；Tsallis互信息中的初始 \(q\) 值和指数滑动系数分别为1.1和0.99。除任务模型初始化外，其余设置沿用原版DUET。这里的“task pass”仅指任务模型优化对目标训练集的一次完整遍历，不包含周期开始时的全量预测扫描和VLM更新遍历。

> **[投稿前必须确认：DUET阶段训练预算]**  
> 附件 `office_home_dac_duet_uniform5_seed2020.csv` 的最终日志字段为 `Cycle 5/5`，且每个任务的最终 `iter` 等于一次目标集遍历，指向“5个cycle × 每cycle 1个task pass”。当前仓库脚本 `tools/run_office_home_dac_duet_handoff_uniform5_all.sh` 则明确设置为“4个cycle × 每cycle 5个task passes”。两者对应5次与20次DUET任务模型遍历，不能在论文中混写。应以生成84.70%结果的原始完整日志为准，并修改脚本、方法正文和实验设置，使三者严格一致。

**算法1：DAC→DUET无源域目标适配**

1. 输入源域训练完成的任务模型、预训练VLM和无标签目标集；
2. 初始化任务模型与DAC提示分支，在完整目标集上获得两路初始概率；
3. 对每个样本初始化全分布记忆、折扣损失和相等的专家权重；
4. 在每个DAC轮次开始时扫描完整目标集；
5. 根据当前双视角一致性和相对历史记忆的稳定性计算反馈可靠性；
6. 使用当前联合观测回看上一轮任务/VLM分布，更新每个样本自己的折扣历史损失；
7. 由历史平均损失计算连续专家权重，并更新所有样本的全分布记忆；
8. 使用一致硬标签、冲突记忆硬标签和双分支全分布互信息目标更新任务模型与提示上下文；
9. 完成固定DAC轮数后，保存任务模型的特征提取器、瓶颈层和分类器；
10. 将三个组件完整载入原版DUET，丢弃DAC记忆、历史权重和提示状态；
11. 按固定周期运行DUET的累积一致伪标签、强弱增广一致性和VLM软蒸馏；
12. 输出最后一个训练步骤的任务模型作为最终模型。

## 3.8 计算开销与适用边界

DAC为每个目标样本保存一组 \(C\) 维分布记忆、上一轮任务/VLM分布、两个历史损失标量和两个专家权重，因此其主要额外存储复杂度为 \(O(NC)\)。每轮信用计算只包含逐样本概率运算，其复杂度同样为 \(O(NC)\)；主要训练开销仍来自任务模型和VLM的目标域前向与反向传播。DAC状态在阶段交接后被删除，最终推理的参数量和计算量与适配后的任务模型相同。

该方法依赖任务模型与VLM在训练过程中能够形成具有一定稳定性的互补预测。如果两个分支长期同时偏向同一错误类别，后续联合观测无法提供外部纠错依据；如果目标类别无法由VLM类别文本合理表达，VLM分支也可能产生系统性偏差。因此，本文将结论限定于闭集无源域分类及具有可用类别名称的设置，不将其直接外推到开放集、类别不匹配或无语义类别名称的场景。

> **[待补充效率结果]**  
> 报告DAC阶段、DUET阶段和完整流程的训练时间、峰值显存、额外状态大小及最终推理时间；同时给出与原版DUET在相同硬件和相同数据遍历预算下的比较。

---

## 作者核对说明（投稿前删除）

1. 本方法部分已经完全删除类别均衡锚点库、单侧翻转合成监督、16维成对证据、比较器、难度匹配、重放和固定覆盖率选择。上述内容不属于当前DAC→DUET方法。
2. DAC只使用每个样本自身的跨轮历史分配任务/VLM权重，不训练跨样本路由器。
3. handoff只传递任务模型的特征提取器、瓶颈层和分类器；DAC memory、历史信用、专家权重和prompt均不传递。
4. 当前摘要、引言、相关工作、贡献列表和结论仍以comparator为核心，必须随后同步重写，否则整篇论文会出现方法与前文主张不一致的问题。
5. Office-Home附件结果为seed 2020单种子结果。方法部分可以写固定协议，但摘要和结论暂时不能写“稳定提升”或“具有统计显著性”。
