# 超越源假设：基础模型与生成式模型时代的视觉无源域自适应综述

> 英文题目建议：**Beyond Source Hypotheses: Visual Source-Free Domain Adaptation in the Era of Foundation and Generative Models**  
> 稿件类型：Survey/Review  
> 文献覆盖时间：2016 年 1 月—2026 年 8 月 13 日  
> 当前状态：可继续扩展的中文投稿初稿。所有必须由作者确认或补充的内容均以 `[作者待补：……]` 标记。

## 投稿前信息

- 作者与单位：`[作者待补：作者姓名、单位、邮箱、ORCID、通讯作者]`
- 作者在该主题上的研究基础：`[作者待补：列出作者已发表或在投的 SFDA 论文。若投 Computer Science Review，需要核对其 Expertise Statement 要求。]`
- 系统检索流程图：`[作者待补：完成最终数据库导出和人工筛选后，填写检索、去重、初筛、全文筛选及最终纳入数量。]`
- 定量结果复核：`[作者待补：由两名作者独立核对各论文实验表，并决定是否加入统一数值对比表。本文当前不跨论文拼接可能不公平的最高准确率。]`

## 摘要

无源域自适应（source-free domain adaptation，SFDA）研究如何在不能重新访问源数据的条件下，利用源模型和无标签目标数据完成跨域适配。早期视觉 SFDA 主要从源模型权重、批归一化统计和目标域几何结构中恢复监督，逐渐形成了信息最大化、伪标签、原型聚类、邻域一致性、对比学习、模型扰动和虚拟源域生成等技术路线。近两年，视觉—语言模型、视觉基础模型、多模态大模型以及扩散生成模型改变了 SFDA 的信息边界：适配不再只依赖“源模型—目标数据”闭环，还可借助文本语义、开放世界视觉知识和生成先验。然而，外部基础模型也带来了新的误差来源、计算成本、数据许可和隐私风险，并使不同工作之间的实验条件更难比较。

本文面向完整的视觉 SFDA，而非仅讨论图像分类或 CLIP 引导方法。我们提出一个四轴统一框架，从**可用知识来源、适配发生位置、部署协议和视觉任务**同时描述现有方法；在此基础上，将领域演进划分为源假设迁移、目标结构学习、基础模型辅助和生成式先验驱动四个阶段。本文系统讨论传统判别式方法、视觉—语言与视觉基础模型、Stable Diffusion 等生成式模型、开放集与通用类别空间、在线与持续适配、多源和联邦适配，以及分类、分割、检测、视频、三维与医学影像等任务。进一步地，我们指出三个常被忽视的问题：源数据不可见并不自动等于隐私安全；基础模型不会消除伪标签噪声，而是将误差来源从单模型扩展到多知识源；当前排行榜常混合不同骨干网络、预训练数据、模型选择规则和外部模型预算，因而未必能反映算法本身的贡献。最后，本文提出面向可复现评测、可信模型选择、类别空间变化、生成数据治理、模型遗忘和资源受限部署的研究议程。

**关键词：** 无源域自适应；源假设迁移；视觉基础模型；视觉—语言模型；扩散模型；生成式人工智能；开放集；持续适配；可信评测

## 1 引言

视觉模型在训练分布与部署分布不一致时容易失效。相机、天气、成像设备、地理区域、画风和数据采集流程的变化，都可能在不改变任务定义的情况下显著改变输入分布。传统无监督域适应（UDA）在训练时同时访问带标签源数据和无标签目标数据，通过分布对齐或目标自训练缓解这种变化。然而，源数据可能受隐私协议、商业授权、医疗合规、存储限制或跨机构数据治理约束，部署方往往只能获得训练完成的源模型。SFDA 因此把问题收紧为：在适配阶段不再访问源样本，只利用源模型和无标签目标数据获得目标模型。

SHOT 将这一问题表述为源假设迁移，并证明冻结源分类器、在目标域优化信息最大化和伪标签即可形成有竞争力的基线 [1]。此后，研究者从模型内部统计、目标域簇结构、近邻关系、对比表示、不确定性估计和伪源数据重建等角度不断提高适配性能。到 2024 年，Li 等和 Fang 等已分别在 TPAMI 与 Neural Networks 发表 SFDA 综述，另有对研究现状和方向的总结 [49–51]。因此，一篇新的综述必须回答的不再是“SFDA 有哪些伪标签方法”，而是“基础模型和生成式模型加入后，SFDA 的问题边界、知识来源、风险与评测标准发生了什么变化”。

视觉 SFDA 正经历三项结构性变化。第一，知识来源从单一源模型扩大到 ImageNet 预训练编码器、CLIP 类视觉—语言模型、SAM 类视觉基础模型和多模态大模型。DIFO、Co-learn++、ProDe 和 DUET 等工作表明，外部语义知识可以修正源模型偏差，但外部模型的预测同样可能失配 [17,19,20,22]。第二，生成式模型从“近似源域”发展到“操作目标域”。早期方法使用 GAN、图像翻译或特征统计构造伪源样本；后续工作开始利用文本到图像扩散、潜在扩散、条件扩散和扩散式稠密预测模型生成、编辑或校正目标监督 [25–27]。第三，研究设置从离线闭集分类扩展到开放集、通用类别空间、类别增量、在线流、联邦客户端、视频与稠密预测，同时暴露出后门、模型泄漏和灾难性遗忘等问题 [42–48]。

本文的主要贡献如下。

1. **范围更新。** 覆盖传统视觉 SFDA、视觉—语言/视觉基础模型、生成式扩散、类别空间变化、持续与联邦适配，以及分类之外的多种视觉任务，重点补充 2024—2026 年发展。
2. **统一分类。** 用“知识来源—适配位置—部署协议—任务形态”四轴框架描述方法，使伪标签、提示学习、扩散生成和模型统计等看似分散的技术能够在同一坐标系中比较。
3. **批判性评测。** 将骨干网络、外部预训练、目标标签模型选择、目标数据访问方式和计算预算视为实验协议的一部分，而不是可忽略的实现细节。
4. **研究议程。** 从可信选择、开放词汇、生成数据治理、隐私与遗忘、持续部署和资源效率等方面给出可验证的开放问题。

## 2 综述方法与范围

### 2.1 检索策略

本综述检索 OpenAlex、Crossref、arXiv、CVF Open Access、OpenReview 和出版商页面，并交叉检查论文正式发表状态。核心检索式包括 `source-free domain adaptation`、`source-free unsupervised domain adaptation`、`source hypothesis transfer`，以及与 `classification`、`segmentation`、`object detection`、`video`、`medical image`、`open set`、`universal`、`continual`、`federated`、`vision-language`、`foundation model`、`diffusion` 和 `generative` 的组合。最后检索日期为 2026 年 8 月 13 日。

纳入标准为：①适配阶段不访问原始源训练样本；②至少使用一个源训练模型或其可发布统计；③目标训练数据无人工标签，主动 SFDA 等扩展设置需单独标记；④任务包含视觉输入或视觉输出；⑤论文提供足够的方法或实验信息。仅讨论传统 UDA、无目标数据的源自由域泛化、纯零样本识别、没有适配过程的分布外检测，以及非视觉图领域方法，不纳入核心方法表，但在边界讨论中按需引用。

`[作者待补：完成正式系统检索后填写：各数据库命中数 n=__；去重后 n=__；题录初筛 n=__；全文筛选 n=__；最终纳入 n=__。建议由两名作者独立筛选并报告分歧解决规则。]`

### 2.2 与已有综述的区别

| 综述 | 主要覆盖 | 时间重点 | 本文新增内容 |
|---|---|---|---|
| Zhang 等，Neurocomputing 2023 | 研究现状与方向 | 传统 SFDA | 更新基础模型与生成式阶段 |
| Li 等，TPAMI 2024 | 模块化分类、分类基准、应用 | 约 2020—2023 | 四轴分类、协议审计、2024—2026 新设置 |
| Fang 等，Neural Networks 2024 | SFUDA 方法、应用和邻近领域 | 约 2020—2023 | VLM/VFM/MLLM、扩散、联邦、安全与遗忘 |
| CLIP-powered DG/DA Survey，TPAMI 2026 | CLIP 驱动的 DG 与 DA | CLIP 为中心 | 以 SFDA 为中心，同时覆盖传统、生成式和多任务分支 |
| **本文** | 完整视觉 SFDA | 2016—2026.08 | 知识预算、生成先验、开放部署和公平评测的统一分析 |

## 3 问题定义与边界

设源域为带标签数据集 \(\mathcal D_s=\{(x_i^s,y_i^s)\}\)，源训练完成后仅发布模型 \(f_s(\cdot;\theta_s)\)。适配阶段可访问无标签目标数据 \(\mathcal D_t=\{x_j^t\}_{j=1}^{n_t}\)，目标是学习 \(f_t(\cdot;\theta_t)\)。一般形式可写为

\[
\theta_t=\mathcal A(\theta_s,\mathcal D_t,\mathcal K;\mathcal B),
\]

其中 \(\mathcal K\) 表示可选外部知识，如预训练视觉编码器、VLM、VFM、MLLM 或扩散模型；\(\mathcal B\) 表示信息与计算预算。经典 SFDA 令 \(\mathcal K=\varnothing\)，增强型 SFDA 允许外部冻结模型或可训练提示。明确写出 \(\mathcal K\) 和 \(\mathcal B\) 很重要，因为一个只用 ResNet-50 的方法和一个调用数十亿参数多模态模型的方法，即使都不访问源数据，也不是同一资源条件。

按类别集合关系，可区分闭集 \(\mathcal Y_s=\mathcal Y_t\)、部分集 \(\mathcal Y_t\subset\mathcal Y_s\)、开放集 \(\mathcal Y_s\subset\mathcal Y_t\) 与通用/开放部分集。按数据到达方式，可区分离线批量、在线流式、持续多域和类别增量。按模型可见性，可区分白盒、灰盒和黑盒。按域的数量，还包括多源无数据、多个目标域和联邦 SFDA。

SFDA 与测试时适配（TTA）存在交叉，但不等价。SFDA 通常允许在一个无标签目标训练集上进行多轮适配；TTA 更强调测试时、在线、小批量或逐样本更新。源自由域泛化则在训练完成后既不访问源数据，也不使用目标训练数据，因此不属于本文核心范围。本文把满足 SFDA 信息条件的在线/TTA 方法纳入扩展分支，但在表格中明确数据访问协议。

## 4 四轴统一分类框架

### 4.1 轴一：可用知识来源

1. **源模型显式知识。** 分类器权重、logits、特征、注意力和教师预测。
2. **源模型隐式统计。** 批归一化均值与方差、参数敏感性、分类器几何和内部激活。
3. **目标域内在结构。** 聚类、原型、近邻图、时序一致性、增强一致性和类先验。
4. **辅助判别式预训练知识。** ImageNet 编码器、CLIP、其他 VLM、SAM/VFM 和 MLLM。
5. **辅助生成式知识。** GAN、图像翻译、扩散模型、生成式反演和合成数据。
6. **分布式或人工知识。** 多个源模型、联邦客户端、少量主动标注或专家约束。

这一轴揭示了 SFDA 的实质：方法之间的主要差别不是是否“无源”，而是在没有源样本后用什么信息替代源监督。

### 4.2 轴二：适配发生位置

- **输入空间：** 风格迁移、频域调整、增强、伪源/伪目标生成与扩散编辑。
- **特征空间：** 聚类、原型对齐、近邻保持、对比学习、最优传输和流形约束。
- **输出空间：** 熵最小化、信息最大化、伪标签、蒸馏、一致性和不确定性校准。
- **参数空间：** 冻结分类器、选择性更新、BN 统计替换、模型扰动、提示或适配器更新。
- **语义空间：** 文本提示、视觉—语言对齐、多模型共识、MLLM 监督和开放词汇映射。

### 4.3 轴三：部署协议

协议包括离线或在线、单域或连续域、单源或多源、中央式或联邦式、闭集或开放类别空间、完整白盒或仅 API 输出。很多方法在算法名称上相近，但协议不同，不能直接比较。例如，离线方法可以反复遍历全部目标数据并建立全局近邻库，在线方法通常不能回看未来样本；VLM 辅助方法可能访问外部预训练数据编码的知识，而纯 SFDA 方法不能。

### 4.4 轴四：视觉任务

任务从图像分类扩展到二维/三维目标检测、语义和医学分割、视频动作识别、人体姿态估计、行人重识别、点云理解、图像增强和遥感。分类方法中的全局簇假设在稠密预测中并不直接成立；检测还面临前景—背景不平衡和定位噪声；视频方法必须处理时间相关性；医学影像则要求结构合理性、跨模态适配和临床安全性。

### 4.5 代表方法在统一框架中的位置

| 方法 | 年份 | 主要知识来源 | 主要适配位置 | 设置/任务 | 核心特点与主要风险 |
|---|---:|---|---|---|---|
| SHOT [1] | 2020 | 源分类器＋目标簇 | 特征/输出 | 闭集分类 | 冻结分类器并进行信息最大化；依赖目标簇假设 |
| USFDA [2] | 2020 | 源模型＋目标结构 | 特征/输出 | 通用类别空间 | 同时处理域与类别偏移；已知/未知易混淆 |
| Image Translation [3] | 2020 | BN 统计＋图像翻译 | 输入 | 分类 | 将目标图像转为源风格；源分布近似有限 |
| Domain Impression [4] | 2021 | 源模型反演 | 输入/输出 | 分类 | 合成源样本；可能出现模型记忆和低多样性 |
| A2Net [5] | 2021 | 源模型＋目标自监督 | 特征/输出 | 分类 | 对抗推理、类别对比和旋转任务；训练模块较多 |
| NRC [6] | 2021 | 目标近邻结构 | 特征/输出 | 分类 | 使用局部邻域一致性；需特征库且可能传播错误 |
| APG [7] | 2021 | 源类别原型 | 输入/特征 | 分类 | 生成 avatar prototypes；质量受源模型约束 |
| Distribution Estimation [8] | 2022 | 源权重/分布估计 | 特征 | 分类 | 构造可对齐的源分布；估计误差会传递 |
| AaD [10] | 2022 | 目标邻域 | 特征/输出 | 分类 | 吸引同类、排斥异类；简单但依赖邻域纯度 |
| BMD [11] | 2022 | 多中心目标原型 | 特征/输出 | 分类 | 类别平衡动态原型；增加原型维护成本 |
| TPDS [12] | 2023 | 目标预测分布 | 输出 | 分类 | 搜索目标预测分布；对搜索和先验敏感 |
| C-SFDA [13] | 2023 | 课程伪标签 | 输出 | 高效分类 | 逐步纳入样本；课程质量取决于初始排序 |
| ProxyMix [14] | 2023 | 代理源域＋mixup | 输入/特征 | 分类 | 类平衡代理样本；合成偏差可能残留 |
| RGV [15] | 2025 | 目标代表性与多样性 | 特征/输出 | 分类 | 理论分析并渐进学习；协议和实现较复杂 |
| BN-SFDA [16] | 2025 | 冻结 BN 源知识＋目标模型 | 参数/特征 | 分类 | 双模型协同；依赖 BN 架构 |
| Co-learn++ [19] | 2024 | 源模型＋视觉预训练＋CLIP | 特征/语义 | 多种类别设置 | 多知识源协同；知识预算高于纯 SFDA |
| DIFO [20] | 2024 | 冻结 VLM | 语义/输出 | 闭集与部分集分类 | 提示定制和蒸馏；VLM 指导可能含噪 |
| ReCLIP [21] | 2024 | CLIP 图文空间 | 特征/语义 | VLM 适配 | 同时修正图文错位；更新大模型成本较高 |
| ProDe [17] | 2025 | VLM 代理空间 | 语义/输出 | 多种 SFDA 设置 | 理论化代理去噪；仍依赖代理空间质量 |
| DUET [22] | 2025 | 任务模型＋CLIP | 输出/语义 | 分类 | 双视角一致伪标签；冲突样本利用有限 |
| VSFOT [23] | 2026 | VLM＋源原型 | 特征/语义 | 分类 | 语义引导最优传输与双向蒸馏；计算和匹配敏感 |
| RCL [24] | 2026 | 多个 MLLM | 语义/输出 | 分类 | 可靠性课程蒸馏；推理成本和输出稳定性成问题 |
| DM-SFDA [25] | 2024 | 文本到图像扩散 | 输入 | 分类，预印本 | 生成伪源数据；不可见源域难以真实重建 |
| DPTM [26] | 2025 | 潜在扩散＋目标参考 | 输入/输出 | 分类 | 渐进编辑伪目标；语义保持与成本需审计 |
| FreeDNA [27] | 2025 | 扩散预测器噪声统计 | 模型内部/输入 | 稠密预测 | 训练自由噪声对齐；适用对象较专门 |
| SFDA-Seg [28] | 2021 | 源模型＋伪源分布 | 输入/输出 | 语义分割 | 开创分割分支；像素噪声易累积 |
| FVP [31] | 2023 | 频域视觉提示 | 输入/参数 | 医学分割 | 参数高效；提示泛化受模态影响 |
| UPL-SFDA [32] | 2023 | 不确定性伪标签 | 输出 | 医学分割 | 对不确定样本分级利用；估计校准关键 |
| Tell2Adapt [33] | 2026 | VFM 结构知识 | 语义/输出 | 多模态医学分割 | 统一多目标适配；依赖基础模型适用性 |
| IRG-SFOD [36] | 2023 | 实例关系图 | 特征/输出 | 目标检测 | 建模实例关系；伪框和图错误耦合 |
| SFDA-HPE [39] | 2023 | 姿态空间和对比结构 | 特征/输出 | 人体姿态 | 利用关键点结构；任务专用性强 |
| DTE [42] | 2025 | 权重条码＋最优传输 | 特征/输出 | 开放集 | 区分后再利用；未知阈值仍需选择 |
| FedWCA [45] | 2025 | 多客户端目标结构 | 参数/联邦 | 联邦分类 | 加权客户端聚类；通信与隐私另需评估 |
| SSDA-Secure [47] | 2023 | 压缩和知识迁移 | 参数 | 安全 SFDA | 抑制源后门；不等同于全面隐私防护 |

## 5 传统判别式 SFDA

### 5.1 信息最大化与伪标签自训练

SHOT 冻结源分类器，通过条件熵最小化和预测多样性最大化学习目标特征，并使用目标特征质心细化伪标签 [1]。其影响在于把分类器权重视为源类别结构的压缩载体。后续方法改进了伪标签的生成、筛选、课程安排和噪声鲁棒性，例如 BMD 使用多中心动态原型 [11]，ProxyMix 构造类别平衡代理域 [14]，C-SFDA 采用课程自训练 [13]。伪标签路线易实现且适合与其他技术组合，但当初始源模型在目标域严重失效时，会出现确认偏差：错误预测被训练过程进一步强化。

### 5.2 聚类、原型和邻域结构

NRC、AaD 和相关方法不只依赖单样本置信度，而是利用目标样本之间的局部几何 [6,10]。其共同假设是同类样本在目标特征空间中形成连通或紧凑区域，邻居关系能够提供比单次 softmax 更稳定的监督。这类方法在 Office-Home 和 VisDA-C 等分类基准上长期构成强基线，但需要保存特征库或近邻图，对流式数据、长尾类别和大规模目标集并不友好。邻域还可能跨越真实类别边界，使早期错误在图上传播。

### 5.3 对比、自监督与一致性学习

A2Net、历史对比学习、AdaContrast 类思想以及面向分割的增强一致性方法，通过旋转预测、强弱增强、教师—学生一致性或类条件对比学习提供额外监督。它们降低了对硬伪标签的单点依赖，但性能强烈受增强策略影响；如果增强破坏语义或与真实域偏移无关，一致性目标会约束模型学习错误的不变性。2026 年的工作进一步从特征拓扑和增强噪声角度分析这一问题，说明“一致”本身并不等于“正确”。

### 5.4 模型统计、参数约束与域重建

另一条路线从 BN 统计、分类器权重或源模型响应中近似源分布。Domain Impression、VDM-DA、分布估计和冻结 BN 协同训练等方法分别在像素、特征或统计层面重建源样信息。图像翻译和虚拟域方法则生成伪源样本，再复用传统 UDA 损失。其优点是重新获得“两个域”以便显式对齐，缺点是模型内部统计只是源数据的有损投影，生成样本可能过度迎合源分类器而缺少真实多样性。

### 5.5 不确定性、噪声鲁棒与选择性学习

由于 SFDA 没有真实标签，可靠性估计决定了错误是否会累积。现有方法使用熵、分类边际、模型集成、贝叶斯或证据深度学习、教师稳定性以及风险—覆盖选择。2025 年的标签校准工作将 Dirichlet 证据和 softmax 校准引入 SFDA。需要注意的是，未校准的模型置信度不能直接解释为正确概率，不同模型的 softmax 也不能天然比较。未来评测应同时报告准确率、校准误差、拒识性能和覆盖率。

## 6 基础模型驱动的 SFDA

### 6.1 从通用视觉预训练到视觉—语言模型

传统流水线常在源训练后丢弃最初的 ImageNet 预训练编码器。Co-learn/Co-learn++ 重新把预训练视觉网络和 CLIP 引入目标适配，通过协同伪标签减轻源模型偏差 [19]。DIFO 进一步对冻结 VLM 进行无监督提示定制，并将任务相关的多模态知识蒸馏给目标模型 [20]。ReCLIP 关注的是 VLM 本身在目标域的视觉—文本错位，通过跨模态自训练共同修正视觉和文本编码器 [21]。ProDe 指出 VLM 监督也会含噪，并把 VLM 视为通向潜在域不变空间的代理，而非绝对教师 [17]。

这类工作代表了一个重要转变：SFDA 不再只问“如何相信源模型”，而是问“如何在源模型和通用模型之间分配信任”。DUET 通过任务模型与 CLIP 的一致样本形成伪标签，并把不一致样本用于不确定性驱动训练 [22]；后续方法开始使用最优传输、双向蒸馏和更细粒度的冲突处理 [23]。基础模型提供外部语义锚点，尤其适合大域差和类别语义明确的任务，但其表现受提示模板、类别名称、预训练语料覆盖和语言偏差影响。

### 6.2 VFM、SAM 与 MLLM

在稠密预测中，SAM 类视觉基础模型可以提供类别无关掩码、边界或形状先验。医学分割工作已利用 SAM 修正伪标签，2026 年的 Tell2Adapt 则用视觉基础模型生成并细化跨模态、跨器官的监督。目标检测中的基础模型先验可用于突出前景并缓解背景不平衡。另一方面，多模态大模型能够输出更丰富的语义判断，但也存在指令不遵循、输出不稳定和推理成本高的问题。RCL 等方法因此不直接部署 MLLM，而是以多模型一致性和课程学习把其知识蒸馏到轻量目标模型。

基础模型分支需要公开至少五项信息：模型名称和版本、预训练数据可见范围、提示模板及搜索过程、是否更新基础模型、单样本推理成本。否则，“SFDA 性能提升”可能主要来自更大外部模型，而非适配算法。

## 7 生成式模型与扩散 SFDA

### 7.1 从 GAN 和图像翻译到扩散模型

生成式 SFDA 的基本动机是恢复缺失的数据支持。早期方法使用生成对抗网络、模型反演、BN 统计匹配或图像翻译产生源样本或源风格图像。扩散模型提供更强的多样性和文本控制能力后，研究重点从“重建一个固定伪源域”转向“构造可渐进接近目标任务的中间域”。

需要区分三种常被混称为“扩散”的技术：①Stable Diffusion/潜在扩散等生成模型；②在标签图或样本图上传播信息的图扩散；③扩散式稠密预测模型内部的去噪过程。它们的优化对象和计算成本完全不同，不能归入同一方法类别。

### 7.2 扩散模型的四种角色

1. **伪源生成。** DM-SFDA 等工作用文本到图像扩散生成源样或源风格数据，再执行域对齐。该路线受生成器是否真实覆盖不可见源分布限制，且部分早期工作目前仍为预印本。
2. **目标域语义编辑。** NeurIPS 2025 的 DPTM 将可靠与不可靠目标样本分开，用潜在扩散把不可靠样本修改为指定类别，同时保持目标域外观，并逐步缩小伪目标与真实目标之间的差异。这比直接生成伪源更少依赖对不可见源域的猜测。
3. **结构或标签补全。** 医学和遥感分割中，条件扩散可从边缘、少量高质量种子或结构先验恢复完整掩码，从而把扩散的去噪能力用于伪标签修正。
4. **扩散模型本身的适配。** FreeDNA 面向扩散式稠密预测模型，通过噪声统计对齐实现训练自由或源自由适配。此时扩散模型既不是数据生成器，也不是外部教师，而是待适配的任务模型。

### 7.3 机会与风险

扩散模型能够覆盖传统增强难以表达的外观变化，并通过文本或结构条件控制语义；但它也可能改变类别身份、生成训练语料中的刻板偏差或泄漏近似训练样本。生成质量指标并不等于下游适配价值。建议同时评估语义保持、域接近度、类别覆盖、样本多样性、隐私记忆和单位增益计算成本。若使用 Stable Diffusion，应报告具体版本、许可证、提示词、负提示词、采样器、步数、随机种子和人工筛选规则。

## 8 扩展设置

### 8.1 开放集、部分集与通用 SFDA

闭集假设在真实部署中经常不成立。Universal SFDA、UMAD、GLC/GLC++ 等方法同时处理域偏移和类别集合偏移；DTE 通过权重条码估计与稀疏标签分配区分已知和未知样本。难点在于高熵既可能表示域偏移下的已知难样本，也可能表示真正未知类。只用一个阈值进行拒识通常不稳定，未来应联合报告已知类准确率、未知检测、H-score、开放集校准和阈值敏感性。

基础模型让开放词汇 SFDA 成为可能，但也模糊了“未知类”的定义：对源分类器未知的类别可能已被 CLIP 预训练见过。因此论文必须分别定义任务标签知识、源模型知识与外部模型知识。

### 8.2 在线、持续与类别增量

在线 SFDA 不能依赖完整目标集的全局聚类；持续 SFDA 还需在域序列中避免遗忘。相关研究使用记忆库、动态教师、提示适配和正则化保持历史知识。CVPR 2025 的类别增量 SFUDA把新类别学习与域迁移结合起来。未来协议应区分是否允许回放、目标域边界是否已知、流是否时间相关、每个样本可见次数以及是否需要保留源性能。

### 8.3 多源、多目标与联邦 SFDA

多源无数据场景提供多个源模型而非多个源数据集，核心是估计每个源模型对当前目标的可迁移性。联邦 SFDA 进一步加入客户端异质性、通信和隐私约束。FedWCA、FedSCAl 等方法通过客户端聚类、加权聚合或服务器—客户端预测对齐降低客户端漂移。需要防止把“原始数据不上传”误写为严格隐私保证；模型更新仍可能泄漏信息，并需与差分隐私或安全聚合区分。

### 8.4 黑盒、主动与安全 SFDA

黑盒 SFDA 只能访问源模型输出，无法使用内部特征或 BN 统计。主动 SFDA 允许选择极少量目标样本标注，属于不同标注预算。SSDA（Secure SFDA）表明恶意源模型可把后门带入目标模型；2026 年进一步工作指出，源模型可能保留本应遗忘的源专有类别。由此可见，source-free 是数据访问协议，不是隐私、安全或合规结论。

## 9 按视觉任务分析

| 任务 | 代表性方向 | 主要困难 | 推荐指标 |
|---|---|---|---|
| 图像分类 | SHOT、NRC、AaD、DIFO、ProDe、DUET、DPTM | 伪标签噪声、长尾和类别偏移 | accuracy、macro-F1、ECE、risk–coverage |
| 语义分割 | SFDA-Seg、AugCo、UPL-SFDA、FVP、Tell2Adapt | 像素相关、边界、小目标、结构合理性 | mIoU、Dice、HD95/ASD、类别均衡指标 |
| 目标检测 | Free Lunch、Overlook Style、IRG、动态教师、VFM 先验 | 定位与分类耦合、背景不平衡 | mAP、AP50/75、校准与伪框质量 |
| 视频识别/分割 | 大型语言—视觉模型、CleanAdapt、Co-STAR | 时间相关、动作语义、在线成本 | top-1、mean class accuracy、时序稳定性 |
| 三维点云/检测 | SF-UDA3D、点云分割和城市级方法 | 稀疏性、传感器差异、几何偏移 | mAP、mIoU、距离分段指标 |
| 医学影像 | Fourier style mining、FVP、UPL-SFDA、SAM/VFM | 模态差、形状约束、临床风险 | Dice、HD95/ASD、灵敏度、外部中心验证 |
| 姿态/重识别/遥感 | SFDA-HPE、ReID、遥感检测与分割 | 结构先验、细粒度身份、地理偏移 | PCK、mAP/CMC、任务专用指标 |

任务扩展说明了分类排行榜不能代表整个领域。分割和检测必须处理结构化输出，视频和在线场景不能随意访问全局目标集，医学任务还需要跨中心外部验证与失败案例审查。

## 10 数据集、评测与可复现性

### 10.1 常用基准

分类通常使用 Office-31、Office-Home、VisDA-C、DomainNet/DomainNet-126、PACS 和 ImageNet-C；分割使用 GTA5→Cityscapes、SYNTHIA→Cityscapes、跨设备或跨模态医学数据；检测常见 Cityscapes、Foggy Cityscapes、Sim10k、KITTI 和不同天气/传感器组合；视频常用 UCF-HMDB、EPIC-Kitchens 等。

这些基准存在明显局限。Office-31 规模小且容易饱和；Office-Home 和 DomainNet 的任务平均可能掩盖难迁移方向；VisDA-C 经常混用总体准确率和类别平均准确率；医学数据的预处理、切分和伦理限制常不一致；不少工作在目标测试标签上选择最佳 epoch，形成隐性监督。

### 10.2 公平比较清单

建议每篇 SFDA 论文报告：

1. 源模型架构、源训练策略及源模型初始目标性能；
2. 是否使用 ImageNet、CLIP、SAM、MLLM、扩散模型或其他外部知识；
3. 目标数据是离线全量、多遍、在线单遍还是 episodic；
4. 模型选择是否访问目标标签，若否，使用何种无监督准则；
5. 所有域方向的均值、标准差和至少三个随机种子；
6. 类别不平衡、未知类别和大域差下的结果；
7. 参数量、训练和推理 FLOPs/时间、显存及外部 API 成本；
8. 校准、风险—覆盖、失败案例及适配导致的负迁移；
9. 代码、配置、源模型权重和提示词；
10. 数据、模型和生成器的许可及隐私说明。

### 10.3 为什么不直接拼接 SOTA 数字

不同论文可能使用不同骨干、源模型训练、数据划分、提示模板、预训练模型和模型选择规则。把各论文报告的最高数值放入一个表中，会制造“同协议比较”的假象。本文建议建立分层排行榜：纯源模型预算、通用视觉预训练预算、VLM/VFM 预算和生成模型预算分别比较；每层再区分目标标签选择与完全无监督选择。

`[作者待补：如期刊要求定量综述，请逐篇复核原始表格和代码，将同骨干、同源模型、同数据划分、同模型选择的结果放入主表；其余结果移至补充材料并标记不可直接比较。]`

## 11 综合讨论

### 11.1 什么真正推动了性能提升

SFDA 的性能提高主要来自三类增量：更可靠的目标监督、更丰富的外部知识和更合适的优化约束。传统方法重点减少单一源模型产生的伪标签误差；基础模型方法增加第二或多个知识视角；扩散方法则改变可训练数据本身。三者并非互斥。未来强方法很可能把目标结构、外部语义和生成增强结合起来，但必须说明每部分的独立贡献和成本。

### 11.2 基础模型没有消除确认偏差

VLM 与源模型的错误相关性决定了融合收益。如果两者在同一困难类别上同时失败，一致性会产生高置信错误；如果两者经常冲突，简单丢弃冲突样本会损失大量信息。因而研究重点应从“使用 CLIP 标签”转向“估计多知识源在每个样本上的相对可靠性”，并允许拒绝决策。MLLM 也不应被当作自动真值生成器。

### 11.3 源自由不等于隐私保护

不传输源样本降低了直接暴露风险，但源模型仍可能遭受成员推断、模型反演、属性推断或后门攻击。生成伪源数据还可能重现训练样本特征。论文应使用“减少源数据访问”这类有限表述，除非给出正式隐私机制和攻击评测。

### 11.4 生成模型改变了 SFDA 的合规边界

当扩散模型作为外部知识源时，其训练数据、版权和记忆风险进入 SFDA 系统。生成图片是否属于“源自由”取决于定义：它不访问原始源样本，但可能包含外部数据先验。更合理的做法是报告信息来源账本，而不是只使用二元的 source-free 标签。

## 12 开放问题与研究方向

1. **可审计的信息预算。** 建立从纯模型到多基础模型/生成模型的分级协议，并报告外部数据和计算来源。
2. **真正无标签的模型选择。** 研究与目标风险稳定相关的无监督验证指标，避免用目标标签挑 epoch 或超参数。
3. **从闭集到开放词汇。** 同时处理域偏移、标签集合偏移、类别名称歧义和层级标签，而不是仅增加一个 unknown 类。
4. **可靠的多模型仲裁。** 学习源模型、VLM、VFM 和 MLLM 的样本级可靠性，显式建模相关错误、冲突和拒绝。
5. **生成数据的语义与隐私验证。** 不只评价图像质量，还要验证类别身份、覆盖、记忆、版权和下游因果贡献。
6. **持续与开放世界部署。** 在未知域边界、时间相关数据和新类别不断到达时兼顾适配、遗忘和恢复能力。
7. **安全与机器遗忘。** 检测恶意源模型、后门和源专有知识泄漏，并在不访问源数据时提供可验证遗忘。
8. **任务统一而非分类迁移。** 发展能跨分类、检测、分割和视频工作的原则，同时保留任务特有结构。
9. **长尾、公平与校准。** 报告少数类、不同人群或设备上的性能，避免平均准确率掩盖系统性失败。
10. **资源受限 SFDA。** 将能耗、延迟、显存和 API 成本纳入目标函数，发展基础模型知识的离线蒸馏和轻量部署。
11. **理论边界。** 解释源模型可识别性、类别先验变化、外部知识偏差和目标簇假设何时足以保证适配。
12. **可持续基准。** 建立带时间顺序、未知类别、真实隐私约束和固定模型选择协议的长期基准。

## 13 结论

视觉 SFDA 已从“在没有源数据时继续训练分类器”发展为一个涉及模型知识恢复、目标结构学习、基础模型协作、生成式数据操作和持续部署的广泛研究领域。传统伪标签、聚类和一致性仍是多数方法的优化基础；VLM、VFM 与 MLLM 扩大了语义信息来源；扩散模型则把适配推进到可控的数据生成和任务模型去噪过程。与此同时，外部模型预算、目标标签模型选择、隐私泄漏和计算成本使“source-free”这一标签本身越来越不足以描述真实系统。未来工作的关键不只是继续提高单一基准上的平均准确率，而是建立能够说明知识从何而来、何时可信、以何种成本发挥作用、在何处失效的可审计 SFDA 方法学。

## 参考文献（首轮核验版）

> 下列条目优先保留正式发表版本；预印本已明确标注。最终投稿前需使用 Zotero/EndNote 统一格式，并逐条核对作者、卷期、页码和 DOI。

1. Liang, J., Hu, D. & Feng, J. Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. *ICML*, 2020.
2. Kundu, J. N. et al. Universal Source-Free Domain Adaptation. *CVPR*, 2020. DOI: 10.1109/CVPR42600.2020.00460.
3. Hou, Y. & Zheng, L. Source Free Domain Adaptation with Image Translation. *arXiv:2008.07514*, 2020.
4. Kurmi, V. K., Subramanian, V. K. & Namboodiri, V. P. Domain Impression: A Source Data Free Domain Adaptation Method. *WACV*, 2021. DOI: 10.1109/WACV48630.2021.00066.
5. Xia, H., Zhao, H. & Ding, Z. Adaptive Adversarial Network for Source-Free Domain Adaptation. *ICCV*, 2021. DOI: 10.1109/ICCV48922.2021.00888.
6. Yang, S. et al. Exploiting the Intrinsic Neighborhood Structure for Source-Free Domain Adaptation. *NeurIPS*, 2021.
7. Qiu, Z. et al. Source-Free Domain Adaptation via Avatar Prototype Generation and Adaptation. *IJCAI*, 2021. DOI: 10.24963/ijcai.2021/402.
8. Ding, N. et al. Source-Free Domain Adaptation via Distribution Estimation. *CVPR*, 2022. DOI: 10.1109/CVPR52688.2022.00707.
9. Wang, F. et al. Exploring Domain-Invariant Parameters for Source Free Domain Adaptation. *CVPR*, 2022. DOI: 10.1109/CVPR52688.2022.00701.
10. Yang, S. et al. Attracting and Dispersing: A Simple Approach for Source-Free Domain Adaptation. *NeurIPS*, 2022.
11. Qu, S. et al. BMD: A General Class-Balanced Multicentric Dynamic Prototype Strategy for Source-Free Domain Adaptation. *ECCV*, 2022.
12. Tang, S. et al. Source-Free Domain Adaptation via Target Prediction Distribution Searching. *IJCV*, 2023. DOI: 10.1007/s11263-023-01892-w.
13. Karim, N. et al. C-SFDA: A Curriculum Learning Aided Self-Training Framework for Efficient Source Free Domain Adaptation. *CVPR*, 2023. DOI: 10.1109/CVPR52729.2023.02310.
14. Ding, Y. et al. ProxyMix: Proxy-Based Mixup Training with Label Refinery for Source-Free Domain Adaptation. *Neural Networks*, 2023. DOI: 10.1016/j.neunet.2023.08.005.
15. Zhu, R. et al. Revisiting Source-Free Domain Adaptation: Insights into Representativeness, Generalization, and Variety. *CVPR*, 2025.
16. Deng, X., Wang, Y. & Xue, Z. Leveraging Frozen Batch Normalization for Co-Training in Source-Free Domain Adaptation. *AISTATS*, 2025.
17. Tang, S. et al. Proxy Denoising for Source-Free Domain Adaptation. *ICLR*, 2025.
18. Lee, J. Y., Nam, H. & Cho, S. I. Measure the Feature Universe: Topology-Based Pseudo Labeling and Gravity Consistency for Source-Free Domain Adaptation. *CVPR*, 2026.
19. Zhang, W., Shen, L. & Foo, C.-S. Source-Free Domain Adaptation Guided by Vision and Vision-Language Pre-Training. *IJCV*, 2024. DOI: 10.1007/s11263-024-02215-3.
20. Tang, S. et al. Source-Free Domain Adaptation with Frozen Multimodal Foundation Model. *CVPR*, 2024. DOI: 10.1109/CVPR52733.2024.02238.
21. Hu, X. et al. ReCLIP: Refine Contrastive Language Image Pre-Training with Source Free Domain Adaptation. *WACV*, 2024. DOI: 10.1109/WACV57701.2024.00297.
22. Lee, J. Y. et al. DUET: Dual-Perspective Pseudo Labeling and Uncertainty-Aware Exploration & Exploitation Training for Source-Free Domain Adaptation. *NeurIPS*, 2025.
23. Han, S. et al. Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport. *CVPR*, 2026.
24. Chen, D. et al. Empowering Source-Free Domain Adaptation via MLLM-Guided Reliability-Based Curriculum Learning. *WACV*, 2026.
25. Chopra, S. et al. Source-Free Domain Adaptation with Diffusion-Guided Source Data Generation. *arXiv:2402.04929*, 2024 (preprint).
26. Huang, Y. et al. Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation. *NeurIPS*, 2025.
27. Xu, H. et al. FreeDNA: Endowing Domain Adaptation of Diffusion-Based Dense Prediction with Training-Free Domain Noise Alignment. *ICCV*, 2025.
28. Liu, Y., Zhang, W. & Wang, J. Source-Free Domain Adaptation for Semantic Segmentation. *CVPR*, 2021. DOI: 10.1109/CVPR46437.2021.00127.
29. Bateson, M., Kervadec, H. & Dolz, J. Source-Free Domain Adaptation for Image Segmentation. *Medical Image Analysis*, 2022. DOI: 10.1016/j.media.2022.102617.
30. Yang, C. et al. Source Free Domain Adaptation for Medical Image Segmentation with Fourier Style Mining. *Medical Image Analysis*, 2022. DOI: 10.1016/j.media.2022.102457.
31. Wang, Y. et al. FVP: Fourier Visual Prompting for Source-Free Unsupervised Domain Adaptation of Medical Image Segmentation. *IEEE TMI*, 2023. DOI: 10.1109/TMI.2023.3306105.
32. Wu, J. et al. UPL-SFDA: Uncertainty-Aware Pseudo Label Guided Source-Free Domain Adaptation for Medical Image Segmentation. *IEEE TMI*, 2023. DOI: 10.1109/TMI.2023.3318364.
33. Shi, Y. et al. Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model. *CVPR*, 2026.
34. Li, X. et al. A Free Lunch for Unsupervised Domain Adaptive Object Detection without Source Data. *AAAI*, 2021. DOI: 10.1609/aaai.v35i10.17029.
35. Li, S. et al. Source-Free Object Detection by Learning to Overlook Domain Style. *CVPR*, 2022. DOI: 10.1109/CVPR52688.2022.00785.
36. VS, V., Oza, P. & Patel, V. M. Instance Relation Graph Guided Source-Free Domain Adaptive Object Detection. *CVPR*, 2023.
37. He, Q. et al. Dual-Rate Dynamic Teacher for Source-Free Domain Adaptive Object Detection. *ICCV*, 2025.
38. Saltori, C. et al. SF-UDA3D: Source-Free Unsupervised Domain Adaptation for LiDAR-Based 3D Object Detection. *3DV*, 2020. DOI: 10.1109/3DV50981.2020.00087.
39. Peng, Q., Zheng, C. & Chen, C. Source-Free Domain Adaptive Human Pose Estimation. *ICCV*, 2023.
40. Zara, G. et al. The Unreasonable Effectiveness of Large Language-Vision Models for Source-Free Video Domain Adaptation. *ICCV*, 2023.
41. Dasgupta, A., Jawahar, C. V. & Alahari, K. Source-Free Video Domain Adaptation by Learning from Noisy Labels. *Pattern Recognition*, 2025. DOI: 10.1016/j.patcog.2024.111328.
42. Liu, W. et al. Distinguish Then Exploit: Source-Free Open Set Domain Adaptation via Weight Barcode Estimation and Sparse Label Assignment. *CVPR*, 2025.
43. Qu, S. et al. GLC++: Source-Free Universal Domain Adaptation Through Global-Local Clustering and Contrastive Affinity Learning. *TPAMI* 47(11), 10646–10663 (2025). DOI: 10.1109/TPAMI.2025.3593669.
44. Deng, P. et al. Multi-Granularity Class Prototype Topology Distillation for Class-Incremental Source-Free Unsupervised Domain Adaptation. *CVPR*, 2025.
45. Mori, J. et al. Federated Source-Free Domain Adaptation for Classification: Weighted Cluster Aggregation for Unlabeled Data. *WACV*, 2025.
46. Yashwanth, M. et al. FedSCAl: Leveraging Server and Client Alignment for Unsupervised Federated Source-Free Domain Adaptation. *WACV*, 2026.
47. Ahmed, S. et al. SSDA: Secure Source-Free Domain Adaptation. *ICCV*, 2023.
48. Devalapally, A. et al. Source Models Leak What They Shouldn't: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization. *CVPR*, 2026.
49. Li, J. et al. A Comprehensive Survey on Source-Free Domain Adaptation. *TPAMI* 46, 5743–5762 (2024). DOI: 10.1109/TPAMI.2024.3370978.
50. Fang, Y. et al. Source-Free Unsupervised Domain Adaptation: A Survey. *Neural Networks* 174, 106230 (2024). DOI: 10.1016/j.neunet.2024.106230.
51. Zhang, N. et al. Source-Free Unsupervised Domain Adaptation: Current Research and Future Directions. *Neurocomputing* 564, 126921 (2024). DOI: 10.1016/j.neucom.2023.126921.

## 作者必须补充或确认的内容

1. `[作者待补：最终作者信息、基金、贡献声明和利益冲突。]`
2. `[作者待补：选择目标期刊后，按期刊要求调整篇幅、摘要格式、Highlights、图形摘要和参考文献样式。]`
3. `[作者待补：完成 PRISMA/系统检索数字及排除理由。]`
4. `[作者待补：至少绘制四幅原创图：研究演进时间线、四轴分类图、基础模型/扩散模型知识流图、公平评测协议图。]`
5. `[作者待补：逐条复核 2025—2026 论文的正式卷期、页码和 DOI；预印本不得写成正式会议论文。]`
6. `[作者待补：决定是否进行统一代码复现。若不复现，应将数值比较明确标为“原论文报告结果”，并避免跨协议排名。]`
7. `[作者待补：根据期刊政策披露生成式 AI 辅助写作，并由作者对全部事实、引文和论证负责。]`
