# Beyond Source Hypotheses: Visual Source-Free Domain Adaptation in the Era of Foundation and Generative Models

> Article type: Survey/Review  
> Literature window: January 2016–13 August 2026  
> Status: English submission draft. Items requiring author input are marked `[AUTHOR INPUT: ...]`.

## Information required before submission

- Authors and affiliations: `[AUTHOR INPUT: names, affiliations, e-mails, ORCIDs, and corresponding author.]`
- Author expertise: `[AUTHOR INPUT: list the authors' published or submitted work on SFDA. This is particularly important if Computer Science Review is selected.]`
- Systematic-search flow: `[AUTHOR INPUT: after the final database export and manual screening, enter the numbers retrieved, deduplicated, screened, excluded, and included.]`
- Quantitative verification: `[AUTHOR INPUT: two authors should independently verify every value extracted from the original papers before a cross-paper results table is added.]`

## Abstract

Source-free domain adaptation (SFDA) adapts a model to an unlabeled target domain without revisiting the data used to train the source model. Early visual SFDA methods recovered supervision from source-model weights, batch-normalization statistics, and the geometry of target representations, giving rise to information maximization, pseudo-labeling, prototype clustering, neighborhood consistency, contrastive learning, model perturbation, and virtual-domain reconstruction. Foundation and generative models have recently changed this information boundary. Adaptation is no longer confined to a closed loop between one source model and target data: vision–language models (VLMs), vision foundation models (VFMs), multimodal large language models (MLLMs), and diffusion models can contribute semantic, structural, or generative priors. These additional knowledge sources can improve adaptation under large domain shifts, but they also introduce new errors, computational costs, licensing constraints, and privacy risks, while making experimental comparisons less controlled.

This survey covers visual SFDA as a whole rather than only image classification or CLIP-guided methods. We organize the field through four complementary axes: **knowledge source, adaptation locus, deployment protocol, and visual task**. From this perspective, we trace four stages of development: source-hypothesis transfer, target-structure learning, foundation-model-assisted adaptation, and generative-prior-driven adaptation. We review conventional discriminative methods; vision–language and vision foundation models; Stable Diffusion and other generative approaches; open-set and universal label spaces; online, continual, multi-source, and federated adaptation; and applications to classification, segmentation, detection, video, three-dimensional vision, and medical imaging. Three conclusions emerge. First, source-data inaccessibility does not by itself provide privacy or security. Second, foundation models do not remove pseudo-label noise; they turn a single-model reliability problem into a multi-source reliability problem. Third, many published leaderboards mix different backbones, pretraining corpora, model-selection rules, and external-model budgets and therefore do not isolate adaptation quality. We conclude with a research agenda for reproducible evaluation, label-free model selection, open-world adaptation, generative-data governance, machine unlearning, and resource-constrained deployment.

**Keywords:** source-free domain adaptation; source hypothesis transfer; vision foundation model; vision–language model; diffusion model; generative artificial intelligence; open-set adaptation; continual adaptation; trustworthy evaluation

## 1. Introduction

Visual recognition systems can fail when the deployment distribution differs from the distribution on which they were trained. Changes in cameras, weather, imaging devices, geographic regions, artistic styles, or acquisition pipelines can alter the input distribution without changing the nominal task. Unsupervised domain adaptation (UDA) addresses this problem by jointly using labeled source data and unlabeled target data to align distributions or train on target-derived supervision. In many deployments, however, the source data cannot be transferred to the target site because of privacy agreements, commercial licensing, medical governance, storage constraints, or institutional separation. The receiving party may obtain only a trained source model. SFDA therefore asks whether a useful target model can be learned from a source model and unlabeled target data without revisiting the source samples.

Source Hypothesis Transfer (SHOT) established an influential formulation of this problem: freeze the source classifier, optimize information maximization on target observations, and refine target pseudo-labels by class centroids [1]. Subsequent studies mined source-model statistics, target clusters, neighborhood graphs, contrastive relations, uncertainty, and reconstructed source-like data. By 2024, two broad surveys had appeared in *IEEE Transactions on Pattern Analysis and Machine Intelligence* and *Neural Networks* [49,50], alongside an earlier review of current research and future directions [51]. A new survey cannot therefore justify itself by listing another set of pseudo-labeling methods. It must explain how the problem has changed since foundation and generative models became practical auxiliary knowledge sources.

Three structural changes motivate this survey. First, the knowledge available at adaptation time has expanded from a single source model to ImageNet-pretrained encoders, CLIP-like VLMs, SAM-like VFMs, and MLLMs. Co-learn++, DIFO, ProDe, and DUET show that external semantic knowledge can correct source bias, but also reveal that external predictions may themselves be noisy [17,19,20,22]. Second, generative adaptation is moving from approximating an inaccessible source domain toward manipulating target-domain support. Earlier work used generative adversarial networks, image translation, inversion, or feature statistics to build pseudo-source data [3,4]. More recent work employs text-to-image diffusion, latent diffusion, conditional diffusion, and diffusion-based dense predictors to generate, edit, or repair adaptation supervision [25–27]. Third, SFDA has expanded beyond offline closed-set classification to open and universal label spaces, class-incremental streams, federated clients, video, three-dimensional sensing, and dense prediction. These settings expose problems—including backdoors, source-knowledge leakage, and catastrophic forgetting—that a classification-only taxonomy cannot capture [42–48].

This survey makes four contributions.

1. **Updated scope.** We jointly cover conventional visual SFDA, vision–language and vision foundation models, generative diffusion, changing label spaces, continual and federated deployment, and visual tasks beyond classification, with particular attention to work from 2024–2026.
2. **Unified taxonomy.** A four-axis framework—knowledge source, adaptation locus, deployment protocol, and task—places pseudo-labeling, prompting, diffusion generation, and model-statistic recovery in one analytical space.
3. **Critical evaluation.** We treat the backbone, external pretraining, target-label model selection, target-data access pattern, and computational budget as components of the protocol rather than incidental implementation details.
4. **Research agenda.** We formulate concrete questions around label-free selection, open vocabulary, generative-data governance, privacy and unlearning, continual deployment, and resource efficiency.

## 2. Review methodology and scope

### 2.1 Search strategy

We searched OpenAlex, Crossref, arXiv, CVF Open Access, OpenReview, and publisher records, and cross-checked whether preprints had acquired a formal publication. The core queries were `source-free domain adaptation`, `source-free unsupervised domain adaptation`, and `source hypothesis transfer`, combined with `classification`, `segmentation`, `object detection`, `video`, `medical image`, `open set`, `universal`, `continual`, `federated`, `vision-language`, `foundation model`, `diffusion`, and `generative`. The last search was performed on 13 August 2026.

A study was eligible for the core review if: (i) raw source training samples were unavailable during adaptation; (ii) at least one source-trained model or releasable source statistic was available; (iii) target training labels were unavailable, with active SFDA and other relaxed variants explicitly separated; (iv) the task involved a visual input or output; and (v) the paper provided sufficient methodological or experimental detail. Conventional UDA, target-free source-free domain generalization, pure zero-shot recognition, distribution-shift detection without adaptation, and non-visual graph-only methods were excluded from the core method tables, although they are cited when defining boundaries.

`[AUTHOR INPUT: complete the final systematic search and enter: database records n=__; records after deduplication n=__; title/abstract screening n=__; full-text screening n=__; final included studies n=__. Two authors should screen independently and report how disagreements were resolved.]`

### 2.2 Distinction from previous surveys

| Survey | Primary coverage | Main period | Increment offered here |
|---|---|---|---|
| Zhang et al., *Neurocomputing* 2023 [51] | Current research and future directions | Conventional SFDA | Foundation and generative stage |
| Li et al., *TPAMI* 2024 [49] | Modular taxonomy, classification benchmarks, applications | Mainly 2020–2023 | Four-axis taxonomy, protocol audit, 2024–2026 settings |
| Fang et al., *Neural Networks* 2024 [50] | SFUDA methods, applications, related fields | Mainly 2020–2023 | VLM/VFM/MLLM, diffusion, federation, security, and unlearning |
| CLIP-powered DG/DA survey, *TPAMI* 2026 | CLIP-driven domain generalization and adaptation | CLIP-centered | SFDA-centered coverage spanning conventional, generative, and task-specific branches |
| **This survey** | Visual SFDA as a whole | 2016–August 2026 | Knowledge budgets, generative priors, open deployment, and fair evaluation |

## 3. Problem formulation and boundaries

Let the labeled source domain be \(\mathcal D_s=\{(x_i^s,y_i^s)\}\). After source training, only a model \(f_s(\cdot;\theta_s)\) is released. During adaptation, the learner can access an unlabeled target set \(\mathcal D_t=\{x_j^t\}_{j=1}^{n_t}\) and seeks a target model \(f_t(\cdot;\theta_t)\). A general formulation is

\[
\theta_t=\mathcal A(\theta_s,\mathcal D_t,\mathcal K;\mathcal B),
\]

where \(\mathcal K\) denotes optional external knowledge—such as a pretrained visual encoder, VLM, VFM, MLLM, or diffusion model—and \(\mathcal B\) denotes information and computation budgets. Classical SFDA sets \(\mathcal K=\varnothing\); foundation-model-augmented SFDA permits frozen auxiliary models or trainable prompts. Explicitly reporting \(\mathcal K\) and \(\mathcal B\) is necessary: a ResNet-50-only method and a method invoking a billion-parameter MLLM obey the same source-data restriction but not the same resource or information conditions.

The relation between source and target label sets defines closed-set (\(\mathcal Y_s=\mathcal Y_t\)), partial-set (\(\mathcal Y_t\subset\mathcal Y_s\)), open-set (\(\mathcal Y_s\subset\mathcal Y_t\)), and universal/open-partial settings. The arrival process separates offline batch adaptation from online streams, continual domains, and class-incremental learning. Model access separates white-box, gray-box, and black-box SFDA. The number and ownership of models further yield multi-source-model, multi-target, and federated variants.

SFDA overlaps with, but is not identical to, test-time adaptation (TTA). SFDA commonly permits repeated passes over an unlabeled target training set; TTA often emphasizes deployment-time, online, small-batch, or instance-wise updates. Source-free domain generalization uses neither source samples nor target training observations after source training and is outside the core scope. We include online/TTA methods when their information conditions satisfy SFDA, but label their access protocol explicitly.

## 4. A four-axis taxonomy

### 4.1 Axis I: knowledge source

The first axis records what replaces the unavailable source supervision.

1. **Explicit source-model knowledge:** classifier weights, logits, intermediate features, attention, and teacher predictions.
2. **Implicit source statistics:** batch-normalization moments, parameter sensitivity, classifier geometry, and internal activations.
3. **Intrinsic target structure:** clusters, prototypes, neighborhood graphs, temporal consistency, augmentation consistency, and estimated class priors.
4. **Auxiliary discriminative pretraining:** ImageNet encoders, CLIP and other VLMs, SAM-like VFMs, and MLLMs.
5. **Auxiliary generative knowledge:** GANs, image translators, diffusion models, inversion procedures, and synthetic data.
6. **Distributed or human knowledge:** multiple source models, federated clients, sparse active labels, or expert constraints.

This axis exposes the central issue in SFDA: methods differ less in whether they are “source-free” than in what information substitutes for source samples.

### 4.2 Axis II: adaptation locus

- **Input space:** style transfer, Fourier processing, augmentation, pseudo-source or pseudo-target generation, and diffusion editing.
- **Feature space:** clustering, prototype alignment, neighborhood preservation, contrastive learning, optimal transport, and manifold regularization.
- **Output space:** entropy and information maximization, pseudo-labels, distillation, consistency, and uncertainty calibration.
- **Parameter space:** classifier freezing, selective updates, BN-statistic replacement, model perturbation, prompts, and adapters.
- **Semantic space:** textual prompts, vision–language alignment, multi-model agreement, MLLM supervision, and open-vocabulary mapping.

Most strong methods occupy several loci. The purpose of this axis is not to force exclusivity but to identify where each source of supervision acts.

### 4.3 Axis III: deployment protocol

The protocol includes offline versus online access, single versus continual domains, single versus multiple source models, centralized versus federated training, closed versus open label spaces, and full white-box versus API-only access. Similar algorithm names may therefore denote incomparable conditions. Offline methods can repeatedly traverse a complete target set and build global neighborhood banks; online methods cannot inspect future samples. VLM-assisted methods access knowledge encoded by external pretraining, whereas pure SFDA methods do not.

### 4.4 Axis IV: visual task

The task axis covers image classification; two- and three-dimensional object detection; semantic, panoramic, and medical segmentation; video recognition and segmentation; pose estimation; person re-identification; point-cloud understanding; image restoration; and remote sensing. A global cluster assumption used for classification does not transfer directly to dense prediction. Detection couples localization noise with foreground–background imbalance, video adds temporal dependence, and medical imaging requires anatomical plausibility, cross-modality adaptation, and safety-oriented validation.

### 4.5 Representative methods in the unified framework

| Method | Year | Main knowledge | Adaptation locus | Setting/task | Core idea and principal risk |
|---|---:|---|---|---|---|
| SHOT [1] | 2020 | source classifier + target clusters | feature/output | closed-set classification | freezes the classifier and maximizes information; relies on target clusters |
| USFDA [2] | 2020 | source model + target structure | feature/output | universal label space | handles domain and category shift; known/unknown ambiguity remains |
| Image Translation [3] | 2020 | BN statistics + translation | input | classification | maps target images toward source style; source approximation is limited |
| Domain Impression [4] | 2021 | source-model inversion | input/output | classification | synthesizes source observations; memory leakage and low diversity are possible |
| A2Net [5] | 2021 | source model + target self-supervision | feature/output | classification | adversarial inference, category contrast, and rotation; multi-module training |
| NRC [6] | 2021 | target neighborhoods | feature/output | classification | local consistency; needs a feature bank and may propagate errors |
| APG [7] | 2021 | source category prototypes | input/feature | classification | generates avatar prototypes; quality is bounded by the source model |
| Distribution Estimation [8] | 2022 | source weights/distribution estimate | feature | classification | reconstructs alignable source support; estimation errors transfer |
| AaD [10] | 2022 | target neighborhoods | feature/output | classification | attracts neighbors and disperses other samples; assumes neighborhood purity |
| BMD [11] | 2022 | multicentric target prototypes | feature/output | classification | class-balanced dynamic centers; adds prototype maintenance |
| TPDS [12] | 2023 | target prediction distribution | output | classification | searches a target prediction distribution; sensitive to search and priors |
| C-SFDA [13] | 2023 | curriculum pseudo-labels | output | efficient classification | progressively admits samples; depends on the initial ordering |
| ProxyMix [14] | 2023 | proxy source + mixup | input/feature | classification | class-balanced proxy observations; synthetic bias may persist |
| RGV [15] | 2025 | representativeness and variety | feature/output | classification | theoretical risk analysis and progressive learning; complex protocol |
| BN-SFDA [16] | 2025 | frozen-BN source + target model | parameter/feature | classification | co-training; depends on BN-based architectures |
| Co-learn++ [19] | 2024 | source + visual pretraining + CLIP | feature/semantic | multiple label settings | collaborative knowledge; larger information budget than pure SFDA |
| DIFO [20] | 2024 | frozen VLM | semantic/output | closed/partial classification | prompt customization and distillation; VLM guidance can be noisy |
| ReCLIP [21] | 2024 | CLIP vision–text space | feature/semantic | VLM adaptation | corrects cross-modal misalignment; updating large encoders is costly |
| ProDe [17] | 2025 | VLM proxy space | semantic/output | multiple SFDA settings | theoretically motivated proxy denoising; depends on proxy quality |
| DUET [22] | 2025 | task model + CLIP | output/semantic | classification | dual-view agreement labels; limited use of conflicts |
| VSFOT [23] | 2026 | VLM + source prototypes | feature/semantic | classification | semantic OT and bidirectional distillation; matching is costly/sensitive |
| RCL [24] | 2026 | multiple MLLMs | semantic/output | classification | reliability curriculum distillation; inference cost and instability |
| DM-SFDA [25] | 2024 | text-to-image diffusion | input | classification, preprint | generates pseudo-source data; cannot directly recover an unseen source |
| DPTM [26] | 2025 | latent diffusion + target references | input/output | classification | progressive pseudo-target editing; semantic fidelity and cost require audit |
| FreeDNA [27] | 2025 | diffusion-predictor noise statistics | internal/input | dense prediction | training-free noise alignment; specialized model class |
| SFDA-Seg [28] | 2021 | source model + pseudo-source support | input/output | segmentation | establishes the segmentation branch; pixel noise accumulates |
| FVP [31] | 2023 | Fourier visual prompts | input/parameter | medical segmentation | parameter-efficient; prompt transfer depends on modality |
| UPL-SFDA [32] | 2023 | uncertainty-aware labels | output | medical segmentation | stratifies uncertain samples; calibration is decisive |
| Tell2Adapt [33] | 2026 | VFM structural knowledge | semantic/output | multimodal medical segmentation | unified multi-target adaptation; depends on VFM applicability |
| IRG-SFOD [36] | 2023 | instance-relation graph | feature/output | object detection | models instance relations; pseudo-box and graph errors interact |
| SFDA-HPE [39] | 2023 | pose structure + contrast | feature/output | pose estimation | uses keypoint structure; task-specific |
| DTE [42] | 2025 | weight barcodes + OT | feature/output | open-set | distinguishes then exploits; unknown selection remains sensitive |
| FedWCA [45] | 2025 | multi-client target structure | parameter/federated | federated classification | weighted client clustering; communication and privacy need separate tests |
| Secure SFDA [47] | 2023 | compression + transfer | parameter | secure SFDA | suppresses source backdoors; not a general privacy guarantee |

## 5. Conventional discriminative SFDA

### 5.1 Information maximization and pseudo-label self-training

SHOT freezes the source classifier, minimizes conditional entropy while maximizing prediction diversity, and refines labels with target centroids [1]. Its lasting contribution is the view that classifier weights are compressed carriers of source category structure. Later methods improve pseudo-label generation, selection, curricula, and noise robustness. BMD uses class-balanced dynamic multi-centers [11], ProxyMix constructs a class-balanced proxy domain [14], and C-SFDA schedules samples through curriculum self-training [13]. Confidence-weighted and uncertainty-aware methods estimate which examples can safely supervise the target model.

Pseudo-labeling is simple, compatible with most architectures, and remains the optimization backbone of many recent approaches. Its weakness is confirmation bias. When the initial source model fails under a large shift, its wrong predictions become training targets and may be reinforced. Thresholding can reduce the immediate error but sacrifices coverage, often excluding minority classes and hard transfer directions.

### 5.2 Clustering, prototypes, and neighborhoods

NRC, AaD, and related approaches use relationships among target observations instead of relying only on isolated softmax scores [6,10]. Their common assumption is that examples from the same category form connected or compact regions in target representation space, and that neighbor relations can provide more stable supervision than a single forward pass. These methods have remained strong baselines on Office-Home and VisDA-C. However, memory banks and nearest-neighbor graphs are costly for large target sets and difficult to maintain in a stream. Neighborhoods can also cross real decision boundaries, propagating early mistakes through a graph.

Recent work revisits this geometry rather than treating Euclidean proximity as sufficient. Representativeness, intra-target generalization, and sample variety jointly affect risk [15], while feature-universe and topology-based methods expand the embedding support before propagating labels [18]. These analyses suggest that pseudo-label confidence and geometric coverage should be evaluated together.

### 5.3 Contrastive, self-supervised, and consistency learning

A2Net, historical contrastive learning, rotation-based objectives, and augmentation-consistency methods obtain supervision from transformations or sample relations [5]. Teacher–student training, weak-to-strong consistency, and class-conditional contrastive losses reduce dependence on one hard label. The central risk is that performance becomes sensitive to augmentation. If a transformation changes task semantics or does not resemble the actual domain shift, consistency regularization encourages the wrong invariance. Later work on augmentation-induced noise and feature-level consistency confirms that agreement alone is not evidence of correctness.

### 5.4 Model statistics, parameter constraints, and domain reconstruction

A second family approximates source support using batch-normalization statistics, classifier geometry, or source-model responses. Domain Impression, VDM-DA, distribution estimation, and frozen-BN co-training reconstruct source-like information in pixel, feature, or statistic space [4,8,16]. Image-translation and virtual-domain approaches generate pseudo-source observations and then reuse conventional UDA losses [3]. This recovers an explicit two-domain problem but relies on a lossy projection of the source distribution. Generated data may satisfy the source classifier while lacking the diversity or causal factors of the real source data.

### 5.5 Uncertainty, noisy-label learning, and selective adaptation

Because target labels are unavailable, reliability estimation determines whether errors accumulate. Existing approaches use entropy, margins, ensembles, Bayesian or evidential uncertainty, teacher stability, and risk–coverage selection. Label-calibration work has introduced Dirichlet evidence and calibrated softmax objectives. Yet an uncalibrated confidence is not a probability of correctness, and probabilities from different models are not directly comparable. Accuracy alone is therefore incomplete; calibration error, selective risk, coverage, and rejection behavior should accompany it.

## 6. Foundation-model-driven SFDA

### 6.1 From general visual pretraining to VLMs

Conventional pipelines often discard the original ImageNet-pretrained encoder after it has been specialized on the source. Co-learn and Co-learn++ bring a pretrained visual network and CLIP back into target adaptation to improve source-model pseudo-labels [19]. DIFO customizes a frozen VLM with unsupervised prompt learning and distills task-relevant multimodal knowledge into the target model [20]. ReCLIP instead adapts a VLM itself, addressing visual–textual misalignment through cross-modal self-training [21]. ProDe observes that VLM supervision may be inaccurate at an unknown rate and treats the VLM as a noisy proxy for a latent invariant space rather than an oracle [17].

This line of work changes the central question from “when should the source model be trusted?” to “how should trust be allocated between a task-specific source model and a general-purpose model?” DUET constructs labels from agreement between the task model and CLIP and routes uncertain examples to consistency learning [22]. Later work incorporates optimal transport, bidirectional distillation, and explicit treatment of conflicts [23]. VLMs provide semantic anchors that can be valuable under large appearance shifts, but their predictions depend on category names, prompts, language, pretraining coverage, and correlated cultural or visual bias.

### 6.2 VFMs, SAM, and MLLMs

For dense prediction, SAM-like VFMs can supply class-agnostic masks, boundaries, or shape priors. Medical segmentation studies use SAM to refine target pseudo-labels, and Tell2Adapt uses a VFM to generate and visually re-ground supervision across modalities and anatomical targets [33]. Detection methods can use foundation-model masks to focus feature learning on objects and reduce foreground–background imbalance. MLLMs offer richer semantic judgments but introduce instruction-following failures, stochastic outputs, latency, and high inference cost. Reliability-based curriculum learning therefore distills agreement among frozen MLLMs into a smaller target model rather than deploying the MLLMs directly [24].

Foundation-model papers should report at least five items: the exact model and version, the relevant pretraining-data scope when known, prompts and prompt-selection procedures, which foundation-model parameters are updated, and per-sample training and inference cost. Otherwise, a gain attributed to “SFDA” may primarily reflect a much larger external model.

## 7. Generative models and diffusion-based SFDA

### 7.1 From GANs and translation to diffusion

Generative SFDA attempts to recover missing data support. Early methods used GANs, inversion, batch-statistic matching, or image translation to generate source-like images [3,4]. Diffusion models add diversity and conditional control, shifting attention from reconstruction of one fixed pseudo-source distribution toward construction of intermediate distributions that support the target task.

Three meanings of “diffusion” must be kept separate: (i) Stable Diffusion, latent diffusion, and other generative denoising models; (ii) graph or label diffusion that propagates information over sample relations; and (iii) the denoising process inside a diffusion-based dense predictor. Their objects, losses, and resource costs are different.

### 7.2 Four roles of diffusion models

1. **Pseudo-source generation.** DM-SFDA uses text-to-image diffusion to synthesize source-like observations and then applies domain alignment [25]. The approach is limited by whether a generator can recover an unseen source distribution, and early versions remain preprints.
2. **Target-domain semantic editing.** DPTM separates reliable and unreliable target observations, uses latent diffusion to edit unreliable examples toward assigned classes while maintaining target appearance, and progressively reduces the discrepancy between pseudo-target and real target support [26]. This avoids fully guessing the inaccessible source domain.
3. **Structural or label completion.** In medical and remote-sensing segmentation, conditional diffusion can recover full masks from edges, sparse high-quality seeds, or anatomical structure, using denoising as a pseudo-label repair mechanism.
4. **Adaptation of diffusion predictors.** FreeDNA adapts diffusion-based dense prediction through domain-noise alignment [27]. Here diffusion is neither a data generator nor an external teacher; it is the task model being adapted.

### 7.3 Opportunities and risks

Diffusion models express appearance shifts that handcrafted augmentation may miss and allow text- or structure-conditioned control. They can also change category identity, reproduce biases in their training corpora, or memorize training-like images. Perceptual generation quality is not equivalent to adaptation utility. Evaluation should include semantic preservation, domain proximity, class coverage, diversity, privacy memorization, and computation per unit of downstream gain. Studies using Stable Diffusion should report the exact checkpoint and license, prompts and negative prompts, sampler, number of steps, random seeds, and any human filtering.

## 8. Extended settings

### 8.1 Open-set, partial-set, and universal SFDA

The closed-set assumption often fails after deployment. Universal SFDA, UMAD, and GLC/GLC++ jointly address domain and category shift, while DTE uses weight-barcode estimation and sparse label assignment to distinguish known and unknown samples [2,42,43]. A fundamental ambiguity remains: high entropy may indicate either a difficult known observation under domain shift or a genuinely unknown category. A single rejection threshold is rarely stable. Evaluation should jointly report known-class accuracy, unknown detection, H-score, open-set calibration, and threshold sensitivity.

Foundation models make open-vocabulary SFDA feasible but also blur the definition of “unknown.” A category unknown to the source classifier may have been well represented in CLIP pretraining. Studies must distinguish knowledge available in task labels, the source model, and the external model.

### 8.2 Online, continual, and class-incremental adaptation

Online SFDA cannot assume access to global target clusters; continual SFDA must additionally prevent forgetting across a sequence of domains. Existing studies use memory banks, dynamic teachers, visual prompts, or stability regularization. Class-incremental SFUDA combines new-category acquisition with domain shift [44]. Protocols should state whether replay is permitted, whether domain boundaries are known, whether the stream is temporally correlated, how many times each observation may be seen, and whether source-domain performance must be preserved.

### 8.3 Multi-source, multi-target, and federated SFDA

Multi-source-free adaptation receives multiple source models rather than multiple source datasets and must estimate each model's relevance to the target. Federated SFDA adds client heterogeneity, communication, and privacy constraints. FedWCA and FedSCAl use client clustering, weighted aggregation, or server–client prediction alignment to reduce client drift [45,46]. “Raw data never leave the client” should not be presented as a formal privacy guarantee; gradients and model updates can still leak information, and secure aggregation or differential privacy are separate mechanisms.

### 8.4 Black-box, active, secure, and unlearning variants

Black-box SFDA can observe source-model outputs but cannot inspect features or BN statistics. Active SFDA permits a small target-label budget and should be compared separately from fully unsupervised adaptation. Secure SFDA demonstrates that a malicious source owner can transfer a backdoor into the target model [47]. More recent work shows that an adapted model may preserve source-exclusive knowledge that should have been forgotten [48]. Source-free is consequently a data-access protocol, not a privacy, security, or compliance conclusion.

## 9. Analysis by visual task

| Task | Representative directions | Main difficulties | Recommended measures |
|---|---|---|---|
| Image classification | SHOT, NRC, AaD, DIFO, ProDe, DUET, DPTM | noisy labels, long tails, category shift | accuracy, macro-F1, ECE, risk–coverage |
| Semantic/medical segmentation | SFDA-Seg, AugCo, UPL-SFDA, FVP, Tell2Adapt | pixel correlation, boundaries, small structures, plausibility | mIoU, Dice, HD95/ASD, class-balanced scores |
| Object detection | Free Lunch, Overlook Style, IRG, dynamic teachers, VFM priors | coupled localization/classification noise, foreground imbalance | mAP, AP50/75, calibration, pseudo-box quality |
| Video recognition/segmentation | language–vision guidance, CleanAdapt, Co-STAR | temporal dependence, action semantics, online cost | top-1, mean class accuracy, temporal stability |
| 3D point clouds and detection | SF-UDA3D, point-cloud and city-scale segmentation | sparsity, sensor changes, geometric shift | mAP, mIoU, distance-stratified performance |
| Medical imaging | Fourier style mining, FVP, UPL-SFDA, SAM/VFM | modality gaps, structural constraints, clinical risk | Dice, HD95/ASD, sensitivity, external-center validation |
| Pose, re-identification, remote sensing | SFDA-HPE, ReID, remote-sensing detection/segmentation | structural priors, fine-grained identity, geographic shift | PCK, mAP/CMC, task-specific metrics |

These task branches show why a classification leaderboard cannot represent the field. Segmentation and detection have structured outputs; video and online systems cannot freely inspect a complete target set; and medical systems require external-center testing, structural checks, and clinically meaningful failure analysis [28–41].

## 10. Datasets, evaluation, and reproducibility

### 10.1 Common benchmarks

Classification commonly uses Office-31, Office-Home, VisDA-C, DomainNet/DomainNet-126, PACS, and ImageNet-C. Segmentation studies use GTA5→Cityscapes, SYNTHIA→Cityscapes, device-shifted medical datasets, and cross-modality medical datasets. Detection studies frequently use Cityscapes, Foggy Cityscapes, Sim10k, KITTI, and weather or sensor variants. Video adaptation uses UCF–HMDB, EPIC-Kitchens, and related action datasets.

These benchmarks have limitations. Office-31 is small and near saturation. Task averages on Office-Home and DomainNet hide difficult transfer directions. VisDA-C results sometimes mix overall and mean-per-class accuracy. Medical preprocessing, splits, and ethical access vary across studies. Most importantly, some work selects the best epoch or hyperparameters on labeled target test data, introducing hidden supervision.

### 10.2 Fair-comparison checklist

Every SFDA study should disclose:

1. source architecture, source-training procedure, and source-only target performance;
2. any ImageNet, CLIP, SAM, MLLM, diffusion, or other external knowledge;
3. whether target access is offline and multi-pass, online single-pass, or episodic;
4. whether model selection uses target labels and, if not, the unsupervised criterion;
5. all transfer directions, standard deviations, and at least three random seeds;
6. performance under class imbalance, unknown categories, and large shifts;
7. parameter count, training and inference cost, memory, energy where possible, and external API cost;
8. calibration, risk–coverage, failures, and negative transfer;
9. code, configurations, source weights, and prompts; and
10. licenses and privacy statements for data, pretrained models, and generators.

### 10.3 Why reported SOTA numbers should not simply be concatenated

Published results often differ in backbone, source training, split, prompt, pretrained model, and model-selection rule. Placing each paper's best number in a single table creates a false impression of a controlled comparison. We recommend stratified leaderboards: a source-model-only budget, general visual-pretraining budget, VLM/VFM budget, and generative-model budget. Within each stratum, target-label selection must be separated from genuinely label-free selection.

`[AUTHOR INPUT: if a quantitative meta-table is required, verify the original tables and code. Put only results with the same backbone, source model, data split, and selection rule in the main comparison. Mark all remaining values as “reported by the original paper” and avoid cross-protocol rankings.]`

## 11. Synthesis and critical discussion

### 11.1 What has actually driven progress?

SFDA improvements come from three broad sources: more reliable target supervision, richer external knowledge, and better optimization constraints. Conventional approaches reduce errors produced by one source model. Foundation-model methods add another semantic or structural view. Diffusion methods alter the available training support. These sources are complementary, and future systems will likely combine target geometry, external semantics, and generated observations. Meaningful ablations must nevertheless isolate their contribution and cost.

### 11.2 Foundation models do not remove confirmation bias

The benefit of a VLM depends on its error correlation with the source model. If both fail on the same difficult classes, agreement creates a confident error. If they disagree frequently, discarding all conflicts wastes information. The research problem should therefore shift from “use CLIP labels” toward sample-wise reliability estimation across multiple knowledge sources, including an explicit option to abstain. MLLM responses likewise cannot be treated as automatic ground truth.

### 11.3 Source-free does not mean privacy-preserving

Not transmitting source examples reduces direct exposure, but a source model may still permit membership inference, inversion, attribute inference, or backdoor transfer. Pseudo-source generation can reproduce training-like information. Unless an explicit privacy mechanism and attack evaluation are provided, papers should make the bounded claim that SFDA reduces source-data access rather than claiming privacy protection.

### 11.4 Generative models change the governance boundary

When a diffusion generator supplies external knowledge, its training data, copyright, and memorization risks enter the adaptation system. Generated images can be source-sample-free while still carrying broad external-data priors. A knowledge provenance ledger is therefore more informative than a binary source-free label.

## 12. Open problems and research directions

1. **Auditable information budgets.** Define levels from source-model-only adaptation to systems using multiple foundation or generative models, with external data and computation disclosed.
2. **Truly label-free model selection.** Develop unsupervised validation criteria that correlate reliably with target risk and remove the use of target labels for epoch or hyperparameter selection.
3. **From closed sets to open vocabularies.** Jointly address domain shift, label-set shift, ambiguous class names, and hierarchical labels instead of adding one generic unknown output.
4. **Reliable multi-model arbitration.** Estimate sample-wise reliability of source models, VLMs, VFMs, and MLLMs while modeling correlated errors, conflicts, and abstention.
5. **Semantic and privacy validation of generated data.** Evaluate class identity, coverage, memorization, licensing, and causal downstream contribution—not only visual fidelity.
6. **Continual open-world deployment.** Adapt under unknown domain boundaries, correlated streams, and arriving categories while controlling forgetting and enabling recovery.
7. **Security and machine unlearning.** Detect malicious source models and source-private knowledge leakage, and provide verifiable forgetting without source access.
8. **Task-level unification.** Seek principles that span classification, detection, segmentation, and video without ignoring the structure of each output.
9. **Long-tail performance, fairness, and calibration.** Report minority-class and subgroup or device performance so that mean accuracy does not conceal systematic failures.
10. **Resource-constrained SFDA.** Include energy, latency, memory, and API cost in the objective, and distill foundation-model knowledge for lightweight deployment.
11. **Theory of identifiability and external priors.** Characterize when source-model geometry, target clusters, and external priors are sufficient—and when adaptation is impossible or harmful.
12. **Sustainable benchmarks.** Build longitudinal benchmarks with time order, unknown categories, genuine privacy constraints, and fixed label-free selection protocols.

## 13. Conclusion

Visual SFDA has evolved from continued classifier training without source samples into a broad field spanning model-knowledge recovery, target-structure learning, foundation-model collaboration, generative data manipulation, and continual deployment. Pseudo-labeling, clustering, and consistency remain the optimization core of most systems; VLMs, VFMs, and MLLMs expand the available semantic evidence; and diffusion models bring controllable generation and task-model denoising into the adaptation loop. At the same time, external-model budgets, label-assisted model selection, privacy leakage, and computation make the term “source-free” insufficient to describe a real system. Progress should therefore be measured not only by mean accuracy on one benchmark, but by whether a method makes its knowledge provenance, reliability, cost, and failure boundaries auditable.

## References (first verified working list)

> Formal versions are preferred where available; preprints are explicitly marked. Before submission, the authors must normalize the reference style in Zotero/EndNote and verify every author list, volume, issue, page range, and DOI.

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

## Author actions required

1. `[AUTHOR INPUT: authors, affiliations, funding, contributions, and competing interests.]`
2. `[AUTHOR INPUT: select a journal and adapt length, abstract, highlights, graphical abstract, and citation style.]`
3. `[AUTHOR INPUT: complete the systematic-search counts and exclusion reasons.]`
4. `[AUTHOR INPUT: prepare at least four original figures: field timeline, four-axis taxonomy, foundation/diffusion knowledge flow, and fair-evaluation protocol.]`
5. `[AUTHOR INPUT: verify formal publication metadata for every 2025–2026 paper; do not present a preprint as a conference or journal paper.]`
6. `[AUTHOR INPUT: decide whether to reproduce selected methods. If not, label numerical comparisons as values reported by the original papers and do not rank incompatible protocols.]`
7. `[AUTHOR INPUT: disclose generative-AI writing assistance according to the target journal's policy and accept responsibility for every fact, citation, and argument.]`
