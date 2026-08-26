"""
Minimal demo: Anchor Memory + Cross-Attention Verifier for SFDA.
最小演示：用于 SFDA 的锚点记忆与交叉注意力验证器。

What this script demonstrates:
1. A CNN produces one feature vector per image.
2. Each category keeps K=64 high-confidence anchor features.
3. Low-confidence/high-entropy samples are routed to a small Transformer-style
   cross-attention verifier.
4. Attention weights are summed by anchor label to obtain category scores.
5. If the verifier is still uncertain, the sample is rejected for this round.

这个脚本演示了：
1. CNN 为每张图像生成一个特征向量。
2. 每个类别保留 K=64 个高置信度锚点特征。
3. 低置信度/高熵样本被送往一个小型 Transformer 风格的交叉注意力验证器。
4. 根据锚点标签汇总注意力权重以获得类别分数。
5. 如果验证器仍然不确定，则本轮拒绝该样本。

Replace the synthetic feature generator with your own CNN features later.
请在稍后用您自己的 CNN 特征替换该合成特征生成器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def prediction_entropy(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalized entropy in [0, 1]. prob shape: [B, C].
    归一化熵，范围为 [0, 1]。prob 形状为 [B, C]。"""
    num_classes = prob.size(-1)  # 类别数
    entropy = -(prob * torch.log(prob.clamp_min(eps))).sum(dim=-1)  # 计算每个样本的熵
    return entropy / torch.log(torch.tensor(float(num_classes), device=prob.device))  # 归一化到 [0, 1]


@dataclass
class AnchorMemory:
    """
    Stores K anchor features for each class.

    features: [C, K, D]
    存储每个类别的 K 个锚点特征。
    """
    features: torch.Tensor

    @property
    def num_classes(self) -> int:
        return self.features.size(0)  # 返回类别数量

    @property
    def anchors_per_class(self) -> int:
        return self.features.size(1)  # 返回每类锚点个数

    @property
    def feature_dim(self) -> int:
        return self.features.size(2)  # 返回特征维度

    def flatten(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            flat_features: [C*K, D]
            flat_labels:   [C*K]
        返回：
            flat_features: [C*K, D]
            flat_labels:   [C*K]
        """
        c, k, d = self.features.shape  # 取出特征形状
        flat_features = self.features.reshape(c * k, d)  # 展平为 [C*K, D]
        flat_labels = torch.arange(c, device=self.features.device).repeat_interleave(k)  # 生成对应标签
        return flat_features, flat_labels

    @torch.no_grad()
    def update_class(self, class_id: int, new_features: torch.Tensor) -> None:
        """
        Simple FIFO update for one class.

        new_features: [N, D]
        简单的 FIFO 更新，用于单个类别。

        new_features: [N, D]
        """
        if new_features.ndim != 2:
            raise ValueError("new_features must have shape [N, D].")
        if new_features.size(1) != self.feature_dim:
            raise ValueError("Feature dimension mismatch.")
        if not (0 <= class_id < self.num_classes):
            raise ValueError("Invalid class_id.")

        old = self.features[class_id]
        merged = torch.cat([old, new_features], dim=0)
        self.features[class_id] = merged[-self.anchors_per_class:]


class CrossAttentionVerifier(nn.Module):
    """
    Query sample attends to all anchors.

    The model does NOT magically know the class.
    We sum attention weights according to each anchor's stored label.
    查询样本对所有锚点进行注意力计算。

    该模型并不“神奇地”知道类别。
    我们根据每个锚点的存储标签对注意力权重求和。
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_heads: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")

        self.num_classes = num_classes  # 保存类别数量

        self.query_norm = nn.LayerNorm(feature_dim)  # 对查询特征做归一化
        self.anchor_norm = nn.LayerNorm(feature_dim)  # 对锚点特征做归一化


        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )  # 多头交叉注意力模块

        # Optional learned refinement of the query after reading the memory.
        # 读取记忆后对查询进行可学习的精炼。
        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # A small learned class head. This is combined with attention-by-class.
        # 一个小型学习分类头，将与按类注意力信号结合。
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        query_features: torch.Tensor,
        anchor_features: torch.Tensor,
        anchor_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_features:  [B, D]
            anchor_features: [M, D]
            anchor_labels:   [M]

        Returns:
            class_prob:      [B, C]
            attention_by_class: [B, C]
        返回：
            class_prob:      [B, C]
            attention_by_class: [B, C]
        """
        if query_features.ndim != 2:
            raise ValueError("query_features must have shape [B, D].")
        if anchor_features.ndim != 2:
            raise ValueError("anchor_features must have shape [M, D].")
        if anchor_labels.ndim != 1:
            raise ValueError("anchor_labels must have shape [M].")

        batch_size = query_features.size(0)

        # [B, 1, D]
        query = self.query_norm(query_features).unsqueeze(1)

        # Repeat the same Anchor Memory for every query in the batch:
        # [B, M, D]
        # 将相同的锚点记忆复制到批次中每个查询上：
        # [B, M, D]
        memory = self.anchor_norm(anchor_features).unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # attn_weights: [B, 1, M]
        # 注意力权重： [B, 1, M]
        attended, attn_weights = self.cross_attention(
            query=query,
            key=memory,
            value=memory,
            need_weights=True,
            average_attn_weights=True,
        )

        refined_query = query.squeeze(1) + attended.squeeze(1)  # 将原始查询与注意力读出的信息相加
        refined_query = refined_query + self.ffn(refined_query)  # 通过FFN进一步精炼查询表示

        # Sum attention weights according to anchor class labels.
        # [B, M]
        # 根据锚点类别标签汇总注意力权重。
        sample_to_anchor = attn_weights.squeeze(1)

        # [B, C]
        # 初始化按类别的注意力权重累加张量
        attention_by_class = torch.zeros(
            batch_size,
            self.num_classes,
            device=query_features.device,
            dtype=query_features.dtype,
        )
        expanded_labels = anchor_labels.unsqueeze(0).expand(batch_size, -1)
        attention_by_class.scatter_add_(
            dim=1,
            index=expanded_labels,
            src=sample_to_anchor,
        )

        # Learned class logits from the refined query.
        # 从精炼后的查询中学习类别 logits。
        learned_logits = self.classifier(refined_query)

        # Convert attention mass to logits and combine the two signals.
        # 将注意力质量转换为 logits，并将两种信号组合。
        attention_logits = torch.log(attention_by_class.clamp_min(1e-8))
        final_logits = learned_logits + attention_logits  # 学习信号与注意力信号融合
        class_prob = F.softmax(final_logits, dim=-1)  # 归一化为类别概率

        return class_prob, attention_by_class


def make_synthetic_data(
    num_classes: int = 3,
    feature_dim: int = 32,
    anchors_per_class: int = 64,
    train_queries_per_class: int = 256,
    device: str = "cpu",
):
    """
    Creates clustered features to imitate CNN embeddings.
    创建聚类特征以模拟 CNN 嵌入。
    """
    torch.manual_seed(7)

    class_centers = F.normalize(
        torch.randn(num_classes, feature_dim, device=device), dim=-1
    )  # 随机生成类别中心，并归一化

    # High-confidence anchors: close to their class center.
    # 高置信度锚点：接近其类别中心。
    anchors = []
    train_queries = []
    train_labels = []

    for class_id in range(num_classes):
        anchor_feature = class_centers[class_id] + 0.15 * torch.randn(
            anchors_per_class, feature_dim, device=device
        )  # 生成该类别的锚点特征
        anchor_feature = F.normalize(anchor_feature, dim=-1)  # 归一化锚点向量
        anchors.append(anchor_feature)

        query_feature = class_centers[class_id] + 0.25 * torch.randn(
            train_queries_per_class, feature_dim, device=device
        )  # 生成该类别的查询特征
        query_feature = F.normalize(query_feature, dim=-1)  # 归一化查询向量
        train_queries.append(query_feature)

        train_labels.append(
            torch.full(
                (train_queries_per_class,),
                class_id,
                dtype=torch.long,
                device=device,
            )  # 为查询生成对应标签
        )

    memory = AnchorMemory(torch.stack(anchors, dim=0))  # 将锚点堆叠成 [C, K, D]
    train_queries = torch.cat(train_queries, dim=0)  # 将训练查询合并成 [C*Q, D]
    train_labels = torch.cat(train_labels, dim=0)  # 合并标签

    return class_centers, memory, train_queries, train_labels


def train_verifier(
    model: nn.Module,
    memory: AnchorMemory,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> None:
    """
    In real SFDA, train_queries/train_labels would come from reliable
    high-confidence pseudo-labeled target samples.
    在真实的 SFDA 中，train_queries/train_labels 应来自可靠的
    高置信度伪标签目标样本。
    """
    model.train()  # 切换到训练模式
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 优化器

    anchor_features, anchor_labels = memory.flatten()  # 将记忆展平为 [M, D] 和 [M]
    num_samples = train_queries.size(0)  # 训练样本总数

    for epoch in range(epochs):
        permutation = torch.randperm(num_samples, device=train_queries.device)  # 随机打乱样本顺序
        total_loss = 0.0

        for start in range(0, num_samples, batch_size):
            index = permutation[start : start + batch_size]
            query_batch = train_queries[index]  # 本批次查询特征
            label_batch = train_labels[index]  # 本批次标签

            prob, _ = model(
                query_features=query_batch,
                anchor_features=anchor_features,
                anchor_labels=anchor_labels,
            )  # 计算概率

            loss = F.nll_loss(
                torch.log(prob.clamp_min(1e-8)),
                label_batch,
            )  # 负对数似然损失

            optimizer.zero_grad(set_to_none=True)  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item() * query_batch.size(0)  # 累计损失

        if epoch in {0, 9, 19, 29, 39}:
            mean_loss = total_loss / num_samples
            print(f"Epoch {epoch + 1:02d} | loss = {mean_loss:.4f}")


@torch.no_grad()
def cnn_prediction(
    features: torch.Tensor,
    source_classifier_weights: torch.Tensor,
    temperature: float = 0.20,
) -> torch.Tensor:
    """
    A toy CNN classifier using cosine similarity.
    Replace this with softmax(cnn_classifier(cnn_features)).
    """
    features = F.normalize(features, dim=-1)  # 归一化输入特征
    weights = F.normalize(source_classifier_weights, dim=-1)  # 归一化分类器权重
    logits = features @ weights.T / temperature  # 余弦相似度除以温度
    return F.softmax(logits, dim=-1)  # 输出类别概率


@torch.no_grad()
def route_and_verify(
    cnn_prob: torch.Tensor,
    query_features: torch.Tensor,
    verifier: CrossAttentionVerifier,
    memory: AnchorMemory,
    cnn_confidence_threshold: float = 0.80,
    cnn_entropy_threshold: float = 0.45,
    verifier_confidence_threshold: float = 0.70,
):
    """
    Three-way decision:
    1. CNN high confidence -> accept CNN.
    2. CNN uncertain, verifier confident -> accept verifier.
    3. Both uncertain -> reject for this training round.
    三分决策：
    1. CNN 高置信度 -> 接受 CNN 结果。
    2. CNN 不确定且验证器置信度高 -> 接受验证器结果。
    3. 两者都不确定 -> 本轮拒绝。
    """
    cnn_confidence, cnn_label = cnn_prob.max(dim=-1)  # CNN 最高概率与对应标签
    cnn_entropy = prediction_entropy(cnn_prob)  # 计算 CNN 预测熵

    direct_mask = (
        (cnn_confidence >= cnn_confidence_threshold)
        & (cnn_entropy <= cnn_entropy_threshold)
    )  # 直接接受的样本掩码
    uncertain_mask = ~direct_mask  # 需要验证器处理的样本掩码

    final_label = torch.full_like(cnn_label, fill_value=-1)  # 初始化最终标签
    source = ["rejected"] * query_features.size(0)  # 默认决策来源为 rejected

    # Directly accept confident CNN predictions.
    # 直接接受 CNN 的高置信度预测。
    final_label[direct_mask] = cnn_label[direct_mask]
    for i in torch.where(direct_mask)[0].tolist():
        source[i] = "cnn"

    verifier_prob = torch.zeros_like(cnn_prob)  # 验证器概率占位
    attention_by_class = torch.zeros_like(cnn_prob)  # 注意力按类的占位

    # Only uncertain samples are sent to the verifier.
    # 只有不确定的样本才会被发送到验证器。
    if uncertain_mask.any():
        anchor_features, anchor_labels = memory.flatten()  # 展平锚点记忆用于验证器

        uncertain_prob, uncertain_attention = verifier(
            query_features=query_features[uncertain_mask],
            anchor_features=anchor_features,
            anchor_labels=anchor_labels,
        )  # 对不确定样本进行验证器推理

        verifier_prob[uncertain_mask] = uncertain_prob  # 保存验证器概率
        attention_by_class[uncertain_mask] = uncertain_attention  # 保存按类注意力

        verifier_confidence, verifier_label = uncertain_prob.max(dim=-1)  # 验证器置信度与标签
        accept_by_verifier = verifier_confidence >= verifier_confidence_threshold  # 验证器接收条件

        uncertain_indices = torch.where(uncertain_mask)[0]  # 不确定样本索引
        accepted_indices = uncertain_indices[accept_by_verifier]  # 验证器接受的索引

        final_label[accepted_indices] = verifier_label[accept_by_verifier]  # 使用验证器标签

        for i in accepted_indices.tolist():
            source[i] = "transformer"  # 决策来自 transformer 验证器

    return {
        "final_label": final_label,
        "decision_source": source,
        "cnn_prob": cnn_prob,
        "cnn_entropy": cnn_entropy,
        "verifier_prob": verifier_prob,
        "attention_by_class": attention_by_class,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    num_classes = 3  # 类别数量
    feature_dim = 32  # 特征维度
    anchors_per_class = 64  # 每个类别的锚点数量

    (
        class_centers,
        memory,
        train_queries,
        train_labels,
    ) = make_synthetic_data(
        num_classes=num_classes,
        feature_dim=feature_dim,
        anchors_per_class=anchors_per_class,
        device=device,
    )  # 生成合成数据和锚点记忆

    verifier = CrossAttentionVerifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_heads=4,
        hidden_dim=64,
    ).to(device)

    train_verifier(
        model=verifier,
        memory=memory,
        train_queries=train_queries,
        train_labels=train_labels,
    )

    verifier.eval()  # 切换到评估模式，不计算梯度

    # Simulate 4 target samples:
    # sample 0: clearly class 0
    # sample 1: mixture of class 0 and class 1
    # sample 2: clearly class 2
    # sample 3: highly ambiguous mixture of all classes
    # 样本 0：明显属于类别 0
    # 样本 1：类别 0 和 类别 1 的混合
    # 样本 2：明显属于类别 2
    # 样本 3：高度模糊，混合所有类别
    test_features = torch.stack(
        [
            class_centers[0] + 0.10 * torch.randn(feature_dim, device=device),
            0.55 * class_centers[0]
            + 0.45 * class_centers[1]
            + 0.20 * torch.randn(feature_dim, device=device),
            class_centers[2] + 0.12 * torch.randn(feature_dim, device=device),
            class_centers.mean(dim=0)
            + 0.40 * torch.randn(feature_dim, device=device),
        ],
        dim=0,
    )
    test_features = F.normalize(test_features, dim=-1)

    # Toy source classifier weights.
    # In your project, use the real CNN classifier output instead.
    source_classifier_weights = class_centers + 0.10 * torch.randn_like(class_centers)

    cnn_prob = cnn_prediction(
        features=test_features,
        source_classifier_weights=source_classifier_weights,
    )

    result = route_and_verify(
        cnn_prob=cnn_prob,
        query_features=test_features,
        verifier=verifier,
        memory=memory,
        cnn_confidence_threshold=0.80,
        cnn_entropy_threshold=0.45,
        verifier_confidence_threshold=0.70,
    )

    print("\nClass names: 0=cat, 1=dog, 2=bird\n")

    for i in range(test_features.size(0)):
        print(f"Sample {i}")
        print("  CNN probability:       ", result["cnn_prob"][i].cpu().numpy().round(3))
        print("  CNN entropy:           ", round(result["cnn_entropy"][i].item(), 3))
        print(
            "  Attention by category:",
            result["attention_by_class"][i].cpu().numpy().round(3),
        )
        print(
            "  Verifier probability: ",
            result["verifier_prob"][i].cpu().numpy().round(3),
        )
        print("  Final label:           ", result["final_label"][i].item())
        print("  Decision source:       ", result["decision_source"][i])
        print()


if __name__ == "__main__":
    main()