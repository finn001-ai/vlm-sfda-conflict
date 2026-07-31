"""CT-DUET：以冲突互补标签补全 DUET 的正标签准入流程。"""

from src.methods.oh import plmatch


def train_target(cfg):
    """运行首轮 prior + 冲突负监督到一致正监督的 CT-DUET。

    主循环、CLIP 更新、monotonic pseudo-label memory、consistency 与 CLIP
    KL 均来自发布版 DUET。新增机制只作用于尚未准入的 task/CLIP 冲突样本。
    """
    return plmatch.train_target(
        cfg,
        first_cycle_prior=True,
        complementary_transition=True,
    )
