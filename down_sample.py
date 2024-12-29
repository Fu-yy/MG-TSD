import torch
import torch.nn as nn
import torch.optim as optim


class LearnableAggregator(nn.Module):
    """
    最小示例：可学习聚合层
    目标：将长序列 T 聚合成 K (K < T) 个“聚合中心（centroid）”，
         并根据到每个中心的距离做一个soft assignment。
    """

    def __init__(self, T, K):
        super(LearnableAggregator, self).__init__()
        # 可学习中心的索引(浮点数)，大小为K
        # 范围初始化到 [0, T-1] 之间
        self.centroids = nn.Parameter(torch.linspace(0, T - 1, steps=K), requires_grad=True)
        self.T = T
        self.K=K
        # 这里简化起见，只聚合时间维度，不聚合batch维度
        # 你可在实际中针对batch或者特征做更多设计

    def forward(self, x):
        """
        x: (B, T, d_model)
        Returns: (B, K, d_model)
        """
        B, T, D = x.shape

        # 计算每个time step到K个centroid的距离
        # step_idx: [0, 1, 2, ..., T-1]
        step_idx = torch.arange(T, device=x.device).view(1, T)  # (1, T)
        centroids = self.centroids.view(self.K, 1)  # (K, 1)

        # 广播计算距离
        # shape: (K, T)
        dist = (step_idx - centroids) ** 2  # 用平方距离做简单示例
        # softmax 让每个time step对K个聚合中心的归一化权重
        # 这里我们要对“每个time step”得到一个分配到K个中心的概率
        # 所以 softmax 应该在dim=0 或 dim=1 ？ 需谨慎
        # 在这里，我们想对 "K" 做 softmax 还是对 "T" 做 softmax?
        # 不同实现方式会带来不同含义
        # 这里演示：对K做 softmax => 每个time step被分配到 K 个centroid中的某一个(或多个)

        # 先把dist取负数，然后做softmax => 距离越小，概率越大
        attn = torch.softmax(-dist, dim=0)  # (K, T)

        # attn: (K, T) => (1, K, T)
        attn = attn.unsqueeze(0).repeat(B, 1, 1)  # (B, K, T)

        # x: (B, T, D) => (B, T, 1, D) => (B, 1, T, D)
        # 方便做批量矩阵乘
        x_expanded = x.unsqueeze(1)  # (B, 1, T, D)

        # (B, K, T) x (B, 1, T, D) => (B, K, D)
        # 先扩展 attn => (B, K, T, 1)
        attn_4d = attn.unsqueeze(-1)  # (B, K, T, 1)

        aggregated = (attn_4d * x_expanded).sum(dim=2)  # sum over T
        # => (B, K, D)

        return aggregated


class DynamicAggregationModel(nn.Module):
    """
    在序列上先做一次自注意力 (或者Conv等)，然后用LearnableAggregator将 T -> K，
    再做一次自注意力或FC，最后输出。
    """

    def __init__(self, seq_len, input_dim, d_model=32, K=8):
        super(DynamicAggregationModel, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        # 简单自注意力
        self.mha = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

        # 可学习聚合: T -> K
        self.aggregator = LearnableAggregator(seq_len, K)

        # 在聚合后序列上再做一次自注意力
        self.mha2 = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)

        # 输出
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        B, T, _ = x.shape

        # 1) 投影
        x = self.input_proj(x)  # (B, T, d_model)

        # 2) 自注意力
        attn_out, _ = self.mha(x, x, x)
        x = self.ln(x + attn_out)  # (B, T, d_model)

        # 3) 动态聚合 => (B, K, d_model)
        x_agg = self.aggregator(x)

        # 4) 在聚合后的序列上再做注意力
        attn_out2, _ = self.mha2(x_agg, x_agg, x_agg)
        x_agg = self.ln2(x_agg + attn_out2)  # (B, K, d_model)

        # 5) Pooling
        x_final = x_agg.mean(dim=1)  # (B, d_model)

        # 6) 输出
        out = self.fc_out(x_final)
        return out


if __name__ == "__main__":
    # --- 简单测试 ---
    batch_size, seq_len, nvar = 4, 20, 5
    model = DynamicAggregationModel(seq_len, nvar, d_model=32, K=5)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = torch.randn(batch_size, seq_len, nvar)
    y = torch.randn(batch_size, 1)

    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    print("[DynamicAggregation] Output shape:", pred.shape)  # (4, 1)
    print("Done DynamicAggregationModel test.\n")
