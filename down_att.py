import torch
import torch.nn as nn
import torch.optim as optim


class PoolAttentionBlock(nn.Module):
    """
    一个最小示例：对输入做pooling后，再做自注意力。
    """

    def __init__(self, d_model, num_heads=2, dropout=0.1):
        super(PoolAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自注意力
        attn_out, _ = self.mha(x, x, x)
        x = self.ln(x + attn_out)

        # 前馈
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class HierarchicalTransformer(nn.Module):
    """
    演示分层下采样 + 上采样 + 多粒度融合
    """

    def __init__(self, input_dim, d_model=32, num_heads=2):
        super(HierarchicalTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # 高分辨率注意力层
        self.high_res_block = PoolAttentionBlock(d_model, num_heads)

        # 低分辨率注意力层
        self.low_res_block = PoolAttentionBlock(d_model, num_heads)

        # 用于在上采样时，先对低分辨率特征做线性变换后与高分辨率特征融合
        self.fusion_linear = nn.Linear(d_model * 2, d_model)

        # 最终输出
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        B, T, _ = x.shape

        # 1) 投影
        x = self.input_proj(x)  # (B, T, d_model)

        # 2) 高分辨率注意力
        x_high = self.high_res_block(x)  # (B, T, d_model)

        # 3) 下采样, 假设 factor=2
        #    如果 T 是奇数，可自行处理，这里简单起见假定 T 是偶数
        factor = 2
        T_low = T // factor
        x_low = x_high.reshape(B, T_low, factor, -1).mean(dim=2)  # (B, T_low, d_model)

        # 4) 低分辨率注意力
        x_low = self.low_res_block(x_low)  # (B, T_low, d_model)

        # 5) 上采样 (最简单的方法：repeat或插值)
        #    这里演示 repeat
        x_low_upsampled = x_low.unsqueeze(2).repeat(1, 1, factor, 1).reshape(B, T_low * factor, -1)  # (B, T, d_model)

        # 6) 融合高低分辨率特征 (逐时刻拼接)
        #    如果尺寸不匹配，可以做插值，这里假设 factor* T_low == T
        fused = torch.cat([x_high, x_low_upsampled], dim=-1)  # (B, T, 2*d_model)
        fused = self.fusion_linear(fused)  # (B, T, d_model)

        # 7) 最终做一个pooling
        out = fused.mean(dim=1)  # (B, d_model)

        # 8) 输出
        out = self.fc_out(out)  # (B, 1)
        return out


if __name__ == "__main__":
    # --- 简单测试 ---
    batch_size, seq_len, nvar = 4, 16, 5
    model = HierarchicalTransformer(input_dim=nvar, d_model=32, num_heads=2)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    x = torch.randn(batch_size, seq_len, nvar)
    y = torch.randn(batch_size, 1)

    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    print("[HierarchicalTransformer] Output shape:", pred.shape)  # (4, 1)
    print("Done HierarchicalTransformer test.\n")
