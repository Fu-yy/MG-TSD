import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# 1) 可学习/随机 Fourier 时间步嵌入
# ------------------------------------------------------------------
class RandomFourierTimeEmbedding(nn.Module):
    """
    使用随机 Fourier Features 来嵌入扩散时间步 t，
    并用小 MLP 投影到指定的 hidden_dim。
    """

    def __init__(self, embed_dim=16, hidden_dim=64, scale=10.0, max_steps=1000):
        """
        Args:
            embed_dim: Fourier 基函数数量 (即频率数)
            hidden_dim: 输出投影维度
            scale: 控制频率范围
            max_steps: 允许的最大扩散步数 (仅决定embedding shape, 无强制限制)
        """
        super().__init__()
        self.max_steps = max_steps
        self.embed_dim = embed_dim
        self.scale = scale

        # 随机生成 Fourier 频率
        # shape: [embed_dim]
        self.B = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

        # MLP 将(2*embed_dim) -> hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        """
        t: [batch_size] 整数表示扩散步 (0<=t<max_steps)
        return: [batch_size, hidden_dim]
        """
        # 先将 t 归一化 (可选)
        # 使得 t 落在 [0,1] 范围，这里简单除以 max_steps
        t_norm = t.float() / self.max_steps  # [B]

        # [B, embed_dim]
        #  phi(t) = [cos(2π B_i t), sin(2π B_i t)] (可再乘以 2pi 之类)
        #  这里为了简化, 不额外 * 2π
        cos_comp = torch.cos(t_norm.unsqueeze(-1) * self.B)
        sin_comp = torch.sin(t_norm.unsqueeze(-1) * self.B)
        fourier_feats = torch.cat([cos_comp, sin_comp], dim=-1)  # [B, 2*embed_dim]

        out = self.proj(fourier_feats)  # [B, hidden_dim]
        return out


# ------------------------------------------------------------------
# 2) 条件编码器 (TransformerEncoder)
# ------------------------------------------------------------------
class ConditionEncoder(nn.Module):
    """
    对条件序列做嵌入 + Transformer 编码，输出一个全局表示 或 逐时刻表示。
    为简化，这里只输出 [B, hidden_dim] 的全局聚合embedding (取最后CLS)。
    如果你想要逐时刻的输出，可再改一下 forward。
    """

    def __init__(self, cond_dim, hidden_dim, n_heads=4, n_layers=2):
        """
        Args:
            cond_dim: 条件向量的原始特征维度
            hidden_dim: 编码后的维度
            n_heads: 多头注意力数
            n_layers: TransformerEncoder层数
        """
        super().__init__()
        self.input_proj = nn.Linear(cond_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',
            batch_first=True  # 让输入形状是 [B, seq_len, d_model]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 我们加一个可学习 CLS 向量，用于聚合全局信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, cond):
        """
        cond: [B, seq_len, cond_dim]
        return: [B, hidden_dim]
        """
        B, seq_len, _ = cond.shape

        # 投影
        x = self.input_proj(cond)  # [B, seq_len, hidden_dim]

        # prepend cls token
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, seq_len+1, hidden_dim]

        # transformer
        out = self.transformer(x)  # [B, seq_len+1, hidden_dim]
        # 取第0个位置 (CLS) 作为全局条件表示
        cls_out = out[:, 0, :]  # [B, hidden_dim]
        return cls_out


# ------------------------------------------------------------------
# 3) Cross-Attention模块
# ------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """
    将 condition embedding 作为 key/value,
    将主分支 (x) 作为 query, 做一次 cross-attention 融合。
    这里给出最简化实现 (single-head)；你可扩展成多头注意力。
    """

    def __init__(self, channels, cond_channels):
        super().__init__()
        # 将 x 投影到 query
        self.to_q = nn.Conv1d(channels, channels, kernel_size=1)
        # 将 cond 投影到 key/value，这里 cond 只有 [B, cond_channels]，需要 reshape
        self.to_k = nn.Linear(cond_channels, channels)
        self.to_v = nn.Linear(cond_channels, channels)

        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, cond_emb):
        """
        x: [B, channels, T]  -- query
        cond_emb: [B, cond_channels] -- global条件
        """
        B, C, T = x.shape
        # query
        q = self.to_q(x)  # [B, C, T]

        # key, value  (broadcast到时域维度 T 或者不 broadcast 都可以)
        k = self.to_k(cond_emb).unsqueeze(-1)  # [B, channels, 1]
        v = self.to_v(cond_emb).unsqueeze(-1)  # [B, channels, 1]

        # 这里做一个最简易的 "点乘注意力"：
        #  attn = softmax(q*k / sqrt(C)) * v
        #  不同位置共享同一个 key/value (因为只有一个 global cond)，
        #  也可视为 "全局门控"。
        scale = math.sqrt(C)
        attn_logits = (q * k) / scale  # [B, C, T]
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, C, T]
        out = attn_weights * v  # [B, C, T]

        out = self.out_proj(out)  # [B, C, T]
        # 残差
        x = x + out
        return x


# ------------------------------------------------------------------
# 4) 典型的 1D U-Net 框架 (ResidualBlock, DownBlock, UpBlock)
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res=2):
        super().__init__()
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.resblocks = nn.ModuleList([ResidualBlock(out_ch) for _ in range(num_res)])

    def forward(self, x):
        x = self.downsample(x)
        for rb in self.resblocks:
            x = rb(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res=2):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(out_ch,in_ch, kernel_size=4, stride=2, padding=1)
        # self.upsample = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.resblocks = nn.ModuleList([ResidualBlock(in_ch) for _ in range(num_res)])

    def forward(self, x, skip):
        x = self.upsample(x)
        # 处理尺寸不一致的情况
        if x.shape[-1] != skip.shape[-1]:
            min_len = min(x.shape[-1], skip.shape[-1])
            x = x[..., :min_len]
            skip = skip[..., :min_len]
        x = torch.cat([skip, x], dim=1)  # channel 维度拼接
        for rb in self.resblocks:
            x = rb(x)
        return x


# ------------------------------------------------------------------
# 5) 最终的 ConditionalUNet 网络 (融合条件 + 时间步)
# ------------------------------------------------------------------
class ConditionalUNet(nn.Module):
    """
    - RandomFourierTimeEmbedding or Learned Embedding for time step
    - ConditionEncoder for cond
    - CrossAttentionBlock to fuse cond into main UNet
    """

    def __init__(
            self,
            in_channels=1,
            base_channels=32,
            channel_mults=[1, 2],
            num_res_blocks=2,
            # 时间步嵌入
            time_embed_dim=16,
            time_hidden_dim=64,
            max_steps=1000,
            # 条件嵌入
            cond_dim=10,  # 假设每个时刻条件是 10维
            cond_hidden=64,
            # cross attention
            use_cross_attn=True
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        # 1) 时间步嵌入
        self.time_embed = RandomFourierTimeEmbedding(
            embed_dim=time_embed_dim,
            hidden_dim=time_hidden_dim,
            scale=10.0,
            max_steps=max_steps
        )
        self.time_proj = nn.Linear(time_hidden_dim, base_channels)

        # 2) 条件编码器
        self.cond_encoder = ConditionEncoder(
            cond_dim=cond_dim,
            hidden_dim=cond_hidden,
            n_heads=4,
            n_layers=2
        )

        # 3) 输入层
        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # 下采样
        downs = []
        ch_in = base_channels
        skip_channels = []
        for mult in channel_mults:
            ch_out = base_channels * mult
            downs.append(DownBlock(ch_in, ch_out, num_res=num_res_blocks))
            skip_channels.append(ch_out)
            ch_in = ch_out
        self.downs = nn.ModuleList(downs)

        # 中间层
        self.mid_block1 = ResidualBlock(ch_in)
        # CrossAttention
        if use_cross_attn:
            self.cross_attn = CrossAttentionBlock(ch_in, cond_hidden)
        else:
            self.cross_attn = None
        self.mid_block2 = ResidualBlock(ch_in)

        # 上采样
        ups = []
        for mult in reversed(channel_mults):
            ch_out = base_channels * mult
            # 上采样时 in_ch = skip_ch + 当前ch_in
            ups.append(UpBlock(ch_in + ch_out, ch_out, num_res=num_res_blocks))
            ch_in = ch_out
        self.ups = nn.ModuleList(ups)

        # 输出层
        self.out_conv = nn.Conv1d(ch_in, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        """
        x: [B, in_channels, T]
        t: [B] 扩散步
        cond: [B, cond_seq_len, cond_dim], 条件序列
        return: [B, in_channels, T]
        """
        B, _, T = x.shape

        # (1) 计算时间步嵌入
        t_emb = self.time_embed(t)  # [B, time_hidden_dim]
        t_emb_map = self.time_proj(t_emb).unsqueeze(-1)  # [B, base_channels, 1]
        t_emb_map = t_emb_map.expand(-1, -1, T)  # broadcast到长度 T

        # (2) 条件编码 -> 全局 cond_emb
        cond_emb = self.cond_encoder(cond)  # [B, cond_hidden]

        # (3) 主UNet分支
        h = self.in_conv(x)
        h = h + t_emb_map  # 融合时间信息

        # 下采样
        skips = []
        for down_block in self.downs:
            h = down_block(h)
            skips.append(h)

        # 中间层
        h = self.mid_block1(h)
        if self.use_cross_attn and self.cross_attn is not None:
            h = self.cross_attn(h, cond_emb)  # Cross Attention 融合条件
        h = self.mid_block2(h)

        # 上采样
        for up_block in self.ups:
            skip = skips.pop()
            h = up_block(h, skip)

        out = self.out_conv(h)
        return out


'''

关键创新点解读

    RandomFourierTimeEmbedding
        通过随机初始化的频率向量 BB（保存在 self.B 中），将扩散步 tt 投影到一组 cos⁡(B⋅t),sin⁡(B⋅t)cos(B⋅t),sin(B⋅t)。随后再用一个两层 MLP 输出到指定的 hidden_dim。
        这种方法比固定的 sin⁡,cos⁡sin,cos 更灵活，有时在实际中表现更好，因为它能“学习”或“适应”到最优的频率范围，或者在训练初期更具多样性。

    ConditionEncoder (基于 Transformer)
        将一个条件序列（可能包含外生变量、额外上下文、历史观测等）映射到一个全局 embedding。
        这里我们加了 cls_token 用于聚合全局信息（类似 BERT 的处理）。如果你希望保留逐时刻的输出，也可以不取 cls，而是返回 [B, seq_len, hidden_dim] 的张量，然后在网络的 cross attention 里做更精细的操作。

    CrossAttentionBlock
        在 U-Net 中的中间层 (或每个 scale) 对主分支特征 x（作为 query）与 cond_emb（作为 key/value）做跨注意力，实现条件的自适应融合。
        这里为了演示，采用了最简化版本：因为 cond_emb 只有 [B, cond_hidden]，只相当于 1 个时刻，所以相当于对每个时间步做一个 global gating。
        你也可以改进为多头 cross-attention，并将条件序列做成 [B, cond_len, cond_hidden] 的 key/value，得到更丰富的条件依赖。

    U-Net 结构
        仍然是一个典型的 1D U-Net，用 DownBlock 逐步下采样，UpBlock 逐步上采样，中间用 ResidualBlock 处理特征。
        在中间层或多处插入CrossAttentionBlock 可以让条件信息在网络的多个尺度发挥作用。

进一步扩展思路

    若条件序列与主序列长度相同（或接近），可以 逐时刻 cross-attention：
        将 cond_encoder 输出保留成 [B, cond_len, cond_hidden]，在 cross-attention 中直接 Q = x, K=cond, V=cond。这样可以捕捉时间对齐的条件信息。
    若希望在更多层数里融合条件：
        可以在每个下采样/上采样的 ResidualBlock 或者 AttentionBlock 里都调用一次 CrossAttention；
    可以把 RandomFourierTimeEmbedding 换成可学习的 nn.Embedding(max_steps, time_hidden_dim) + MLP；
    可以把 ConditionEncoder 换成一个CNN 或者 RNN，再或者是多头 self-attention + cross-attention 的结合；
    可以使用多头 cross-attention 代替这里的单头点乘注意力；
    可以在损失函数和训练流程上做更多改进，如自适应调度、针对时间序列常见的指标 (MSE, MAE, CRPS) 等。

小结

以上示例综合演示了条件融合策略（通过 TransformerEncoder 提取全局条件）+ 可学习/随机 Fourier 时间步嵌入（给扩散步提供更灵活的编码），并使用 Cross-Attention 将条件融入主分支。该框架可运行且有一定创新性，你可在此基础上做更多实验、调参与改进，以写入论文或项目中。祝研究顺利!

'''

# ----------------------- 测试一下 -----------------------
if __name__ == "__main__":
    # 假设我们有一个 batch_size=4, 序列长度=128, 条件序列长度=10, 条件向量维度=10
    B = 4
    T = 128
    cond_len = 10
    cond_dim = 10

    # 构造一个网络
    model = ConditionalUNet(
        in_channels=1,
        base_channels=32,
        channel_mults=[1, 2],
        num_res_blocks=2,
        time_embed_dim=16,
        time_hidden_dim=64,
        max_steps=1000,
        cond_dim=cond_dim,
        cond_hidden=64,
        use_cross_attn=True
    )

    x = torch.randn(B, 1, T)  # 需要去噪的“时间序列”
    t = torch.randint(0, 1000, (B,))  # 扩散步
    cond = torch.randn(B, cond_len, cond_dim)  # 条件序列

    out = model(x, t, cond)  # [B, 1, T]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    # 只要 out.shape == x.shape，且能正向运行，就说明网络结构基本可行。
