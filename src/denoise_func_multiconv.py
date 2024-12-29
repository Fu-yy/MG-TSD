import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleResidualBlock(nn.Module):
    """
    类似WaveNet的门控激活结构，用于多尺度分支。
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, 2*channels, kernel_size=3, padding=1)
        self.out_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: [B, channels, T]
        y = self.conv(x)
        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.out_conv(y)
        return (x + y) / math.sqrt(2)

class DownSampleBlock(nn.Module):
    """
    用 stride=2 的卷积进行下采样
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)  # [B, out_channels, T//2]

class UpSampleBlock(nn.Module):
    """
    反卷积(转置卷积) 或者 可以插值后卷积
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)  # [B, out_channels, T*2]

class DiffusionEmbedding(nn.Module):
    # 与之前相同
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(dim).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class MultiScaleEpsilonTheta(nn.Module):
    """
    多尺度结构示例:
      1) 输入 x 先卷积 -> residual blocks
      2) 下采样 -> residual blocks -> 下采样 -> residual blocks ...
      3) 再逐级上采样 -> residual blocks ...
      4) 最终输出
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        num_scales=2,
        residual_blocks_per_scale=2,
        diffusion_emb_dim=16,
        diffusion_hidden_dim=64,
        max_steps=500,
    ):
        super().__init__()

        self.diffusion_embedding = DiffusionEmbedding(diffusion_emb_dim, diffusion_hidden_dim, max_steps)
        self.time_proj = nn.Linear(diffusion_hidden_dim, base_channels)

        # 初始投影
        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # 下采样/上采样模块列表
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 每个尺度的 residual blocks
        self.down_resblocks = nn.ModuleList()
        self.up_resblocks = nn.ModuleList()

        ch = base_channels
        for s in range(num_scales):
            # 在down之前做 residual blocks
            res_list = nn.ModuleList([MultiScaleResidualBlock(ch) for _ in range(residual_blocks_per_scale)])
            self.down_resblocks.append(res_list)

            # downsample
            self.downs.append(DownSampleBlock(ch, ch*2))
            ch = ch * 2

        # 最底层再来几个res
        self.mid_res = nn.ModuleList([MultiScaleResidualBlock(ch) for _ in range(2)])

        for s in range(num_scales):
            # upsample
            self.ups.append(UpSampleBlock(ch, ch//2))
            ch = ch // 2
            # up之后做 residual blocks
            res_list = nn.ModuleList([MultiScaleResidualBlock(ch) for _ in range(residual_blocks_per_scale)])
            self.up_resblocks.append(res_list)

        # 最终输出
        self.out_conv = nn.Conv1d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: [B, 1, T]
        t: [B]
        """
        B, _, T = x.shape

        # 时间步嵌入
        t_emb = self.diffusion_embedding(t)  # [B, diffusion_hidden_dim]
        t_emb_map = self.time_proj(t_emb).unsqueeze(-1)  # [B, base_channels, 1]
        # broadcast到时序长度
        t_emb_map = t_emb_map.expand(-1, -1, T)

        # 输入投影
        h = self.in_conv(x)
        h = h + t_emb_map  # 融合时间步嵌入

        # encoder/downsample path
        skips = []
        for i, resblocks in enumerate(self.down_resblocks):
            for rb in resblocks:
                h = rb(h)
            skips.append(h)  # 保留特征图供上采样时融合
            h = self.downs[i](h)

        # mid
        for rb in self.mid_res:
            h = rb(h)

        # decoder/upsample path
        for i, resblocks in enumerate(self.up_resblocks):
            h = self.ups[i](h)
            # 和skip相加(或concat也可以)
            skip_h = skips[-(i+1)]
            # 需要考虑两者长度不一致时的crop/pad
            if h.shape[-1] != skip_h.shape[-1]:
                # 假设h更长(一般都更短, 但以防万一)
                min_len = min(h.shape[-1], skip_h.shape[-1])
                h = h[..., :min_len]
                skip_h = skip_h[..., :min_len]
            h = h + skip_h
            for rb in resblocks:
                h = rb(h)

        out = self.out_conv(h)
        return out
