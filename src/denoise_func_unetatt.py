import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention1D(nn.Module):
    """
    自注意力模块 for 1D sequence: [B, C, T]
    将它 reshape 成 [B, T, C] 做多头注意力, 再 reshape回来
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv_proj = nn.Linear(channels, channels*3)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        # x shape: [B, C, T]
        B, C, T = x.shape
        # -> [B, T, C]
        x_t = x.permute(0, 2, 1)
        x_norm = self.norm(x_t)
        qkv = self.qkv_proj(x_norm)  # [B, T, 3C]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each [B, T, C]

        # 分多头
        head_dim = C // self.num_heads
        q = q.view(B, T, self.num_heads, head_dim).permute(0,2,1,3)  # [B,H,T,hdim]
        k = k.view(B, T, self.num_heads, head_dim).permute(0,2,1,3)
        v = v.view(B, T, self.num_heads, head_dim).permute(0,2,1,3)

        # 注意力
        attn_weights = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(head_dim)  # [B,H,T,T]
        attn = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,T,hdim]

        # -> [B,T,C]
        out = out.permute(0,2,1,3).contiguous().view(B, T, C)
        out = self.out_proj(out)

        # 残差
        x_out = x_t + out
        # -> [B, C, T]
        x_out = x_out.permute(0,2,1)
        return x_out

class UNetBlock(nn.Module):
    """
    简单的Conv1D残差Block + 自注意力
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.attn = SelfAttention1D(channels)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        # self.conv3 = nn.Conv1d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x: [B, channels, T]
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        # 残差
        x = x + h
        # 注意力
        x = self.attn(x)

        # x = self.conv3(x)
        return x

class DownBlock(nn.Module):
    """
    下采样: Conv stride=2 => 通道 from in_ch to out_ch => T 减半
    然后若干UNetBlock
    """
    def __init__(self, in_channels, out_channels, num_res=2):
        super().__init__()
        self.downsample = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=4, stride=2, padding=1)
        self.resblocks = nn.ModuleList([UNetBlock(out_channels) for _ in range(num_res)])

    def forward(self, x):
        x = self.downsample(x)  # shape: [B, out_channels, T//2]
        for rb in self.resblocks:
            x = rb(x)
        return x

class UpBlock(nn.Module):
    """
    上采样: ConvTranspose stride=2 => 通道 from in_channels to out_channels => T 翻倍
    然后cat对应 skip => 通道 = out_channels + skip_ch
    再过若干UNetBlock(...), 这里让 UNetBlock 的 channels = out_channels + skip_ch
    """
    def __init__(self, in_channels, out_channels, skip_channels, num_res=2):
        """
        Args:
            in_channels: 当前输入（底部）的通道
            out_channels: 卷积反转后期望的通道
            skip_channels: 要和 skip 拼接的通道
        """
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels,
                                           kernel_size=4, stride=2, padding=1)
        # 拼接后通道 = out_channels + skip_channels
        total_ch = out_channels + skip_channels
        self.resblocks = nn.ModuleList([UNetBlock(out_channels) for _ in range(num_res)])
        self.cat_conv = nn.Conv1d(total_ch,out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self, x, skip):
        # x: [B, in_channels, T_bottom]
        # skip: [B, skip_channels, T_skip]
        x = self.upsample(x)  # -> [B, out_channels, T*2]

        # 保证 T 对齐（理论上应相等，如果下采样和上采样都对称）
        if x.shape[-1] != skip.shape[-1]:
            # 如果有 1 步误差，可裁剪或补零
            min_len = min(x.shape[-1], skip.shape[-1])
            x = x[..., :min_len]
            skip = skip[..., :min_len]

        # 通道拼接
        x = torch.cat([skip, x], dim=1)  # [B, out_channels+skip_channels, T]
        x = self.cat_conv(x)
        # 再过若干残差块
        for rb in self.resblocks:
            x = rb(x)
        return x

class DiffusionEmbedding(nn.Module):
    """
    时间步嵌入 (与原版本相同)
    """
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]  # [B, 2*dim]
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


class UNetEpsilonTheta(nn.Module):
    """
    修正后的 1D U-Net + 自注意力 + 时间步嵌入
    确保 skip 和 upBlock 对齐，不出现强行裁剪
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        diffusion_emb_dim=16,
        diffusion_hidden_dim=64,
        max_steps=500,
    ):
        super().__init__()
        # 时间步嵌入
        self.diffusion_embedding = DiffusionEmbedding(
            diffusion_emb_dim, diffusion_hidden_dim, max_steps
        )
        self.time_proj = nn.Linear(diffusion_hidden_dim, base_channels)

        # 输入层
        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---------------------- DownBlocks -----------------------
        downs = []
        ch_in = base_channels
        skip_channels_list = []
        for mult in channel_mults:
            ch_out = base_channels * mult
            downs.append(DownBlock(ch_in, ch_out, num_res=num_res_blocks))
            skip_channels_list.append(ch_out)  # 记录下来供up时使用
            ch_in = ch_out
        self.downs = nn.ModuleList(downs)

        # 最底层(中间层)
        self.mid_block1 = UNetBlock(ch_in)
        self.mid_block2 = UNetBlock(ch_in)

        # ---------------------- UpBlocks -----------------------
        # 反向遍历 channel_mults
        ups = []
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            # skip 通道也应该是 out_ch
            skip_ch = out_ch
            # 这里 in_channels = 当前ch_in (最底层开始)
            # 第一次 up: in_channels = ch_in (底层), out_channels= out_ch
            up_block = UpBlock(in_channels=ch_in,
                               out_channels=out_ch,
                               skip_channels=skip_ch,
                               num_res=num_res_blocks)
            ups.append(up_block)
            ch_in = out_ch  # 下一轮输入通道 = 这轮的 out_channels
        self.ups = nn.ModuleList(ups)

        # 输出
        self.out_conv = nn.Conv1d(ch_in, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: [B, in_channels, T]
        t: [B] (diffusion steps)
        """
        B, _, T = x.shape

        # 1) 时间步嵌入
        t_emb = self.diffusion_embedding(t)  # [B, diffusion_hidden_dim]
        t_emb_map = self.time_proj(t_emb).unsqueeze(-1)  # [B, base_channels, 1]
        t_emb_map = t_emb_map.expand(-1, -1, T)          # broadcast到长度 T

        # 2) 输入卷积
        h = self.in_conv(x)  # [B, base_channels, T]
        h = h + t_emb_map

        # 3) 下采样
        skips = []
        for down_block in self.downs:
            h = down_block(h)
            skips.append(h)  # 每次下采样后的输出都进 skip

        # 4) 中间层
        h = self.mid_block1(h)
        h = self.mid_block2(h)

        # 5) 上采样 (反向使用 skip)
        for i, up_block in enumerate(self.ups):
            # 每次 pop 最后一个 skip
            skip_h = skips.pop()  # 对应最后一次 down 的输出
            h = up_block(h, skip_h)

        # 6) 输出卷积
        out = self.out_conv(h)
        return out


if __name__ == '__main__':
    # 测试一下
    model = UNetEpsilonTheta(
        in_channels=1,
        base_channels=32,
        channel_mults=[1, 2, 4],
        num_res_blocks=2,
        diffusion_emb_dim=16,
        diffusion_hidden_dim=64,
        max_steps=1000,
    )

    B = 4
    T = 128
    x = torch.randn(B, 1, T)      # 需要去噪的“时间序列”
    t = torch.randint(0, 1000, (B,))  # 扩散步
    out = model(x, t)            # [B, 1, T]

    print("Input shape:", x.shape)   # [4, 1, 128]
    print("Output shape:", out.shape)# [4, 1, 128]
