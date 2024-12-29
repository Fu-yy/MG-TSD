import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math


class DiffusionEmbedding(nn.Module):
    """
    与原版类似的时间步嵌入
    """
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
        return x  # [batch_size, proj_dim]

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [T,2*dim]
        return table


class FrequencyResidualBlock(nn.Module):
    """
    在频域进行卷积 + 残差。
    这里我们用2D卷积处理 (batch, channels=2, freq, time) 的复数谱(实部/虚部分开)。
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=(3,3), dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=tuple(k//2 for k in kernel_size),
            dilation=dilation
        )
        self.gate_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=tuple(k//2 for k in kernel_size),
            dilation=dilation
        )
        self.skip_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x):
        # x shape: [B, in_channels, F, T]
        y = self.conv(x)  # -> [B, hidden_channels, F, T]
        y = self.activation(y)
        y = self.gate_conv(y)  # -> [B, hidden_channels, F, T]
        # 简单门控: 分chunk或通道拆分也可以，这里用一个激活来模拟
        y = torch.sigmoid(y) * y
        skip = self.skip_conv(y)  # [B, in_channels, F, T]
        return (x + skip) / math.sqrt(2)


class FrequencyDomainEpsilonTheta(nn.Module):
    """
    频域EpsilonTheta: 先对输入x做STFT -> 2D卷积网络 -> iSTFT恢复到时域
    注意：这里仅仅是一个示例版本。可根据需要改变STFT参数、网络深度等。
    """
    def __init__(
        self,
        n_fft=64,
        hop_length=32,
        num_res_blocks=5,
        diffusion_emb_dim=16,
        diffusion_hidden_dim=64,
    ):
        """
        Args:
            n_fft (int): STFT窗长度
            hop_length (int): hop大小
            num_res_blocks (int): 频域残差块数量
            diffusion_emb_dim (int): 时间步嵌入维度
            diffusion_hidden_dim (int): 时间步嵌入映射后维度
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 时间步嵌入
        self.diffusion_embedding = DiffusionEmbedding(
            dim=diffusion_emb_dim,
            proj_dim=diffusion_hidden_dim
        )

        # 将时间步嵌入映射到频域网络的通道维度
        self.time_proj = nn.Linear(diffusion_hidden_dim, 8)

        # 频域网络, 输入通道=2 (实部+虚部)
        freq_blocks = []
        in_channels = 2 + 8  # concat时间步嵌入的通道->(B,8,F,T)后cat
        hidden_channels = 16
        freq_blocks.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        freq_blocks.append(nn.SiLU())
        for _ in range(num_res_blocks):
            freq_blocks.append(FrequencyResidualBlock(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels
            ))
        freq_blocks.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1))  # 回到 2通道(实+虚)
        self.freq_net = nn.Sequential(*freq_blocks)

    def forward(self, x, t):
        """
        x: [B, 1, T] 时间序列
        t: [B] 扩散时间步
        return: 去噪后的 [B, 1, T]
        """
        B, _, T = x.shape

        # 1) STFT
        #  x_stft_realimag shape: [batch, freq_bins, time_frames, 2]
        #  我们转成 [batch, 2, freq_bins, time_frames] 以符合 Conv2d (channels, H, W)
        x_stft_complex = []
        for b in range(B):
            stft_res = torchaudio.functional.stft(
                x[b, 0],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft, device=x.device),
                return_complex=True
            )  # shape: [freq_bins, time_frames]
            x_stft_complex.append(stft_res.unsqueeze(0))
        # -> [B, 1, freq_bins, time_frames], complex
        x_stft_complex = torch.cat(x_stft_complex, dim=0)

        # 分离实部虚部到2个通道
        x_stft_real = x_stft_complex.real
        x_stft_imag = x_stft_complex.imag
        # stack后 -> [B, 2, freq_bins, time_frames]
        x_freq = torch.stack([x_stft_real, x_stft_imag], dim=1)

        # 2) 时间步嵌入并与频域拼接
        t_emb = self.diffusion_embedding(t)  # [B, diffusion_hidden_dim]
        t_emb_map = self.time_proj(t_emb)    # [B, 8]
        # 扩展到频域同大小: broadcast到 (B,8,freq_bins,time_frames)
        t_emb_4d = t_emb_map.unsqueeze(-1).unsqueeze(-1)
        t_emb_4d = t_emb_4d.expand(-1, -1, x_freq.shape[2], x_freq.shape[3])

        # 拼接通道
        x_freq_in = torch.cat([x_freq, t_emb_4d], dim=1)  # [B, 2+8, F, T]

        # 3) 频域网络
        x_freq_out = self.freq_net(x_freq_in)  # [B, 2, F, T]

        # 4) iSTFT 回到时域
        #    先还原成复数 [B, F, T]
        out_real = x_freq_out[:, 0, :, :]
        out_imag = x_freq_out[:, 1, :, :]
        out_complex = torch.complex(out_real, out_imag)

        # 做逆变换, 需要逐样本处理
        x_denoised_list = []
        for b in range(B):
            istft_res = torchaudio.functional.istft(
                out_complex[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft, device=x.device),
                length=T  # 保证输出和原输入长度相同
            )
            x_denoised_list.append(istft_res.unsqueeze(0))  # [1, T]
        x_denoised = torch.stack(x_denoised_list, dim=0)  # [B, 1, T]

        return x_denoised
