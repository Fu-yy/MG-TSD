import torch
from torch import nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class Average_pool_upsampler(nn.Module):
    def __init__(self,kernel_size,upsample_mode='linear'):
        super(Average_pool_upsampler,self).__init__()
        self.kernel_size=kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size)
        self.upsample_mode=upsample_mode
    def forward(self,x):
        """
            对 1D 序列 x 做平均池化，然后上采样回原始长度。
            假设 x.shape = [seq_len], 这里演示单通道情况。
            如果是多通道或批量，可扩展到 [B, C, seq_len] 再使用 F.avg_pool1d / F.interpolate.

            参数:
            -------
            x : torch.Tensor
                形状 [seq_len], 单通道 1D 信号
            kernel_size : int
                池化核大小
            upsample_mode : str
                上采样模式，'linear' 或 'nearest' 等

            返回:
            -------
            out : torch.Tensor
                形状与 x 相同 ([seq_len]) 的上采样后信号
            """
        B, seq_len, dim = x.shape

        # (1) 先 reshape 成 [1, 1, seq_len] 方便使用 F.avg_pool1d
        # x_3d = x.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, L=seq_len]
        x_3d = x
        # (2) 平均池化，这里 stride = kernel_size，做最大程度的下采样
        pooled = self.pool(x_3d)
        # pooled.shape = [1, 1, L_out], 其中 L_out = ceil(seq_len / kernel_size)

        # (3) 上采样回原始长度
        #     F.interpolate 输入格式是 [B, C, L], 输出大小用 size=(seq_len,)
        upsampled = F.interpolate(
            pooled, size=seq_len, mode=self.upsample_mode, align_corners=False
        )

        # (4) squeeze 回到 [seq_len]
        out = upsampled
        return out

class DonwSample_Fourier_high_low(nn.Module):
    def __init__(self,rate):
        super(DonwSample_Fourier_high_low, self).__init__()
        self.rate=rate

    def forward(self,x):
        """
            给定 2D 实数序列 x，做 rFFT，然后只保留 [freq_range[0], freq_range[1]) 的频率成分，
            最后 iFFT 回时域并返回。

            freq_range 以频谱下标来表示区间。例如 (0, 10) 表示只保留前 0~9 的频率 bin。
            """
        B, T, N = x.shape
        X_f = torch.fft.rfft(x, dim=1)  # [freq_bins], freq_bins = floor(seq_len/2) + 1

        # 构造掩码
        mask = torch.zeros_like(X_f,device=x.device)
        start, end = 0,int(self.rate * (T // 2 + 1))
        end = min(end, (T // 2) + 1)  # 防止越界
        start = min(start, (T // 2) + 1)  # 防止越界
        if end < 0:
            mask[:, end:, :] = 1.0

        else:
            mask[:, start:end, :] = 1.0
        mask_high = 1-mask
        X_f_masked = X_f * mask
        X_f_masked_high = X_f * mask
        # irfft 需要指定 n=seq_len 才能恢复到原长度
        x_recover = torch.fft.irfft(X_f_masked, n=T, dim=1)
        x_recover_high = torch.fft.irfft(X_f_masked_high, n=T, dim=1)
        return x_recover,x_recover_high


class DonwSample_Fourier(nn.Module):
    def __init__(self,rate):
        super(DonwSample_Fourier, self).__init__()
        self.rate=rate

    def forward(self,x):
        """
            给定 2D 实数序列 x，做 rFFT，然后只保留 [freq_range[0], freq_range[1]) 的频率成分，
            最后 iFFT 回时域并返回。

            freq_range 以频谱下标来表示区间。例如 (0, 10) 表示只保留前 0~9 的频率 bin。
            """
        B, T, N = x.shape
        X_f = torch.fft.rfft(x, dim=1)  # [freq_bins], freq_bins = floor(seq_len/2) + 1

        # 构造掩码
        mask = torch.zeros_like(X_f,device=x.device)
        start, end = 0,int(self.rate * (T // 2 + 1))
        end = min(end, (T // 2) + 1)  # 防止越界
        start = min(start, (T // 2) + 1)  # 防止越界
        if end < 0:
            mask[:, end:, :] = 1.0

        else:
            mask[:, start:end, :] = 1.0

        X_f_masked = X_f * mask
        # irfft 需要指定 n=seq_len 才能恢复到原长度
        x_recover = torch.fft.irfft(X_f_masked, n=T, dim=1)
        return x_recover


class DownSample_Fourier_Learnable(nn.Module):
    def __init__(self, initial_rate=0.2, temperature=10.0):
        """
        初始化可学习的傅里叶下采样模块。

        参数:
        - initial_rate (float): 初始的下采样比例，范围在 [0, 1]。
        - temperature (float): 控制 Sigmoid 函数的斜率，用于生成更接近二值的掩码。
        """
        super(DownSample_Fourier_Learnable, self).__init__()
        # 使用一个参数 theta，通过 Sigmoid 函数映射到 [0, 1] 作为 rate
        self.theta = nn.Parameter(torch.tensor(initial_rate))
        self.temperature = temperature  # 温度参数用于控制掩码的锐利度

    def forward(self, x):
        """
        前向传播，分离低频和高频部分。

        参数:
        - x (torch.Tensor): 输入数据，形状为 [B, T, N]

        返回:
        - x_low (torch.Tensor): 低频部分，形状为 [B, T, N]
        - x_high (torch.Tensor): 高频部分，形状为 [B, T, N]
        """
        B, T, N = x.shape
        X_f = torch.fft.rfft(x, dim=1)  # [B, freq_bins, N]
        freq_bins = X_f.shape[1]

        # 计算当前的 rate，确保在 [0, 1] 范围内
        rate = torch.sigmoid(self.theta)  # [1]

        # 创建频率索引
        freq_indices = torch.arange(freq_bins, device=x.device).float()

        # 计算掩码的中心点
        cutoff = rate * (freq_bins - 1)

        # 使用 Sigmoid 函数生成平滑掩码
        mask = torch.sigmoid((cutoff - freq_indices) * self.temperature)
        mask = mask.unsqueeze(-1)  # [freq_bins, 1]

        # 扩展掩码到批次和特征维度
        mask = mask.unsqueeze(0).expand(B, -1, N)  # [B, freq_bins, N]

        # 应用掩码
        X_f_masked = X_f * mask  # 低频逐渐保留，高频逐渐抑制

        # 逆傅里叶变换回时域
        x_low = torch.fft.irfft(X_f_masked, n=T, dim=1)
        x_high = x - x_low  # 高频部分

        return x_low, x_high
def fourier_mask_subband(x: torch.Tensor, freq_range: tuple) -> torch.Tensor:
    """
    给定 2D 实数序列 x，做 rFFT，然后只保留 [freq_range[0], freq_range[1]) 的频率成分，
    最后 iFFT 回时域并返回。

    freq_range 以频谱下标来表示区间。例如 (0, 10) 表示只保留前 0~9 的频率 bin。
    """
    B, T, N = x.shape
    X_f = torch.fft.rfft(x,dim=1)  # [freq_bins], freq_bins = floor(seq_len/2) + 1

    # 构造掩码
    mask = torch.zeros_like(X_f)
    start, end = freq_range
    end = min(end, T // 2 + 1)  # 防止越界
    start = min(start, T // 2 + 1)  # 防止越界
    mask[:,start:end,:] = 1.0

    X_f_masked = X_f * mask
    # irfft 需要指定 n=seq_len 才能恢复到原长度
    x_recover = torch.fft.irfft(X_f_masked, n=T,dim=1)
    return x_recover



def average_pool_and_upsample(x: torch.Tensor, kernel_size: int, upsample_mode='linear') -> torch.Tensor:
    """
    对 1D 序列 x 做平均池化，然后上采样回原始长度。
    假设 x.shape = [seq_len], 这里演示单通道情况。
    如果是多通道或批量，可扩展到 [B, C, seq_len] 再使用 F.avg_pool1d / F.interpolate.

    参数:
    -------
    x : torch.Tensor
        形状 [seq_len], 单通道 1D 信号
    kernel_size : int
        池化核大小
    upsample_mode : str
        上采样模式，'linear' 或 'nearest' 等

    返回:
    -------
    out : torch.Tensor
        形状与 x 相同 ([seq_len]) 的上采样后信号
    """
    B,seq_len,dim = x.shape

    # (1) 先 reshape 成 [1, 1, seq_len] 方便使用 F.avg_pool1d
    # x_3d = x.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, L=seq_len]
    x_3d = x
    # (2) 平均池化，这里 stride = kernel_size，做最大程度的下采样
    pooled = F.avg_pool1d(x_3d, kernel_size=kernel_size, stride=kernel_size)
    # pooled.shape = [1, 1, L_out], 其中 L_out = ceil(seq_len / kernel_size)

    # (3) 上采样回原始长度
    #     F.interpolate 输入格式是 [B, C, L], 输出大小用 size=(seq_len,)
    upsampled = F.interpolate(
        pooled, size=seq_len, mode=upsample_mode, align_corners=False
    )

    # (4) squeeze 回到 [seq_len]
    out = upsampled
    return out