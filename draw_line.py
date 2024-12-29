import torch
import torch.nn.functional as F
import torch.fft as fft
import matplotlib.pyplot as plt
from torch import nn


class Average_pool_upsampler(nn.Module):
    def __init__(self,kernel_size):
        super(Average_pool_upsampler,self)
        self.kernel_size=kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size,stride=kernel_size)
    def forward(self,x,upsample_mode='linear'):
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
            pooled, size=seq_len, mode=upsample_mode, align_corners=False
        )

        # (4) squeeze 回到 [seq_len]
        out = upsampled
        return out

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


def fourier_mask_subband(x: torch.Tensor, freq_range: tuple) -> torch.Tensor:
    """
    给定 2D 实数序列 x，做 rFFT，然后只保留 [freq_range[0], freq_range[1]) 的频率成分，
    最后 iFFT 回时域并返回。

    freq_range 以频谱下标来表示区间。例如 (0, 10) 表示只保留前 0~9 的频率 bin。
    """
    B, T, N = x.shape
    X_f = fft.rfft(x,dim=1)  # [freq_bins], freq_bins = floor(seq_len/2) + 1

    # 构造掩码
    mask = torch.zeros_like(X_f)
    start, end = freq_range
    end = min(end, T // 2 + 1)  # 防止越界
    start = min(start, T // 2 + 1)  # 防止越界
    mask[:,start:end,:] = 1.0

    X_f_masked = X_f * mask
    # irfft 需要指定 n=seq_len 才能恢复到原长度
    x_recover = fft.irfft(X_f_masked, n=T,dim=1)
    return x_recover


def extract_multiscale_signals(x: torch.Tensor, num_scales: int = 5,mask_begin_zero:bool = True) -> torch.Tensor:
    B, T, N = x.shape
    X_f = torch.fft.rfft(x, dim=1)  # [B, freq_bins, N]
    freq_bins = X_f.shape[1]  # = floor(T/2) + 1

    band_size = freq_bins // num_scales
    scale_signals = []
    start_idx = 0

    for i in range(num_scales):
        end_idx = start_idx + band_size
        if i == num_scales - 1:
            end_idx = freq_bins  # 最后一段包含剩余所有频率

        mask = torch.zeros_like(X_f)
        if mask_begin_zero:
            mask[:, 0:end_idx, :] = 1.0
        else:
            mask[:, start_idx:end_idx, :] = 1.0

        X_f_sub = X_f * mask
        x_sub = torch.fft.irfft(X_f_sub, n=T, dim=1).real

        scale_signals.append(x_sub)
        start_idx = end_idx

    return scale_signals


if __name__ == "__main__":
    # ======================
    # 1) 生成一个测试信号 (可替换成你的真实数据)
    # ======================
    torch.manual_seed(0)
    seq_len = 256
    # 人工构造一个混合波，含有多种频率成分
    t = torch.linspace(0, 2 * 3.14159, seq_len)
    # 低频 + 高频 + 噪声
    x_orig = 0.7 * torch.sin(2 * t) + 0.3 * torch.sin(15 * t) + 0.1 * torch.randn_like(t)

    # ======================
    # 2) 多种平均池化 + 上采样
    # ======================
    kernel_sizes = [2, 4, 8]
    pooled_upsampled_signals = []
    for k in kernel_sizes:
        x_pu = average_pool_and_upsample(x_orig, kernel_size=k, upsample_mode='linear')
        pooled_upsampled_signals.append(x_pu)

    # ======================
    # 3) 傅里叶掩码 (示例：截取不同范围的低频)
    #    这里简单把频域分为三段: [0, 10), [0, 20), [0, 40)
    #    表示只保留 0~9, 0~19, 0~39 这几段低频
    # ======================
    x_orig_torch = x_orig.unsqueeze(0).unsqueeze(-1)
    res_from_zero = extract_multiscale_signals(x_orig_torch,num_scales=3,mask_begin_zero=True)
    res_from_middle = extract_multiscale_signals(x_orig_torch,num_scales=3,mask_begin_zero=False)




    # freq_ranges = [(0, 10), (0, 20), (0, 40)]
    # fourier_masked_signals = []
    # for fr in res_from_zero:
    #     x_fm = fourier_mask_subband(x_orig, freq_range=fr)
    #     fourier_masked_signals.append(x_fm)
    # freq_ranges = [(0, 10), (11, 20), (21, 40)]
    # fourier_masked_signals_split = []
    # for fr in freq_ranges:
    #     x_fm = fourier_mask_subband(x_orig, freq_range=fr)
    #     fourier_masked_signals_split.append(x_fm)

    # ======================
    # 4) 画图对比
    # ======================
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # --- (a) 平均池化 + 上采样 对比 ---
    axes[0].plot(x_orig.detach().numpy(), label='Original', color='k', linestyle='--')
    for i, x_pu in enumerate(pooled_upsampled_signals):
        axes[0].plot(x_pu.detach().numpy(), label=f'Pool+Upsample k={kernel_sizes[i]}')
    axes[0].set_title("平均池化 + 上采样 不同 kernel_size 对比")
    axes[0].legend()
    axes[0].grid(True)

    # --- (b) 傅里叶掩码 对比 ---
    axes[1].plot(x_orig.detach().numpy(), label='Original', color='k', linestyle='--')
    for i, x_fm in enumerate(res_from_zero):
        x_fm = x_fm.squeeze(0).squeeze(-1)
        fr_str = f"{i}"
        axes[1].plot(x_fm.detach().numpy(), label=f'Fourier Mask freq={fr_str}')
    axes[1].set_title("傅里叶掩码 不同低频范围 对比")
    axes[1].legend()
    axes[1].grid(True)
    # --- (c) 傅里叶掩码 平均分 对比 ---
    axes[2].plot(x_orig.detach().numpy(), label='Original', color='k', linestyle='--')
    for i, x_fm in enumerate(res_from_middle):
        fr_str = f"{i}"
        x_fm = x_fm.squeeze(0).squeeze(-1)

        axes[2].plot(x_fm.detach().numpy(), label=f'Fourier Mask freq={fr_str}')
    axes[2].set_title("傅里叶掩码 不同低频范围 对比")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
