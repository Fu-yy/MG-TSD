import numpy as np
import torch
import torch.nn.functional as F
import torch.fft as fft
import matplotlib.pyplot as plt
from torch import nn
import matplotlib as mpl

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


# 哪里需要显示中文就在哪里设置

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
def fourier_mask_subband_invert(x: torch.Tensor, freq_range: tuple) -> torch.Tensor:
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
    mask[:,start:,:] = 1.0

    X_f_masked = X_f * mask
    # irfft 需要指定 n=seq_len 才能恢复到原长度
    x_recover = fft.irfft(X_f_masked, n=T,dim=1)
    return x_recover


def gaussian_noise(shape, mean=0.0, std=1.0):
    """
    生成高斯噪声。

    参数:
    - shape: 噪声的形状 (如: [batch_size, num_features])。
    - mean: 噪声的均值，默认 0。
    - std: 噪声的标准差，默认 1。

    返回:
    - np.ndarray: 高斯噪声数组。
    """
    return np.random.normal(mean, std, shape)


def add_noise(data, t, alpha_t):
    """
    为数据添加高斯噪声。

    参数:
    - data: 原始数据 (x0)。
    - t: 时间步 (0 到 T)。
    - alpha_t: 时间相关的噪声强度参数。

    返回:
    - xt: 添加噪声后的数据。
    """
    noise = gaussian_noise(data.shape)
    xt = torch.sqrt(alpha_t) * data + torch.sqrt(1 - alpha_t) * noise
    return xt, noise
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



def plot_fourier_masked_signals(data_list, titles, x_orig, figsize=(10, 8), sharex=True,dpi=300,save_path='test.png'):
    """
    Plots multiple subplots for Fourier masked signals.

    Args:
        data_list (list): A list of lists, where each inner list contains the Fourier masked signals for one subplot.
        titles (list): A list of titles for each subplot.
        x_orig (torch.Tensor or numpy.ndarray): The original signal for comparison, should be 1D.
        figsize (tuple): Size of the figure. Default is (10, 8).
        sharex (bool): Whether to share the x-axis among subplots. Default is True.

    Returns:
        None
    """
    num_subplots = len(data_list)
    fig, axes = plt.subplots(num_subplots, 1, figsize=figsize, sharex=sharex)

    if num_subplots == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one subplot

    for idx, (signals, title) in enumerate(zip(data_list, titles)):
        axes[idx].plot(x_orig, label='Original', color='k', linestyle='--')
        for i, signal in enumerate(signals):
            signal = signal.squeeze(0).squeeze(-1)
            fr_str = f"{i}"
            axes[idx].plot(signal, label=f'Fourier Mask freq={fr_str}')
        axes[idx].set_title(title)
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')  # Save the figure as a high-resolution file
    print(f"Figure saved to {save_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np


    def visualize_masks(rate, temperature, freq_bins=100):
        freq_indices = np.arange(freq_bins)
        cutoff = rate * (freq_bins - 1)

        # 全一掩码
        all_one_mask = np.ones(freq_bins)

        # 平滑掩码
        smooth_mask = 1 / (1 + np.exp((cutoff - freq_indices) * temperature))

        plt.figure(figsize=(10, 4))
        plt.plot(freq_indices, all_one_mask, label='All-One Mask')
        plt.plot(freq_indices, smooth_mask, label='Smooth Mask', color='orange')
        plt.axvline(x=cutoff, color='green', linestyle='--', label='Cutoff Frequency')
        plt.xlabel('Frequency Index')
        plt.ylabel('Mask Value')
        plt.title('All-One Mask vs. Smooth Mask')
        plt.legend()
        plt.grid(True)
        plt.show()


    # 示例
    visualize_masks(rate=0.2, temperature=10.0)

    # ======================
    # 1) 生成一个测试信号 (可替换成你的真实数据)
    # ======================
    torch.manual_seed(0)
    seq_len = 128
    alpha=0.1
    # alpha=0.3
    # alpha=0.5
    # alpha=0.7
    # alpha=0.9

    # 人工构造一个混合波，含有多种频率成分
    t = torch.linspace(0, 2 * 3.14159, seq_len)
    # 低频 + 高频 + 噪声
    x_orig = 0.7 * torch.sin(2 * t) + 0.3 * torch.sin(15 * t) + 0.1 * torch.randn_like(t)

    # ======================
    # 2) 多种平均池化 + 上采样
    # ======================
    # kernel_sizes = [2, 4, 8]
    # pooled_upsampled_signals = []
    # for k in kernel_sizes:
    #     x_pu = average_pool_and_upsample(x_orig, kernel_size=k, upsample_mode='linear')
    #     pooled_upsampled_signals.append(x_pu)

    # ======================
    # 3) 傅里叶掩码 (示例：截取不同范围的低频)
    #    这里简单把频域分为三段: [0, 10), [0, 20), [0, 40)
    #    表示只保留 0~9, 0~19, 0~39 这几段低频
    # ======================
    # x_orig_torch = x_orig.unsqueeze(0).unsqueeze(-1)
    # res_from_zero = extract_multiscale_signals(x_orig_torch,num_scales=3,mask_begin_zero=True)
    # res_from_middle = extract_multiscale_signals(x_orig_torch,num_scales=3,mask_begin_zero=False)

    index_end = seq_len  //2 + 1
    index_end_int1  = int(index_end*0.1)
    index_end_int4  = int(index_end*0.4)
    index_end_int7  = int(index_end*0.7)
    index_end_int9  = int(index_end*0.9)

    x_orig_tensor = x_orig.unsqueeze(0).unsqueeze(-1)
    freq_ranges = [(0, index_end_int1), (0, index_end_int4), (0, index_end_int7), (0, index_end_int9)]
    freq_ranges = [(0, index_end_int1), (0, index_end_int9)]
    fourier_masked_signals = []
    for fr in freq_ranges:
        x_fm = fourier_mask_subband(x_orig_tensor, freq_range=fr)
        fourier_masked_signals.append(x_fm)


    freq_ranges = [(0, index_end_int1), (index_end_int1, index_end_int4), (index_end_int4, index_end_int7),(index_end_int7,index_end_int9)]
    freq_ranges = [(0, index_end_int1), (index_end_int4, index_end_int7)]
    fourier_masked_signals_split = []
    for fr in freq_ranges:
        x_fm = fourier_mask_subband(x_orig_tensor, freq_range=fr)

        fourier_masked_signals_split.append(x_fm)

    freq_ranges = [(-index_end_int1, 0), (-index_end_int4, 0), (-index_end_int7, 0), (-index_end_int9, 0)]
    freq_ranges = [(-index_end_int1, 0), (-index_end_int7, 0)]
    fourier_masked_signals_split_invert = []
    for fr in freq_ranges:
        x_fm = fourier_mask_subband_invert(x_orig_tensor, freq_range=fr)

        fourier_masked_signals_split_invert.append(x_fm)



    # ======================
    # 4) 画图对比
    # ======================

    data_list = [
        fourier_masked_signals,
        fourier_masked_signals_split,
        fourier_masked_signals_split_invert
    ]

    titles = [
        "傅里叶掩码 不同低频范围 对比",
        "傅里叶掩码 对比",
        "傅里叶掩码 平均分 对比"
    ]

    plot_fourier_masked_signals(data_list, titles, x_orig,dpi=600,save_path='01.长度为_'+str(seq_len)+'_原始数据分解后的，最上方是前百分之多少，中间的是分段，最下面的是后百分之多少.png')

    alpha_t = torch.tensor(alpha)
    # 添加噪声
    noisy_data, noise = add_noise(x_orig_tensor, t=0, alpha_t=alpha_t)

    list_01=[]
    list_01_sum=0
    list_01_noise=[]
    list_02=[]
    list_02_sum=0

    list_02_noise=[]
    list_03=[]
    list_03_sum=0

    list_03_noise=[]
    for fourier_masked_signals_item,fourier_masked_signals_split_item,fourier_masked_signals_split_invert_item in zip(fourier_masked_signals, fourier_masked_signals_split,fourier_masked_signals_split_invert):
        i01, noise01 = add_noise(fourier_masked_signals_item, t=0, alpha_t=alpha_t)
        i02, noise02 = add_noise(fourier_masked_signals_split_item, t=0, alpha_t=alpha_t)
        i03, noise03 = add_noise(fourier_masked_signals_split_invert_item, t=0, alpha_t=alpha_t)
        list_01.append(i01)
        list_01_noise.append(noise01)
        list_02.append(i02)
        list_02_noise.append(noise02)
        list_03.append(i03)
        list_03_noise.append(noise03)
        list_01_sum += i01
        list_02_sum += i02
        list_03_sum += i03

    titles_fourier_noise = [
        "傅里叶掩码 不同低频范围 对比 加噪后",
        "傅里叶掩码 对比  加噪后",
        "傅里叶掩码 平均分 对比  加噪后"
    ]

    data_list_fourier_noise=[list_01,list_02,list_03]
    noisy_data01 = noisy_data.squeeze(0).squeeze(-1)

    plot_fourier_masked_signals(data_list_fourier_noise, titles_fourier_noise, noisy_data01,dpi=600,save_path='02.长度为_'+str(seq_len)+'噪声alpha为'+str(alpha)+'_原始数据分解后的，最上方是前百分之多少，中间的是分段，最下面的是后百分之多少.png')


    noise_list=[list_01,list_02,list_03]
    titles_noise = [
        "傅里叶掩码 不同低频范围 对比 噪声",
        "傅里叶掩码 对比  噪声",
        "傅里叶掩码 平均分 对比  噪声"
    ]

    noise01 = noise.squeeze(0).squeeze(-1)

    plot_fourier_masked_signals(noise_list, titles_noise, noise01,dpi=600,save_path='03.长度为_'+str(seq_len)+'噪声alpha为'+str(alpha)+'_原始数据分解后的，最上方是前百分之多少，中间的是分段，最下面的是后百分之多少.png')

    titles_noise = [
        "相加",
    ]

    sum_noise = [[torch.stack(list_01,dim=-1).mean(-1),torch.stack(list_02,dim=-1).mean(-1),torch.stack(list_03,dim=-1).mean(-1)]]
    plot_fourier_masked_signals(sum_noise, titles_noise, noisy_data01,dpi=600,save_path='04.长度为_'+str(seq_len)+'噪声alpha为'+str(alpha)+'_原始数据分解后的，最上方是前百分之多少，中间的是分段，最下面的是后百分之多少,三部分噪声加和.png')






