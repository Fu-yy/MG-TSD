import torch
import numpy as np
import matplotlib.pyplot as plt


def draw_chart(time_series):
    # 示例输入数据 (batchsize=2, seqlen=100, dim=1)
    batchsize, seqlen, dim = time_series.shape
    # time_series = torch.sin(torch.linspace(0, 2 * np.pi, seqlen)).unsqueeze(0).unsqueeze(-1)
    # time_series = time_series.repeat(batchsize, 1, dim)  # 模拟 batch 数据

    # 傅里叶变换
    fft_result = torch.fft.fft(time_series, dim=1)  # 对序列长度维度 (seqlen) 进行傅里叶变换
    real_part = fft_result.real  # 实部
    imag_part = fft_result.imag  # 虚部

    # 绘制图像
    batch_idx = 0  # 选择第一个 batch 进行绘图
    dim_idx = 0  # 选择第一个特征维度

    time_values = torch.linspace(0, 1, seqlen).numpy()
    frequencies = np.fft.fftfreq(seqlen, d=(time_values[1] - time_values[0]))

    # 原始数据
    plt.figure(figsize=(15, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_values, time_series[batch_idx, :, dim_idx].numpy(), label="Original Data")
    plt.title("Original Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()

    # 实部
    plt.subplot(3, 1, 2)
    plt.plot(frequencies, real_part[batch_idx, :, dim_idx].numpy(), label="Real Part")
    plt.title("Fourier Transform - Real Part")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    # 虚部
    plt.subplot(3, 1, 3)
    plt.plot(frequencies, imag_part[batch_idx, :, dim_idx].numpy(), label="Imaginary Part", color="orange")
    plt.title("Fourier Transform - Imaginary Part")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()