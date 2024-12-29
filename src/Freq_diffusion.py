import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import torch.nn.functional as F
import numpy as np

###############################################################################
# 1. 一些工具函数
###############################################################################
def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦退火版本的 beta schedule
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

def extract(a, t, x_shape):
    """
    从向量 a 中取出 t 对应位置的值，并 reshape 成 x_shape 便于广播
    a: [num_timesteps]
    t: [B] (time index)
    x_shape: 用于构造输出的形状
    """
    # t 的形状是 [B]
    batch_size = t.shape[0]
    out = a.gather(0, t)  # 取出对应 beta / alpha
    # 重塑形状: 在后面维度补 1 以便和 x 对应
    return out.reshape(batch_size, *([1]*(len(x_shape)-1)))
    # return out.reshape(batch_size, ([1]*(len(x_shape)-1)))


###############################################################################
# 2. 频域下的噪声预测网络
###############################################################################
class FreqDenoiseNet(nn.Module):
    """
    一个简单的频域噪声预测网络 (MLP) 示例：
      输入:  (B, T_freq, D, 2)  其中最后一维是(实部, 虚部)
      条件:  (B, cond_dim)      可选
    输出:  同形状 (B, T_freq, D, 2)，即对频域噪声的预测
    """
    def __init__(self, T_freq, D, cond_dim=0, hidden_dim=64):
        super().__init__()
        # 每个样本 flatten 后的维度 = T_freq * D * 2 (实部+虚部)
        in_dim = T_freq * D * 2
        if cond_dim > 0:
            in_dim += cond_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_freq * D * 2)
        )
        self.T_freq = T_freq
        self.D = D
        self.cond_dim = cond_dim

    def forward(self, x_freq, t, cond=None):
        """
        x_freq: (B, T_freq, D, 2)
        t:      (B,)  -> 这里仅演示，不拼到网络，或可做成 embedding
        cond:   (B, cond_dim) 可选附加信息
        """
        B, T_f, D, _ = x_freq.shape
        # Flatten: [B, T_freq, D, 2] -> [B, T_freq*D*2]
        x_flat = x_freq.view(B, -1)

        # 如果有 cond，就拼起来
        if cond is not None:
            # 简单拼接 (也可考虑更复杂的 cross-attention 等)
            x_input = torch.cat([x_flat, cond], dim=1)  # [B, T_freq*D*2 + cond_dim]
        else:
            x_input = x_flat

        # 前向传播
        out = self.net(x_input)  # [B, T_freq*D*2]
        # reshape 回原形状
        out = out.view(B, T_f, D, 2)
        return out


###############################################################################
# 3. 在频域里进行前向(加噪)和反向(去噪)的扩散
###############################################################################
class FrequencyDiffusion(nn.Module):
    """
    频域扩散模型：
      - 不继承任何已有 Diffusion 类，完全自写
      - 输入:  (B, T, D) 的时域序列
      - 内部:
         1) rfft -> (B, T_freq, D) 复数 -> (B, T_freq, D, 2) 实部+虚部
         2) 在频域执行 q_sample、p_sample 等操作
         3) 训练时用频域噪声预测损失
      - 采样:
         1) 在频域中从噪声反推
         2) irfft 回到时域
    """
    def __init__(
            self,
            seq_len,
            dim,
            denoise_net,
            diff_steps=100,
            beta_schedule="cosine",
            beta_end=0.1,
            loss_type="l2"
    ):
        super().__init__()
        self.seq_len = seq_len     # T
        self.dim = dim             # D
        self.denoise_net = denoise_net
        self.num_timesteps = diff_steps
        self.loss_type = loss_type

        # 1) 生成 betas
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(diff_steps)
        elif beta_schedule == "linear":
            betas = np.linspace(1e-4, beta_end, diff_steps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        betas = torch.from_numpy(betas).float()
        self.register_buffer("betas", betas)

        # 2) 计算 alphas 及其累乘
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu().numpy(), axis=0)
        alphas_cumprod = torch.from_numpy(alphas_cumprod).float()
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
        self.__scale = None

        # 注册成 buffer 以便在 GPU 上同步
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", 1. / torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1.))

        # 后验分布的一些系数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale
    ############################
    # 频域变换辅助函数
    ############################
    def time_to_freq(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] 实数时域
        返回: [B, T_freq, D, 2]  (实部+虚部)
        """
        # 对时间维 T 做 rfft
        # 结果形状: [B, T_freq, D] (复数)
        X_cplx = torch.fft.rfft(x, dim=1)
        # 将复数分为 实部 和 虚部
        X_real = X_cplx.real  # (B, T_freq, D)
        X_imag = X_cplx.imag  # (B, T_freq, D)
        # 拼在一起
        X_out = torch.stack([X_real, X_imag], dim=-1)  # (B, T_freq, D, 2)
        return X_out

    def freq_to_time(self, X: torch.Tensor,seq_len=1) -> torch.Tensor:
        """
        X: [B, T_freq, D, 2]  (实部+虚部)
        返回: [B, T, D] 实数时域
        """
        # 拆回复数
        X_real = X[..., 0]    # [B, T_freq, D]
        X_imag = X[..., 1]    # [B, T_freq, D]
        X_cplx = torch.complex(X_real, X_imag)  # [B, T_freq, D]
        # irfft 回到时域
        # x_time = torch.fft.irfft(X_cplx, n=self.seq_len, dim=1)  # [B, T, D]
        x_time = torch.fft.irfft(X_cplx, n=seq_len, dim=1)  # [B, T, D]
        return x_time

    ############################
    # 前向扩散 (q_sample)
    ############################
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise_freq=None):
        """
        在频域进行 q(x_t|x_0) = sqrt(alpha_cumprod)*X_0 + sqrt(1-alpha_cumprod)*noise
        x_start: [B, T, D]
        t: [B]
        noise_freq: [B, T_freq, D, 2]  (可选,若 None 则随机)
        返回: X_t (B, T_freq, D, 2)
        """
        # 1) 转到频域
        X_0 = self.time_to_freq(x_start)  # (B, T_freq, D, 2)

        # 2) 如果没给噪声，就随机生成
        if noise_freq is None:
            noise_freq = torch.randn_like(X_0)

        # 3) 取出 sqrt_alphas_cumprod[t], sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, X_0.shape)
        sqrt_1_m_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, X_0.shape)

        # 4) 加权得到 X_t
        X_t = X_0 * sqrt_alpha + noise_freq * sqrt_1_m_alpha
        return X_t

    ############################
    # 反向扩散 (p_sample)
    ############################
    def predict_x0_from_noise_freq(self, X_t, t, noise_freq):
        """
        根据 x_t, 噪声, 还原 x_0:
          x_0 = 1/sqrt(alpha_cumprod) * (x_t - sqrt(1/alpha_cumprod -1)* noise)
        """
        sqrt_rec = extract(self.sqrt_recip_alphas_cumprod, t, X_t.shape)
        sqrt_rec_m1 = extract(self.sqrt_recipm1_alphas_cumprod, t, X_t.shape)
        X_0 = sqrt_rec * X_t - sqrt_rec_m1 * noise_freq
        return X_0

    def p_mean_variance_freq(self, X_t, t, cond=None,seq_len=1):
        """
        计算后验分布的 mean, variance in freq domain
        """
        # 1) 用频域网络预测噪声
        noise_freq_pred = self.denoise_net(X_t, t, cond)  # (B, T_freq, D, 2)

        # 2) 拟合出 x_0
        X_0_recon = self.predict_x0_from_noise_freq(X_t, t, noise_freq_pred)

        # 先回到时域 clip，然后再转回频域 (防止幅度过大)
        x_0_time = self.freq_to_time(X_0_recon,seq_len)  # (B, T, D)
        x_0_time = torch.clamp(x_0_time, -1., 1.)
        X_0_recon_clipped = self.time_to_freq(x_0_time)  # (B, T_freq, D, 2)

        # 3) 计算后验均值
        coef1 = extract(self.posterior_mean_coef1, t, X_t.shape)  # (B,1,1,1)
        coef2 = extract(self.posterior_mean_coef2, t, X_t.shape)
        model_mean = coef1 * X_0_recon_clipped + coef2 * X_t

        # 4) 后验方差
        model_var = extract(self.posterior_variance, t, X_t.shape)
        model_log_var = extract(self.posterior_log_variance_clipped, t, X_t.shape)

        return model_mean, model_var, model_log_var, noise_freq_pred

    @torch.no_grad()
    def p_sample_freq(self, X_t, t, cond=None,seq_len=1):
        """
        X_{t-1} ~ N(model_mean, model_var)
        """
        model_mean, _, model_log_var, _ = self.p_mean_variance_freq(X_t, t, cond,seq_len)
        # 只有 t>0 时才加噪
        noise = torch.randn_like(X_t)
        nonzero_mask = (1 - (t == 0).float()).reshape(X_t.shape[0], 1, 1, 1)
        X_prev = model_mean + nonzero_mask * (0.5 * model_log_var).exp() * noise
        return X_prev



    def log_prob(self, x, cond, share_ratio: float, *args, **kwargs):
        B, T, _ = x.shape
        time = torch.randint(0, self.num_timesteps,
                             (B,), device=x.device).long()
        loss = self.p_losses(
            x_start=x, cond=cond, t=time,
        )

        return loss

    ############################
    # (训练) 损失函数
    ############################
    def p_losses(self, x_start, t, cond=None):
        """
        1) 在频域加噪
        2) 预测噪声
        3) MSE or L1
        """
        B = x_start.shape[0]
        X_0 = self.time_to_freq(x_start)  # (B, T_freq, D, 2)
        cond_freq = self.time_to_freq(cond)  # (B, T_freq, D, 2)
        noise_freq = torch.randn_like(X_0)
        # 前向扩散: X_t
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, X_0.shape)
        sqrt_1_m_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, X_0.shape)
        X_t = X_0 * sqrt_alpha + noise_freq * sqrt_1_m_alpha

        # 预测噪声
        noise_freq_pred = self.denoise_net(X_t, t, cond_freq)

        if self.loss_type == "l1":
            loss = F.l1_loss(noise_freq_pred, noise_freq)
        else:
            loss = F.mse_loss(noise_freq_pred, noise_freq)

        return loss

    ############################
    # 对外采样接口
    ############################
    @torch.no_grad()
    def sample(self, shape=None, cond=None,sample_shape=torch.Size(),share_ratio=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.dim,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape
        """
        shape: (B, T, D)
        return: (B, T, D) 生成后的时域序列
        """
        B, T, D = shape
        device = self.betas.device

        # 初始化频域噪声: X_T
        T_freq = (T // 2) + 1
        X_t = torch.randn(B, T_freq, D, 2, device=device)
        cond_freq = self.time_to_freq(cond)  # (B, T_freq, D, 2)
        for i in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), i, device=device, dtype=torch.long)
            X_t = self.p_sample_freq(X_t, t_batch, cond_freq,seq_len=T)

        # 频域 -> 时域
        x_0 = self.freq_to_time(X_t,seq_len=T)
        return x_0


###############################################################################
# 4. 训练与测试示例
###############################################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------
    # 超参数
    # --------------------------
    B = 2      # batch_size
    T = 8      # 序列长度 (大于 1)
    D = 3      # 每个时刻的特征维度 (大于 1)
    cond_dim = 5   # 条件向量维度 (可选，可设0表示无条件)
    diff_steps = 20
    beta_schedule = "cosine"
    loss_type = "l2"
    n_epochs = 3

    # --------------------------
    # 构造数据 & 条件
    # --------------------------
    # 假设数据范围在 [-1,1]
    x_data = torch.rand(B, T, D) * 2 - 1.0
    # 随机条件
    cond = torch.randn(B, cond_dim)

    x_data = x_data.to(device)
    cond = cond.to(device)

    # --------------------------
    # 定义频域网络 & 扩散模型
    # --------------------------
    T_freq = (T // 2) + 1  # rfft 之后的频率长度
    denoise_net = FreqDenoiseNet(T_freq=T_freq, D=D, cond_dim=cond_dim, hidden_dim=64).to(device)

    diffusion = FrequencyDiffusion(
        seq_len=T,
        dim=D,
        denoise_net=denoise_net,
        diff_steps=diff_steps,
        beta_schedule=beta_schedule,
        beta_end=0.2,
        loss_type=loss_type
    ).to(device)

    # --------------------------
    # 优化器
    # --------------------------
    optimizer = optim.Adam(diffusion.parameters(), lr=1e-3)

    # --------------------------
    # 训练循环
    # --------------------------
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # 随机抽取一个时间步 t
        t = torch.randint(0, diff_steps, (B,), device=device).long()
        loss = diffusion.p_losses(x_data, t, cond=cond)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}")

    # --------------------------
    # 采样测试
    # --------------------------
    with torch.no_grad():
        x_sampled = diffusion.sample(shape=(B, T, D), cond=cond)
    print("Sampled shape:", x_sampled.shape)
    print("Sampled data (first item):\n", x_sampled[0])


if __name__ == "__main__":
    main()
