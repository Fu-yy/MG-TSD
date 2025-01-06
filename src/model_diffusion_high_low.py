import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from inspect import isfunction


# =================================================
# 一些辅助函数
# =================================================

def default(val, d):
    return val if val is not None else (d() if isfunction(d) else d)

def extract(a, t, x_shape):
    """
    从 a (shape=[timesteps]) 中索引出对应时刻 t 的数值，
    并 reshape 成 x_shape 的广播形式 (批次维度适配)。
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    """
    产生噪声张量。如果 repeat=True，则只在 batch=1 采样一次，然后在 batch 方向复制。
    """
    if repeat:
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    else:
        return torch.randn(shape, device=device)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    参考 DDPM / Nichol 等论文的余弦调度函数
    https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


# =================================================
# 一个演示用的 "拉普拉斯" 分解函数
# 这里仅用 “三通道平分” 代替真正的高/中/低频拆分
# =================================================

def laplacian_decompose(x):
    """
    将 x 拆分成 (x_high, x_mid, x_low)，三者 shape 相同。
    这里仅以 channel 维度三等分为例。
    注意: 你需要根据实际的频域分解替换此函数。
    例如：wavelet, FFT, 或真正的 Laplacian pyramid 等。
    """
    B, C = x.shape[0], x.shape[1]
    if C % 3 != 0:
        raise ValueError("通道数必须能被3整除，这里只是示例拆分!")

    c_part = C // 3
    x_high = x[:, 0:c_part, ...]
    x_mid = x[:, c_part:2*c_part, ...]
    x_low = x[:, 2*c_part:3*c_part, ...]

    return x_high, x_mid, x_low

def laplacian_reconstruct(x_high, x_mid, x_low):
    """
    将 (x_high, x_mid, x_low) 拼回原维度。
    这里同样仅做三通道拼接。
    """
    return torch.cat([x_high, x_mid, x_low], dim=1)


# =================================================
# 构造多频段衰减因子 alpha^{(1)}_t, alpha^{(2)}_t, alpha^{(3)}_t
# 在不同时刻 t，频段衰减到 0 的快慢不同
# =================================================

def create_linear_alpha_schedule(num_timesteps, end_step):
    """
    在 [0, end_step] 区间内: alpha 从 1 线性降到 0
    在 (end_step, num_timesteps) 区间: alpha = 0
    """
    alpha = np.zeros(num_timesteps, dtype=np.float32)
    for i in range(num_timesteps):
        if i < end_step:
            alpha[i] = 1.0 - float(i) / float(end_step)  # 线性衰减
        else:
            alpha[i] = 0.0
    return alpha


# =================================================
# 主 Diffusion 类（带三频段 alpha 衰减）
# =================================================

class LaplacianGaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        input_size,      # 比如通道数*C, 空间维度等
        diff_steps=100,  # 总扩散步数
        beta_end=0.1,
        beta_schedule="linear",
        loss_type="l2",
        t_high=20,       # 高频衰减截止步
        t_mid=60         # 中频衰减截止步
    ):
        """
        参数含义:
          - denoise_fn: 你自定义的网络，用于预测噪声
          - input_size: (channels, height, width) 或者 (channels,seq_len) 等
          - diff_steps: 总扩散步数 T
          - beta_end: beta 的终值
          - beta_schedule: beta 调度类型 (linear, cosine, etc.)
          - loss_type: l1, l2, huber
          - t_high: 高频衰减在 step=t_high 时变为 0
          - t_mid:  中频衰减在 step=t_mid 时变为 0
        """
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size,)
        self.loss_type = loss_type

        # ====== 准备 betas ======
        if beta_schedule == "linear":
            betas = np.linspace(1e-4, beta_end, diff_steps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(diff_steps)
        else:
            raise NotImplementedError(f"Unknown beta_schedule: {beta_schedule}")
        self.num_timesteps = diff_steps
        betas = betas.astype(np.float32)

        # ====== 计算各种常用量 ======
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        # posterior q(x_{t-1} | x_t, x_0) 的参数
        # variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # 可能出现数值下溢，log 做下截断
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        # 均值系数
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

        # ====== 生成三条衰减 schedule: alpha_high, alpha_mid, alpha_low ======
        # t_high < t_mid < diff_steps
        alpha_high = create_linear_alpha_schedule(diff_steps, end_step=t_high)
        alpha_mid  = create_linear_alpha_schedule(diff_steps, end_step=t_mid)
        alpha_low  = create_linear_alpha_schedule(diff_steps, end_step=diff_steps)  # 低频一直到最后

        self.register_buffer("alpha_high", to_torch(alpha_high))  # shape=[diff_steps]
        self.register_buffer("alpha_mid",  to_torch(alpha_mid))
        self.register_buffer("alpha_low",  to_torch(alpha_low))

    @property
    def device(self):
        return self.betas.device

    # ---------------------------------------------
    # 前向 q(x_t | x_0) 的均值与方差
    # ---------------------------------------------
    def q_mean_variance(self, x_start, t):
        """
        针对普通DDPM (不考虑多频段时),
        q(x_t | x_0) = N(x_t; sqrt_alpha_cumprod * x_0,  (1-alpha_cumprod)*I )
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # ---------------------------------------------
    # 由网络预测的 noise 反推出 x_0
    # ---------------------------------------------
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # ---------------------------------------------
    # 后验分布 q(x_{t-1}| x_t, x_0)
    # ---------------------------------------------
    def q_posterior(self, x_start, x_t, t):
        """
        posterior_mean = ( coef1 * x_start + coef2 * x_t )
        posterior_variance = ...
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ---------------------------------------------
    # p_theta(x_{t-1} | x_t)
    # ---------------------------------------------
    def p_mean_variance(self, x, cond, t, clip_denoised):
        # 1) 由 denoise_fn 预测噪声
        noise_pred = self.denoise_fn(x, t, cond=cond)  # 你的网络需要实现 forward(x, t, cond=...)
        # 2) 反推 x_0
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        # 3) 算后验分布
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    # ---------------------------------------------
    # 单步采样 p(x_{t-1}|x_t)
    # ---------------------------------------------
    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # 当 t==0 时，不再添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # ---------------------------------------------
    # 反向采样循环
    # ---------------------------------------------
    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None):
        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)  # 从标准高斯开始
        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                img,
                cond=cond,
                t=torch.full((b,), i, device=device, dtype=torch.long),
            )
        return img

    # ---------------------------------------------
    # 整体采样接口
    # ---------------------------------------------
    @torch.no_grad()
    def sample(self, sample_shape, cond=None):
        """
        sample_shape: [B, C, ...]
        cond: 你的条件，若无则 None
        """
        img = self.p_sample_loop(sample_shape, cond=cond)
        return img

    # ---------------------------------------------
    # 关键：在前向加噪中，引入三频段衰减
    # ---------------------------------------------
    def q_sample_multi_freq(self, x_start, t, noise=None):
        """
        多频段的前向扩散:
            x_t = alpha^1_t * x0_high + alpha^2_t * x0_mid + alpha^3_t * x0_low + sigma_t * eps

        其中:
            sigma_t = sqrt(1 - alpha_cumprod[t])  (这里沿用普通DDPM的概念, 也可灵活改)
            alpha^i_t 来自 self.alpha_high, self.alpha_mid, self.alpha_low
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 1) 先做“拉普拉斯”拆分 (你可换成真正的频域拆分)
        x0_high, x0_mid, x0_low = laplacian_decompose(x_start)

        # 2) 取出对应时刻的 alpha_high[t], alpha_mid[t], alpha_low[t]
        alpha1_t = extract(self.alpha_high, t, x_start.shape)
        alpha2_t = extract(self.alpha_mid,  t, x_start.shape)
        alpha3_t = extract(self.alpha_low,  t, x_start.shape)

        # 3) 取出 sqrt(1 - alpha_cumprod[t]) 作为噪声系数
        #    这里为了和普通DDPM兼容, 我们就用 (1 - alphas_cumprod[t]) 做方差
        sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # 4) 多频段加权后叠加噪声
        x_t_high = alpha1_t * x0_high
        x_t_mid  = alpha2_t * x0_mid
        x_t_low  = alpha3_t * x0_low
        x_t = x_t_high + x_t_mid + x_t_low + sigma_t * noise

        return x_t

    # ---------------------------------------------
    # 训练时： 计算损失
    # ---------------------------------------------
    def p_losses(self, x_start, cond, t, noise=None):
        """
        标准DDPM中:
           x_noisy = q_sample(x_start, t) = sqrt_alpha_cumprod[t]*x_start + ...
           net预测的噪声要和真实 noise 一致 (MSE or L1)
        这里用多频段加噪 q_sample_multi_freq.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 1) 构造带噪声的 x_noisy
        x_noisy = self.q_sample_multi_freq(x_start, t, noise=noise)
        # 2) 网络预测噪声
        noise_recon = self.denoise_fn(x_noisy, t, cond=cond)
        # 3) 计算损失
        if self.loss_type == "l1":
            loss = F.l1_loss(noise_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise_recon, noise)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")
        return loss

    # ---------------------------------------------
    # 用于训练时计算 log_prob 或 loss
    # ---------------------------------------------
    def log_prob(self, x, cond=None):
        """
        x: [B, C, ...]
        cond: [B, ...], 条件可选
        这里简单地随机采样一个 t, 再计算 p_losses
        """
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        loss = self.p_losses(x, cond, t)
        return loss
