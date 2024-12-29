from functools import partial
from inspect import isfunction

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# 需要安装: pip install torchdiffeq
from torchdiffeq import odeint  # 核心接口


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class OdeGaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            input_size,
            beta_end=0.1,
            diff_steps=100,
            loss_type="l2",
            betas=None,
            beta_schedule="linear",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.__scale = None

        # -- 确定 betas --
        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), ...
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # 注册缓冲
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """
        由 x_t、预测噪声 noise，反推 x_0 (即 x_start)
        """
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        """
        用网络的输出(预测噪声)反推 x_0，再返回 q(x_{t-1}|x_t,x_0) 的均值、方差
        """
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, cond=cond)
        )
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        """
        单步随机采样(传统离散反演)
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        """
        传统离散采样循环
        """
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )
        return img

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None):
        """
        原有的随机扩散采样接口
        """
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
        else:
            shape = sample_shape

        x_hat = self.p_sample_loop(shape, cond)
        if self.scale is not None:
            x_hat *= self.scale
        return x_hat

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        """
        插值示例
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散: q(x_t | x_0)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t, noise=None,*args, **kwargs):
        """
        用于训练时的 loss
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)

        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def log_prob(self, x, cond, *args, **kwargs):
        """
        在训练时被调用, 返回 loss
        """
        if self.scale is not None:
            x /= self.scale

        B, T, _ = x.shape
        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, *args, **kwargs
        )
        return loss

    # -----------------------------------------------------------
    # 下面开始: 用 torchdiffeq 求解 Probability Flow ODE
    # -----------------------------------------------------------

    def _continuous_beta(self, t_continuous):
        """
        将连续时间 t_continuous ∈ [0, 1] (或 [1, 0])
        映射到离散 steps [0, ..., self.num_timesteps - 1] 上的 beta。

        这里用 简单“最近邻”策略: round 后 clamp 到 [0, num_timesteps-1].
        你也可以改成线性插值, 做得更平滑.
        """
        steps_f = t_continuous * (self.num_timesteps - 1)
        step_i = torch.round(steps_f).long()
        step_i = torch.clamp(step_i, 0, self.num_timesteps - 1)
        return self.betas[step_i]

    class ProbabilityFlowFunc(nn.Module):
        """
        ODE函数: dx/dt = 0.5 * beta(t) * [ (predicted score) - x ]
        但是你的网络实际预测的是 "原噪声" epsilon,
        所以我们在这里补上 factor = 1 / sqrt(1 - alpha_cumprod[t])
        来将其转换为相应的 ( -score ).
        """

        def __init__(self, diffusion_model, cond):
            super().__init__()
            self.diff = diffusion_model
            self.cond = cond

        def forward(self, t, x):
            """
            t: shape (), scalar float in [0,1] (由 odeint 传入)
            x: shape (batch, dim, ...)
            """
            # ---------------------------
            # 1) 计算 beta(t)
            # ---------------------------
            beta_t = self.diff._continuous_beta(t)  # shape: scalar

            # ---------------------------
            # 2) 找到离散步 step_i 用于调用 denoise_fn
            # ---------------------------
            steps_f = t * (self.diff.num_timesteps - 1)
            step_i = torch.round(steps_f).long().clamp(0, self.diff.num_timesteps - 1)

            # ---------------------------
            # 3) 预测 "原始噪声" (network outputs e_\theta)
            # ---------------------------
            b = x.shape[0]
            step_i_batched = step_i.expand(b)
            pred_noise = -  self.diff.denoise_fn(x, step_i_batched, cond=self.cond)
            # pred_noise.shape == (batch, dim, ...)

            # ---------------------------
            # 4) 缩放 pred_noise => 变成 ODE 需要的项
            #    如果模型没对输出做过任何符号 / 缩放,
            #    则此处要做 factor = 1 / sqrt(1 - alpha_cumprod[t])
            # ---------------------------
            #   在 OdeGaussianDiffusion 里有 self.sqrt_one_minus_alphas_cumprod
            #   其 shape = [num_timesteps],  对应离散 steps
            #   我们先提取下:
            factor = 1.0 / extract(
                self.diff.sqrt_one_minus_alphas_cumprod,
                step_i_batched,  # discrete time steps
                x.shape  # 用于 reshape
            )
            # pred_noise * factor 即对噪声做了 1 / sqrt(1 - alpha_cumprod[t]) 的缩放
            #
            # 另外, 如果你对分母还需要负号(例如 score = - e_\theta / sigma),
            # 则相当于 dx/dt = 0.5 * beta_t * [ + e_\theta / sigma - x ].
            # 参照 VP-SDE derivation:
            #    dx/dt = -0.5 beta x + 0.5 beta * ( e_\theta / sqrt(...))
            #    = 0.5 beta [ e_\theta / sqrt(...) - x ]
            # 这个正负号是对齐主流公式的常见写法.

            # ---------------------------
            # 5) 计算漂移 drift = 0.5 * beta(t) * [ scaled_noise - x ]
            # ---------------------------
            drift = 0.5 * beta_t * (factor * pred_noise - x)

            return drift

    @torch.no_grad()
    def sample_ode_torchdiffeq(
            self,
            sample_shape=torch.Size(),
            cond=None,
            t_min=1.0,
            t_max=0.0,
            method="dopri5",
            rtol=1e-5,
            atol=1e-5
    ):
        """
        使用 torchdiffeq.odeint 来对 Probability Flow ODE 做数值求解。
        默认从 t=1 -> t=0, 也可反过来调 t_min, t_max.

        - sample_shape: (batch_size, time, ???) or (batch_size, ???),
                        具体得看你如何组织 x.
        - cond: 条件向量 / 序列 / ...
        - method: 可以是 "dopri5", "rk4", ...
        - rtol, atol: ODE 误差容忍度
        """
        device = self.betas.device
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
        else:
            shape = sample_shape

        # 1) 先从 高斯噪声开始 x(t=1)
        x_init = torch.randn(shape, device=device)

        # 2) 定义 ODE func
        ode_func = self.ProbabilityFlowFunc(self, cond)

        # 3) 定义时间边界 [t_min, t_max], 例如 [1, 0]
        t_span = torch.tensor([t_min, t_max], device=device, dtype=torch.float32)

        # 4) 调用 odeint 做积分
        #    返回 shape 为 (time_points, batch_size, ...)
        x_sol = odeint(
            ode_func,  # ODE函数
            x_init,  # 初值
            t_span,  # 时间张量 [start, end]
            method=method,
            rtol=rtol,
            atol=atol
        )

        # 5) 取最后时刻 x(t=0)
        x_0 = x_sol[-1]

        if self.scale is not None:
            x_0 = x_0 * self.scale
        return x_0
