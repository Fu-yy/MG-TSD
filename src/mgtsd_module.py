from inspect import isfunction
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F

from gluonts.core.component import validated
from gluonts.torch.modules.distribution_output import DistributionOutput
# from gluonts.torch.distributions.distribution_output import DistributionOutput
from typing import Tuple
import copy

from torchdiffeq import odeint

"""
diffusion models
"""


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    def noise(): return torch.randn(shape, device=device)
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


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        input_size,
        # control the diffusion and sampling(reverse diffusion) procedures
        share_ratio_list,
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
        self.share_ratio_list = share_ratio_list  # ratio of betas are shared

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
                betas = np.linspace(1e-4 ** 0.5, beta_end **
                                    0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1)
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("diff_steps", to_torch(diff_steps))

        for cur_share_ratio in self.share_ratio_list:
            start_index = int(len(betas)*(1-cur_share_ratio))
            betas_sub = copy.deepcopy(betas)
            betas_sub[:start_index] = 0  # share the latter part of the betas
            alphas_sub = 1.0 - betas_sub
            alphas_cumprod_sub = np.cumprod(alphas_sub, axis=0)
            alphas_cumprod_prev_sub = np.append(1.0, alphas_cumprod_sub[:-1])
            suffix = int(cur_share_ratio * 100)
            self.register_buffer(
                f"alphas_cumprod_{suffix}", to_torch(alphas_cumprod_sub))
            self.register_buffer(
                f"alphas_cumprod_prev_{suffix}", to_torch(alphas_cumprod_prev_sub))

            self.register_buffer(f"sqrt_alphas_cumprod_{suffix}", to_torch(
                np.sqrt(alphas_cumprod_sub)))
            self.register_buffer(
                f"sqrt_one_minus_alphas_cumprod_{suffix}", to_torch(
                    np.sqrt(1.0 - alphas_cumprod_sub))
            )
            self.register_buffer(
                f"log_one_minus_alphas_cumprod_{suffix}", to_torch(
                    np.log(1.0 - alphas_cumprod_sub))
            )
            self.register_buffer(
                f"sqrt_recip_alphas_cumprod_{suffix}", to_torch(
                    np.sqrt(1.0 / alphas_cumprod_sub))
            )
            self.register_buffer(
                f"sqrt_recipm1_alphas_cumprod_{suffix}", to_torch(
                    np.sqrt(1.0 / alphas_cumprod_sub - 1))
            )
            self.register_buffer(

                f"posterior_mean_coef1_{suffix}",
                to_torch(betas_sub * np.sqrt(alphas_cumprod_prev_sub) / (1.0 - alphas_cumprod_sub)),)

            self.register_buffer(
                f"posterior_mean_coef2_{suffix}",

                to_torch(
                    (1.0 - alphas_cumprod_prev_sub) *
                    np.sqrt(alphas_sub) / (1.0 - alphas_cumprod_sub)

                ),)
            posterior_variance_sub = (
                betas_sub * (1.0 - alphas_cumprod_prev_sub) / (1.0 - alphas_cumprod_sub))
            self.register_buffer(
                f"posterior_variance_{suffix}", to_torch(posterior_variance_sub))

            self.register_buffer(
                f"posterior_log_variance_clipped_{suffix}",
                to_torch(np.log(np.maximum(posterior_variance_sub, 1e-20))),)

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def q_mean_variance(self, x_start, t, share_ratio: float):
        # get q(x_t|x_0) distribution foward process
        # q(x_t|x_0)=N(sqrt_alphas_cumprod*x0, (1-alphas_cumprod)I)
        suffix = int(share_ratio * 100)

        mean = extract(
            getattr(self, f'sqrt_alphas_cumprod_{suffix}'), t, x_start.shape) * x_start
        variance = extract(
            1.0 - getattr(self, f'alphas_cumprod_{suffix}'), t, x_start.shape)
        log_variance = extract(
            getattr(self, f'log_one_minus_alphas_cumprod_{suffix}'), t, x_start.shape)

        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, share_ratio: float):
        # x_0=1/sqrt(alphas_cumprod)*x_t - \sqrt{1/alphas_cumprod -1 }* eps
        suffix = int(share_ratio * 100)
        return (
            extract(
                getattr(self, f'sqrt_recip_alphas_cumprod_{suffix}'), t, x_t.shape) * x_t
            - extract(getattr(self, f'sqrt_recipm1_alphas_cumprod_{suffix}'), t, x_t.shape) * noise)

    def q_posterior(self, x_start, x_t, t, share_ratio: float):
        suffix = int(share_ratio * 100)

        posterior_mean = (
            extract(
                getattr(self, f'posterior_mean_coef1_{suffix}'), t, x_t.shape) * x_start
            + extract(getattr(self, f'posterior_mean_coef2_{suffix}'), t, x_t.shape) * x_t
        )
        posterior_variance = extract(
            getattr(self, f'posterior_variance_{suffix}'), t, x_t.shape)
        posterior_log_variance_clipped = extract(
            getattr(
                self, f'posterior_log_variance_clipped_{suffix}'), t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool, share_ratio: float):

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, cond=cond), share_ratio=share_ratio,
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)  # changed

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t,  share_ratio=share_ratio,
        )

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, share_ratio: float, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised, share_ratio=share_ratio,
        )

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        sample = model_mean + nonzero_mask * \
            (0.5 * model_log_variance).exp() * noise

        return sample

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, share_ratio: float):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        inter_steps = int(self.num_timesteps*(1-share_ratio))
        for i in reversed(range(inter_steps, self.num_timesteps)):
            img = self.p_sample(
                x=img, cond=cond, t=torch.full(
                    (b,), i, device=device, dtype=torch.long),
                share_ratio=share_ratio,
            )

        return img

    @torch.no_grad()
    def sample_ode_old(
            self,
            shape: Tuple[int, ...],
            cond,
            share_ratio: float,
            method: str = 'dopri5',
            atol: float = 1e-5,
            rtol: float = 1e-5
    ) -> torch.Tensor:
        """
        Samples from the model using an ODE-based approach.

        Args:
            shape (Tuple[int, ...]): The shape of the sample to generate.
            cond: Conditioning information for the denoise function.
            share_ratio (float): The share ratio to select the appropriate buffers.
            method (str): The ODE solver method to use.
            atol (float): Absolute tolerance for the ODE solver.
            rtol (float): Relative tolerance for the ODE solver.

        Returns:
            torch.Tensor: The generated sample.
        """
        device = self.betas.device
        x_init = torch.randn(shape, device=device)
        t_start = 1.0
        t_end = 0.0

        # Determine the suffix based on share_ratio
        suffix = int(share_ratio * 100)

        # Retrieve precomputed buffers for the given share_ratio
        alphas_cumprod = getattr(self, f"alphas_cumprod_{suffix}").to(device)
        sqrt_one_minus_alphas_cumprod = getattr(self, f"sqrt_one_minus_alphas_cumprod_{suffix}").to(device)
        betas = self.betas.to(device)

        def interpolate_time(t: float, buffer: torch.Tensor) -> torch.Tensor:
            """
            Linearly interpolates the buffer at continuous time t.

            Args:
                t (float): Continuous time between 0 and 1.
                buffer (torch.Tensor): The buffer to interpolate.

            Returns:
                torch.Tensor: The interpolated value.
            """
            t_scaled = t * (self.num_timesteps - 1)
            t0 = int(torch.floor(torch.tensor(t_scaled)).item())
            t1 = min(t0 + 1, self.num_timesteps - 1)
            alpha0 = buffer[t0]
            alpha1 = buffer[t1]
            alpha = t_scaled - t0
            return alpha0 + (alpha1 - alpha0) * alpha

        def ode_func(t: float, x: torch.Tensor) -> torch.Tensor:
            """
            Defines the ODE for the probability flow.

            Args:
                t (float): Current time in [0, 1].
                x (torch.Tensor): Current state.

            Returns:
                torch.Tensor: The derivative dx/dt.
            """
            # Interpolate beta(t) and sqrt_one_minus_alpha(t)
            beta_t = interpolate_time(t, betas)
            beta_t = beta_t.view(1, *([1] * (x.ndim - 1)))  # Reshape for broadcasting

            sqrt_one_minus_alpha_t = interpolate_time(t, sqrt_one_minus_alphas_cumprod)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(1, *([1] * (x.ndim - 1)))

            # Scale t to the model's timestep scale
            t_scaled = t * (self.num_timesteps - 1)
            t_tensor = torch.tensor([t_scaled], device=device, dtype=torch.long)

            # Predict noise using the denoise function
            eps = self.denoise_fn(x, t_tensor, cond=cond)

            # Compute the score
            score = -eps / sqrt_one_minus_alpha_t

            # Compute the derivative dx/dt
            dxdt = -0.5 * beta_t * x + 0.5 * beta_t * score
            return dxdt

        # Solve the ODE from t=1 to t=0
        solution = odeint(
            func=ode_func,
            y0=x_init,
            t=torch.tensor([t_start, t_end], device=device),
            method=method,
            atol=atol,
            rtol=rtol
        )

        # The final state is the sample at t=0
        x_final = solution[-1]
        return x_final

    def sample_ode(self, shape, cond, share_ratio: float, solver='rk4', atol=1e-6, rtol=1e-3):
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)

        suffix = int(share_ratio * 100)
        sqrt_one_minus_alphas_cumprod = getattr(self, f'sqrt_one_minus_alphas_cumprod_{suffix}')
        alphas_cumprod = getattr(self, f'alphas_cumprod_{suffix}')
        betas = self.betas.cpu().numpy()  # Assuming uniform time steps

        # Define the ODE function for the probability flow ODE
        def ode_func(t, x):
            # t ranges from 0 to 1
            # Convert t to the corresponding step
            t_step = t * (self.num_timesteps - 1)
            step_low = int(torch.floor(t_step))
            step_high = min(step_low + 1, self.num_timesteps - 1)
            dt = t_step - step_low

            # Linear interpolation of parameters
            beta = self.betas[step_low] + dt * (self.betas[step_high] - self.betas[step_low])
            alpha_cumprod = alphas_cumprod[step_low] + dt * (alphas_cumprod[step_high] - alphas_cumprod[step_low])
            sqrt_one_minus_alpha_cumprod_val = sqrt_one_minus_alphas_cumprod[step_low] + dt * (
                    sqrt_one_minus_alphas_cumprod[step_high] - sqrt_one_minus_alphas_cumprod[step_low])

            # Estimate the noise using the denoise function
            # Here, t is continuous, so we map it to the nearest step for denoising
            t_tensor = torch.full((b,), step_low, device=device, dtype=torch.long)
            noise = self.denoise_fn(x, t_tensor, cond=cond)

            # Compute the score
            score = - noise / sqrt_one_minus_alpha_cumprod_val.view(-1, 1, 1)

            # Probability flow ODE: dx/dt = 0.5 * beta(t) * (x - sigma(t)^2 * score)
            # Here, sigma(t)^2 = 1 - alpha_cumprod
            sigma_sq = 1.0 - alpha_cumprod
            # dxdt = 0.5 * beta.view(-1, 1, 1) * (x - sigma_sq.view(-1, 1, 1) * score)
            dxdt = 0.5 * beta.view(-1, 1, 1) * (x -  score)

            return dxdt

        # Integrate the ODE from t=1 to t=0
        # t_span = torch.tensor([1.0, 0.01], device=device)
        t_span = torch.linspace(1.0, 0.01, steps=100, device=device)
        # t_eval = torch.linspace(1.0, 0.0, steps=self.num_timesteps, device=device)

        # Use CPU for integration if necessary
        # ordent = odeint(ode_func, x, t_span, method=solver, atol=atol, rtol=rtol)
        with torch.no_grad():
            x = odeint(ode_func, x, t_span, method=solver, atol=atol, rtol=rtol)[-1]

        return x
    @torch.no_grad()
    def sample(self, share_ratio: float, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape

        # x_hat = self.p_sample_loop(
        #     shape=shape, cond=cond, share_ratio=share_ratio)
        x_hat = self.sample_ode(
            shape=shape, cond=cond, share_ratio=share_ratio)
        # x_hat = self.sample_flow(
        #     shape=shape, cond=cond, share_ratio=share_ratio)
        return x_hat
    def gen_pflow(self,time_series, t, t_next, model, betas, total_step, cond):
        """
        生成下一个时间步的时间序列数据，使用概率流方法（PF）。

        参数：
        - time_series: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
        - t: 当前时间步标量或张量
        - t_next: 下一个时间步标量或张量
        - model: 模型，用于预测噪声
        - betas: Beta 系数列表或张量
        - total_step: 总步数

        返回：
        - img_next: 下一个时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
        """
        n = time_series.shape[0]  # batch_size
        betas_list = betas[0]
        beta_0, beta_1 = betas_list[0], betas_list[-1]
        # beta_0, beta_1 = betas[0][0], betas[0][-1]

        # 假设 t 是标量，如果 t 是张量，则根据需要调整
        if isinstance(t, torch.Tensor):
            t_start = t
        else:
            t_start = torch.ones(n, device=time_series.device) * t
        beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

        log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

        # 计算漂移（drift）和扩散（diffusion）
        # 调整形状以适应时间序列数据 (batch_size, seqlen, dim)
        drift = -0.5 * beta_t.view(-1, 1, 1) * time_series  # (batch_size, seqlen, dim)
        diffusion = torch.sqrt(beta_t).view(-1, 1, 1)  # (batch_size, 1, 1)

        # 预测噪声
        score = - model(inputs=time_series, cond=cond, time=t_start * (total_step - 1)) / std.view(-1, 1,
                                                                                                   1)  # (batch_size, seqlen, dim)

        # 更新漂移
        drift = drift - (diffusion ** 2) * score * 0.5  # (batch_size, seqlen, dim)

        # 生成下一个时间步的时间序列数据
        # img_next = time_series + drift  # 简化处理，具体更新方式可根据需求调整

        return drift

    @torch.no_grad()
    def sample_flow(self, share_ratio: float, seq_timestep=None, sample_shape=torch.Size(), cond=None,
                    sample_speed=None, noise_x=None,shape=None, device=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape
        noise = torch.randn(shape,device=device)
        # cond_shape = cond.shape if cond is not None else cond_shape = cond.shape
        tol = 1e-5
        rtol = 1e-3
        atol = 1e-6
        x = noise_x
        betas = getattr(self, f'betas'),
        diff_steps = getattr(self, f'diff_steps')

        # print("sample_flow")
        # call_count = 0
        def drift_func(t, x):
            # x = torch.from_numpy(x.reshape(shape)).to(device).type(torch.float32)
            drift = self.gen_pflow(time_series=x, t=t, t_next=t, cond=cond, model=self.denoise_fn, betas=betas,
                                   total_step=diff_steps)
            return drift

        # 使用 torchdiffeq 的 odeint 进行积分
        t_span = torch.tensor([1.0, 1e-2], device=device)

        solution = odeint(drift_func, noise, t_span, rtol=rtol, atol=atol, method='rk4')
        # print(f"drift_func was called {call_count} times.")
        # res = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32).to(device)
        res = solution[-1].view(shape)
        return res

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
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

    def q_sample(self, x_start, t, share_ratio: float, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        suffix = int(share_ratio * 100)
        return (
            extract(
                getattr(self, f'sqrt_alphas_cumprod_{suffix}'), t, x_start.shape) * x_start
            + extract(getattr(self, f'sqrt_one_minus_alphas_cumprod_{suffix}'), t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t, share_ratio: float, noise=None):
        # if share betas, means only part of the betas are used.
        noise = default(noise, lambda: torch.randn_like(x_start))
        # x_t = a x0 + b \eps
        x_noisy = self.q_sample(x_start=x_start, t=t,
                                noise=noise, share_ratio=share_ratio)
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

    def log_prob(self, x, cond, share_ratio: float, *args, **kwargs):
        B, T, _ = x.shape

        time = torch.randint(0, self.num_timesteps,
                             (B * T,), device=x.device).long()
        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, share_ratio=share_ratio,
            *args, **kwargs
        )

        return loss


"""
diffusion output  
"""


class DiffusionOutput(DistributionOutput):
    @validated()
    def __init__(self, diffusion, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.diffusion = diffusion
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.diffusion.scale = scale
        self.diffusion.cond = cond

        return self.diffusion

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
