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
from scipy import integrate
import torch
from torchdiffeq import odeint
from method import choose_method

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
        # self.method = choose_method('PF')  # add pflow
        # self.method = self.gen_pflow()  # add pflow
        self.ets = None

    def gen_pflow_modify(self, t, x, model, betas, cond):
        """
        计算概率流 ODE 的漂移项。

        参数：
        - t: 当前时间步（连续时间）
        - x: 当前时间步的数据
        - model: 用于预测噪声的模型
        - betas: Beta 系数的张量，形状为 [num_timesteps]
        - cond: 条件信息（如果有的话）

        返回：
        - 漂移项，形状与 x 相同
        """
        # 将连续时间 t 映射到离散时间步
        discrete_t = (t * (self.num_timesteps - 1)).long()

        # 获取对应的 beta_t
        beta_t = betas[discrete_t].view(-1, 1, 1)  # 根据需要调整维度

        # 预测得分函数（score = ∇x log p_t(x)）
        score = model(x, discrete_t, cond=cond)

        # 计算漂移项
        drift = 0.5 * beta_t * x - 0.5 * (beta_t ** 2) * score

        return drift
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
    def sample(self, share_ratio: float, sample_shape=torch.Size(), cond=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape

        x_hat = self.p_sample_loop(
            shape=shape, cond=cond, share_ratio=share_ratio)
        return x_hat



    # @torch.no_grad()
    #
    # def sample_flow(self, share_ratio: float,seq_timestep=None, sample_shape=torch.Size(), cond=None,sample_speed=None):
    #     if cond is not None:
    #         shape = cond.shape[:-1] + (self.input_size,)
    #         # TODO reshape cond to (B*T, 1, -1)
    #     else:
    #         shape = sample_shape
    #     device = cond.device
    #     cond_shape = cond.shape
    #     tol = 1e-5 if sample_speed > 1 else sample_speed
    #     rtol  = 1e-3 if sample_speed > 1 else sample_speed
    #     atol  = 1e-6 if sample_speed > 1 else sample_speed
    #     x = torch.randn(shape, device=device)
    #     betas = getattr(self, f'betas'),
    #     diff_steps = getattr(self, f'diff_steps')
    #
    #     # print("sample_flow")
    #     # call_count = 0
    #     def drift_func(t, x):
    #         x = torch.from_numpy(x.reshape(shape)).to(device).type(torch.float32)
    #         # cond = torch.from_numpy(cond.reshape(cond_shape)).to(device).type(torch.float32)
    #         # nonlocal call_count
    #         # call_count += 1
    #         # print(f"inner called {call_count} times.")
    #
    #         # drift = self.method(img=x, t=None, t_next=t,cond=cond, model=self.denoise_fn, alphas_cump=)
    #         drift = self.method(time_series=x, t=t, t_next=t,cond=cond, model=self.denoise_fn, betas=betas, total_step=diff_steps)
    #
    #
    #         # drift = self.schedule.denoising(x, None, t, model)
    #         drift = drift.cpu().numpy().reshape((-1,))
    #         return drift
    #
    #     solution = integrate.solve_ivp(drift_func, (1, 1e-3),x.cpu().numpy().reshape((-1,)),rtol=rtol, atol=atol, method='RK45')
    #     # print(f"drift_func was called {call_count} times.")
    #     res = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32).to(device)
    #     return res
    @torch.no_grad()

    def sample_flow(self, share_ratio: float,seq_timestep=None, sample_shape=torch.Size(), cond=None,sample_speed=None,noise_x=None,device=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape

        # cond_shape = cond.shape if cond is not None else cond_shape = cond.shape
        tol = 1e-5 if sample_speed > 1 else sample_speed
        rtol  = 1e-3 if sample_speed > 1 else sample_speed
        atol  = 1e-6 if sample_speed > 1 else sample_speed
        x = noise_x
        betas = getattr(self, f'betas'),
        diff_steps = getattr(self, f'diff_steps')

        # print("sample_flow")
        # call_count = 0
        def drift_func(t, x):
            # x = torch.from_numpy(x.reshape(shape)).to(device).type(torch.float32)
            drift = self.gen_pflow(time_series=x, t=t, t_next=t,cond=cond, model=self.denoise_fn, betas=betas, total_step=diff_steps)
            return drift
        # 使用 torchdiffeq 的 odeint 进行积分
        t_span = torch.tensor([1.0, 1e-2], device=device)

        solution = odeint(drift_func,noise_x, t_span,rtol=rtol, atol=atol, method='rk4')
        # print(f"drift_func was called {call_count} times.")
        # res = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32).to(device)
        res = solution[-1].view(shape)
        return res

    def sample_flow_inner(self, share_ratio: float, seq_timestep=None, sample_shape=torch.Size(), cond=None,
                    sample_speed=None):
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape
        device = cond.device if cond is not None else torch.device('cpu')
        x = torch.randn(shape, device=device)
        return self.sample_flow(cond=cond,noise_x = x, share_ratio=share_ratio,
                                                    seq_timestep=None,sample_speed=sample_speed)



    @torch.no_grad()
    def sample_flow_modify(self, share_ratio: float, sample_shape=torch.Size(), cond=None, sample_speed=None):
        """
        使用概率流 ODE 方法生成样本。

        参数：
        - share_ratio: 分享比例，用于调整扩散过程
        - sample_shape: 生成样本的形状
        - cond: 条件信息（如果有的话）
        - sample_speed: 采样速度控制参数

        返回：
        - 生成的样本，形状为 sample_shape
        """
        if cond is not None:
            shape = cond.shape[:-1] + (self.input_size,)
        else:
            shape = sample_shape
        device = cond.device if cond is not None else torch.device('cpu')

        # 初始化 x 在 t=1（纯噪声）
        x = torch.randn(shape, device=device)

        betas = getattr(self, 'betas')  # 确保 betas 是形状为 [num_timesteps] 的张量
        diff_steps = getattr(self, 'diff_steps')  # 总扩散步数

        # 定义漂移函数用于 ODE 积分
        def drift_func(t, x):
            return self.gen_pflow_modify(t, x, model=self.denoise_fn, betas=betas, cond=cond)

        # 定义时间跨度从 t=1 到 t=0
        t_span = torch.tensor([1.0, 0.0], device=device)

        # 执行 ODE 积分
        solution = odeint(drift_func, x, t_span, rtol=1e-3, atol=1e-6, method='rk4')

        # 最终状态是 t=0 时的样本
        x_hat = solution[-1]

        # 可选：限制输出范围
        # x_hat = x_hat.clamp(-1.0, 1.0)

        return x_hat
    # @torch.no_grad()
    # def sample_flow(self, share_ratio: float, seq_timestep=None, sample_shape=torch.Size(), cond=None,
    #                 sample_speed=None):
    #     """
    #     使用微分方程求解器在 CUDA 上生成样本。
    #
    #     参数:
    #         share_ratio (float): 共享比例。
    #         seq_timestep (optional): 序列时间步。
    #         sample_shape (torch.Size): 生成样本的形状。
    #         cond (torch.Tensor, optional): 条件张量。
    #         sample_speed (float, optional): 采样速度，影响容差。
    #
    #     返回:
    #         torch.Tensor: 生成的样本。
    #     """
    #
    #     # 确定生成样本的形状
    #     if cond is not None:
    #         # 假设 cond 的最后一个维度是输入大小，将其替换为 self.input_size
    #         shape = cond.shape[:-1] + (self.input_size,)
    #         # TODO: 根据需要调整 cond 的形状，例如 reshape 为 (B*T, 1, -1)
    #     else:
    #         shape = sample_shape
    #
    #     # 设置设备为 GPU
    #     device = cond.device if cond is not None else torch.device('cuda')
    #
    #     # 设置容差
    #     tol = 1e-5 if sample_speed is not None and sample_speed > 1 else sample_speed if sample_speed is not None else 1e-5
    #
    #     # 初始化噪声
    #     x = torch.randn(shape, device=device)
    #
    #     # 获取模型参数
    #     betas = getattr(self, 'betas')  # 确保 'betas' 属性存在
    #     diff_steps = getattr(self, 'diff_steps')  # 确保 'diff_steps' 属性存在
    #
    #     # 定义 drift 函数
    #     class DriftFunc(torch.nn.Module):
    #         def __init__(self, method, cond, model, betas, diff_steps):
    #             super(DriftFunc, self).__init__()
    #             self.method = method
    #             self.cond = cond
    #             self.model = model
    #             self.betas = betas
    #             self.diff_steps = diff_steps
    #
    #         def forward(self, t, x):
    #             """
    #             计算 drift = dx/dt。
    #
    #             参数:
    #                 t (torch.Tensor): 当前时间。
    #                 x (torch.Tensor): 当前状态，形状为 (N,)。
    #
    #             返回:
    #                 torch.Tensor: 计算得到的 drift，形状为 (N,)。
    #             """
    #             # 将 x 从 (N,) 恢复到原始形状
    #             x_reshaped = x.view(shape)
    #
    #             # 计算 drift，确保所有操作在 GPU 上
    #             drift = self.method(
    #                 time_series=x_reshaped,
    #                 t=t,
    #                 t_next=t,
    #                 cond=self.cond,
    #                 model=self.model,
    #                 betas=self.betas,
    #                 total_step=self.diff_steps
    #             )
    #
    #             # 将 drift 展平以匹配 ODE 求解器的输入格式
    #             drift_flat = drift.view(-1)
    #             return drift_flat
    #
    #     # 初始化 DriftFunc 模块
    #     drift_func = DriftFunc(
    #         method=self.method,
    #         cond=cond,
    #         model=self.denoise_fn,
    #         betas=betas,
    #         diff_steps=diff_steps
    #     ).to(device)
    #
    #     # 定义时间跨度，从 t=1 到 t=1e-3，步数根据需要调整
    #     t_span = torch.linspace(1, 1e-3, steps=100, device=device)
    #
    #     # 初始条件，展平为 (N,)
    #     x0 = x.view(-1)
    #
    #     # 执行 ODE 积分，选择合适的求解方法（例如 'rk4', 'dopri5'）
    #     solution = odeint(drift_func, x0, t_span, rtol=tol, atol=tol, method='rk4')
    #
    #     # 获取最后一个时间点的结果，并恢复原始形状
    #     res = solution[-1].view(shape)
    #
    #     return res



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
        a = extract(
                getattr(self, f'sqrt_alphas_cumprod_{suffix}'), t, x_start.shape)
        d = a * x_start

        e = extract(getattr(self, f'sqrt_one_minus_alphas_cumprod_{suffix}'), t, x_start.shape)
        f = e * noise
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

        # time = torch.randint(0, self.num_timesteps,
        #                      (B * T,), device=x.device).long()
        # loss = self.p_losses(
        #     x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, share_ratio=share_ratio,
        #     *args, **kwargs
        # )

        time = torch.randint(0, self.num_timesteps,
                             (B ,), device=x.device).long()
        loss = self.p_losses(
            x, cond, time, share_ratio=share_ratio,
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
