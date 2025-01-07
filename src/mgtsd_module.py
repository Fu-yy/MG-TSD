from inspect import isfunction
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F

from gluonts.core.component import validated
# from gluonts.torch.modules.distribution_output import DistributionOutput
from gluonts.torch.distributions.distribution_output import DistributionOutput
from typing import Tuple
import copy

from down_up_fourier_pool import DonwSample_Fourier_high_low

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



def create_custom_beta_schedule(num_timesteps, beta_begin=1e-4,beta_end=0.1,end_ratio=0.2, decay_type="linear"):
    """
    创建 beta 调度:
    - 前 `end_ratio * num_timesteps` 使用不同的速率函数从 0 增长到 1。
    - 后续保持为 1。

    Parameters:
        num_timesteps: int, 总时间步数
        end_ratio: float, 前 end_ratio 的范围内增长到 1
        decay_type: str, 增长速率类型, 可选 'linear', 'quadratic', 'exponential'

    Returns:
        beta: np.array, 增长调度
    """
    end_step = int(num_timesteps * end_ratio)  # 前 end_ratio 时间步的长度
    beta = np.zeros(num_timesteps)  # 初始化 beta
    beta.fill(beta_end)

    # 根据不同的速率定义增长函数
    if decay_type == "linear":
        growth = np.linspace(beta_begin, beta_end, end_step)  # 线性增长
    elif decay_type == "quadratic":
        growth = np.linspace(beta_begin, beta_end, end_step) ** 2  # 平方增长
    elif decay_type == "exponential":
        growth = 1 - np.exp(-np.linspace(beta_begin, beta_end, end_step) * 5)  # 指数增长
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")

    # 填充前 end_step 部分和后续部分
    beta[:end_step] = growth
    # 后续部分保持为 1（已经初始化为 1，无需重复赋值）

    return beta


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
        denoise_fn_low,
        denoise_fn_high,
        input_size,
        # control the diffusion and sampling(reverse diffusion) procedures
        share_ratio_list,
        beta_end=0.1,
        beta_begin=1e-4,
        diff_steps=100,
        loss_type="l2",
        end_ratio=0.3,
        rate=0.5,
        betas_high=None,
        betas_low=None,
        beta_schedule="linear",
    ):
        super().__init__()
        self.denoise_fn_low = denoise_fn_low
        self.denoise_fn_high = denoise_fn_high
        self.input_size = input_size
        self.__scale = None
        self.share_ratio_list = share_ratio_list  # ratio of betas are shared
        self.end_ratio = end_ratio
        self.diff_steps = diff_steps

        betas_high = create_custom_beta_schedule(num_timesteps=diff_steps,beta_begin=beta_begin,beta_end=beta_end,end_ratio=end_ratio,decay_type=beta_schedule)
        betas_low = create_custom_beta_schedule(num_timesteps=diff_steps,beta_begin=beta_begin,beta_end=beta_end,end_ratio=1,decay_type=beta_schedule)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        (timesteps,) = betas_low.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.register_buffer("betas_low", to_torch(betas_low))
        self.register_buffer("betas_high", to_torch(betas_high))
        self.fourier_low_high = DonwSample_Fourier_high_low(rate=rate)

        for cur_share_ratio in self.share_ratio_list:
            start_index_low = int(len(betas_low)*(1-cur_share_ratio))
            betas_sub_low = copy.deepcopy(betas_low)
            betas_sub_low[:start_index_low] = 0  # share the latter part of the betas
            alphas_sub_low = 1.0 - betas_sub_low
            alphas_cumprod_sub_low = np.cumprod(alphas_sub_low, axis=0)
            alphas_cumprod_prev_sub_low = np.append(1.0, alphas_cumprod_sub_low[:-1])
            suffix = int(cur_share_ratio * 100)
            self.register_buffer(
                f"alphas_cumprod_low_{suffix}", to_torch(alphas_cumprod_sub_low))
            self.register_buffer(
                f"alphas_cumprod_prev_low_{suffix}", to_torch(alphas_cumprod_prev_sub_low))

            self.register_buffer(f"sqrt_alphas_cumprod_low_{suffix}", to_torch(
                np.sqrt(alphas_cumprod_sub_low)))
            self.register_buffer(
                f"sqrt_one_minus_alphas_cumprod_low_{suffix}", to_torch(
                    np.sqrt(1.0 - alphas_cumprod_sub_low))
            )
            self.register_buffer(
                f"log_one_minus_alphas_cumprod_low_{suffix}", to_torch(
                    np.log(1.0 - alphas_cumprod_sub_low))
            )
            self.register_buffer(
                f"sqrt_recip_alphas_cumprod_low_{suffix}", to_torch(
                    np.sqrt(1.0 / alphas_cumprod_sub_low))
            )
            self.register_buffer(
                f"sqrt_recipm1_alphas_cumprod_low_{suffix}", to_torch(
                    np.sqrt(1.0 / alphas_cumprod_sub_low - 1))
            )
            self.register_buffer(

                f"posterior_mean_coef1_low_{suffix}",
                to_torch(betas_sub_low * np.sqrt(alphas_cumprod_prev_sub_low) / (1.0 - alphas_cumprod_sub_low)),)

            self.register_buffer(
                f"posterior_mean_coef2_low_{suffix}",

                to_torch(
                    (1.0 - alphas_cumprod_prev_sub_low) *
                    np.sqrt(alphas_sub_low) / (1.0 - alphas_cumprod_sub_low)

                ),)
            posterior_variance_sub_low = (
                betas_sub_low * (1.0 - alphas_cumprod_prev_sub_low) / (1.0 - alphas_cumprod_sub_low))
            self.register_buffer(
                f"posterior_variance_low_{suffix}", to_torch(posterior_variance_sub_low))

            self.register_buffer(
                f"posterior_log_variance_clipped_low_{suffix}",
                to_torch(np.log(np.maximum(posterior_variance_sub_low, 1e-20))),)






            # ------  high

            start_index_high = int(len(betas_high) * (1 - cur_share_ratio))
            betas_sub_high = copy.deepcopy(betas_high)
            betas_sub_high[:start_index_high] = 0  # share the latter part of the betas
            alphas_sub_high = 1.0 - betas_sub_high
            alphas_cumprod_sub_high = np.cumprod(alphas_sub_high, axis=0)
            alphas_cumprod_prev_sub_high = np.append(1.0, alphas_cumprod_sub_high[:-1])
            suffix = int(cur_share_ratio * 100)
            self.register_buffer(
                f"alphas_cumprod_high_{suffix}", to_torch(alphas_cumprod_sub_high))
            self.register_buffer(
                f"alphas_cumprod_prev_high_{suffix}", to_torch(alphas_cumprod_prev_sub_high))

            self.register_buffer(f"sqrt_alphas_cumprod_high_{suffix}", to_torch(
                np.sqrt(alphas_cumprod_sub_high)))
            self.register_buffer(
                f"sqrt_one_minus_alphas_cumprod_high_{suffix}", to_torch(
                    np.sqrt(1.0 - alphas_cumprod_sub_high))
            )
            self.register_buffer(
                f"log_one_minus_alphas_cumprod_high_{suffix}", to_torch(
                    np.log(1.0 - alphas_cumprod_sub_high))
            )
            self.register_buffer(
                f"sqrt_recip_alphas_cumprod_high_{suffix}", to_torch(
                    np.sqrt(1.0 / alphas_cumprod_sub_high))
            )
            self.register_buffer(
                f"sqrt_recipm1_alphas_cumprod_high_{suffix}", to_torch(
                    np.sqrt(1.0 / alphas_cumprod_sub_high - 1))
            )
            self.register_buffer(

                f"posterior_mean_coef1_high_{suffix}",
                to_torch(betas_sub_high * np.sqrt(alphas_cumprod_prev_sub_high) / (1.0 - alphas_cumprod_sub_high)), )

            self.register_buffer(
                f"posterior_mean_coef2_high_{suffix}",

                to_torch(
                    (1.0 - alphas_cumprod_prev_sub_high) *
                    np.sqrt(alphas_sub_high) / (1.0 - alphas_cumprod_sub_high)

                ), )
            posterior_variance_sub_high = (
                    betas_sub_high * (1.0 - alphas_cumprod_prev_sub_high) / (1.0 - alphas_cumprod_sub_high))
            self.register_buffer(
                f"posterior_variance_high_{suffix}", to_torch(posterior_variance_sub_high))

            self.register_buffer(
                f"posterior_log_variance_clipped_high_{suffix}",
                to_torch(np.log(np.maximum(posterior_variance_sub_high, 1e-20))), )

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

    def predict_start_from_noise(self, x_t, t,index, noise, share_ratio: float):
        # x_0=1/sqrt(alphas_cumprod)*x_t - \sqrt{1/alphas_cumprod -1 }* eps
        suffix = int(share_ratio * 100)
        t_val=index
        if t_val > self.end_ratio * self.diff_steps: # low

            sqrt_recip_alphas_cumprod_low =  extract(getattr(self, f'sqrt_recip_alphas_cumprod_low_{suffix}'), t, x_t.shape)

            sqrt_recipm1_alphas_cumprod_low = extract(getattr(self, f'sqrt_recipm1_alphas_cumprod_low_{suffix}'), t, x_t.shape)
            return (sqrt_recip_alphas_cumprod_low * x_t - sqrt_recipm1_alphas_cumprod_low*noise)
        else:  # high
            sqrt_recip_alphas_cumprod_high = extract(getattr(self, f'sqrt_recip_alphas_cumprod_high_{suffix}'), t, x_t.shape)

            sqrt_recipm1_alphas_cumprod_high = extract(getattr(self, f'sqrt_recipm1_alphas_cumprod_high_{suffix}'), t, x_t.shape)
            return (sqrt_recip_alphas_cumprod_high * x_t - sqrt_recipm1_alphas_cumprod_high*noise)

    def q_posterior(self, x_start, x_t, t,index, share_ratio: float):
        suffix = int(share_ratio * 100)
        t_val=index
        if t_val > self.end_ratio * self.diff_steps: # low
            posterior_mean_coef1 = extract(getattr(self, f'posterior_mean_coef1_low_{suffix}'), t, x_t.shape)
            posterior_mean_coef2 = extract(getattr(self, f'posterior_mean_coef2_low_{suffix}'), t, x_t.shape)
            posterior_variance = extract(getattr(self, f'posterior_variance_low_{suffix}'), t, x_t.shape)

            posterior_log_variance_clipped = extract(getattr(self, f'posterior_log_variance_clipped_low_{suffix}'), t, x_t.shape)
        else:
            posterior_mean_coef1 = extract(getattr(self, f'posterior_mean_coef1_high_{suffix}'), t, x_t.shape)
            posterior_mean_coef2 = extract(getattr(self, f'posterior_mean_coef2_high_{suffix}'), t, x_t.shape)
            posterior_variance = extract(getattr(self, f'posterior_variance_high_{suffix}'), t, x_t.shape)

            posterior_log_variance_clipped = extract(getattr(self, f'posterior_log_variance_clipped_high_{suffix}'), t,
                                                     x_t.shape)
        posterior_mean = (posterior_mean_coef1 * x_start+ posterior_mean_coef2 * x_t)


        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t,index, clip_denoised: bool, share_ratio: float):
        t_val = index
        if t_val > self.end_ratio * self.diff_steps:
            noise = self.denoise_fn_low(x, t, cond=cond)
        else:
            noise = self.denoise_fn_high(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=noise,index=index, share_ratio=share_ratio,
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)  # changed

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x,index=index, t=t,  share_ratio=share_ratio,
        )

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, index,share_ratio: float, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t,index=index, clip_denoised=clip_denoised, share_ratio=share_ratio,
        )

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        sample = model_mean + nonzero_mask * \
            (0.5 * model_log_variance).exp() * noise

        return sample

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, share_ratio: float):
        device = self.betas_low.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        inter_steps = int(self.num_timesteps*(1-share_ratio))
        for i in reversed(range(inter_steps, self.num_timesteps)):
            img = self.p_sample(
                x=img, cond=cond, t=torch.full(
                    (b,), i, device=device, dtype=torch.long),index=i,
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


    def q_sample(self, x_start_low,x_start_high, t, share_ratio: float,noise_low =None,noise_high=None):
        noise_low = default(noise_low, lambda: torch.randn_like(x_start_low))
        noise_high = default(noise_high, lambda: torch.randn_like(x_start_low))
        suffix = int(share_ratio * 100)

        sqrt_low_alpha_cumprod  = extract(getattr(self, f'sqrt_alphas_cumprod_low_{suffix}'), t, x_start_low.shape)
        sqrt_one_minus_low_alpha_cumprod  = extract(getattr(self, f'sqrt_one_minus_alphas_cumprod_low_{suffix}'), t, x_start_low.shape)
        # low_alpha_cumprod  = extract(getattr(self, f'sqrt_alphas_cumprod_low_{suffix}'), t, x_start_low.shape)

        sqrt_high_alpha_cumprod  = extract(getattr(self, f'sqrt_alphas_cumprod_high_{suffix}'), t, x_start_high.shape)
        sqrt_one_minus_high_alpha_cumprod  = extract(getattr(self, f'sqrt_one_minus_alphas_cumprod_high_{suffix}'), t, x_start_high.shape)

        x_t_low = sqrt_low_alpha_cumprod * x_start_low +sqrt_one_minus_low_alpha_cumprod * noise_low
        x_t_high = sqrt_high_alpha_cumprod * x_start_high +sqrt_one_minus_high_alpha_cumprod * noise_low
        x_t = x_t_low + x_t_high
        return x_t

    def p_losses(self, x_start_low,x_start_high, cond, t, share_ratio: float, noise=None):
        # if share betas, means only part of the betas are used.
        noise_low = default(noise, lambda: torch.randn_like(x_start_low))
        noise_high = default(noise, lambda: torch.randn_like(x_start_high))
        # x_t = a x0 + b \eps
        x_t = self.q_sample(x_start_low=x_start_low,x_start_high=x_start_high, t=t,
                                noise_low=noise_low,noise_high=noise_high, share_ratio=share_ratio)
        x_recon_low = self.denoise_fn_low(x_t, t, cond=cond)
        x_recon_high = self.denoise_fn_high(x_recon_low, t, cond=cond)

        if self.loss_type == "l1":
            # loss_low = F.l1_loss(x_recon_low, noise_low)
            loss_high = F.l1_loss(x_recon_high, noise_low)
        elif self.loss_type == "l2":
            # loss_low = F.mse_loss(x_recon_low, noise_low)
            loss_high = F.mse_loss(x_recon_high, noise_low)
        elif self.loss_type == "huber":
            # loss_low = F.smooth_l1_loss(x_recon_low, noise_low)
            loss_high = F.smooth_l1_loss(x_recon_high, noise_low)
        else:
            raise NotImplementedError()
        loss = loss_high
        return loss

    def log_prob(self, x, cond, share_ratio: float, *args, **kwargs):
        B, T, _ = x.shape

        time = torch.randint(0, self.num_timesteps,
                             (B * T,), device=x.device).long()

        x_low,x_high =self.fourier_low_high(x)


        loss = self.p_losses(
            x_low.reshape(B * T, 1, -1),x_high.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, share_ratio=share_ratio,
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
