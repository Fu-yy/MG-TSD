# Copyright 2022 Luping Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import copy
import torch as th


def choose_method(name):
    if name == 'DDIM':
        return gen_order_1
    elif name == 'S-PNDM':
        return gen_order_2
    elif name == 'F-PNDM':
        return gen_order_4
    elif name == 'FON':
        return gen_fon
    elif name == 'PF':
        return gen_pflow
    else:
        return None


def gen_pflow(time_series, t, t_next, model, betas, total_step,cond):
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
    # betas_list = betas.tolist()
    # beta_0, beta_1 = betas_list[0], betas_list[-1]
    beta_0, beta_1 = betas[0][0], betas[0][-1]

    # 假设 t 是标量，如果 t 是张量，则根据需要调整
    if isinstance(t, th.Tensor):
        t_start = t
    else:
        t_start = th.ones(n, device=time_series.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # 计算漂移（drift）和扩散（diffusion）
    # 调整形状以适应时间序列数据 (batch_size, seqlen, dim)
    drift = -0.5 * beta_t.view(-1, 1, 1) * time_series  # (batch_size, seqlen, dim)
    diffusion = th.sqrt(beta_t).view(-1, 1, 1)  # (batch_size, 1, 1)

    # 预测噪声
    score = - model(inputs=time_series, cond=cond,time=t_start * (total_step - 1)) / std.view(-1, 1, 1)  # (batch_size, seqlen, dim)

    # 更新漂移
    drift = drift - (diffusion ** 2) * score * 0.5  # (batch_size, seqlen, dim)

    # 生成下一个时间步的时间序列数据
    # img_next = time_series + drift  # 简化处理，具体更新方式可根据需求调整

    return drift


def gen_fon(time_series, t, t_next, model, alphas_cump, ets):
    """
    生成下一个时间步的时间序列数据，使用 FON 方法。

    参数：
    - time_series: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    - t: 当前时间步标量或张量
    - t_next: 下一个时间步标量或张量
    - model: 模型，用于预测噪声
    - alphas_cump: Alpha 累积系数，形状为 (T+1,)
    - ets: 噪声缓存列表，用于高阶方法

    返回：
    - img_next: 下一个时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    """
    t_list = [t, (t + t_next) / 2.0, t_next]
    if len(ets) > 2:
        noise = model(time_series, t)
        img_next = transfer(time_series, t, t - 1, noise, alphas_cump)
        delta1 = img_next - time_series
        ets.append(delta1)
        delta = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = model(time_series, t_list[0])
        img_ = transfer(time_series, t, t - 1, noise, alphas_cump)
        delta_1 = img_ - time_series
        ets.append(delta_1)

        img_2 = time_series + delta_1 * (t - t_next) / 2.0
        noise = model(img_2, t_list[1])
        img_ = transfer(time_series, t, t - 1, noise, alphas_cump)
        delta_2 = img_ - time_series

        img_3 = time_series + delta_2 * (t - t_next) / 2.0
        noise = model(img_3, t_list[1])
        img_ = transfer(time_series, t, t - 1, noise, alphas_cump)
        delta_3 = img_ - time_series

        img_4 = time_series + delta_3 * (t - t_next)
        noise = model(img_4, t_list[2])
        img_ = transfer(time_series, t, t - 1, noise, alphas_cump)
        delta_4 = img_ - time_series
        delta = (1 / 6.0) * (delta_1 + 2 * delta_2 + 2 * delta_3 + delta_4)

    img_next = time_series + delta * (t - t_next)
    return img_next


def gen_order_4(time_series, t, t_next, model, alphas_cump, ets):
    """
    生成下一个时间步的时间序列数据，使用四阶方法。

    参数：
    - time_series: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    - t: 当前时间步标量或张量
    - t_next: 下一个时间步标量或张量
    - model: 模型，用于预测噪声
    - alphas_cump: Alpha 累积系数，形状为 (T+1,)
    - ets: 噪声缓存列表，用于高阶方法

    返回：
    - img_next: 下一个时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    """
    t_list = [t, (t + t_next) / 2, t_next]
    if len(ets) > 2:
        noise_ = model(time_series, t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = runge_kutta(time_series, t_list, model, alphas_cump, ets)

    img_next = transfer(time_series, t, t_next, noise, alphas_cump)
    return img_next


def runge_kutta(x, t_list, model, alphas_cump, ets):
    """
    运行 Runge-Kutta 方法来计算噪声。

    参数：
    - x: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    - t_list: 时间步列表
    - model: 模型，用于预测噪声
    - alphas_cump: Alpha 累积系数，形状为 (T+1,)
    - ets: 噪声缓存列表，用于高阶方法

    返回：
    - et: 计算得到的噪声，形状为 (batch_size, seqlen, dim)
    """
    e_1 = model(x, t_list[0])
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(x_2, t_list[1])
    x_3 = transfer(x_2, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(x_3, t_list[1])
    x_4 = transfer(x_3, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(x_4, t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
    return et


def gen_order_2(time_series, t, t_next, model, alphas_cump, ets):
    """
    生成下一个时间步的时间序列数据，使用二阶方法。

    参数：
    - time_series: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    - t: 当前时间步标量或张量
    - t_next: 下一个时间步标量或张量
    - model: 模型，用于预测噪声
    - alphas_cump: Alpha 累积系数，形状为 (T+1,)
    - ets: 噪声缓存列表，用于高阶方法

    返回：
    - img_next: 下一个时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    """
    if len(ets) > 0:
        noise_ = model(time_series, t)
        ets.append(noise_)
        noise = 0.5 * (3 * ets[-1] - ets[-2])
    else:
        noise = improved_eular(time_series, t, t_next, model, alphas_cump, ets)

    img_next = transfer(time_series, t, t_next, noise, alphas_cump)
    return img_next


def improved_eular(x, t, t_next, model, alphas_cump, ets):
    """
    改进的欧拉方法，用于二阶生成方法。

    参数：
    - x: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    - t: 当前时间步标量或张量
    - t_next: 下一个时间步标量或张量
    - model: 模型，用于预测噪声
    - alphas_cump: Alpha 累积系数，形状为 (T+1,)
    - ets: 噪声缓存列表，用于高阶方法

    返回：
    - et: 计算得到的噪声，形状为 (batch_size, seqlen, dim)
    """
    e_1 = model(x, t)
    ets.append(e_1)
    x_2 = transfer(x, t, t_next, e_1, alphas_cump)

    e_2 = model(x_2, t_next)
    et = (e_1 + e_2) / 2
    return et


def gen_order_1(time_series, t, t_next, model, alphas_cump, ets):
    """
    生成下一个时间步的时间序列数据，使用一阶方法（类似于 Euler 方法）。

    参数：
    - time_series: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    - t: 当前时间步标量或张量
    - t_next: 下一个时间步标量或张量
    - model: 模型，用于预测噪声
    - alphas_cump: Alpha 累积系数，形状为 (T+1,)
    - ets: 噪声缓存列表，用于高阶方法

    返回：
    - img_next: 下一个时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    """
    noise = model(time_series, t)
    ets.append(noise)
    img_next = transfer(time_series, t, t_next, noise, alphas_cump)
    return img_next


def transfer(x, t, t_next, et, alphas_cump):
    """
    转移函数，用于更新时间序列数据。

    参数：
    - x: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    - t: 当前时间步标量或张量
    - t_next: 下一个时间步标量或张量
    - et: 预测的噪声，形状为 (batch_size, seqlen, dim)
    - alphas_cump: Alpha 累积系数，形状为 (T+1,)

    返回：
    - x_next: 更新后的时间序列数据，形状为 (batch_size, seqlen, dim)
    """
    # 确保 t 和 t_next 是张量
    if not isinstance(t, th.Tensor):
        t = th.tensor(t, device=x.device)
    if not isinstance(t_next, th.Tensor):
        t_next = th.tensor(t_next, device=x.device)

    at = alphas_cump[t.long() + 1].view(-1, 1, 1)  # (batch_size, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1)  # (batch_size, 1, 1)

    # 更新公式调整为适应时间序列数据
    x_delta = (at_next - at) * (
            (1 / (th.sqrt(at) * (th.sqrt(at) + th.sqrt(at_next)))) * x -
            1 / (th.sqrt(at) * ((1 - at_next).sqrt() + ((1 - at) * at_next).sqrt())) * et
    )

    x_next = x + x_delta
    return x_next


def transfer_dev(x, t, t_next, et, alphas_cump):
    """
    开发版的转移函数，用于更新时间序列数据（可选）。

    参数：
    - x: 当前时间步的时间序列数据，形状为 (batch_size, seqlen, dim)
    - t: 当前时间步标量或张量
    - t_next: 下一个时间步标量或张量
    - et: 预测的噪声，形状为 (batch_size, seqlen, dim)
    - alphas_cump: Alpha 累积系数，形状为 (T+1,)

    返回：
    - x_next: 更新后的时间序列数据，形状为 (batch_size, seqlen, dim)
    """
    # 确保 t 和 t_next 是张量
    if not isinstance(t, th.Tensor):
        t = th.tensor(t, device=x.device)
    if not isinstance(t_next, th.Tensor):
        t_next = th.tensor(t_next, device=x.device)

    at = alphas_cump[t.long() + 1].view(-1, 1, 1)  # (batch_size, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1)  # (batch_size, 1, 1)

    # 计算 x_start
    x_start = th.sqrt(1.0 / at) * x - th.sqrt(1.0 / at - 1) * et
    x_start = x_start.clamp(-1.0, 1.0)

    # 更新 x_next
    x_next = x_start * th.sqrt(at_next) + th.sqrt(1 - at_next) * et

    return x_next
