import math
import torch
import torch.nn as nn
import torch.nn.functional as F


########################################################################
# 1. 球面几何工具函数 (支持任意维度 dim)
########################################################################

def normalize_to_unit_sphere(x, eps=1e-8):
    """
    将 x 归一化到单位球面, x: [N, dim].
    返回与 x 同形状, 且每个向量范数=1.
    """
    norm = x.norm(dim=-1, keepdim=True) + eps
    return x / norm


def project_to_tangent(vec, base, eps=1e-8):
    """
    将 vec 投影到 base 的切空间.
    base: [N, dim], 要满足 \|base\|=1
    vec:  [N, dim]
    返回与 base 正交的向量: vec - <vec, base>* base
    """
    dot_ = (vec * base).sum(dim=-1, keepdim=True)
    return vec - dot_ * base


def log_map_on_sphere(x, base_point, eps=1e-8):
    """
    球面对数映射 (log_{base_point}(x)).
    x, base_point: [N, dim], 都在单位球面 (范数=1).
    返回切向量 v, 使得 exp_{base_point}(v) = x.
    数学公式:
      angle = arccos( <x, base_point> )
      v = ( angle / sin(angle) ) * ( x - <x,base_point> * base_point )
    """
    dot_ = (x * base_point).sum(dim=-1, keepdim=True)
    dot_clamped = torch.clamp(dot_, -1.0, 1.0)
    angle = torch.acos(dot_clamped)  # [N,1]

    sin_angle = torch.sin(angle)
    v = x - dot_clamped * base_point
    v = angle / (sin_angle + eps) * v
    return v


def exp_map_on_sphere(v, base_point, eps=1e-8):
    """
    球面指数映射 (exp_{base_point}(v)).
    v: [N, dim], base_point 的切向量
    base_point: [N, dim], 在单位球面上
    返回在球面上的向量 x = cos(\|v\|)*base_point + sin(\|v\|)*(v/\|v\|)
    并做一次 normalize, 防止数值漂移.
    """
    norm_v = v.norm(dim=-1, keepdim=True) + eps
    unit_v = v / norm_v
    x = torch.cos(norm_v) * base_point + torch.sin(norm_v) * unit_v
    return normalize_to_unit_sphere(x)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

########################################################################
# 2. SphereDiffusion 类: 在任意维度球面上做扩散
########################################################################

class SphereDiffusion(nn.Module):
    def __init__(self, denoise_fn, dim, num_timesteps=10, loss_type="l2",input_size=137):
        """
        denoise_fn: 预测噪声的网络, (x_noisy, t) -> shape same as x_noisy
        dim: 数据维度, 如 137 表示在 S^{136} 球面
        num_timesteps: 扩散步数
        loss_type: "l1" 或 "l2"
        """
        super().__init__()
        self.denoise_fn = denoise_fn
        self.dim = dim
        self.__scale = None
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.input_size = input_size
        self.auto_loss = AutomaticWeightedLoss(2)
        # 简易 schedule: alpha_t = (t+1)/num_timesteps, 逐步从 0->1
        alpha_list = [(t + 1) / float(num_timesteps) for t in range(num_timesteps)]
        self.register_buffer("alpha_list", torch.tensor(alpha_list, dtype=torch.float32))
    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def _get_alpha_t(self, t_index):
        """
        根据时间索引 t_index (0 <= t < num_timesteps),
        取 alpha_t 并返回, shape=[N].
        """
        return self.alpha_list.gather(0, t_index)

    def q_sample(self, x_start, t, noise=None):
        """
        正向加噪: x_t = exp_{x_start}( dt * noise_in_tangent )
        x_start: [B,S,dim], 已在球面上
        t: [B,S], 时间步
        noise: [B,S,dim], 标准正态
        """
        B, S, D = x_start.shape
        assert D == self.dim

        if noise is None:
            noise = torch.randn_like(x_start)

        # flatten
        x_ = x_start.view(B * S, D)
        noise_ = noise.view(B * S, D)
        t_ = t.view(B * S)

        # alpha_t -> dt = sqrt(1 - alpha_t)
        alpha_t = self._get_alpha_t(t_)
        dt_ = torch.sqrt(1.0 - alpha_t).unsqueeze(-1)  # [B*S,1]

        # 保证 x_ 在球面上
        x_ = normalize_to_unit_sphere(x_)

        # 投影噪声到切空间, 乘以 dt_
        v = project_to_tangent(noise_, x_)
        v_norm = v.norm(dim=-1, keepdim=True) + 1e-8
        v = v / v_norm * (dt_ * v_norm)

        # exp_map 回球面
        x_noisy_ = exp_map_on_sphere(v, x_)

        x_noisy = x_noisy_.view(B, S, D)
        return x_noisy

    @torch.no_grad()
    def p_sample(self, x_t, t, cond=None,):
        """
        反向去噪: x_{t-1} = exp_{x_t}( -dt * predicted_noise + dt * eps )
        x_t: [B,S,dim], t: [B,S]
        cond: 可选条件
        """
        B, S, D = x_t.shape
        x_t_ = x_t.view(B * S, D)
        t_ = t.view(B * S)

        alpha_t = self._get_alpha_t(t_)
        dt_ = torch.sqrt(1.0 - alpha_t).unsqueeze(-1)  # [B*S,1]

        # 预测噪声
        predicted_noise = self.denoise_fn(x_t, t, cond=cond)
        predicted_noise_ = predicted_noise.view(B * S, D)
        predicted_noise_ = project_to_tangent(predicted_noise_, x_t_)

        # drift
        drift = - dt_ * predicted_noise_

        # 随机噪声
        eps = torch.randn_like(x_t_)
        eps = project_to_tangent(eps, x_t_)
        # 只有当 t>0 时才加随机噪声
        mask = (t_ > 0).float().unsqueeze(-1)
        random_step = dt_ * eps * mask

        v = drift + random_step
        x_tm1_ = exp_map_on_sphere(v, x_t_)
        x_tm1 = x_tm1_.view(B, S, D)
        return x_tm1


    def log_prob(self, x, cond, share_ratio: float, *args, **kwargs):
        B, T, _ = x.shape

        # time = torch.randint(0, self.num_timesteps,(B , T,), device=x.device).long()
        time = torch.randint(0, self.num_timesteps,(B * T,), device=x.device).long()
        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time,
        )

        return loss

    def p_losses(self, x_start,cond=None, t=None, ):
        """
        训练时, 先用 q_sample 得到 x_noisy,
        再让网络去预测加到 x_start 上的噪声(切向量),
        和真实噪声做 L1/L2 损失.
        """
        B, S, D = x_start.shape
        noise = torch.randn_like(x_start)

        # 得到 x_noisy
        x_noisy = self.q_sample(x_start, t, noise=noise)

        # 网络输出
        predicted_noise = self.denoise_fn(x_noisy, t, cond=cond)

        # 计算真正加到球面上的噪声
        x_ = x_start.view(B * S, D)
        n_ = noise.view(B * S, D)
        t_ = t.view(B * S)
        alpha_t = self._get_alpha_t(t_)
        dt_ = torch.sqrt(1.0 - alpha_t).unsqueeze(-1)

        tangent_noise = project_to_tangent(n_, x_)
        tn_norm = tangent_noise.norm(dim=-1, keepdim=True) + 1e-8
        tangent_noise = tangent_noise / tn_norm * (dt_ * tn_norm)
        tangent_noise = tangent_noise.view(B, S, D)

        # loss
        # if self.loss_type == "l1":
        #     loss = F.l1_loss(predicted_noise, tangent_noise)
        # else:  # default l2
        #     loss = F.mse_loss(predicted_noise, tangent_noise)

        manifold_loss = self.manifold_combined_loss(predicted_noise, tangent_noise,alpha=0.3)

        return manifold_loss

    def manifold_combined_loss(self,predicted_noise, tangent_noise, alpha=0.5):
        mse_loss = F.mse_loss(predicted_noise, tangent_noise)
        cosine_loss = 1 - F.cosine_similarity(predicted_noise, tangent_noise, dim=-2).mean()
        res = self.auto_loss(mse_loss, cosine_loss)
        return res
    @torch.no_grad()
    def forward_process_demo(self, x0):
        """
        演示: 从 x0 开始, 连续做 q_sample(num_timesteps 次),
        返回一个列表: [x_1, x_2, ..., x_T].
        """
        B, S, D = x0.shape
        x = x0
        seq = []
        for step in range(self.num_timesteps):
            t_val = torch.full((B, S), step, device=x0.device, dtype=torch.long)
            x = self.q_sample(x, t_val)
            seq.append(x)
        return seq
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

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, share_ratio: float):
        device = cond.device

        B,S,D = shape
        img = torch.randn(shape, device=device)
        # inter_steps = int(self.num_timesteps*(1-share_ratio))
        inter_steps = 0
        for i in reversed(range(inter_steps, self.num_timesteps)):
            img = self.p_sample(
                x_t=img, cond=cond, t=torch.full(
                    (B,), i, device=device, dtype=torch.long),
            )
            # img = self.p_sample(
            #     x_t=img, cond=cond, t=torch.full(
            #         (B, S), i, device=device, dtype=torch.long),
            # )

        return img
    @torch.no_grad()
    def backward_process_demo(self, xT, cond=None):
        """
        演示: 从 xT 开始, 连续做 p_sample(num_timesteps 次),
        返回一个列表: [x_{T-1}, x_{T-2}, ..., x_0].
        """
        B, S, D = xT.shape
        x = xT
        seq = []
        for step in reversed(range(self.num_timesteps)):
            t_val = torch.full((B, S), step, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t_val, cond=cond)
            seq.append(x)
        return seq


########################################################################
# 3. 一个简单去噪网络: SimpleDenoiseMLP (可替换成你自己的网络)
########################################################################

class SimpleDenoiseMLP(nn.Module):
    """
    任意去噪网络皆可, 只要输出 shape 与 x_noisy 相同, 表示预测噪声.
    这里演示一个简单 MLP(不含复杂时间嵌入等).
    """

    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 留给时间 t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x_noisy, t, cond=None):
        """
        x_noisy: [B,S,dim], t: [B,S]
        """
        B, S, D = x_noisy.shape
        # flatten
        x_ = x_noisy.view(B * S, D)
        t_ = t.view(B * S, 1).float()
        inp = torch.cat([x_, t_], dim=-1)  # [B*S, dim+1]
        out_ = self.net(inp)
        out = out_.view(B, S, D)
        return out


########################################################################
# 4. 测试示例
########################################################################

def example_usage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 假设你要处理 dim=137 的球面:
    dim = 137
    # 批大小 B=2, 序列长度 S=3
    B, S = 2, 3

    # 1) 构造一个简单的去噪网络
    denoise_net = SimpleDenoiseMLP(dim=dim, hidden_dim=64).to(device)

    # 2) 构造 SphereDiffusion
    diffusion = SphereDiffusion(denoise_fn=denoise_net, dim=dim,
                                num_timesteps=5, loss_type="l2").to(device)

    # 3) 生成随机数据 x0: [B,S,dim], 并归一化到球面
    x0 = torch.randn(B, S, dim, device=device)
    x0 = normalize_to_unit_sphere(x0.view(B * S, dim)).view(B, S, dim)

    # 4) 简易训练循环
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)

    for step in range(1):
        t_rand = torch.randint(0, diffusion.num_timesteps, (B, S), device=device)
        loss = diffusion.p_losses(x0, t_rand)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Step[{step + 1}] loss={loss.item():.6f}")

    # 5) 演示正向过程
    forward_seq = diffusion.forward_process_demo(x0)
    print(f"Forward sequence length: {len(forward_seq)}. Last shape={forward_seq[-1].shape}")

    # 6) 演示反向过程: 从 xT -> 回到 x0
    xT = forward_seq[-1]
    backward_seq = diffusion.backward_process_demo(xT)
    print(f"Backward sequence length: {len(backward_seq)}. Last shape={backward_seq[-1].shape}")
    print("Done!")


if __name__ == "__main__":
    example_usage()
