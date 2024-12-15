import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x  # [batch_size, proj_dim]

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat(
            [torch.sin(table), torch.cos(table)], dim=1)  # [T,2*dim]
        return table

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(
            residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(
            diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip

class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, up_factor=2, in_dim=137, out_dim=137,in_pad=4,out_pad=0,target_dim=137):
        super().__init__()
        self.up_factor = up_factor
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.up_linear = nn.Linear(in_dim + in_pad, (in_dim+out_pad) * up_factor)

    def forward(self, x):
        # 使用线性插值进行上采样
        # x: [B, C, T]
        res = x
        line_res_x = self.up_linear(res)
        return line_res_x
'''
class ProgressiveDenoisingModel(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        num_stages=4,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
        downsample_factor=2
    ):
        """
        多级(渐进式)扩散模型框架，集成去噪逻辑.

        Args:
            target_dim (int): 原始目标长度
            cond_length (int): 条件长度
            num_stages (int): 分几个阶段进行去噪
            downsample_factor (int): 每阶段下采样因子
            time_emb_dim (int, optional): 时间嵌入维度。默认为16。
            residual_layers (int, optional): 残差层数。默认为8。
            residual_channels (int, optional): 残差通道数。默认为8。
            dilation_cycle_length (int, optional): 膨胀循环长度。默认为2。
            residual_hidden (int, optional): 残差隐藏层大小。默认为64。
        """
        super().__init__()
        self.num_stages = num_stages
        self.downsample_factor = downsample_factor

        # 时间嵌入
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )  # Time embedding shape [batch_size, residual_hidden]

        # 条件上采样
        self.cond_upsamplers = nn.ModuleList([
            CondUpsampler(
                cond_length=cond_length // (downsample_factor ** i),
                target_dim=target_dim // (downsample_factor ** i)
            ) for i in range(num_stages)
        ])

        # 定义每个阶段的残差层
        self.stages = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(
                    hidden_size=residual_hidden,
                    residual_channels=residual_channels,
                    dilation=2 ** (j % dilation_cycle_length)
                ) for j in range(residual_layers)
            ]) for _ in range(num_stages)
        ])

        # 输入投影和跳跃连接投影
        self.input_projections = nn.ModuleList([
            nn.Conv1d(1, residual_channels, 1, padding=2, padding_mode="circular")
            for _ in range(num_stages)
        ])

        self.skip_projections = nn.ModuleList([
            nn.Conv1d(residual_channels, residual_channels, 3)
            for _ in range(num_stages)
        ])

        # 下采样和上采样模块
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
            for _ in range(num_stages - 1)
        ])

        self.upsamplers = nn.ModuleList([
            UpsampleBlock(up_factor=downsample_factor, in_dim=target_dim // (downsample_factor ** (i)))
            for i in range(num_stages )
        ])

        # 跳跃连接的卷积层以匹配通道数
        # self.skip_convs = nn.ModuleList([
        #     nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        #     for _ in range(num_stages - 1)
        # ])

        # 输出投影
        self.output_projections = nn.ModuleList([
            nn.Conv1d(residual_channels, 1, 3)
            for _ in range(num_stages)
        ])

        self.skip_output_projections = nn.ModuleList([
            nn.Conv1d(residual_channels, 1, 1)
            for _ in range(num_stages)
        ])

        # 初始化权重
        for proj in self.input_projections + self.skip_projections:
            nn.init.kaiming_normal_(proj.weight)
        for out_proj in self.output_projections:
            nn.init.zeros_(out_proj.weight)

    def forward(self, x, time, cond):
        """
        多级渐进式去噪:
        1. 按阶段下采样输入和条件
        2. 从最低分辨率开始逐级去噪
        3. 上采样并融合跳跃连接

        Args:
            x (Tensor): 输入噪声，[B, 1, T]
            time (Tensor): 时间步，[B]
            cond (Tensor): 条件输入，[B, cond_length]

        Returns:
            Tensor: 去噪后的输出，[B, 1, T]
        """
        # 下采样输入和条件到各个阶段
        inputs = [x]
        conditions = [cond]
        for i in range(1, self.num_stages):
            x_down = self.downsamplers[i-1](inputs[i-1])
            cond_down = self.downsamplers[i-1](conditions[i-1])
            inputs.append(x_down)
            conditions.append(cond_down)

        # 存储每个阶段的跳跃连接
        skip_connections = [0] * self.num_stages

        # 时间嵌入
        diffusion_step = self.diffusion_embedding(time)  # [B, residual_hidden]

        # 从最低分辨率开始去噪
        # denoised = inputs[-1]
        # cond_up = self.cond_upsamplers[-1](conditions[-1])  # [B, target_dim, T_down]
        #
        # # 输入投影
        # denoised = self.input_projections[-1](denoised)  # [B, residual_channels, T_down]
        # denoised = F.leaky_relu(denoised, 0.4)
        #
        # # 残差层
        # for layer in self.stages[-1]:
        #     denoised, skip = layer(denoised, cond_up, diffusion_step)
        #     skip_connections[-1] += skip
        #
        # # 跳跃连接投影
        # denoised = self.skip_projections[-1](denoised)
        # denoised = F.leaky_relu(denoised, 0.4)
        # denoised = self.output_projections[-1](denoised)  # [B, 1, T_down]

        # 逐级上采样并融合跳跃连接
        for i in reversed(range(self.num_stages)):
            # 上采样
            denoised = self.upsamplers[i](inputs[i])  # [B, C, T]

            # 确保上采样后的长度与对应输入匹配
            target_length = inputs[i].size(-1)
            if denoised.size(-1) != target_length:
                denoised = F.interpolate(denoised, size=target_length, mode='linear', align_corners=True)

            # 条件上采样
            cond_up = self.cond_upsamplers[i](conditions[i])  # [B, target_dim, T]

            # 输入投影
            stage_input = self.input_projections[i](inputs[i])  # [B, residual_channels, T]
            stage_input = F.leaky_relu(stage_input, 0.4)

            # 残差层
            for layer in self.stages[i]:
                denoised, skip = layer(stage_input, cond_up, diffusion_step)
                skip_connections[i] += skip

            # 跳跃连接投影
            denoised = self.skip_projections[i](denoised)
            denoised = F.leaky_relu(denoised, 0.4)
            denoised = self.output_projections[i](denoised)  # [B, 1, T]
            skip_connection_projections = self.skip_output_projections[i](skip_connections[i][...,-denoised.size(-1):])
            # 融合跳跃连接（可以选择加权、拼接等方式）
            # 这里选择简单相加
            denoised = denoised + skip_connection_projections

        return denoised  # [B, 1, T]




'''
# 备份
def gaussian_low_pass_filter_gpu(data, cutoff_ratio=0.1, sigma=1.0):
    """
    对输入的时间序列数据应用高斯低通滤波器，去除高频噪声（使用 rFFT）。

    参数:
    - data: 输入时间序列，形状为 (batchsize, seqlen, dim)
    - cutoff_ratio: 控制高斯滤波器的中心频率 (0 < cutoff_ratio < 0.5)
    - sigma: 高斯滤波器的标准差，控制滤波器的平滑程度

    返回:
    - filtered_data: 经过高斯低通滤波后的时间序列，形状与输入相同
    """
    batchsize, seqlen, dim = data.shape
    # 使用 rFFT 进行傅里叶变换
    fft = torch.fft.rfft(data, dim=1)

    # 创建高斯低通滤波器掩码
    freqs = torch.fft.rfftfreq(seqlen, d=1.0).to(data.device)  # 频率范围 [0, 0.5]
    gaussian_mask = torch.exp(-0.5 * ((freqs / cutoff_ratio) / sigma) ** 2).unsqueeze(0).unsqueeze(-1)  # 形状 (1, seqlen//2+1, 1)

    # 应用掩码
    fft_filtered = fft * gaussian_mask

    # 反傅里叶变换，使用 irfft 恢复时域信号
    filtered_data = torch.fft.irfft(fft_filtered, n=seqlen, dim=1)

    return filtered_data



class ProgressiveDenoisingModel(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        num_stages=4,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
        downsample_factor=2
    ):
        """
        多级(渐进式)扩散模型框架，集成去噪逻辑.

        Args:
            target_dim (int): 原始目标长度
            cond_length (int): 条件长度
            num_stages (int): 分几个阶段进行去噪
            downsample_factor (int): 每阶段下采样因子
            time_emb_dim (int, optional): 时间嵌入维度。默认为16。
            residual_layers (int, optional): 残差层数。默认为8。
            residual_channels (int, optional): 残差通道数。默认为8。
            dilation_cycle_length (int, optional): 膨胀循环长度。默认为2。
            residual_hidden (int, optional): 残差隐藏层大小。默认为64。
        """
        super().__init__()
        self.num_stages = num_stages
        self.downsample_factor = downsample_factor

        # 时间嵌入
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )  # Time embedding shape [batch_size, residual_hidden]

        # 条件上采样
        self.cond_upsamplers = nn.ModuleList([
            CondUpsampler(
                cond_length=cond_length // (downsample_factor ** i),
                target_dim=target_dim // (downsample_factor ** i)
            ) for i in range(num_stages)
        ])

        # 定义每个阶段的残差层
        self.stages = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(
                    hidden_size=residual_hidden,
                    residual_channels=residual_channels,
                    dilation=2 ** (j % dilation_cycle_length)
                ) for j in range(residual_layers)
            ]) for _ in range(num_stages)
        ])

        # 输入投影和跳跃连接投影
        self.input_projections = nn.ModuleList([
            nn.Conv1d(1, residual_channels, 1, padding=2, padding_mode="circular")
            for _ in range(num_stages)
        ])

        self.skip_projections = nn.ModuleList([
            nn.Conv1d(residual_channels, residual_channels, 3)
            for _ in range(num_stages)
        ])

        # 下采样和上采样模块
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
            for _ in range(num_stages - 1)
        ])

        self.upsamplers = nn.ModuleList([
            UpsampleBlock(up_factor=downsample_factor, in_dim=target_dim // (downsample_factor ** (i+1)),target_dim=target_dim,in_pad=0,out_pad=0)
            for i in range(num_stages - 1)
        ])

        # 跳跃连接的卷积层以匹配通道数
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
            for _ in range(num_stages - 1)
        ])

        # 输出投影
        self.output_projections = nn.ModuleList([
            nn.Conv1d(residual_channels, 1, 3)
            for _ in range(num_stages)
        ])

        self.skip_output_projections = nn.ModuleList([
            nn.Conv1d(residual_channels, 1, 1)
            for _ in range(num_stages)
        ])

        # 初始化权重
        for proj in self.input_projections + self.skip_projections:
            nn.init.kaiming_normal_(proj.weight)
        for out_proj in self.output_projections:
            nn.init.zeros_(out_proj.weight)

    def forward(self, x, time, cond):
        """
        多级渐进式去噪:
        1. 按阶段下采样输入和条件
        2. 从最低分辨率开始逐级去噪
        3. 上采样并融合跳跃连接

        Args:
            x (Tensor): 输入噪声，[B, 1, T]
            time (Tensor): 时间步，[B]
            cond (Tensor): 条件输入，[B, cond_length]

        Returns:
            Tensor: 去噪后的输出，[B, 1, T]
        """
        # 下采样输入和条件到各个阶段
        inputs = [x]
        cond = gaussian_low_pass_filter_gpu(cond.permute(0,2,1).contiguous(),cutoff_ratio=0.4).permute(0,2,1).contiguous()
        conditions = [cond]
        for i in range(1, self.num_stages):
            x_down = self.downsamplers[i - 1](inputs[i - 1])
            cond_down = self.downsamplers[i - 1](conditions[i - 1])
            inputs.append(x_down)
            conditions.append(cond_down)

        # 存储每个阶段的跳跃连接
        skip_connections = [0] * self.num_stages

        # 时间嵌入
        diffusion_step = self.diffusion_embedding(time)  # [B, residual_hidden]

        # 从最低分辨率开始去噪
        denoised = inputs[-1]
        cond_up = self.cond_upsamplers[-1](conditions[-1])  # [B, target_dim, T_down]

        # 输入投影
        denoised = self.input_projections[-1](denoised)  # [B, residual_channels, T_down]
        denoised = F.leaky_relu(denoised, 0.4)

        # 残差层
        for layer in self.stages[-1]:
            denoised, skip = layer(denoised, cond_up, diffusion_step)
            # 跳跃连接投影
            skip = self.skip_projections[-1](skip)
            skip = F.leaky_relu(skip, 0.4)
            skip = self.output_projections[-1](skip)  # [B, 1, T_down]

            skip_connections[-1] += skip


        # 逐级上采样并融合跳跃连接
        for i in reversed(range(self.num_stages - 1)):
            # 上采样
            up_skip = self.upsamplers[i](skip_connections[i+1])  # [B, C, T]


            # 条件上采样
            cond_up = self.cond_upsamplers[i](conditions[i])  # [B, target_dim, T]
            # 确保上采样后的长度与对应输入匹配
            target_length = cond_up.size(-1)
            if up_skip.size(-1) != target_length:
                up_skip = F.interpolate(up_skip, size=target_length, mode='linear', align_corners=True)


            # 输入投影 + 残差
            stage_input = self.input_projections[i](inputs[i] )  # [B, residual_channels, T]

            cond_up =cond_up + up_skip[...,-stage_input.size(-1):]
            stage_input = F.leaky_relu(stage_input, 0.4)

            # 残差层
            for layer in self.stages[i]:
                stage_input, skip = layer(stage_input, cond_up, diffusion_step)

                # 跳跃连接投影
                skip = self.skip_projections[i](skip)
                skip = F.leaky_relu(skip, 0.4)
                skip = self.output_projections[i](skip)  # [B, 1, T_down]
                skip_connections[i] += skip
            if i == 0:
                # 跳跃连接投影
                # denoised = self.skip_projections[i](denoised)
                # denoised = F.leaky_relu(denoised, 0.4)
                # denoised = self.output_projections[i](denoised)  # [B, 1, T]
                # skip_connection_projections = self.skip_output_projections[i](skip_connections[i][...,-denoised.size(-1):])
                # 融合跳跃连接（可以选择加权、拼接等方式）
                # 这里选择简单相加
                # denoised = denoised + skip_connection_projections
                denoised = skip_connections[i]

        return denoised  # [B, 1, T]

# # 备份
#
#
# class ProgressiveDenoisingModel(nn.Module):
#     def __init__(
#         self,
#         target_dim,
#         cond_length,
#         num_stages=4,
#         time_emb_dim=16,
#         residual_layers=8,
#         residual_channels=8,
#         dilation_cycle_length=2,
#         residual_hidden=64,
#         downsample_factor=2
#     ):
#         """
#         多级(渐进式)扩散模型框架，集成去噪逻辑.
#
#         Args:
#             target_dim (int): 原始目标长度
#             cond_length (int): 条件长度
#             num_stages (int): 分几个阶段进行去噪
#             downsample_factor (int): 每阶段下采样因子
#             time_emb_dim (int, optional): 时间嵌入维度。默认为16。
#             residual_layers (int, optional): 残差层数。默认为8。
#             residual_channels (int, optional): 残差通道数。默认为8。
#             dilation_cycle_length (int, optional): 膨胀循环长度。默认为2。
#             residual_hidden (int, optional): 残差隐藏层大小。默认为64。
#         """
#         super().__init__()
#         self.num_stages = num_stages
#         self.downsample_factor = downsample_factor
#
#         # 时间嵌入
#         self.diffusion_embedding = DiffusionEmbedding(
#             time_emb_dim, proj_dim=residual_hidden
#         )  # Time embedding shape [batch_size, residual_hidden]
#
#         # 条件上采样
#         self.cond_upsamplers = nn.ModuleList([
#             CondUpsampler(
#                 cond_length=cond_length // (downsample_factor ** i),
#                 target_dim=target_dim // (downsample_factor ** i)
#             ) for i in range(num_stages)
#         ])
#
#         # 定义每个阶段的残差层
#         self.stages = nn.ModuleList([
#             nn.ModuleList([
#                 ResidualBlock(
#                     hidden_size=residual_hidden,
#                     residual_channels=residual_channels,
#                     dilation=2 ** (j % dilation_cycle_length)
#                 ) for j in range(residual_layers)
#             ]) for _ in range(num_stages)
#         ])
#
#         # 输入投影和跳跃连接投影
#         self.input_projections = nn.ModuleList([
#             nn.Conv1d(1, residual_channels, 1, padding=2, padding_mode="circular")
#             for _ in range(num_stages)
#         ])
#
#         self.skip_projections = nn.ModuleList([
#             nn.Conv1d(residual_channels, residual_channels, 3)
#             for _ in range(num_stages)
#         ])
#
#         # 下采样和上采样模块
#         self.downsamplers = nn.ModuleList([
#             nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
#             for _ in range(num_stages - 1)
#         ])
#
#         self.upsamplers = nn.ModuleList([
#             UpsampleBlock(up_factor=downsample_factor, in_dim=target_dim // (downsample_factor ** (i+1)))
#             for i in range(num_stages - 1)
#         ])
#
#         # 跳跃连接的卷积层以匹配通道数
#         self.skip_convs = nn.ModuleList([
#             nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
#             for _ in range(num_stages - 1)
#         ])
#
#         # 输出投影
#         self.output_projections = nn.ModuleList([
#             nn.Conv1d(residual_channels, 1, 3)
#             for _ in range(num_stages)
#         ])
#
#         self.skip_output_projections = nn.ModuleList([
#             nn.Conv1d(residual_channels, 1, 1)
#             for _ in range(num_stages)
#         ])
#
#         # 初始化权重
#         for proj in self.input_projections + self.skip_projections:
#             nn.init.kaiming_normal_(proj.weight)
#         for out_proj in self.output_projections:
#             nn.init.zeros_(out_proj.weight)
#
#     def forward(self, x, time, cond):
#         """
#         多级渐进式去噪:
#         1. 按阶段下采样输入和条件
#         2. 从最低分辨率开始逐级去噪
#         3. 上采样并融合跳跃连接
#
#         Args:
#             x (Tensor): 输入噪声，[B, 1, T]
#             time (Tensor): 时间步，[B]
#             cond (Tensor): 条件输入，[B, cond_length]
#
#         Returns:
#             Tensor: 去噪后的输出，[B, 1, T]
#         """
#         # 下采样输入和条件到各个阶段
#         inputs = [x]
#         conditions = [cond]
#         for i in range(1, self.num_stages):
#             x_down = self.downsamplers[i-1](inputs[i-1])
#             cond_down = self.downsamplers[i-1](conditions[i-1])
#             inputs.append(x_down)
#             conditions.append(cond_down)
#
#         # 存储每个阶段的跳跃连接
#         skip_connections = [0] * self.num_stages
#
#         # 时间嵌入
#         diffusion_step = self.diffusion_embedding(time)  # [B, residual_hidden]
#
#         # 从最低分辨率开始去噪
#         denoised = inputs[-1]
#         cond_up = self.cond_upsamplers[-1](conditions[-1])  # [B, target_dim, T_down]
#
#         # 输入投影
#         denoised = self.input_projections[-1](denoised)  # [B, residual_channels, T_down]
#         denoised = F.leaky_relu(denoised, 0.4)
#
#         # 残差层
#         for layer in self.stages[-1]:
#             denoised, skip = layer(denoised, cond_up, diffusion_step)
#             skip_connections[-1] += skip
#
#         # 跳跃连接投影
#         denoised = self.skip_projections[-1](denoised)
#         denoised = F.leaky_relu(denoised, 0.4)
#         denoised = self.output_projections[-1](denoised)  # [B, 1, T_down]
#
#         # 逐级上采样并融合跳跃连接
#         for i in reversed(range(self.num_stages - 1)):
#             # 上采样
#             denoised = self.upsamplers[i](denoised)  # [B, C, T]
#
#             # 确保上采样后的长度与对应输入匹配
#             target_length = inputs[i].size(-1)
#             if denoised.size(-1) != target_length:
#                 denoised = F.interpolate(denoised, size=target_length, mode='linear', align_corners=True)
#
#             # 条件上采样
#             cond_up = self.cond_upsamplers[i](conditions[i])  # [B, target_dim, T]
#
#             # 输入投影
#             stage_input = self.input_projections[i](inputs[i])  # [B, residual_channels, T]
#             stage_input = F.leaky_relu(stage_input, 0.4)
#
#             # 残差层
#             for layer in self.stages[i]:
#                 denoised, skip = layer(stage_input, cond_up, diffusion_step)
#                 skip_connections[i] += skip
#
#             # 跳跃连接投影
#             denoised = self.skip_projections[i](denoised)
#             denoised = F.leaky_relu(denoised, 0.4)
#             denoised = self.output_projections[i](denoised)  # [B, 1, T]
#             skip_connection_projections = self.skip_output_projections[i](skip_connections[i][...,-denoised.size(-1):])
#             # 融合跳跃连接（可以选择加权、拼接等方式）
#             # 这里选择简单相加
#             denoised = denoised + skip_connection_projections
#
#         return denoised  # [B, 1, T]



class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        dimension = 128
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

        self.flatt = nn.Flatten(start_dim=-2),
        self.l1 =nn.Linear(pn, pn // 2),
        self.bn1 =nn.BatchNorm1d(pn // 2),
        # self.bn1 =nn.BatchNorm1d(pn // 2),
        self.act =nn.ReLU(),
        self.l2 =nn.Linear(pn // 2, dimension),
        self.dr = nn.Dropout(head_dropout),



    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        # x= self.flatt(x)
        # x = self.l1(x)
        # x = self.bn1(x)
        # x = self.act(x)
        # x = self.l2(x)
        # x = self.dr(x)


        x = self.pooler(x) # [(bs * n_vars) x dimension]  # 224 96 32 -- 224 128
        return x





class ContrastiveWeight(nn.Module):

    def __init__(self, temperature=0.2,positive_nums=3):
        super(ContrastiveWeight, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size) #  主对角线  448*448
        mask = torch.from_numpy(diag) #  448*448
        mask = mask.type(torch.bool)#  主对角线  448*448  True False

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        positives_mask = np.zeros(similarity_matrix.size()) #
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)  # k偏移相对位置
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0  # 主对角线置为0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0  # 主对角线置为0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat((torch.ones(cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(batch_emb_om.device).float()

        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix, logits, positives_mask

def get_config():
    import argparse
    import torch
    import random
    import numpy as np
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SimMTM')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    parser.add_argument('--train_only', type=bool, required=False, default=False,
                        help='perform training on full input dataset without validation and testing')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./outputs/checkpoints/',
                        help='location of model fine-tuning checkpoints')
    parser.add_argument('--pretrain_checkpoints', type=str, default='./outputs/pretrain_checkpoints/',
                        help='location of model pre-training checkpoints')
    parser.add_argument('--transfer_checkpoints', type=str, default='ckpt_best.pth',
                        help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
    parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')
    parser.add_argument('--select_channels', type=float, default=1, help='select the rate of channels to train')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--patch_len', type=int, default=12, help='path length')
    parser.add_argument('--stride', type=int, default=12, help='stride')

    # optimization
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # Pre-train
    parser.add_argument('--lm', type=int, default=3, help='average masking length')
    parser.add_argument('--positive_nums', type=int, default=3, help='masking series numbers')
    parser.add_argument('--rbtp', type=int, default=1,
                        help='0: rebuild the embedding of oral series; 1: rebuild oral series')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
    parser.add_argument('--masked_rule', type=str, default='geometric',
                        help='geometric, random, masked tail, masked head')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    batch = 128
    seqlen=48
    dim=128
    # x = torch.randn([batch, seqlen, dim])
    # x_r = torch.reshape(x,[batch*seqlen,1,dim])
    x_r = torch.randn(batch*seqlen,1,dim)
    configs = get_config()
    configs.seq_len = seqlen
    configs.d_model = dim
    pooler = Pooler_Head(1, configs.d_model, head_dropout=0.1)
    contra = ContrastiveWeight()

    res = pooler(x_r)
    loss_cl, similarity_matrix, logits, positives_mask = contra(res)

    c = '[end'