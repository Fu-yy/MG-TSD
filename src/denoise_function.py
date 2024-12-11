import math
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
    def __init__(self, up_factor=2, in_dim=137, out_dim=137):
        super().__init__()
        self.up_factor = up_factor
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.up_linear = nn.Linear(in_dim, in_dim * up_factor)

    def forward(self, x):
        # 使用线性插值进行上采样
        # x: [B, C, T]
        res = x
        line_res_x = self.up_linear(res)
        return line_res_x

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
备份


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
            UpsampleBlock(up_factor=downsample_factor, in_dim=target_dim // (downsample_factor ** (i+1)))
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
        denoised = inputs[-1]
        cond_up = self.cond_upsamplers[-1](conditions[-1])  # [B, target_dim, T_down]

        # 输入投影
        denoised = self.input_projections[-1](denoised)  # [B, residual_channels, T_down]
        denoised = F.leaky_relu(denoised, 0.4)

        # 残差层
        for layer in self.stages[-1]:
            denoised, skip = layer(denoised, cond_up, diffusion_step)
            skip_connections[-1] += skip

        # 跳跃连接投影
        denoised = self.skip_projections[-1](denoised)
        denoised = F.leaky_relu(denoised, 0.4)
        denoised = self.output_projections[-1](denoised)  # [B, 1, T_down]

        # 逐级上采样并融合跳跃连接
        for i in reversed(range(self.num_stages - 1)):
            # 上采样
            denoised = self.upsamplers[i](denoised)  # [B, C, T]

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

