import math

import torch
from torch import nn
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


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        """Denoising Network

        Args:
            target_dim (int): Target dimension 1
            cond_length (int): Condition length 100
            time_emb_dim (int, optional): Time embedding. Defaults to 16.
            residual_layers (int, optional): Number of residual layers. Defaults to 8.
            residual_channels (int, optional):  Residual channels. Defaults to 8.
            dilation_cycle_length (int, optional): Dilation cycle length. Defaults to 2.
            residual_hidden (int, optional): Residual hidden size. Defaults to 64.
        """
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )  # 1D convolution shape [batch_size, residual_channels, target_dim]
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )  # Time embedding shape [batch_size, proj_dim]
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )  # Condition upsampling shape [batch_size, target_dim]
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(
            residual_channels, residual_channels, 3)  # Skip connection
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)  # Output

        # Kaiming initialization
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        # Initialize output weights to 0
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        x = self.input_projection(inputs)  # [B,8,T]
        x = F.leaky_relu(x, 0.4)  # [B,8,T]

        diffusion_step = self.diffusion_embedding(time)  # [B,64]
        cond_up = self.cond_upsampler(cond)  # [B,1,T]
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / \
            math.sqrt(len(self.residual_layers))  # [B,8,T]
        x = self.skip_projection(x)  # [B,8,T]
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)  # [B,1,T]
        return x  # [B,1,T]



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
        多级(渐进式)扩散模型框架

        Args:
            target_dim (int): 原始目标长度
            cond_length (int): 条件长度
            num_stages (int): 分几个阶段进行去噪
            downsample_factor (int): 每阶段下采样因子
        """
        super().__init__()
        self.num_stages = num_stages
        self.downsample_factor = downsample_factor

        # 定义每个阶段的 EpsilonTheta 模型
        self.stages = nn.ModuleList([
            EpsilonTheta(
                target_dim=target_dim // (downsample_factor ** i),
                cond_length=cond_length // (downsample_factor ** i),
                time_emb_dim=time_emb_dim,
                residual_layers=residual_layers,
                residual_channels=residual_channels,
                dilation_cycle_length=dilation_cycle_length,
                residual_hidden=residual_hidden
            ) for i in range(num_stages)
        ])

        # 下采样和上采样模块
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
            for _ in range(num_stages - 1)
        ])

        self.upsamplers = nn.ModuleList([
            UpsampleBlock(up_factor=downsample_factor,in_dim = target_dim // (downsample_factor ** (i+1)))
            for i in range(num_stages - 1)
        ])

        # 跳跃连接的卷积层以匹配通道数
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
            for _ in range(num_stages - 1)
        ])

    def forward(self, x, time, cond):
        """
        多级渐进式去噪:
        1. 按阶段下采样输入和条件
        2. 从最低分辨率开始逐级去噪
        3. 上采样并融合跳跃连接
        """
        inputs = [x]
        conditions = [cond]

        # 下采样输入和条件到各个阶段
        for i in range(1, self.num_stages):
            x_down = self.downsamplers[i-1](inputs[i-1])
            cond_down = self.downsamplers[i-1](conditions[i-1])
            inputs.append(x_down)
            conditions.append(cond_down)

        # 存储每个阶段的跳跃连接
        skip_connections = []

        # 从最低分辨率开始去噪
        denoised = self.stages[-1](inputs[-1], time, conditions[-1])

        # 逐级上采样并融合跳跃连接
        for i in reversed(range(self.num_stages - 1)):
            # 上采样
            denoised = self.upsamplers[i](denoised)

            # 确保上采样后的长度与对应输入匹配
            target_length = inputs[i].size(-1)
            if denoised.size(-1) != target_length:
                denoised = F.interpolate(denoised, size=target_length, mode='linear', align_corners=True)

            # 跳跃连接：融合来自下采样阶段的特征
            skip = self.stages[i](inputs[i], time, conditions[i])

            # 融合方式可以是相加、拼接后卷积等，这里选择相加
            denoised = denoised + skip

            # 可选：如果需要，可以添加更多融合操作

        return denoised

class UpsampleBlock(nn.Module):
    def __init__(self, up_factor=2,in_dim=137,out_dim=137):
        super().__init__()
        self.up_factor = up_factor
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.up_linear = nn.Linear(in_dim,in_dim * up_factor)

    def forward(self, x):
        # 使用线性插值进行上采样
        # x: [B, C, T]
        res = x
        # T = x.size(-1)
        # new_T = T * self.up_factor
        # x = F.interpolate(x, size=new_T, mode='linear', align_corners=True)
        line_res_x = self.up_linear(res)
        return line_res_x



if __name__ == '__main__':
    # ------------------------------------------------------------
    # 示例使用
    # ------------------------------------------------------------

    # 假设 EpsilonTheta 等类已正确定义

    # 初始化模型
    progressive_model = ProgressiveDenoisingModel(
        target_dim=128,
        cond_length=100,
        num_stages=2,  # 可以根据需求增加阶段数
        downsample_factor=2
    )

    # 创建示例输入
    x = torch.randn(16, 1, 128)  # [batch_size, channels, length]
    cond = torch.randn(16, 1, 100)  # [batch_size, channels, cond_length]
    time = torch.randint(0, 500, (16,))  # [batch_size]

    # 前向传播
    output = progressive_model(x, time, cond)
    print(output.shape)  # 应该为 [16, 1, 128]