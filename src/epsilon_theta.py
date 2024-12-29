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

class CrossAttentionBlock(nn.Module):
    """
    将 condition embedding 作为 key/value,
    将主分支 (x) 作为 query, 做一次 cross-attention 融合。
    这里给出最简化实现 (single-head)；你可扩展成多头注意力。
    """

    def __init__(self, channels, cond_channels,residual_channels):
        super().__init__()
        # 将 x 投影到 query
        self.to_q = nn.Linear(channels, channels)
        # 将 cond 投影到 key/value，这里 cond 只有 [B, cond_channels]，需要 reshape
        self.to_k = nn.Linear(cond_channels, channels)
        self.to_v = nn.Linear(cond_channels, channels)

        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)

        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
    )
        self.norm1 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
        )

    def compute_attention(self, q, k, v):
        scores = torch.matmul(q, k) / torch.sqrt(torch.tensor(q.size(-1)))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
    def forward(self, x, cond_emb):
        """
        x: [B, channels, T]  -- query
        cond_emb: [B, cond_channels] -- global条件
        """
        B, C, T = x.shape
        cond_emb = self.conditioner_projection(cond_emb)



        # query
        q = self.to_q(x)  # [B, C, T]

        # key, value  (broadcast到时域维度 T 或者不 broadcast 都可以)
        k = self.to_k(cond_emb).permute(0,2,1)  # [B, channels, 1]
        v = self.to_v(cond_emb)  # [B, channels, 1]
        attn_output,_ = self.compute_attention(q, k, v)

        query = self.norm1(q + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        output = self.norm2(query + self.dropout(ff_output))
        return output+ x
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation,x_dim,cond_dim):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.cond_att = CrossAttentionBlock(channels=x_dim+4,cond_channels=cond_dim+4,residual_channels=residual_channels)
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(
            residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step,shape=None):
        # ------------ freqdiffusion begin ------------------------
        # B,L,N,X_D,C_D=shape
        # ------------ freqdiffusion end ------------------------


        diffusion_step = self.diffusion_projection(
            diffusion_step).unsqueeze(-1)
        # conditioner = self.conditioner_projection(conditioner)


        # ------------ freqdiffusion begin ------------------------

        # x_reshape = torch.reshape(x,[B,L,N,self.residual_channels,-1])

        # diffusion_step = diffusion_step.unsqueeze(1).unsqueeze(1)

        # y = x_reshape + diffusion_step
        # y = torch.reshape(y,[ B*L*N,self.residual_channels,-1])
        # ------------ freqdiffusion end ------------------------

        y = x + diffusion_step
        y = self.dilated_conv(y)

        y = self.cond_att(y,conditioner)

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
                    x_dim=target_dim,
                    cond_dim=cond_length,
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

        # ------------------------ freq diffusion begin ---------------------------
        # B,L,X_D,N = inputs.size()
        # _,_,C_D,_ = cond.size()
        # shape = [B,L,N,X_D,C_D]
        # inputs = inputs.permute(0,1,3,2).contiguous() # B,L,N,X_D
        # cond = cond.permute(0,1,3,2).contiguous() # B,L,C_N,D
        # inputs = torch.reshape(inputs, [B*L*N,1,X_D])
        # cond = torch.reshape(cond, [B*L*N,1,C_D])

        # shape = None
        # ------------------------ freq diffusion end ---------------------------


        x = self.input_projection(inputs)  # [B,8,T]
        x = F.leaky_relu(x, 0.4)  # [B,8,T]

        diffusion_step = self.diffusion_embedding(time)  # [B,64]
        # cond_up = self.cond_upsampler(cond)  # [B,1,T]
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / \
            math.sqrt(len(self.residual_layers))  # [B,8,T]
        x = self.skip_projection(x)  # [B,8,T]
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)  # [B,1,T]


        # ------------------------ freq diffusion begin ---------------------------\
        # x = torch.reshape(x, [B,L,N,X_D])
        # x = x.permute(0,1,3,2).contiguous()
        # ------------------------ freq diffusion end ---------------------------

        return x  # [B,1,T]
