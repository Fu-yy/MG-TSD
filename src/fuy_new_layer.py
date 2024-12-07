import torch
import torch.nn as nn
import torch.nn.functional as F
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim,input_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.linear = nn.Linear(input_size * self.patch_size, embed_dim)

    def forward(self, sequence, time_feat,repeated_index_embeddings):
        """
        Args:
            sequence: (batch_size, seq_len, input_dim)
            time_feat: (batch_size, seq_len, num_features)
        Returns:
            patches: (batch_size, num_patches, embed_dim)
        """
        batch_size, seq_len, input_dim = sequence.size()
        _, _, num_features = time_feat.size()

        # 确保序列长度可以被 patch_size 整除
        assert seq_len % self.patch_size == 0, "seq_len must be divisible by patch_size"

        num_patches = seq_len // self.patch_size

        # 重塑为 patches
        sequence_patches = sequence.view(batch_size, num_patches, self.patch_size,
                                         input_dim)  # (batch_size, num_patches, patch_size, input_dim)
        time_patches = time_feat.view(batch_size, num_patches, self.patch_size,
                                      num_features)  # (batch_size, num_patches, patch_size, num_features)
        repeated_index_embeddings_patches = repeated_index_embeddings.view(batch_size, num_patches, self.patch_size,-1)  # (batch_size, num_patches, patch_size, num_features)

        # 展平 patch 内部的时间步
        sequence_patches = sequence_patches.view(batch_size, num_patches,
                                                 -1)  # (batch_size, num_patches, patch_size * input_dim)
        time_patches = time_patches.view(batch_size, num_patches,
                                         -1)  # (batch_size, num_patches, patch_size * num_features)
        repeated_index_embeddings_patches = torch.reshape(repeated_index_embeddings_patches,(batch_size, num_patches, -1)) # (batch_size, num_patches, patch_size * num_features)

        # 拼接序列和时间特征
        combined_patches = torch.cat((sequence_patches, time_patches,repeated_index_embeddings_patches),
                                     dim=-1)  # (batch_size, num_patches, patch_size * (input_dim + num_features))

        # 线性映射
        patches = self.linear(combined_patches)  # (batch_size, num_patches, embed_dim)

        return patches

class TimeSeriesTransformerWithPatches(nn.Module):
    def __init__(self, patch_size,seqlen, embed_dim, num_heads, num_layers, dim_feedforward,input_size,
                 dropout=0.1, max_seq_len=1000):
        super(TimeSeriesTransformerWithPatches, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim,input_size=input_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (max_seq_len // patch_size), embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, input_size * patch_size),  # 根据任务调整输出层
        )
        self.out_put_layers_finel = nn.Linear(input_size,embed_dim)
        self.patch_size= patch_size

    def forward(self, sequence, time_feat,repeated_index_embeddings):
        """
        Args:
            sequence: (batch_size, seq_len, input_dim)
            time_feat: (batch_size, seq_len, num_features)
        Returns:
            outputs: (batch_size, num_patches, input_dim)
        """
        batch_size, seq_len ,_ = sequence.shape
        # Patch Embedding
        patches = self.patch_embedding(sequence, time_feat,repeated_index_embeddings)  # (batch_size, num_patches, embed_dim)

        # 添加位置编码
        patches = patches + self.positional_encoding[:, :patches.size(1), :]  # (batch_size, num_patches, embed_dim)

        # Transformer 需要 (num_patches, batch_size, embed_dim)
        transformer_input = patches.permute(1, 0, 2)  # (num_patches, batch_size, embed_dim)
        transformer_output = self.transformer_encoder(transformer_input)  # (num_patches, batch_size, embed_dim)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, num_patches, embed_dim)

        # 输出层
        outputs_patchs = self.output_layer(transformer_output)  # (batch_size, num_patches, input_dim)
        outputs_patchs = outputs_patchs.view( batch_size,seq_len // self.patch_size,self.patch_size,-1 )
        outputs_patchs = outputs_patchs.view(batch_size,seq_len,-1)
        outputs = self.out_put_layers_finel(outputs_patchs)
        return outputs


class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes):
        super(LearnableAdjacency, self).__init__()
        # 定义一个基础邻接矩阵，形状为 (num_nodes, num_nodes)
        # 初始化为单位矩阵
        self.adj = nn.Parameter(torch.eye(num_nodes))

    def forward(self, batch_size):
        """
        根据批次大小扩展邻接矩阵
        Args:
            batch_size (int): 当前批次的大小
        Returns:
            adj (torch.Tensor): 扩展后的邻接矩阵，形状为 (batch_size, num_nodes, num_nodes)
        """
        # 通过 Softmax 进行行归一化
        adj = F.softmax(self.adj, dim=1)  # (num_nodes, num_nodes)
        # 扩展到 (batch_size, num_nodes, num_nodes)
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
        return adj


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, adj):
        """
        Args:
            X: (batch_size, num_nodes, in_features)
            adj: (batch_size, num_nodes, num_nodes)  # 归一化的邻接矩阵
        Returns:
            out: (batch_size, num_nodes, out_features)
        """
        out = torch.bmm(adj, X)  # (batch_size, num_nodes, in_features)
        out = self.linear(out)  # (batch_size, num_nodes, out_features)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class GCNModel(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_features, out_features, dropout=0.1):
        super(GCNModel, self).__init__()
        self.adj_module = LearnableAdjacency(num_nodes)
        self.gcn1 = GCNLayer(in_features, hidden_features, dropout)
        self.gcn2 = GCNLayer(hidden_features, out_features, dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_nodes, in_features)
        Returns:
            out: (batch_size, num_nodes, out_features)
        """
        batch_size = x.size(0)
        adj = self.adj_module(batch_size)  # 获取归一化并扩展的邻接矩阵
        out = self.gcn1(x, adj)
        out = self.gcn2(out, adj)
        return out



class LagsAttention(nn.Module):
    def __init__(self, target_dim, num_lags, embed_dim, num_heads,dropout):
        super(LagsAttention, self).__init__()
        self.target_dim = target_dim
        self.num_lags = num_lags
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 线性变换用于查询、键、值
        # self.query = nn.Linear(target_dim, embed_dim)
        # self.key = nn.Linear(target_dim, embed_dim)
        # self.value = nn.Linear(target_dim, embed_dim)
        #
        # # 多头注意力
        # self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.fc = nn.Linear(embed_dim, target_dim)




        # -----

        self.self_attention = nn.MultiheadAttention(
            embed_dim=target_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(target_dim)
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=target_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(target_dim)
        self.ff = nn.Sequential(
            nn.Linear(target_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, target_dim),
        )
        self.norm3 = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(dropout)


    def compute_attention(self, q, k, v):
        scores = torch.matmul(q, k) / torch.sqrt(torch.tensor(q.size(-1)))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self,lags):
        # lags: (batch_size, sub_seq_len, target_dim, num_lags)
        batch_size, sub_seq_len, target_dim, num_lags = lags.size()
        # 重塑为 (batch_size, sub_seq_len * num_lags, target_dim)
        lags = lags.permute(0,1,3,2).contiguous()
        lags = torch.reshape(lags, (batch_size, sub_seq_len * num_lags, target_dim))
        query= lags
        key = lags.permute(0,2,1,).contiguous()
        value = lags
        # Self-attention
        # attn_output, _ = self.self_attention(query, query, query, attn_mask=None)
        attn_output,_ = self.compute_attention(query, key, value)

        query = self.norm1(query + self.dropout(attn_output))

        # # Encoder attention
        # attn_output, _ = self.encoder_attention(query, key, value, attn_mask=None)
        # query = self.norm2(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        output = self.norm3(query + self.dropout(ff_output))



        # 重塑回 (batch_size, sub_seq_len, target_dim, num_lags)
        output = torch.reshape(output, (batch_size, sub_seq_len , num_lags, target_dim)).permute(0, 1, 3, 2)

        return output



class SeqAttention(nn.Module):
    def __init__(self, target_dim, num_lags, embed_dim, num_heads,dropout):
        super(SeqAttention, self).__init__()
        self.target_dim = target_dim
        self.num_lags = num_lags
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 线性变换用于查询、键、值
        # self.query = nn.Linear(target_dim, embed_dim)
        # self.key = nn.Linear(target_dim, embed_dim)
        # self.value = nn.Linear(target_dim, embed_dim)
        #
        # # 多头注意力
        # self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.fc = nn.Linear(embed_dim, target_dim)




        # -----

        self.self_attention = nn.MultiheadAttention(
            embed_dim=target_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(target_dim)
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=target_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(target_dim)
        self.ff = nn.Sequential(
            nn.Linear(target_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, target_dim),
        )
        self.norm3 = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(dropout)


    def compute_attention(self, q, k, v):
        scores = torch.matmul(q, k) / torch.sqrt(torch.tensor(q.size(-1)))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self,lags):
        # lags: (batch_size, sub_seq_len, target_dim, num_lags)
        batch_size, sub_seq_len, target_dim = lags.size()
        # 重塑为 (batch_size, sub_seq_len * num_lags, target_dim)
        # lags = lags.permute(0,1,3,2).contiguous()
        # lags = torch.reshape(lags, (batch_size, sub_seq_len , target_dim))
        query= lags
        key = lags.permute(0,2,1).contiguous()
        value = lags
        # Self-attention
        # attn_output, _ = self.self_attention(query, query, query, attn_mask=None)
        attn_output,_ = self.compute_attention(query, key, value)

        query = self.norm1(query + self.dropout(attn_output))

        # # Encoder attention
        # attn_output, _ = self.encoder_attention(query, key, value, attn_mask=None)
        # query = self.norm2(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        output = self.norm3(query + self.dropout(ff_output))



        # 重塑回 (batch_size, sub_seq_len, target_dim, num_lags)
        # output = torch.reshape(output, (batch_size, sub_seq_len , num_lags, target_dim)).permute(0, 1, 3, 2)

        return output


class FourierAtt(nn.Module):
    def __init__(self, embed_size,seq_length,feature_size):
        super(FourierAtt, self).__init__()
        self.embed_size = embed_size #embed_size
        self.hidden_size = embed_size #hidden_size
        self.seq_length = seq_length #hidden_size

        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.channel_independence = 0
        # self.fc = nn.Sequential(
        #     nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.pre_length)
        # )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)   # b,sqlen,nvar,dim
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x



if __name__ == '__main__':
    fourieratt = FourierAtt(embed_size=128,seq_length=137,feature_size=137)
    x = torch.randn(128,48,137)
    res = fourieratt(x)
    c = 'end'

    # 示例使用
    batchsize = 128
    seqlen = 24  # 每个节点的特征维度 (in_features)
    nvar = 500  # 节点数 (num_nodes)

    # 定义模型
    model = GCNModel(
        num_nodes=nvar,
        in_features=seqlen,
        hidden_features=64,
        out_features=seqlen,
        dropout=0.1
    )

    # 随机输入数据
    x = torch.randn(batchsize, nvar, seqlen)  # (batch_size, num_nodes, in_features)

    # 前向传播
    res = model(x)
    print(res.shape)  # 应输出 (128, 500, 32)



    x = torch.randn(1, 1, 224)
    f = torch.fft.fft(x,dim=1)
    c = f


    # patch_embed = PatchEmbedding(patch_size=seqlen // 4, embed_dim=embed_dim,num_features=num_features,input_dim=input_dim,input_size=input_size)

    # out = patch_embed(sequence,time_feat,repeated_index_embeddings)
    # # 目的：-- 128 48 128
    # c= out

    # 示例使用
    batch_size = 128
    seq_len = 48  # 必须能被 patch_size 整除

    patch_size = 12  # 例如，每 12 个时间步为一个 patch
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    max_seq_len = 240
    num_features = 4
    seqlen = 48
    input_dim = 411
    input_size = 552
    # 输入数据
    # sequence = torch.randn(batch_size, seq_len, input_dim)
    # time_feat = torch.randn(batch_size, seq_len, num_features)

    # 初始化模型
    model = TimeSeriesTransformerWithPatches(
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
        input_size=input_size
    )
    sequence = torch.randn(batch_size, seqlen, input_dim)
    time_feat = torch.randn(batch_size, seqlen, num_features)
    repeated_index_embeddings = torch.randn(batch_size, seqlen, 137)
    # 前向传播
    outputs = model(sequence, time_feat, repeated_index_embeddings)  # (batch_size, num_patches, input_dim)
    c = 'end'