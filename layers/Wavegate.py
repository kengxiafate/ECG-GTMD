import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WaveDetector(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_waves=3):
        super(WaveDetector, self).__init__()
        self.num_waves = num_waves  # P波、QRS波、T波
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, num_waves, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        wave_probs = self.conv3(x)  # [batch_size, num_waves, seq_len]
        wave_probs = wave_probs.transpose(1, 2)  # [batch_size, seq_len, num_waves]
        wave_mask = self.softmax(wave_probs)  # 每个时间点属于各个波段的概率
        return wave_mask


class WaveTypeEncoding(nn.Module):

    def __init__(self, d_model, num_waves=3):
        super(WaveTypeEncoding, self).__init__()
        self.wave_embedding = nn.Embedding(num_waves, d_model)
        self.num_waves = num_waves

    def forward(self, wave_mask):
        # wave_mask shape: [batch_size, seq_len, num_waves]
        batch_size, seq_len, _ = wave_mask.shape

        wave_labels = torch.argmax(wave_mask, dim=-1)  # [batch_size, seq_len]

        wave_encoding = self.wave_embedding(wave_labels)  # [batch_size, seq_len, d_model]

        return wave_encoding


class DynamicWaveAttention(nn.Module):

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(DynamicWaveAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_heads),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, wave_mask, subgraph_adj=None):
        # x shape: [batch_size, seq_len, d_model]
        # wave_mask shape: [batch_size, seq_len, num_waves]
        # subgraph_adj: 子图邻接矩阵 [batch_size, seq_len, seq_len]

        residual = x
        batch_size, seq_len, _ = x.shape

        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        attn_scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / (self.head_dim ** 0.5)

        wave_gates = self.gate_mlp(x)  # [batch_size, seq_len, n_heads]
        wave_gates = wave_gates.permute(0, 2, 1).unsqueeze(2)  # [batch_size, n_heads, 1, seq_len]

        wave_similarity = torch.bmm(wave_mask, wave_mask.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        wave_enhance = wave_similarity.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        attn_scores = attn_scores * wave_gates + attn_scores * wave_enhance

        if subgraph_adj is not None:
            attn_scores = attn_scores + subgraph_adj.unsqueeze(1)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, V)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)

        output = self.layer_norm(attn_output + residual)

        return output


class ECGSubGraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_waves=3):
        super(ECGSubGraphConv, self).__init__()
        self.num_waves = num_waves
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.wave_convs = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(num_waves)
        ])

        # 波段重要性权重
        self.wave_weights = nn.Parameter(torch.ones(num_waves) / num_waves)

        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x, wave_mask, adj_matrices=None):
        # x shape: [batch_size, seq_len, in_channels]
        # wave_mask shape: [batch_size, seq_len, num_waves]
        # adj_matrices: 各波段的邻接矩阵列表 [num_waves, seq_len, seq_len]

        batch_size, seq_len, _ = x.shape
        wave_outputs = []

        for wave_idx in range(self.num_waves):
            wave_prob = wave_mask[:, :, wave_idx]  # [batch_size, seq_len]

            if adj_matrices is not None and wave_idx < len(adj_matrices):
                adj = adj_matrices[wave_idx].unsqueeze(0)  # [1, seq_len, seq_len]
                wave_feat = torch.bmm(adj, x)  # [batch_size, seq_len, in_channels]
            else:
                wave_feat = x

            wave_feat = self.wave_convs[wave_idx](wave_feat)  # [batch_size, seq_len, out_channels]

            weight = self.wave_weights[wave_idx]
            wave_feat = wave_feat * wave_prob.unsqueeze(-1) * weight

            wave_outputs.append(wave_feat)

        output = sum(wave_outputs)  # [batch_size, seq_len, out_channels]
        output = self.activation(output)
        output = self.layer_norm(output)

        return output


class ECGGraphTransformer(nn.Module):
    """ECG Graph Transformer with multi-band subgraph mechanism"""

    def __init__(self, d_model, n_heads, num_layers, num_waves=3, dropout=0.1):
        super(ECGGraphTransformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.num_waves = num_waves

        self.wave_detector = WaveDetector(d_model, d_model // 2, num_waves)

        self.wave_encoding = WaveTypeEncoding(d_model, num_waves)

        self.pos_encoding = PositionalEncoding(d_model)

        # Graph Transformer
        self.layers = nn.ModuleList([
            ECGTransformerLayer(d_model, n_heads, num_waves, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_matrix=None):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape

        wave_mask = self.wave_detector(x)  # [batch_size, seq_len, num_waves]

        pos_enc = self.pos_encoding(torch.zeros(batch_size, seq_len, self.d_model).to(x.device))
        wave_enc = self.wave_encoding(wave_mask)
        x = x + pos_enc + wave_enc
        x = self.dropout(x)

        subgraph_adj = self.build_subgraph_adjacency(wave_mask, adj_matrix)

        for layer in self.layers:
            x = layer(x, wave_mask, subgraph_adj)

        return x, wave_mask

    def build_subgraph_adjacency(self, wave_mask, base_adj=None):
        batch_size, seq_len, num_waves = wave_mask.shape

        if base_adj is None:
            base_adj = torch.ones(seq_len, seq_len).to(wave_mask.device)

        wave_sim = torch.bmm(wave_mask, wave_mask.transpose(1, 2))  # [batch_size, seq_len, seq_len]

        subgraph_adj = base_adj.unsqueeze(0) * wave_sim

        return subgraph_adj


class ECGTransformerLayer(nn.Module):

    def __init__(self, d_model, n_heads, num_waves, dropout=0.1):
        super(ECGTransformerLayer, self).__init__()

        self.wave_attention = DynamicWaveAttention(d_model, n_heads, dropout)

        self.subgraph_conv = ECGSubGraphConv(d_model, d_model, num_waves)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, wave_mask, subgraph_adj):

        attn_out = self.wave_attention(x, wave_mask, subgraph_adj)
        x = self.norm1(x + attn_out)

        conv_out = self.subgraph_conv(x, wave_mask)
        x = self.norm2(x + conv_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :].transpose(0, 1)


class ECGClassificationModel(nn.Module):

    def __init__(self, input_dim, d_model, n_heads, num_layers, num_classes, num_waves=3):
        super(ECGClassificationModel, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        # ECG Graph Transformer
        self.ecg_transformer = ECGGraphTransformer(d_model, n_heads, num_layers, num_waves)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)

        # 通过ECG Graph Transformer
        x, wave_mask = self.ecg_transformer(x)

        # 分类
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        output = self.classifier(x)

        return output, wave_mask

'''
if __name__ == "__main__":

    batch_size = 4
    seq_len = 1000  
    input_dim = 12  
    d_model = 64
    n_heads = 8
    num_layers = 3
    num_classes = 5

    model = ECGClassificationModel(input_dim, d_model, n_heads, num_layers, num_classes)

    # 模拟ECG输入
    x = torch.randn(batch_size, seq_len, input_dim)

    # 前向传播
    output, wave_mask = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"波段掩码形状: {wave_mask.shape}")
    print(f"预测概率: {F.softmax(output, dim=-1)}")'''