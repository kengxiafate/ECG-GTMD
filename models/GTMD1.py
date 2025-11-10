import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import Multi_Resolution_Data, Frequency_Embedding
from layers.GTMD_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FormerLayer, DifferenceFormerlayer
from layers.Multi_Resolution_GNN import MRGNN
from layers.Difference_Pre import DifferenceDataEmb, DataRestoration
import numpy as np
class WaveDetector(nn.Module):

    def __init__(self, d_model, num_waves=3, hidden_dim=64):
        super(WaveDetector, self).__init__()
        self.num_waves = num_waves  # P波、QRS波、T波
        self.conv1 = nn.Conv1d(d_model, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, num_waves, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
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


class ECGSubGraphLayer(nn.Module):

    def __init__(self, configs, res_len):
        super(ECGSubGraphLayer, self).__init__()
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.num_waves = 3  # P波、QRS波、T波
        self.res_len = res_len

        self.wave_detector = WaveDetector(self.d_model, self.num_waves)

        self.wave_encoding = WaveTypeEncoding(self.d_model, self.num_waves)

        self.wave_attention = DynamicWaveAttention(self.d_model, self.n_heads)

        self.subgraph_conv = ECGSubGraphConv(self.d_model, self.d_model, self.num_waves)

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(configs.dropout)
        )

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

    def build_subgraph_adjacency(self, wave_mask, base_adj=None):

        batch_size, seq_len, num_waves = wave_mask.shape

        if base_adj is None:

            base_adj = torch.ones(seq_len, seq_len).to(wave_mask.device)

        wave_sim = torch.bmm(wave_mask, wave_mask.transpose(1, 2))  # [batch_size, seq_len, seq_len]

        subgraph_adj = base_adj.unsqueeze(0) * wave_sim

        return subgraph_adj

    def forward(self, x, adj_matrix=None):
        # x shape: [batch_size, seq_len, d_model]


        wave_mask = self.wave_detector(x)  # [batch_size, seq_len, num_waves]


        wave_enc = self.wave_encoding(wave_mask)
        x = x + wave_enc


        subgraph_adj = self.build_subgraph_adjacency(wave_mask, adj_matrix)


        attn_out = self.wave_attention(x, wave_mask, subgraph_adj)
        x = self.norm1(x + attn_out)


        conv_out = self.subgraph_conv(x, wave_mask)
        x = self.norm2(x + conv_out)


        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x, wave_mask, subgraph_adj


class MRGNN(nn.Module):
    def __init__(self, configs, res_len):
        super(MRGNN, self).__init__()
        self.d_model = configs.d_model
        self.res_num = len(res_len)
        self.res_len = res_len


        self.subgraph_layers = nn.ModuleList([
            ECGSubGraphLayer(configs, res_len[i]) for i in range(self.res_num)
        ])


        self.adjacency_learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(res_len[i], res_len[i] // 2),
                nn.ReLU(),
                nn.Linear(res_len[i] // 2, res_len[i] * res_len[i]),
                nn.Tanh()
            ) for i in range(self.res_num)
        ])

        self.fusion_weights = nn.Parameter(torch.ones(self.res_num) / self.res_num)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, enc_out):

        B = enc_out[0].shape[0]
        outputs = []
        adjacency_matrix_list = []
        wave_masks = []

        for i, x in enumerate(enc_out):
            # x shape: [batch_size, res_len[i], d_model]

            adj_logits = self.adjacency_learners[i](x.mean(dim=-1))  # 平均池化时间维度
            adj_matrix = adj_logits.view(B, self.res_len[i], self.res_len[i])

            x_out, wave_mask, subgraph_adj = self.subgraph_layers[i](x, adj_matrix)

            x_pooled = self.pool(x_out.transpose(1, 2)).squeeze(-1)  # [batch_size, d_model]

            outputs.append(x_pooled)
            adjacency_matrix_list.append(adj_matrix)
            wave_masks.append(wave_mask)

        weighted_outputs = [outputs[i] * self.fusion_weights[i] for i in range(self.res_num)]
        final_output = sum(weighted_outputs)

        return final_output, adjacency_matrix_list


class PositionalEncoding(nn.Module):
    """位置编码"""

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


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.output_attention = configs.output_attention
        self.activation = configs.activation
        self.resolution_list = list(map(int, configs.resolution_list.split(",")))

        self.res_num = len(self.resolution_list)
        self.stride_list = self.resolution_list
        self.res_len = [int(self.seq_len // res) + 1 for res in self.resolution_list]
        self.augmentations = configs.augmentations.split(",")

        # step1: multi_resolution_data
        self.multi_res_data = Multi_Resolution_Data(self.enc_in, self.resolution_list, self.stride_list)

        # step2.1: frequency convolution network
        self.freq_embedding = Frequency_Embedding(self.d_model, self.res_len, self.augmentations)

        # step2.2: difference attention network
        self.diff_data_emb = DifferenceDataEmb(self.res_num, self.enc_in, self.d_model)
        self.difference_attention = Encoder(
            [
                EncoderLayer(
                    DifferenceFormerlayer(
                        self.enc_in,
                        self.res_num,
                        self.d_model,
                        self.n_heads,
                        self.dropout,
                        self.output_attention
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.data_restoration = DataRestoration(self.res_num, self.enc_in, self.d_model)
        self.embeddings = nn.ModuleList([nn.Linear(res_len, self.d_model) for res_len in self.res_len])

        self.pos_encoding = PositionalEncoding(configs.d_model)

        # step 3: transformer (保持原有代码)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FormerLayer(
                        len(self.resolution_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )


        self.mrgnn = MRGNN(configs, self.res_len)

        self.projection = nn.Linear(self.d_model * self.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, C = x_enc.shape

        multi_res_data = self.multi_res_data(x_enc)

        enc_out_1 = self.freq_embedding(multi_res_data)

        x_diff_emb, x_padding = self.diff_data_emb(multi_res_data)
        x_diff_enc, attns = self.difference_attention(x_diff_emb, attn_mask=None)
        enc_out_2 = self.data_restoration(x_diff_enc, x_padding)
        enc_out_2 = [self.embeddings[l](enc_out_2[l]) for l in range(self.res_num)]


        data_enc = [enc_out_1[l] + enc_out_2[l] for l in range(self.res_num)]

        for l in range(self.res_num):
            pos_enc = self.pos_encoding(data_enc[l])
            data_enc[l] = data_enc[l] + pos_enc

        enc_out, attns = self.encoder(data_enc, attn_mask=None)

        output, adjacency_matrix_list = self.mrgnn(enc_out)

        # step 5: projection
        output = output.reshape(B, -1)
        output = self.projection(output)

        return output