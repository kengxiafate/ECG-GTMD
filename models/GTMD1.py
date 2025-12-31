import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import Multi_Resolution_Data, Frequency_Embedding
from layers.GTMD_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FormerLayer, DifferenceFormerlayer
from layers.Multi_Resolution_GNN import MRGNN
from layers.Difference_Pre import DifferenceDataEmb, DataRestoration



class InstanceGating(nn.Module):
    def __init__(self, in_channels):
        super(InstanceGating, self).__init__()

    
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

       
        self.gate_net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x, mask):
        if mask is None:
            return x, None

  
        x_in = x.permute(0, 2, 1)
        feature = self.feature_extractor(x_in).squeeze(-1)


        weights = torch.softmax(self.gate_net(feature), dim=-1)

   
        mask_expanded = mask.unsqueeze(-1).to(x.device)
        w_p = weights[:, 0].view(-1, 1, 1)
        w_qrs = weights[:, 1].view(-1, 1, 1)
        w_t = weights[:, 2].view(-1, 1, 1)

        m_p = (mask_expanded == 1).float()
        m_qrs = (mask_expanded == 2).float()
        m_t = (mask_expanded == 3).float()

        x_gated = (x * m_p * w_p) + \
                  (x * m_qrs * w_qrs) + \
                  (x * m_t * w_t)

        return x_gated, weights


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

 
        self.waveform_gating = InstanceGating(self.enc_in)


        self.multi_res_data = Multi_Resolution_Data(self.enc_in, self.resolution_list, self.stride_list)

  
        self.freq_embedding = Frequency_Embedding(self.d_model, self.res_len, self.augmentations)

      
        self.diff_data_emb = DifferenceDataEmb(self.res_num, self.enc_in, self.d_model)

      
        self.difference_attention = Encoder(
            [EncoderLayer(
                DifferenceFormerlayer(self.enc_in, self.res_num, self.d_model, self.n_heads, self.dropout,
                                      self.output_attention),
                self.d_model,
                self.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )


        self.data_restoration = DataRestoration(self.res_num, self.enc_in, self.d_model)
        self.embeddings = nn.ModuleList([nn.Linear(res_len, self.d_model) for res_len in self.res_len])


        self.encoder = Encoder(
            [EncoderLayer(
                FormerLayer(len(self.resolution_list), self.d_model, self.n_heads, self.dropout, self.output_attention),
                self.d_model,
                self.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

  
        self.mrgnn = MRGNN(configs, self.res_len)


        self.projection = nn.Linear(self.d_model * self.enc_in, configs.num_class)

        self.cur_weights = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, C = x_enc.shape

  
        if mask is not None:
            x_enc, weights = self.waveform_gating(x_enc, mask)
            self.cur_weights = weights
        else:
            self.cur_weights = None


        multi_res_data = self.multi_res_data(x_enc)


        enc_out_1 = self.freq_embedding(multi_res_data)

    
        x_diff_emb, x_padding = self.diff_data_emb(multi_res_data)
        x_diff_enc, attns = self.difference_attention(x_diff_emb, attn_mask=None)

        enc_out_2 = self.data_restoration(x_diff_enc, x_padding)
        enc_out_2 = [self.embeddings[l](enc_out_2[l]) for l in range(self.res_num)]

     
        data_enc = [enc_out_1[l] + enc_out_2[l] for l in range(self.res_num)]

     
        enc_out, attns = self.encoder(data_enc, attn_mask=None)

    
        output, adjacency_matrix_list = self.mrgnn(enc_out)

      
        output = output.reshape(B, -1)
        output = self.projection(output)

        return output
