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
