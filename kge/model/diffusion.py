import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class GNDiff(nn.Module):
    def __init__(self, dim, num_steps=10, beta_start=0.0001, beta_end=0.02, mask_prob=0.1):
        super(GNDiff, self).__init__()
        self.dim = dim
        self.num_steps = num_steps
        self.mask_prob = mask_prob

        # 噪声调度
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # 注册为缓冲区
        self.register_buffer('alpha_bars', alpha_bars)

        # 掩码嵌入
        self.mask_embed = nn.Parameter(torch.randn(1, dim))

        # 去噪网络
        self.denoise_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

        # MSE损失函数
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # 随机选择扩散步数 t
        t = torch.randint(0, self.num_steps, (batch_size,), device=x.device)

        # 前向扩散
        noise = torch.randn_like(x, device=x.device)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        # 掩码机制
        mask = torch.bernoulli(torch.full((batch_size, seq_len), self.mask_prob, device=x.device)).bool()
        x_t[mask] = self.mask_embed.expand_as(x_t[mask])

        # 反向扩散
        predicted_noise = self.denoise_net(x_t.view(batch_size * seq_len, dim)).view(batch_size, seq_len, dim)
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        loss = self.mse_loss(x_0_pred, x)

        return x_0_pred, loss
