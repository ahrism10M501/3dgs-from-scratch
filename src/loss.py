import torch
import torch.nn as nn
import torch.nn.functional as F

def _gaussian_window(window_size=11, sigma=1.5):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size//2
    g = torch.exp(-(coords ** 2) / (2 * sigma **2))
    g = g / g.sum()
    return torch.outer(g, g)

class D_SSIM_Loss(nn.Module):
    def __init__(self, lamda=0.2, window_size=11, sigma=1.5, channels=3):
        super().__init__()
        self.lamda = lamda
        self.window_size = window_size
        self.l1 = nn.L1Loss()

        window = _gaussian_window(window_size, sigma)
        window = window.expand(channels, 1, window_size, window_size).contiguous()

        self.register_buffer("window", window)

    def ssim(self, x, y):
        pad = self.window_size // 2
        C = x.shape[1]

        mu_x = F.conv2d(x, self.window, padding=pad, groups=C)
        mu_y = F.conv2d(y, self.window, padding=pad, groups=C)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        # 가우시안의 분산이네, E[X^2] - E[X]^2
        sigma_x = F.conv2d(x * x, self.window, padding=pad, groups=C) - mu_x2
        sigma_y = F.conv2d(y * y, self.window, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, self.window, padding=pad, groups=C) - mu_xy

        C1, C2 = 0.01 **2, 0.03**2
        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)

        return (num/ den).mean()

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        x = pred.permute(2, 0, 1).unsqueeze(0)
        y = target.permute(2, 0, 1).unsqueeze(0)
        d_ssim = 1.0 - self.ssim(x, y)

        return (1 - self.lamda) * l1 + self.lamda * d_ssim