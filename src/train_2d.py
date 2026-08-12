from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision.utils import save_image

from gaussian_model import GaussianModel
from camera_projection import project_guassians, Camera, look_at
from rasterize import rasterize
from loss import D_SSIM_Loss
import config

def train(target, cam, save_path, num_points=None, iters=None, device="cpu"):
    """
        args:
            target: (H, W, 3), values: [0, 1] requires normamlization
            num_points, iters: None이면 config 값을 쓴다
    """
    iters = config.iters if iters is None else iters
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    g = GaussianModel(num_points=num_points, device=device)

    optimizer = optim.Adam([
        {"params": [g._means],   "lr": config.means_lr},
        {"params": [g._scales],  "lr": config.scales_lr},
        {"params": [g._quats],   "lr": config.quats_lr},
        {"params": [g._opacity], "lr": config.opacity_lr},
        {"params": [g._colors],  "lr": config.colors_lr},
    ],  eps=1e-15)

    criterion = D_SSIM_Loss(lamda=0.2).to(device)

    pbar = tqdm(range(iters))
    for it in pbar:
        means2d, cov2d, depths, mask = project_guassians(g._means, g.covariance(), cam)
        img = rasterize(means2d, cov2d, g.colors, g.opacity, depths, mask, cam)

        # loss = F.l1_loss(img, target)
        loss = criterion(img, target)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if it % config.save_every == 0:
            save_image(img.detach().permute(2, 0, 1), save_path / f"{it:04d}.png")
    return g


def load_target(path, size=None, device="cpu"):
    size = config.size if size is None else size
    im = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(im)).float().to(device) / 255.0

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    target_img = load_target(config.target_path, device=device)
    cam = Camera.from_fov(look_at(eye=config.eye, device=device))

    train(target_img, cam, config.save_path, device=device)
