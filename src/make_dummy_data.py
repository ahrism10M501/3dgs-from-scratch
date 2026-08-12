import math

import torch

from camera_projection import Camera, look_at, project_guassians
from gaussian_model import GaussianModel, inverse_sigmoid
from rasterize import rasterize

def orbit_cam(num_view=16, radius=4.0, elevation=20.0, device="cpu"):
    cams = []
    phi = math.radians(elevation)

    for i in range(num_view):
        theta = 2 * math.pi * i / num_view

        eye = (
            radius * math.cos(phi) * math.sin(theta),
            -radius * math.sin(phi),
            radius * math.cos(phi) * math.cos(theta)
        )

        cams.append(Camera.from_fov(look_at(eye=eye, device=device)))

    return cams

@torch.no_grad()
def make_target_scene(points_per_ring=300, radius=0.6, device="cpu"):
    g = GaussianModel(num_points=points_per_ring * 3, device=device)

    t = torch.linspace(0, 2 * math.pi, points_per_ring, device=device)
    zero, cos, sin = torch.zeros_like(t), radius * torch.cos(t), radius * torch.sin(t)

    means = torch.cat([
        torch.stack([cos, sin, zero], dim=-1),   # xy 평면
        torch.stack([cos, zero, sin], dim=-1),   # xz 평면
        torch.stack([zero, cos, sin], dim=-1),   # yz 평면
    ], dim=0)

    colors = torch.zeros_like(means)
    colors[0 * points_per_ring:1 * points_per_ring, 0] = 1.0   # R
    colors[1 * points_per_ring:2 * points_per_ring, 1] = 1.0   # G
    colors[2 * points_per_ring:3 * points_per_ring, 2] = 1.0   # B

    g._means.copy_(means)
    g._colors.copy_(inverse_sigmoid(colors.clamp(0.01, 0.99)))
    g._scales.fill_(math.log(0.04))
    g._opacity.fill_(inverse_sigmoid(torch.tensor(0.9)))
    return g

@torch.no_grad()
def render_views(g, cams):
    imgs = []
    for cam in cams:
        means2d, cov2d, depths, mask = project_guassians(g._means, g.covariance(), cam)
        imgs.append(rasterize(means2d, cov2d, g.colors, g.opacity, depths, mask, cam))
    return torch.stack(imgs)      # (V, H, W, 3)