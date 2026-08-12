import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision.utils import save_image

from gaussian_model import GaussianModel
from camera_projection import Camera, look_at, project_guassians
from rasterize import rasterize
from loss import D_SSIM_Loss
from make_dummy_data import orbit_cam, make_target_scene, render_views
from export_ply import save_ply

import config


def render(g, cam):
    means2d, cov2d, depths, mask = project_guassians(g._means, g.covariance(), cam)
    return rasterize(means2d, cov2d, g.colors, g.opacity, depths, mask, cam)


def train(targets, cams, save_path, eval_cam=None, num_points=None, iters=None, device="cpu"):
    """
        args:
            targets: (V, H, W, 3), values: [0, 1]. cams와 같은 순서로 대응된다
            cams: 학습용 카메라 리스트
            eval_cam: 진행 상황 저장용 카메라. 학습에 쓰지 않은 시점을 넣으면
                      저장 이미지가 곧 일반화 성능 테스트가 된다
            num_points, iters: None이면 config 값을 쓴다
    """
    iters = config.iters if iters is None else iters
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if eval_cam is None:
        eval_cam = cams[0]

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
        # 매 스텝 시점 하나만 뽑는다 (미니배치 SGD와 같은 논리)
        v = torch.randint(len(cams), (1,)).item()
        cam, target = cams[v], targets[v]

        img = render(g, cam)
        loss = criterion(img, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 매번 같은 시점으로 렌더해야 진행 상황을 비교할 수 있다
        if it % config.save_every == 0:
            with torch.no_grad():
                eval_img = render(g, eval_cam)
            save_image(eval_img.permute(2, 0, 1), save_path / f"{it:04d}.png")

    return g


def held_out_camera(num_view, radius, elevation, device="cpu"):
    """학습 시점 0번과 1번의 정확히 중간 각도 — 한 번도 본 적 없는 뷰"""
    phi = math.radians(elevation)
    theta = math.pi / num_view
    eye = (
        radius * math.cos(phi) * math.sin(theta),
        -radius * math.sin(phi),
        radius * math.cos(phi) * math.cos(theta),
    )
    return Camera.from_fov(look_at(eye=eye, device=device))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    save_path = Path(config.save_path_3d)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1) 학습용 카메라 궤도
    cams = orbit_cam(num_view=config.num_views, radius=config.orbit_radius,
                     elevation=config.elevation, device=device)

    # 2) 정답 장면을 만들어 각 시점에서 렌더 → 이게 학습 데이터가 된다
    gt_scene = make_target_scene(device=device)
    targets = render_views(gt_scene, cams)

    # 학습 전에 데이터부터 눈으로 확인할 것
    for i in range(0, len(cams), 4):
        save_image(targets[i].permute(2, 0, 1), save_path / f"gt_{i:02d}.png")

    # 3) 학습에 쓰지 않은 시점
    eval_cam = held_out_camera(config.num_views, config.orbit_radius,
                               config.elevation, device=device)
    save_image(render_views(gt_scene, [eval_cam])[0].permute(2, 0, 1),
               save_path / "gt_eval.png")

    g = train(targets, cams, save_path, eval_cam=eval_cam, device=device)

    save_ply(g, save_path / "scene.ply")
