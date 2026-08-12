import torch
import torch.nn as nn
import torch.nn.functional as F

import config

def rasterize(means2d, cov2d, colors, opacity, depths, mask, cam, bg=None):
    """
    means2d: (N, 2) 가우시안 중심 위치
    cov2d: (N, 2, 2) 가우시안 공분산 행렬
    colors: (N, 3) 가우시안 색상
    opacity: (N, 1) 가우시안 불투명도
    depths: (N,) 가우시안 깊이
    mask: (N,) 가우시안 유효성 마스크
    cam: Camera 객체
    bg: 배경 색상

    return:
        image: (H, W, 3) 렌더링된 이미지
        depth_map: (H, W) 깊이 맵
        alpha_map: (H, W) 알파 맵
    """

    bg = config.background if bg is None else bg

    H, W = cam.height, cam.width
    device = means2d.device

    # 가까운 순으로 정렬
    idx = mask.nonzero(as_tuple=True)[0]
    idx = idx[depths[idx].argsort()]  # 깊이 기준으로 정렬

    means2d = means2d[idx]
    cov2d = cov2d[idx]
    colors = colors[idx]
    opacity = opacity[idx]

    # 공분산 역행렬
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]

    # tensor 연산에서 det가 0이 되는 경우를 방지하기 위해 clamp 사용, det 0이면 역행렬 없음
    det = (a * c - b * b).clamp(min=1e-8)

    conic_a, conic_b, conic_c = c / det, -b / det, a / det

    # 2D 그리드 생성, 논문에서는 16x16로 가우시안 렌더링, 하지만 우린 이미지 크기에 맞춰서 한다
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    px = torch.stack([xs, ys], dim=-1) + 0.5 # (H, W, 2) 픽셀 중심

    # 가우시안 * 각 픽셀의 alpha
    d = px.unsqueeze(0) - means2d[:, None, None, :]
    dx, dy = d.unbind(-1)

    power = -0.5 * (
        conic_a[:, None, None] * dx * dx + 2 * conic_b[:, None, None] * dx *dy + conic_c[:, None, None] * dy * dy

    ) # 이거 그거 아닌가, 다중 랜덤 변수(2개)의 가우시안의 exp 텀

    alpha = (opacity[:,:, None] * torch.exp(power)).clamp(max=0.99)

    # alpha 값 섞기
    # 1. 전체 투과율 계산 (cumprod 연산은 여기서 딱 한 번만 수행)
    T_full = torch.cumprod(1.0 - alpha, dim=0)

    # 2. 끝까지 남은 투과율 미리 빼두기
    T_final = T_full[-1]  # (H, W)

    # 3. 원래 위치의 투과율을 구하기 위해 한 칸 shift
    T = torch.cat([torch.ones_like(T_full[:1]), T_full[:-1]], dim=0)

    # 4. 이미지 렌더링
    weights = alpha * T
    img = (weights.unsqueeze(-1) * colors[:, None, None, :]).sum(dim=0)

    # 5. 배경 합성
    img = img + T_final.unsqueeze(-1) * bg
    img = img.clamp(0.0, 1.0)

    return img

if __name__ == "__main__":
    from torchvision.utils import save_image

    from camera_projection import Camera, look_at, project_guassians
    from gaussian_model import GaussianModel

    save_path = config.ROOT / "results"
    save_path.mkdir(exist_ok=True)

    cam = Camera.from_fov(look_at(eye=config.eye))
    g = GaussianModel(num_points=200)

    means2d, cov2d, depths, mask = project_guassians(g._means, g.covariance(), cam)
    img = rasterize(means2d, cov2d, g.colors, g.opacity, depths, mask, cam)

    save_image(img.permute(2, 0, 1), save_path / "results01.png")
