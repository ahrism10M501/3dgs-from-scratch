import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 카메라 세팅하기
class Camera:
    def __init__(self, viewmat, fx, fy, cx, cy, width, height):
        self.viewmat = viewmat
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    @staticmethod
    def from_fov(viewmat, fov, width, height, device="cpu"):
        fx = width / (2 * math.tan(math.radians(fov) / 2))
        cx = width / 2 # 보통 이미지 중심은 좌상단인데, 카메라 보정에서는 이미지 중심을 쓰니까...
        cy = height / 2
        # 왜 fx, fy를 다르게 안하고 fx만 쓰는거지?
        return Camera(viewmat, fx, fx, cx, cy, width, height)

# 보고 있는 방향에 따른 viewmat 만들기
def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, -1.0, 0.0), device="cpu"):
    """
    eye: 카메라 위치
    target: 카메라가 바라보는 위치
    up: 카메라의 위쪽 방향

    """
    eye = torch.tensor(eye, dtype=torch.float32, device=device)
    target = torch.tensor(target, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)

    # 카메라 좌표계 정의
    f = F.normalize(target - eye, dim=0) # 정면, eye는 알고있는 방향이니까 target - eye로 정면 방향을 구한다.
    r = F.normalize(torch.cross(f, up), dim=0) # 오른쪽, 좌표계에서 앞과 위를 외적하면 오른쪽이지
    u = torch.cross(f, r, dim=0) # 위쪽 cross: 외적

    R = torch.stack([r, u, f], dim=0)
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = -R @ eye
    return viewmat


def project_guassians(means, cov3d, cam, near=0.2):
    W = cam.viewmat[:3, :3]
    t = cam.viewmat[:3, 3]

    p_cam = means @ W.T + t
    x, y, z = p_cam.unbind(-1)

    mask = z > near
    z = z.clamp(min=near) # 물체까지 거리?

    # 중심 투영
    u = cam.fx * x/ z + cam.cx # z * u = fx * x, 즉 (필름 위 위치/ f = 실제 위치 / z)
    v = cam.fy * y/ z + cam.cy
    means2d = torch.stack([u, v], dim=-1)

    zeros = torch.zeros_like(z)

    # 자코비안, 공분산은 선형변환에서만 정의되는데
    # 원근 투영(카메라 값을 우리 값으로 받아오는건)은 비선형 변환이기 때문에, 자코비안으로 근사해서 공분산을 구한다.
    # 이는 좁은 영역에서 직선으로 보인다는 전제이므로 가우시안이 너무 크거나 화면 가장자리에 있으면 부정확해진다.
    # 그래서 3DGS 렌더링에서 화면 끝 splat이 부정확해진다. 하지만 대부분의 경우에는 충분히 정확하며 매우 빠르다...
    J = torch.stack([
        cam.fx / z, zeros, -cam.fx * x / (z **2),
        zeros, cam.fy / z, -cam.fy * y / (z **2)
    ], dim=-1).reshape(-1, 2, 3)

    T = J @ W
    cov2d = T @ cov3d @ T.transpose(1, 2) # 공분산 구하기

    # low pass filter, 적어도 1픽셀 이상은 될 수 있도록 보장
    cov2d = cov2d + 0.3 * torch.eye(2, device=cov2d.device)

    return means2d, cov2d, p_cam[:, 2], mask