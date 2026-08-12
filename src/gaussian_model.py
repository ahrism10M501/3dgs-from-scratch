import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def inverse_sigmoid(x:torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))

class GaussianModel(nn.Module):
    """
        가우시안들을 저장하고 있는 클래스 객체를 만든다.
        논문대로 means, scales, quats, opacity, colors를 초기화하고 저장한다.

        args:
            num_points: 가우시안의 개수
            extent: 가우시안의 초기 위치를 결정하는 범위
            device: 모델을 저장할 장치 (cpu, gpu)
        defaults:
            num_points=2000, extent=1.0, device="cpu"
    """
    def __init__(self, num_points=2000, extent=1.0, device="cpu"):
        super().__init__()
        self.num_points = num_points
        self.extent = extent
        self.device = device

        # rand는 [0, 1) 이니까 이를 [-0.5, 0.5)로 바꾸고, extent를 곱해서 가우시안 중심 위치를 결정한다.
        means = (torch.rand(num_points, 3, device=device) - 0.5) *2 * extent # 가우시안 중심 위치

        scales = torch.full((num_points, 3), math.log(0.05), device=device)  # 축 방향 크기

        quats = torch.zeros(num_points, 4, device=device) # quaternions, 회전 -> [1, 0, 0, 0] 이면 회전 없음
        quats[:, 0] = 1.0 # 그래서 초기값을 [1, 0, 0, 0]으로 설정

        # 0.1로 채워진 (num_points, 1) 크기의 텐서를 만들고(torch.full) inverse_sigmoid를 적용하여 초기 opacity를 설정한다!
        # 시그모이드 역함수 씌우는건, 아래 property에서 sigmoid를 씌우니까, 초기값을 원하는 0.1로 하려면 역함수 필요하다
        opacity = inverse_sigmoid(torch.full((num_points, 1), 0.1, device=device)) # 가우시안의 불투명도

        colors = torch.rand(num_points, 3, device=device) # uniform 랜덤

        # 실제 학습 가능한 파라미터로 등록하기 위해 nn.Parameter로 만든다. 즉, 위에건 초기값, 아래는 학습하는거
        self._means = nn.Parameter(means)
        self._scales = nn.Parameter(scales)
        self._quats = nn.Parameter(quats)
        self._opacity = nn.Parameter(opacity)
        self._colors = nn.Parameter(colors)

    # 학습할 때 각각의 파라미터를 쓸 수 있게 property로 만들어준다.
    # 이때! 유효 범위를 가지도록 변환 (scales는 exp, quats는 normalize, opacity와 colors는 sigmoid)
    # Adam 같은 옵티마이저가 어떤 범위로 바꾸어도, 항상 물리적으로 유효한 값으로 반환하도록 만드는 것임
    @property
    def scales(self):
        return torch.exp(self._scales)

    @property
    def quats(self):
        return F.normalize(self._quats, dim=-1)

    @property
    def opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def colors(self):
        return torch.sigmoid(self._colors)

    # 논문에서 언급된 covariance를 계산하는 함수
    def covariance(self):
        # 회전 행렬 구하고
        q = F.normalize(self._quats, dim=-1)
        w, x, y, z = q.unbind(-1)

        # 회전행렬, 9개 성분을 각각 (N, ) 텐서로 계산하고 쌓아서, (N, 9) -> reshape -> (N, 3, 3)
        R = torch.stack([
            1 -2 * ( y **2 + z **2), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 -2 * (x **2 + z **2), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 -2 * (x **2 + y **2)
        ], dim=-1).reshape(-1, 3, 3)

        # 원래는 torch.diag(scales)인데, scales가 (N, 3)이라서 unsqueeze(1)로 (N, 1, 3)으로 만들어서 곱해주면
        # (N, 3, 3) 대각행렬이 됨. 이게 코드상 더 값싸서 쓰는거.
        M = R * self.scales.unsqueeze(1)
        return M @ M.transpose(1, 2) # 논문에서 언급된 (R@S@S.T@R.T)

