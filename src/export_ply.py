"""학습된 GaussianModel을 3DGS 표준 PLY로 내보낸다.

superspl.at/editor 같은 웹 뷰어에 드래그&드롭하면 바로 볼 수 있다.
"""
from pathlib import Path

import numpy as np
import torch

# 구면조화(SH) 0차 기저의 상수. 뷰어는 color = 0.5 + SH_C0 * f_dc 로 복원한다
SH_C0 = 0.28209479177387814

PROPERTIES = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]


@torch.no_grad()
def save_ply(g, path):
    """
        args:
            g: GaussianModel
            path: 저장할 .ply 경로

        주의: scale과 opacity는 activation을 통과시키지 않은 raw 값을 넣는다.
        뷰어가 읽을 때 exp/sigmoid를 적용하는 것이 3DGS의 규약이라,
        여기서 g.scales나 g.opacity를 넣으면 두 번 적용되어 망가진다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = g._means.detach().cpu().numpy()
    normals = np.zeros_like(xyz)                              # 안 쓰지만 포맷상 필요
    f_dc = (g.colors.detach().cpu().numpy() - 0.5) / SH_C0    # 색 -> SH 0차 계수
    opacity = g._opacity.detach().cpu().numpy()               # logit 그대로
    scale = g._scales.detach().cpu().numpy()                  # log 그대로
    rot = g.quats.detach().cpu().numpy()                      # 정규화된 (w, x, y, z)

    data = np.concatenate(
        [xyz, normals, f_dc, opacity, scale, rot], axis=1
    ).astype(np.float32)

    assert data.shape[1] == len(PROPERTIES), \
        f"속성 개수 불일치: {data.shape[1]} != {len(PROPERTIES)}"

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        + "".join(f"property float {name}\n" for name in PROPERTIES)
        + "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    print(f"saved {len(xyz)} gaussians -> {path}")


if __name__ == "__main__":
    # 정답 장면을 내보내서 뷰어에서 좌표계/색이 맞게 나오는지 먼저 확인한다.
    # 학습 결과를 보려면 train_3d.py를 실행할 것.
    import config
    from make_dummy_data import make_target_scene

    gt_scene = make_target_scene()
    save_ply(gt_scene, Path(config.save_path_3d) / "gt_scene.ply")
