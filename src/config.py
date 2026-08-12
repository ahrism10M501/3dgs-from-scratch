# Train Hyper Params

from pathlib import Path

# 저장소 루트. 경로들을 여기 기준 절대경로로 만들어서 실행 위치에 상관없이 동작하게 한다.
ROOT = Path(__file__).resolve().parent.parent

## Training
iters = 3000
save_every = 100
save_path = ROOT / "results" / "exp05_adam_dssim_img3"
target_path = ROOT / "src" / "images" / "images.jpeg"

## Multi-view (train_3d.py)
save_path_3d = ROOT / "results" / "exp06_multiview"
num_views = 16      # 궤도 위에 배치할 학습용 카메라 개수
orbit_radius = 4.0  # 원점에서 카메라까지 거리
elevation = 20.0    # 궤도의 고도각 (도)

## Gaussian Initialization
num_points = 2000
extent = 0.8        # 가우시안 초기 위치가 뿌려지는 월드 공간 범위
init_scale = 0.05   # log를 취하기 전의 실제 스케일
init_opacity = 0.1  # sigmoid를 통과한 뒤의 실제 불투명도

## Gaussian Optimizer Learning Rates
means_lr = 5e-3
scales_lr = 5e-3
quats_lr = 1e-3
opacity_lr = 5e-2
colors_lr = 2.5e-2

## Camera / Rendering
size = 128          # 렌더 해상도이자 target 이미지 크기 (둘은 항상 같아야 함)
fov = 60.0
eye = (0.0, 0.0, -4.0)
near = 0.2          # 이보다 가까운 가우시안은 컬링
blur_eps = 0.3      # 2D 공분산 low-pass filter, 최소 픽셀 크기 보장
background = 0.0