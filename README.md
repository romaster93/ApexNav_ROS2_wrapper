<div align="center">
    <img src="assets/apexnav_logo_white.png" alt="ApexNav Logo" width="200">
    <h2>ApexNav ROS2 Wrapper</h2>
    <h4>An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion</h4>
    <strong>
      <em>ROS2 Jazzy Migration of <a href="https://github.com/Robotics-STAR-Lab/ApexNav">ApexNav</a></em>
    </strong>
    <br>
    <br>
    <a href="https://ieeexplore.ieee.org/document/11150727"><img alt="Paper" src="https://img.shields.io/badge/Paper-IEEE-blue"/></a>
    <a href="https://arxiv.org/abs/2504.14478"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-red"/></a>
    <a href='https://robotics-star.com/ApexNav'><img src='https://img.shields.io/badge/Project_Page-ApexNav-green' alt='Project Page'></a>
    <a href='https://github.com/Robotics-STAR-Lab/ApexNav'><img src='https://img.shields.io/badge/Original-ROS1_Version-orange' alt='Original'></a>

<br>
<br>

<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="assets/video_plant.gif" alt="apexnav_demo" width="80%">
  </a>
  <br>
  <em>
    ApexNav ensures highly <strong>reliable</strong> object navigation by leveraging <strong>Target-centric Semantic Fusion</strong>, and boosts <strong>efficiency</strong> with its <strong>Adaptive Exploration Strategy</strong>.
  </em>
</p>

</div>

## About This Repository

This is the **ROS2 Jazzy** port of [ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav), originally built on ROS1 Noetic. The entire codebase has been migrated to work with **Ubuntu 24.04 + ROS2 Jazzy**.

---

### ⚡ `isaacsim-ffw-swerve` Branch — IsaacSim + Swerve Drive 통합

> **이 branch는 NVIDIA IsaacSim 5.1.0 + ROBOTIS FFW-SG2 Swerve 로봇 전용 수정사항을 포함합니다.**
> 원본 ApexNAV 코드는 [`main` branch](https://github.com/romaster93/ApexNav_ROS2_wrapper)에 보존되어 있습니다.
>
> 📖 **전체 가이드**: [isaacsim-aiworker-guide Step 10](https://github.com/romaster93/isaacsim-aiworker-guide/blob/main/guides/10-apexnav-autonomous.md)

#### 주요 변경사항

| 파일 | 변경 내용 |
|------|----------|
| `kino_astar.h/cpp` | `is_swerve` 플래그 추가 — axis-aligned 충돌 체크 (Ackermann 회전 박스 대신) |
| `map_ros.cpp/h` | max-range depth 전용 `free_ray_cloud2d_` 분리 — 개방 환경 가짜 벽 방지 |
| `sdf_map2d.cpp` | `checkCollision` OPTIMISTIC 모드 — UNKNOWN 영역 통과 허용 |
| `exploration_fsm_traj.cpp` | odom_far 임계값 0.6→1.5m 완화 (swerve P제어 특성 반영) |
| `exploration_manager.h` | `searchFrontierPath` Astar2D OPTIMISTIC + FREE 셀 walkback |
| `algorithm_traj.launch.py` | IsaacSim 파라미터 (fx/fy=245.33, frame_id=World, max_depth yaml 참조) |
| `planning_param_ffw.yaml` | FFW-SG2 실측 치수 (length=0.56, width=0.51, is_swerve=true) |

#### 왜 KinoAstar를 수정했는가?

ApexNAV의 KinoAstar는 **Ackermann(자동차형) 로봇** 전제 설계:
- 이동 방향 = 로봇 정면 → yaw 기반 회전 박스 충돌 체크
- **Swerve 로봇**은 횡이동 가능 → yaw ≠ 이동 방향 → 회전 박스가 과잉 충돌 판정
- 결과: `isCollisionPosYaw occ!!!!!!!!` 폭주 → trajectory 실패 → 조기 FINISH

`is_swerve: true`로 axis-aligned 박스 사용 시 해결. 기존 로직은 `else` branch에 보존.

#### IsaacSim 실행 방법

```bash
# 빌드
source /opt/ros/jazzy/setup.bash
colcon build --packages-select path_searching exploration_manager trajectory_manager --symlink-install
source install/setup.bash

# 실행 (IsaacSim Play 상태에서)
ros2 launch exploration_manager exploration_traj.launch.py
```

> 전체 실행 순서 (bridge, swerve_controller 등 포함)는
> [Step 10 가이드](https://github.com/romaster93/isaacsim-aiworker-guide/blob/main/guides/10-apexnav-autonomous.md)를 참고하세요.

---

### What's Changed from the Original
- **Build system**: catkin → ament_cmake + colcon
- **C++ API**: roscpp → rclcpp
- **Python API**: rospy → rclpy
- **Launch files**: XML → Python launch files
- **Message generation**: catkin message generation → rosidl_generate_interfaces
- **TF**: tf → tf2_ros + tf_transformations (transforms3d)
- **Executor**: MultiThreadedExecutor with separate callback groups for service calls

## Installation
> Tested on Ubuntu 24.04 with ROS2 Jazzy and Python 3.12

You need to install [ROS2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html), and it is recommended to use [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your Python environment.

### 1. Prerequisites

#### 1.1 System Dependencies
``` bash
sudo apt update
sudo apt-get install libarmadillo-dev libompl-dev
sudo apt install ros-jazzy-desktop ros-jazzy-cv-bridge ros-jazzy-pcl-conversions \
    ros-jazzy-message-filters ros-jazzy-tf2-ros ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-visualization-msgs ros-jazzy-nav-msgs
```

#### 1.2 LLM (Optional)
> You can skip LLM configuration and directly use our pre-generated LLM output results in `llm/answers`.

ollama
``` bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
```

#### 1.3 External Code Dependencies
```bash
git clone git@github.com:WongKinYiu/yolov7.git # yolov7
git clone https://github.com/IDEA-Research/GroundingDINO.git # GroundingDINO
```

> **Note**: GroundingDINO's CUDA extension may need rebuilding for PyTorch 2.10+. If you see `NameError: name '_C' is not defined`, edit `groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu` and replace `value.type()` with `value.scalar_type()`, then reinstall with:
> ```bash
> cd GroundingDINO && CUDA_HOME=/usr/local/cuda pip install --no-build-isolation -e .
> ```

#### 1.4 Model Weights Download

Download the following model weights and place them in the `data/` directory:
- `mobile_sam.pt`: https://github.com/ChaoningZhang/MobileSAM/tree/master/weights/mobile_sam.pt
- `groundingdino_swint_ogc.pth`:
  ```bash
  wget -O data/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  ```
- `yolov7-e6e.pt`:
  ```bash
  wget -O data/yolov7-e6e.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
  ```


### 2. Setup Python Environment

#### 2.1 Clone Repository
``` bash
git clone git@github.com:romaster93/ApexNav_ROS2_wrapper.git
cd ApexNav_ROS2_wrapper
```

#### 2.2 Create Conda Environment
``` bash
conda env create -f apexnav_environment.yaml -y
conda activate apexnav_ros2
```

#### 2.3 Pytorch
``` bash
# You can use 'nvcc --version' to check your CUDA version.
# CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 2.4 Habitat Simulator
> We recommend using habitat-lab v0.3.1
``` bash
# habitat-lab v0.3.1
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/v0.3.1;
pip install -e habitat-lab

# habitat-baselines v0.3.1
pip install -e habitat-baselines
```

#### 2.5 tf_transformations
``` bash
pip install tf-transformations
```
> This installs the `tf_transformations` Python package (which depends on `transforms3d`) providing the same API as ROS1's `tf.transformations`.

#### 2.6 Others
``` bash
pip install salesforce-lavis==1.0.2
pip install -e .
```

## Datasets Download
> Official Reference: https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md

### Scene Datasets
**Note:** Both HM3D and MP3D scene datasets require applying for official permission first. Please refer to the [original ApexNav README](https://github.com/Robotics-STAR-Lab/ApexNav#-datasets-download) for detailed download instructions.

### Task Datasets
``` bash
# Create necessary directory structure
mkdir -p data/datasets/objectnav/hm3d
mkdir -p data/datasets/objectnav/mp3d

# HM3D-v0.1
wget -O data/datasets/objectnav/hm3d/v1.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
unzip data/datasets/objectnav/hm3d/v1.zip -d data/datasets/objectnav/hm3d && mv data/datasets/objectnav/hm3d/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1 && rm data/datasets/objectnav/hm3d/v1.zip

# HM3D-v0.2
wget -O data/datasets/objectnav/hm3d/v2.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip data/datasets/objectnav/hm3d/v2.zip -d data/datasets/objectnav/hm3d && mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2 && rm data/datasets/objectnav/hm3d/v2.zip

# MP3D
wget -O data/datasets/objectnav/mp3d/v1.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip
unzip data/datasets/objectnav/mp3d/v1.zip -d data/datasets/objectnav/mp3d/v1 && rm data/datasets/objectnav/mp3d/v1.zip
```

## Usage
> All following commands should be run in the conda environment with ROS2 sourced:
> ```bash
> conda activate apexnav_ros2
> source /opt/ros/jazzy/setup.bash
> source install/setup.bash
> ```

### ROS2 Build
``` bash
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

### Run VLMs Servers
Each command should be run in a separate terminal.
``` bash
python -m vlm.detector.grounding_dino --port 12181
python -m vlm.itm.blip2itm --port 12182
python -m vlm.segmentor.sam --port 12183
python -m vlm.detector.yolov7 --port 12184
```

### Launch Visualization and Main Algorithm (Discrete Action Mode)
Each command should be run in a separate terminal.
```bash
ros2 launch exploration_manager rviz.launch.py        # RViz2 visualization
ros2 launch exploration_manager exploration.launch.py  # ApexNav main algorithm
```

### Evaluate Datasets in Habitat
```bash
# Choose one dataset to evaluate
python habitat_evaluation.py --dataset hm3dv1
python habitat_evaluation.py --dataset hm3dv2  # default
python habitat_evaluation.py --dataset mp3d

# Evaluate on one specific episode
python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

# Generate evaluation videos
python habitat_evaluation.py --dataset hm3dv2 need_video=true
```

### Keyboard Control in Habitat
```bash
python habitat_manual_control.py --dataset hm3dv1              # Default episode_id = 0
python habitat_manual_control.py --dataset hm3dv1 test_epi_num=10  # episode_id = 10
```

### Real-world Deployment Example
If you want to run the real-world test example inside the Habitat simulator, please refer to the [Real World README](./real_world_test_example/README.md) for more details.

### Trajectory Mode (Velocity Control)
For trajectory planning with MPC control (used in real-world deployment testing):
```bash
# Terminal 1: RViz visualization (trajectory version)
ros2 launch exploration_manager rviz_traj.launch.py

# Terminal 2: ApexNav main algorithm (trajectory version)
ros2 launch exploration_manager exploration_traj.launch.py

# Terminal 3: Habitat simulator with velocity control
python habitat_vel_control.py
```
> In this mode, use RViz's **"2D Goal Pose"** tool to manually trigger autonomous exploration.
> For detailed real-world deployment instructions, see the [Real World README](./real_world_test_example/README.md).

<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="assets/auto_search.gif" alt="apexnav_demo2" width="80%">
  </a>
  <br>
  <em>
    Trajectory Planning and MPC Control in Real-world Deployment Example in Habitat Simulator.
  </em>
</p>

## TODO List

- [x] Release the main algorithm of ApexNav
- [x] Complete Installation and Usage documentation
- [x] Add datasets download documentation
- [x] Release the code of real-world deployment
- [x] Add ROS2 support

## License

This repository is a derivative work of [ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav), licensed under the [GNU General Public License v3.0](./LICENSE). The original code and algorithm were developed by:

> Mingjie Zhang, Yuheng Du, Chengkai Wu, Jinni Zhou, Zhenchao Qi, Jun Ma, Boyu Zhou
> **Robotics-STAR Lab** — HKUST(GZ) & SUSTech

This wrapper only provides ROS2 Jazzy compatibility. All core algorithms and intellectual property belong to the original authors.

## Acknowledgment

- **[ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav)**: The original ROS1 implementation by Robotics-STAR Lab.
- **[VLFM](https://github.com/bdaiinstitute/vlfm)**: For the concept of Vision-Language Frontier Maps.
- **[FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL)**: For the TSP-based efficient frontier exploration framework.

## Citation

```bibtex
@ARTICLE{zhang2025apexnav,
  author={Zhang, Mingjie and Du, Yuheng and Wu, Chengkai and Zhou, Jinni and Qi, Zhenchao and Ma, Jun and Zhou, Boyu},
  journal={IEEE Robotics and Automation Letters},
  title={ApexNAV: An Adaptive Exploration Strategy for Zero-Shot Object Navigation With Target-Centric Semantic Fusion},
  year={2025},
  volume={10},
  number={11},
  pages={11530-11537},
  keywords={Semantics;Navigation;Training;Robustness;Detectors;Noise measurement;Geometry;Three-dimensional displays;Object recognition;Faces;Search and rescue robots;vision-based navigation;autonomous agents},
  doi={10.1109/LRA.2025.3606388}}
```
