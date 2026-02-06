# ApexNav Real-world Test Example

This release contains trajectory-generation and MPC control code. Using the supplied scripts, you can perform a real-world test within the Habitat simulator, and the implementation is readily portable to a physical mobile robot.

## Dependencies

Please follow [the main README](../README.md) to set up the ApexNav
environment and the Habitat simulator before running this example.

## Quick Start

> All following commands should be run in the conda environment with ROS2 sourced:
> ```bash
> conda activate apexnav_ros2
> source /opt/ros/jazzy/setup.bash
> source install/setup.bash
> ```

### ROS2 Build
```bash
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

### Run VLM servers
Run each of the following commands in a separate terminal:
```bash
python -m vlm.detector.grounding_dino --port 12181
python -m vlm.itm.blip2itm --port 12182
python -m vlm.segmentor.sam --port 12183
python -m vlm.detector.yolov7 --port 12184
```

### Launch visualization and main algorithm
```bash
ros2 launch exploration_manager rviz_traj.launch.py          # RViz2 visualization
ros2 launch exploration_manager exploration_traj.launch.py    # ApexNav main algorithm
```

### Run Habitat simulator (velocity-control version)
```bash
python habitat_vel_control.py
```

### Run the real-world example node
```bash
python ./real_world_test_example/real_world_test_habitat.py
```


### Visualization with RViz

<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="../assets/rviz_show.jpg" alt="piperviz_showline" width="80%">
  </a>
</p>

#### Manually Set Goal in RViz

You can use RViz's *<strong>"2D Pose Estimate"</strong>* tool to set a target pose so the robot can navigate to a specified location.

<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="../assets/point2point.gif" alt="apexnav_demo1" width="80%">
  </a>
  <br>
</p>

#### Autonomous Object Navigation

You can use RViz's *<strong>"2D Nav Goal"</strong>* tool to start autonomous exploration.

<p align="center" style="font-size: 1.0em;">
  <a href="">
    <img src="../assets/auto_search.gif" alt="apexnav_demo2" width="80%">
  </a>
  <br>
</p>

## Acknowledgment

We would like to acknowledge the contributions of the following projects:
- **[REMANI-Planner](https://github.com/Robotics-STAR-Lab/REMANI-Planner)**: Real-time Whole-body Motion Planning for Mobile Manipulators.
