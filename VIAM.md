# VIAM.md

Reference documentation for using Viam robotics platform in this project.

## Platform Overview

Viam is a unified software platform for robotics that provides:
- `viam-server`: Open-source binary managing hardware and software
- Standardized APIs compatible across hardware types
- Built-in services for motion, machine learning, vision, and data
- Cloud-based machine management
- SDKs: Python, Go, TypeScript, C++, Flutter

Supported platforms: Linux (AArch64/x86-64/Armv7l), macOS, Windows, ESP32 microcontrollers.

## Architecture Concepts

**Components**: Physical hardware - cameras, motors, sensors, arms, grippers, etc.

**Services**: Higher-level functionality - vision detection, motion planning, data management, navigation, SLAM.

**Modules**: Plugins containing components and services for hardware integration.

**Frame System**: Hierarchical coordinate system with "world" as root. Each component has a frame and an "_origin" frame. Coordinates in millimeters. For bases: +X=right, +Y=forward, +Z=up.

## CLI Installation

```bash
# macOS
brew tap viamrobotics/brews && brew install viam

# Linux
# Download binary via curl from docs

# From source (requires Go)
go install go.viam.com/rdk/cli/viam@latest
```

Authentication: `viam login` (browser) or `viam login api-key` (API keys). Sessions valid 24 hours.

## Python SDK

Install: `pip install viam-sdk`

### Connection Pattern

```python
import asyncio
from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions

async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key='<API-KEY>',
        api_key_id='<API-KEY-ID>'
    )
    return await RobotClient.at_address('<ROBOT-ADDRESS>', opts)

async def main():
    machine = await connect()
    # ... use components and services
    await machine.close()

if __name__ == '__main__':
    asyncio.run(main())
```

## Arm Component API

Controls robotic arms with linear motion planning and self-collision prevention.

```python
from viam.components.arm import Arm
from viam.proto.common import Pose
from viam.proto.component.arm import JointPositions

my_arm = Arm.from_robot(robot=machine, name="my_arm")

# Get current end-effector position (6 DOF pose)
pos = await my_arm.get_end_position()

# Move end-effector in straight line to target pose
target = Pose(x=5, y=5, z=5, o_x=5, o_y=5, o_z=5, theta=20)
await my_arm.move_to_position(pose=target)

# Move joints to specific angles (collision checks disabled)
joint_pos = JointPositions(values=[0.0, 45.0, 0.0, 0.0, 0.0])
await my_arm.move_to_joint_positions(positions=joint_pos)

# Get current joint positions
joints = await my_arm.get_joint_positions()

# Stop motion
await my_arm.stop()

# Check if moving
moving = await my_arm.is_moving()
```

Key methods:
- `get_end_position()`: Current pose (x, y, z, orientation)
- `move_to_position(pose)`: Linear motion to pose
- `move_to_joint_positions(positions)`: Direct joint control
- `get_joint_positions()`: Current joint angles
- `get_kinematics()`: Returns URDF or SVA kinematics data
- `stop()`, `is_moving()`

## Gripper Component API

```python
from viam.components.gripper import Gripper

my_gripper = Gripper.from_robot(robot=machine, name="my_gripper")

# Open gripper
await my_gripper.open()

# Close until grabbing something or fully closed
# Returns True if grabbed something
grabbed = await my_gripper.grab()

# Check if holding something
holding = await my_gripper.is_holding_something()

# Check movement status
moving = await my_gripper.is_moving()

# Stop gripper
await my_gripper.stop()
```

## Motor Component API

For controlling motors (e.g., dice rolling mechanism).

```python
from viam.components.motor import Motor

my_motor = Motor.from_robot(robot=machine, name="my_motor")

# Set power (-1 to 1)
await my_motor.set_power(power=0.4)  # 40% forward

# Run at specific RPM (negative = reverse)
await my_motor.set_rpm(rpm=75)

# Move specific number of rotations at given RPM
await my_motor.go_for(rpm=50, revolutions=2)

# Move to absolute position
await my_motor.go_to(rpm=50, position_revolutions=5)

# Get current position (revolutions from home)
position = await my_motor.get_position()

# Reset zero position
await my_motor.reset_zero_position(offset=0)

# Stop immediately
await my_motor.stop()
```

## Camera Component API

```python
from viam.components.camera import Camera

my_camera = Camera.from_robot(robot=machine, name="my_camera")

# Get images (recommended for depth + color cameras)
images, metadata = await my_camera.get_images()
timestamp = metadata.captured_at

# Get point cloud data
data, mime_type = await my_camera.get_point_cloud()

# Get camera properties (intrinsics, distortion)
properties = await my_camera.get_properties()

await my_camera.close()
```

Utility functions: `viam_to_pil_image()` for PIL conversion, `bytes_to_depth_array()` for depth processing.

## Vision Service API

For object detection, classification, and segmentation.

```python
from viam.services.vision import VisionClient

my_detector = VisionClient.from_robot(robot=machine, name="my_detector")

# Get detections from live camera feed
detections = await my_detector.get_detections_from_camera("my_camera")

# Get detections from specific image
my_camera = Camera.from_robot(robot=machine, name="my_camera")
images, _ = await my_camera.get_images()
img = images[0]
detections = await my_detector.get_detections(img)

# Each detection has:
# - x_min, y_min, x_max, y_max (bounding box in pixels)
# - class_name (label string)
# - confidence (0.0 to 1.0)

# Get classifications
classifications = await my_detector.get_classifications_from_camera("my_camera", count=5)

# Get 3D object point clouds (requires frame system config)
objects = await my_detector.get_object_point_clouds("my_camera")

# Get all at once
result = await my_detector.capture_all_from_camera(
    "my_camera",
    return_image=True,
    return_detections=True,
    return_classifications=True
)
```

### Vision Service Models

**mlmodel**: Uses TensorFlow Lite, TensorFlow, PyTorch, or ONNX models. Configure with:
- `mlmodel_name`: Reference to deployed ML model service
- `default_minimum_confidence`: Threshold for detections
- `label_confidences`: Per-label confidence thresholds

Tensor naming for detectors: `image` (input), `location`, `category`, `score` (outputs).

## Motion Service API

For motion planning with obstacle avoidance.

```python
from viam.services.motion import MotionClient

motion = MotionClient.from_robot(robot=machine, name="builtin")

# Move component to pose (constructs kinematic chains, avoids collisions)
success = await motion.move(
    component_name=my_arm.get_resource_name(),
    destination=target_pose,
    world_state=world_state  # Optional: obstacles and transforms
)

# Get pose of component in frame system
pose = await motion.get_pose(
    component_name=my_arm.get_resource_name(),
    destination_frame="world"
)

# For mobile bases - move on SLAM map
execution_id = await motion.move_on_map(
    component_name="my_base",
    destination=Pose(y=10),
    slam_service_name="my_slam_service"
)

# Stop ongoing motion plan
await motion.stop_plan(component_name="my_base")
```

## Frame System Configuration

Configure in machine's CONFIGURE tab. Each component needs:
- `parent`: Reference frame name (default: "world")
- `translation`: Origin coordinates in mm `{x, y, z}`
- `orientation`: Rotation (Euler angles or axis vectors)
- `geometry`: Optional collision boundaries (sphere, box, capsule)

Access via Machine Management API:
- `FrameSystemConfig()`: Get all reference frames
- `TransformPose()`: Convert poses between frames

## ML Model Deployment

Supported frameworks:
- TensorFlow Lite (`tflite_cpu`): Optimized for edge devices
- ONNX (`onnx-cpu`, `triton`): Universal format
- TensorFlow (`tensorflow-cpu`, `triton`)
- PyTorch (`torch-cpu`, `triton`)

Deployment steps:
1. Train model or upload to Viam Cloud
2. Add ML model service in CONFIGURE tab
3. Select model from organization or registry
4. Configure vision service to use the ML model

## Data Capture for Training

Data flows: Capture -> Sync -> Cloud

```
# Default capture directory: ~/.viam/capture
# Data synced via encrypted gRPC, deleted after sync
```

Configure in Builder mode:
1. Navigate to resource (e.g., camera)
2. Enable data capture section
3. Select method (e.g., GetImages) and frequency (Hz)
4. Save configuration

Use captured data to create datasets for model training.

## Kinematic Chain Configuration

For robot arms, provide kinematics via JSON or URDF files.

Two approaches:
- **Spatial Vector Algebra (SVA)**: Preferred. Allows arbitrary link frame specification.
- **Denavit-Hartenberg (DH) Parameters**: Alternative mathematical approach.

Access via `arm.get_kinematics()` which returns format and byte contents.

## Relevant Documentation Links

- Main docs: https://docs.viam.com/
- Arm API: https://docs.viam.com/dev/reference/apis/components/arm/
- Gripper API: https://docs.viam.com/dev/reference/apis/components/gripper/
- Camera API: https://docs.viam.com/dev/reference/apis/components/camera/
- Motor API: https://docs.viam.com/dev/reference/apis/components/motor/
- Vision Service: https://docs.viam.com/dev/reference/apis/services/vision/
- Motion Service: https://docs.viam.com/dev/reference/apis/services/motion/
- CLI: https://docs.viam.com/dev/tools/cli/
- Configure Arm: https://docs.viam.com/operate/mobility/move-arm/configure-arm/
