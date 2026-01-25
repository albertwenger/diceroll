"""
Vision-based cube and bowl detection using VIAM vision services.

This module uses VIAM's color_detector vision service to find colored cubes.
The vision services must be configured in the VIAM app.
"""

import asyncio
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.components.arm import Arm
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame


# Cube colors and their detector service names
CUBE_DETECTORS = {
    "green": "vision-detect-green",
    "blue": "vision-detect-blue",
    "red": "vision-detect-red",
    "yellow": "vision-detect-yellow",
}

# Bowl positions (hardcoded since color detector can't detect black)
# These are in arm world coordinates (mm)
BOWL_POSITIONS = {
    "left": {"x": 250, "y": 200, "z": 20},
    "right": {"x": 250, "y": -100, "z": 20},
}

# Scanning position - arm looks down at the center of the tray
SCAN_POSITION = {
    "x": 250,
    "y": 0,
    "z": 350,  # High enough to see the whole workspace
}

# Camera intrinsics for RealSense (approximate, may need calibration)
# These convert pixel coordinates to 3D coordinates
# Higher focal length = smaller world offset for same pixel offset
CAMERA_INTRINSICS = {
    "fx": 900.0,  # Focal length x (increased to reduce offset)
    "fy": 900.0,  # Focal length y (increased to reduce offset)
    "cx": 320.0,  # Principal point x
    "cy": 240.0,  # Principal point y
}


@dataclass
class Detection:
    """A detected object with its position."""
    color: str
    x_px: int  # Pixel x coordinate (center of bounding box)
    y_px: int  # Pixel y coordinate
    width_px: int
    height_px: int
    confidence: float
    # 3D position in arm world frame (mm), if depth available
    world_x: Optional[float] = None
    world_y: Optional[float] = None
    world_z: Optional[float] = None


def load_credentials():
    creds_path = Path(__file__).parent / "credentials.json"
    with open(creds_path) as f:
        return json.load(f)


async def connect():
    creds = load_credentials()
    opts = RobotClient.Options.with_api_key(
        api_key=creds["api_key"],
        api_key_id=creds["api_key_id"]
    )
    return await RobotClient.at_address(creds["robot_address"], opts)


async def list_vision_services(machine) -> list[str]:
    """List available vision services on the robot."""
    # Use the resource_names property (not a method)
    resources = machine.resource_names
    vision_services = [
        r.name for r in resources
        if r.subtype == "vision"
    ]
    return vision_services


async def move_to_scan_position(machine) -> bool:
    """Move arm to scanning position above the tray."""
    print("\nMoving to scan position...")

    arm = Arm.from_robot(machine, "lite6")
    motion = MotionClient.from_robot(machine, "builtin")

    scan_pose = Pose(
        x=SCAN_POSITION["x"],
        y=SCAN_POSITION["y"],
        z=SCAN_POSITION["z"],
        o_x=0, o_y=0, o_z=-1, theta=0  # Gripper pointing down
    )

    destination = PoseInFrame(
        reference_frame="world",
        pose=scan_pose
    )

    try:
        success = await motion.move(
            component_name=arm.name,
            destination=destination
        )
        if success:
            pos = await arm.get_end_position()
            print(f"  At scan position: x={pos.x:.1f}, y={pos.y:.1f}, z={pos.z:.1f}")
        return success
    except Exception as e:
        print(f"  Move error: {e}")
        return False


async def detect_cubes_with_service(
    machine,
    detector_name: str,
    camera_name: str = "realsense"
) -> list[Detection]:
    """
    Detect cubes using a configured VIAM vision service.

    Args:
        machine: Connected robot client
        detector_name: Name of the vision service (e.g., "green_detector")
        camera_name: Camera to use for detection

    Returns:
        List of Detection objects
    """
    try:
        detector = VisionClient.from_robot(machine, detector_name)
        detections = await detector.get_detections_from_camera(camera_name)

        results = []
        for det in detections:
            # Calculate center of bounding box
            x_center = (det.x_min + det.x_max) // 2
            y_center = (det.y_min + det.y_max) // 2
            width = det.x_max - det.x_min
            height = det.y_max - det.y_min

            results.append(Detection(
                color=detector_name.replace("_detector", ""),
                x_px=x_center,
                y_px=y_center,
                width_px=width,
                height_px=height,
                confidence=det.confidence
            ))

        return results
    except Exception as e:
        print(f"  Warning: Could not use detector '{detector_name}': {e}")
        return []


async def detect_all_cubes(
    machine,
    camera_name: str = "realsense"
) -> dict[str, list[Detection]]:
    """
    Detect all colored cubes using configured vision services.
    """
    results = {}

    for color, detector_name in CUBE_DETECTORS.items():
        detections = await detect_cubes_with_service(
            machine, detector_name, camera_name
        )
        if detections:
            results[color] = detections
            print(f"  Found {len(detections)} {color} cube(s)")

    return results


async def pixel_to_world(
    machine,
    x_px: int,
    y_px: int,
    rgb_width: int = 1280,
    rgb_height: int = 720,
    camera_name: str = "realsense",
    arm_name: str = "lite6"
) -> Optional[tuple[float, float, float]]:
    """
    Convert pixel coordinates to world coordinates using depth.

    This uses the RealSense depth data and the arm's current position
    to estimate the 3D world position.

    Returns:
        (x, y, z) in arm world frame (mm), or None if failed
    """
    try:
        camera = Camera.from_robot(machine, camera_name)
        arm = Arm.from_robot(machine, arm_name)

        # Get arm position first
        arm_pos = await arm.get_end_position()

        # Get depth image
        images, _ = await camera.get_images()

        depth_image = None
        depth_width = 0
        depth_height = 0
        for img in images:
            if hasattr(img, 'data') and len(img.data) > 8 and img.data[:8] == b"DEPTHMAP":
                # Parse depth image - try header first, then common resolutions
                import struct
                raw_bytes = img.data
                width = struct.unpack('<I', raw_bytes[8:12])[0]
                height = struct.unpack('<I', raw_bytes[12:16])[0]
                pixel_data = raw_bytes[16:]
                expected_size = width * height * 2

                if abs(expected_size - len(pixel_data)) < 100:
                    depth_image = np.frombuffer(
                        pixel_data[:expected_size], dtype=np.uint16
                    ).reshape((height, width))
                    depth_width, depth_height = width, height
                else:
                    # Fallback: infer from common resolutions
                    for header_size in [16, 24]:
                        total_pixels = len(raw_bytes[header_size:]) // 2
                        for w, h in [(1280, 720), (640, 480), (848, 480), (640, 360)]:
                            if w * h == total_pixels:
                                data = raw_bytes[header_size:header_size + w*h*2]
                                depth_image = np.frombuffer(data, dtype=np.uint16).reshape((h, w))
                                depth_width, depth_height = w, h
                                break
                        if depth_image is not None:
                            break
                break

        if depth_image is None:
            print("  Warning: Could not get depth image")
            return None

        print(f"  RGB: {rgb_width}x{rgb_height}, Depth: {depth_width}x{depth_height}")

        # Scale pixel coordinates from RGB resolution to depth resolution
        scale_x = depth_width / rgb_width
        scale_y = depth_height / rgb_height
        depth_x = int(x_px * scale_x)
        depth_y = int(y_px * scale_y)
        print(f"  Pixel ({x_px}, {y_px}) -> Depth ({depth_x}, {depth_y})")

        # Get depth at pixel location (in mm)
        if 0 <= depth_y < depth_image.shape[0] and 0 <= depth_x < depth_image.shape[1]:
            depth_mm = float(depth_image[depth_y, depth_x])
        else:
            print(f"  Warning: Depth pixel ({depth_x}, {depth_y}) out of bounds")
            return None

        # Estimate distance to tray surface (arm is looking down)
        # Arm is at ~350mm, tray surface is at ~0mm, so distance is ~350mm
        # Use depth for validation but estimate distance from arm position
        distance_to_surface = arm_pos.z  # Approximate: arm height = distance to surface

        # Convert pixel to camera frame 3D point
        # Use depth image center as principal point
        fx, fy = CAMERA_INTRINSICS["fx"], CAMERA_INTRINSICS["fy"]
        cx = depth_width / 2
        cy = depth_height / 2

        # Camera frame coordinates (mm)
        # Assuming camera points straight down when gripper points down
        cam_z = distance_to_surface
        cam_x = (depth_x - cx) * cam_z / fx
        cam_y = (depth_y - cy) * cam_z / fy

        # Transform from camera frame to world frame
        # Camera is mounted on arm, pointing same direction as gripper (down)
        # Camera X = right (arm -Y), Camera Y = down in image (arm +X), Camera Z = forward (arm -Z)
        world_x = arm_pos.x + cam_y  # Camera Y (down in img) -> arm forward
        world_y = arm_pos.y - cam_x  # Camera X (right) -> arm left
        world_z = 15  # Cube pick height above tray surface

        print(f"  Distance: {distance_to_surface:.0f}mm, Offset: ({cam_x:.1f}, {cam_y:.1f})")

        return (world_x, world_y, world_z)

    except Exception as e:
        print(f"  Warning: Failed to convert pixel to world: {e}")
        import traceback
        traceback.print_exc()
        return None


async def find_cube_position(
    machine,
    color: str,
    camera_name: str = "realsense"
) -> Optional[Pose]:
    """
    Find a specific colored cube and return its world position.

    Args:
        machine: Connected robot client
        color: Cube color ("green", "blue", "red", "yellow")
        camera_name: Camera to use

    Returns:
        Pose of the cube in world frame, or None if not found
    """
    if color not in CUBE_DETECTORS:
        print(f"  Unknown color: {color}")
        return None
    detector_name = CUBE_DETECTORS[color]
    detections = await detect_cubes_with_service(machine, detector_name, camera_name)

    if not detections:
        print(f"  No {color} cube detected")
        return None

    # Use the detection with highest confidence
    best = max(detections, key=lambda d: d.confidence)
    print(f"  Found {color} cube at pixel ({best.x_px}, {best.y_px}), "
          f"confidence: {best.confidence:.2f}")

    # Convert to world coordinates
    world_pos = await pixel_to_world(machine, best.x_px, best.y_px, camera_name=camera_name)

    if world_pos is None:
        return None

    world_x, world_y, world_z = world_pos
    print(f"  World position: x={world_x:.1f}, y={world_y:.1f}, z={world_z:.1f}")

    # Return pose with gripper pointing down
    return Pose(
        x=world_x,
        y=world_y,
        z=world_z + 15,  # Slightly above surface for grip
        o_x=0, o_y=0, o_z=-1, theta=0
    )


def get_bowl_pose(side: str) -> Pose:
    """Get the pose for placing into a bowl (hardcoded positions)."""
    if side not in BOWL_POSITIONS:
        raise ValueError(f"Unknown bowl: {side}. Available: {list(BOWL_POSITIONS.keys())}")
    bowl = BOWL_POSITIONS[side]
    return Pose(
        x=bowl["x"],
        y=bowl["y"],
        z=bowl["z"] + 80,  # Above bowl to drop
        o_x=0, o_y=0, o_z=-1, theta=0
    )


async def main():
    """Test vision detection."""
    print("=" * 60)
    print("VISION DETECTION TEST")
    print("=" * 60)

    print("\nConnecting to robot...")
    machine = await connect()
    print("Connected!")

    try:
        # List available vision services
        print("\nAvailable vision services:")
        services = await list_vision_services(machine)
        if services:
            for svc in services:
                print(f"  - {svc}")
        else:
            print("  None found!")
            print("\n  To use vision detection, configure color_detector services")
            print("  in your VIAM app for each cube color.")
            return

        # Move to scanning position
        await move_to_scan_position(machine)

        # Wait a moment for camera to stabilize
        await asyncio.sleep(1.0)

        # Try to detect cubes
        print("\nDetecting cubes...")
        all_detections = await detect_all_cubes(machine)

        if not all_detections:
            print("  No cubes detected. Check vision service configuration.")
        else:
            total = sum(len(d) for d in all_detections.values())
            print(f"\nDetected {total} object(s) total")

            # Try to get world position for each color
            for color, detections in all_detections.items():
                if detections:
                    # Filter to reasonable cube-sized detections
                    cube_sized = [d for d in detections
                                  if 20 < d.width_px < 200 and 20 < d.height_px < 200]
                    if cube_sized:
                        print(f"\n{color.upper()}: {len(cube_sized)} cube-sized detection(s)")
                        best = max(cube_sized, key=lambda d: d.confidence)
                        print(f"  Best at pixel ({best.x_px}, {best.y_px}), "
                              f"size {best.width_px}x{best.height_px}, "
                              f"conf: {best.confidence:.2f}")

                        pose = await find_cube_position(machine, color)
                        if pose:
                            print(f"  World: x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}")
                    else:
                        print(f"\n{color.upper()}: {len(detections)} detection(s) but none cube-sized")

    finally:
        await machine.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
