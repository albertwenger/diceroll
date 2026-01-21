"""
Camera-to-arm calibration script.

Moves the arm through multiple poses, captures the gripper position from both:
1. The arm's own kinematics (end-effector position in arm frame)
2. The RealSense depth camera (gripper position in camera frame)

Then solves for the rigid transform between camera and arm coordinate frames.
"""

import asyncio
import json
import struct
from pathlib import Path

import numpy as np
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.common import Pose
from viam.proto.component.arm import JointPositions


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


def parse_pcd_bytes(pcd_bytes: bytes) -> np.ndarray:
    """Parse PCD (Point Cloud Data) bytes into numpy array of XYZ points."""
    # PCD format: header lines followed by binary data
    # Find end of header (empty line or DATA line)
    header_end = pcd_bytes.find(b"DATA binary\n")
    if header_end == -1:
        raise ValueError("Could not find DATA binary marker in PCD")

    data_start = header_end + len(b"DATA binary\n")

    # Parse header to get point count and fields
    header = pcd_bytes[:header_end].decode('utf-8')
    points_count = 0
    for line in header.split('\n'):
        if line.startswith('POINTS'):
            points_count = int(line.split()[1])
            break

    if points_count == 0:
        return np.array([]).reshape(0, 3)

    # Parse binary data - assuming XYZ float32 format
    binary_data = pcd_bytes[data_start:]
    # Each point is 3 floats (x, y, z) = 12 bytes, but RealSense often includes RGB
    # Try 16 bytes per point first (XYZRGB with padding), fall back to 12
    bytes_per_point = len(binary_data) // points_count

    points = []
    for i in range(points_count):
        offset = i * bytes_per_point
        x, y, z = struct.unpack('fff', binary_data[offset:offset+12])
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            points.append([x, y, z])

    return np.array(points)


def find_gripper_position(points: np.ndarray, arm_pos: Pose) -> np.ndarray | None:
    """
    Find the gripper in the point cloud.

    Strategy: The gripper should be the highest point (smallest Z in camera frame,
    since camera looks down) in the region where we expect it based on rough estimates.
    We also look for a cluster of points that are elevated above the desk.
    """
    if len(points) == 0:
        return None

    # Filter out points that are too far away (likely background)
    # RealSense depth is in mm, points should be within ~1m of camera
    distances = np.linalg.norm(points, axis=1)
    valid_mask = (distances > 100) & (distances < 1500)  # 10cm to 1.5m
    valid_points = points[valid_mask]

    if len(valid_points) == 0:
        return None

    # The gripper should be one of the highest points (closest to camera = smallest Z typically,
    # but depends on camera orientation). Let's find the centroid of the top 1% of points
    # sorted by distance from camera (closest points)
    distances = np.linalg.norm(valid_points, axis=1)
    threshold_idx = max(1, len(valid_points) // 100)  # Top 1%
    closest_indices = np.argsort(distances)[:threshold_idx]

    # Get centroid of these closest points
    closest_points = valid_points[closest_indices]
    gripper_pos = np.mean(closest_points, axis=0)

    return gripper_pos


def solve_transform(arm_points: np.ndarray, camera_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for rigid transform from camera frame to arm frame.

    Uses Kabsch algorithm (SVD-based) to find rotation R and translation t
    such that: arm_point = R @ camera_point + t

    Returns (R, t) where R is 3x3 rotation matrix, t is 3x1 translation vector.
    """
    assert len(arm_points) == len(camera_points) >= 3, "Need at least 3 point pairs"

    # Compute centroids
    arm_centroid = np.mean(arm_points, axis=0)
    cam_centroid = np.mean(camera_points, axis=0)

    # Center the points
    arm_centered = arm_points - arm_centroid
    cam_centered = camera_points - cam_centroid

    # Compute covariance matrix
    H = cam_centered.T @ arm_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = arm_centroid - R @ cam_centroid

    return R, t


def rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert rotation matrix to axis-angle representation."""
    # Angle
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if np.abs(angle) < 1e-6:
        return np.array([0, 0, 1]), 0.0

    # Axis
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))

    return axis / np.linalg.norm(axis), np.degrees(angle)


# Calibration poses - joint positions that move the gripper around the workspace
# These are safe poses for the Lite 6 that keep the gripper over the desk
CALIBRATION_POSES = [
    # [j1, j2, j3, j4, j5, j6] in degrees
    [0, -30, 60, 0, 90, 0],      # Center, forward
    [30, -30, 60, 0, 90, 0],     # Right
    [-30, -30, 60, 0, 90, 0],    # Left
    [0, -45, 75, 0, 90, 0],      # Center, closer
    [0, -15, 45, 0, 90, 0],      # Center, further
    [20, -40, 70, 0, 90, 0],     # Right-close
    [-20, -40, 70, 0, 90, 0],    # Left-close
    [15, -20, 50, 0, 90, 0],     # Right-far
    [-15, -20, 50, 0, 90, 0],    # Left-far
]


async def main():
    print("Connecting to robot...")
    machine = await connect()
    print("Connected!")

    arm = Arm.from_robot(machine, "lite6")
    realsense = Camera.from_robot(machine, "realsense")

    # Get current position for safety check
    current_joints = await arm.get_joint_positions()
    print(f"Current joint positions: {[f'{v:.1f}' for v in current_joints.values]}")

    arm_positions = []  # Positions in arm frame (mm)
    camera_positions = []  # Positions in camera frame (mm)

    print(f"\nStarting calibration with {len(CALIBRATION_POSES)} poses...")
    print("Watch the webcam to monitor arm movement.\n")

    for i, joints in enumerate(CALIBRATION_POSES):
        print(f"Pose {i+1}/{len(CALIBRATION_POSES)}: joints={joints}")

        # Move to pose
        joint_pos = JointPositions(values=[float(j) for j in joints])
        await arm.move_to_joint_positions(positions=joint_pos)

        # Wait for arm to settle
        await asyncio.sleep(1.0)

        # Get arm's end-effector position
        end_pos = await arm.get_end_position()
        arm_point = np.array([end_pos.x, end_pos.y, end_pos.z])
        print(f"  Arm end-effector: x={end_pos.x:.1f}, y={end_pos.y:.1f}, z={end_pos.z:.1f} mm")

        # Get point cloud from RealSense
        pcd_bytes, mime = await realsense.get_point_cloud()
        points = parse_pcd_bytes(pcd_bytes)
        print(f"  Point cloud: {len(points)} valid points")

        # Find gripper in point cloud
        camera_point = find_gripper_position(points, end_pos)

        if camera_point is not None:
            print(f"  Camera detected: x={camera_point[0]:.1f}, y={camera_point[1]:.1f}, z={camera_point[2]:.1f} mm")
            arm_positions.append(arm_point)
            camera_positions.append(camera_point)
        else:
            print("  WARNING: Could not detect gripper in point cloud")

        print()

    # Return to neutral position
    print("Returning to neutral position...")
    neutral = JointPositions(values=[0.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    await arm.move_to_joint_positions(positions=neutral)

    await machine.close()

    # Solve for transform
    if len(arm_positions) < 3:
        print(f"\nERROR: Only {len(arm_positions)} valid correspondences. Need at least 3.")
        return

    print(f"\nSolving transform with {len(arm_positions)} correspondences...")

    arm_points = np.array(arm_positions)
    camera_points = np.array(camera_positions)

    R, t = solve_transform(arm_points, camera_points)
    axis, angle = rotation_matrix_to_axis_angle(R)

    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)

    print(f"\nTranslation (camera origin in arm frame):")
    print(f"  x: {t[0]:.1f} mm")
    print(f"  y: {t[1]:.1f} mm")
    print(f"  z: {t[2]:.1f} mm")

    print(f"\nRotation (axis-angle):")
    print(f"  axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    print(f"  angle: {angle:.1f} degrees")

    # Compute reprojection error
    errors = []
    for arm_pt, cam_pt in zip(arm_points, camera_points):
        projected = R @ cam_pt + t
        error = np.linalg.norm(projected - arm_pt)
        errors.append(error)

    print(f"\nReprojection error:")
    print(f"  Mean: {np.mean(errors):.2f} mm")
    print(f"  Max:  {np.max(errors):.2f} mm")

    # Output VIAM frame config
    print("\n" + "="*60)
    print("VIAM FRAME CONFIGURATION")
    print("="*60)
    print("""
Add this to your realsense camera's frame configuration:

"frame": {
    "parent": "world",
    "translation": {
        "x": %.1f,
        "y": %.1f,
        "z": %.1f
    },
    "orientation": {
        "type": "ov_degrees",
        "value": {
            "x": %.4f,
            "y": %.4f,
            "z": %.4f,
            "th": %.1f
        }
    }
}
""" % (t[0], t[1], t[2], axis[0], axis[1], axis[2], angle))

    # Save results to file
    results = {
        "translation_mm": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
        "rotation_axis": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
        "rotation_angle_degrees": float(angle),
        "reprojection_error_mm": {"mean": float(np.mean(errors)), "max": float(np.max(errors))},
        "num_correspondences": len(arm_positions)
    }

    with open("calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to calibration_results.json")


if __name__ == "__main__":
    asyncio.run(main())
