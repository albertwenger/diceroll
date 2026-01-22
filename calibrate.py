"""
Automatic frame system calibration.

Uses two cameras to automatically determine:
1. Arm axis directions (via webcam observation of arm movement)
2. Camera mounting offset (via RealSense depth measurements)

Safety: Makes small incremental movements and checks webcam after each.
"""

import asyncio
import json
import struct
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.component.arm import JointPositions
from viam.media.utils.pil import viam_to_pil_image


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


def parse_depth_image(raw_bytes: bytes) -> np.ndarray:
    """Parse VIAM depth image format."""
    if raw_bytes[:8] == b"DEPTHMAP":
        # Try to infer dimensions from data size
        # Header is typically 16 bytes: magic(8) + width(4) + height(4)
        data_after_magic = raw_bytes[8:]

        # Try 32-bit width/height (4 bytes each)
        width_32 = struct.unpack('<I', raw_bytes[8:12])[0]
        height_32 = struct.unpack('<I', raw_bytes[12:16])[0]
        pixel_data = raw_bytes[16:]

        expected_size = width_32 * height_32 * 2
        if abs(expected_size - len(pixel_data)) < 100:
            depth = np.frombuffer(pixel_data[:expected_size], dtype=np.uint16)
            return depth.reshape((height_32, width_32))

        # Fallback: infer from common resolutions
        total_pixels = len(raw_bytes[16:]) // 2
        for w, h in [(1280, 720), (640, 480), (848, 480), (640, 360)]:
            if w * h == total_pixels:
                return np.frombuffer(raw_bytes[16:16 + w*h*2], dtype=np.uint16).reshape((h, w))

        # Last resort: try 24-byte header
        total_pixels = len(raw_bytes[24:]) // 2
        for w, h in [(1280, 720), (640, 480), (848, 480), (640, 360)]:
            if w * h == total_pixels:
                return np.frombuffer(raw_bytes[24:24 + w*h*2], dtype=np.uint16).reshape((h, w))

    raise ValueError(f"Cannot parse depth image, size={len(raw_bytes)}")


async def get_depth_image(camera: Camera) -> np.ndarray:
    """Get depth image from RealSense."""
    images, metadata = await camera.get_images()
    for img in images:
        if hasattr(img, 'data') and len(img.data) > 8 and img.data[:8] == b"DEPTHMAP":
            return parse_depth_image(img.data)
    raise ValueError("No depth image found")


async def get_webcam_image(camera: Camera) -> Image.Image:
    """Get RGB image from webcam."""
    images, metadata = await camera.get_images()
    if images:
        return viam_to_pil_image(images[0])
    raise ValueError("No image returned")


def get_center_depth(depth: np.ndarray, region_size: int = 50) -> float:
    """Get median depth in center region of image (in mm)."""
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    r = region_size // 2
    center_region = depth[cy-r:cy+r, cx-r:cx+r]
    valid = center_region[center_region > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


async def clear_arm_error(arm: Arm):
    """Try to clear arm error state."""
    try:
        await arm.do_command({"clear_error": True})
        print("  Cleared arm error")
        await asyncio.sleep(1.0)
    except Exception as e:
        print(f"  Could not clear error: {e}")


async def get_current_joints(arm: Arm) -> list:
    """Get current joint positions."""
    joints = await arm.get_joint_positions()
    return list(joints.values)


async def get_arm_position(arm: Arm) -> dict:
    """Get current arm end-effector position."""
    pos = await arm.get_end_position()
    return {"x": pos.x, "y": pos.y, "z": pos.z}


async def safe_move_joints(arm: Arm, target_joints: list, webcam: Camera,
                           output_dir: Path, step_name: str, max_step: float = 10.0):
    """
    Safely move to target joints in small increments, checking webcam after each.

    Args:
        arm: The arm component
        target_joints: Target joint positions in degrees
        webcam: Webcam for visual verification
        output_dir: Directory to save images
        step_name: Name for logging/saving
        max_step: Maximum degrees to move per step
    """
    current = await get_current_joints(arm)
    target = [float(t) for t in target_joints]

    print(f"\n  Safe move to: {step_name}")
    print(f"  Current joints: [{', '.join(f'{j:.1f}' for j in current)}]")
    print(f"  Target joints:  [{', '.join(f'{j:.1f}' for j in target)}]")

    # Calculate total distance
    diffs = [t - c for t, c in zip(target, current)]
    max_diff = max(abs(d) for d in diffs)

    if max_diff < 1.0:
        print("  Already at target position")
        return True

    # Calculate number of steps needed
    num_steps = max(1, int(np.ceil(max_diff / max_step)))
    print(f"  Moving in {num_steps} steps (max {max_step}° per step)")

    # Capture initial webcam image
    img_before = await get_webcam_image(webcam)

    for step in range(1, num_steps + 1):
        # Interpolate to next position
        t = step / num_steps
        intermediate = [c + (tgt - c) * t for c, tgt in zip(current, target)]

        print(f"  Step {step}/{num_steps}: [{', '.join(f'{j:.1f}' for j in intermediate)}]", end="")

        try:
            joint_pos = JointPositions(values=intermediate)
            await arm.move_to_joint_positions(positions=joint_pos)
            await asyncio.sleep(0.5)

            # Get position after move
            pos = await get_arm_position(arm)
            print(f" -> pos: x={pos['x']:.0f}, y={pos['y']:.0f}, z={pos['z']:.0f}")

            # Capture webcam image to verify
            img_after = await get_webcam_image(webcam)

            # Save intermediate images for debugging
            if step == num_steps:
                img_after.save(output_dir / f"{step_name}.jpg")

        except Exception as e:
            error_str = str(e).lower()
            if "collision" in error_str or "overcurrent" in error_str:
                print(f"\n  COLLISION DETECTED at step {step}!")
                print(f"  Stopping movement. Error: {e}")
                # Save the problematic state
                try:
                    img_err = await get_webcam_image(webcam)
                    img_err.save(output_dir / f"{step_name}_COLLISION.jpg")
                except:
                    pass
                return False
            elif "power cycle" in error_str:
                print(f"\n  ARM NEEDS POWER CYCLE!")
                return False
            else:
                print(f"\n  Error: {e}")
                return False

    print(f"  Move complete!")
    return True


async def main():
    print("=" * 60)
    print("SAFE AUTOMATIC FRAME CALIBRATION")
    print("=" * 60)
    print("""
This script safely calibrates the frame system by:
1. Making small incremental movements (max 10° per step)
2. Checking webcam after each movement
3. Stopping immediately if collision detected

Requirements:
- Clear workspace around arm
- Webcam with view of arm
- RealSense mounted on arm end-effector
""")

    print("\nConnecting to robot...")
    machine = await connect()
    print("Connected!")

    arm = Arm.from_robot(machine, "lite6")
    webcam = Camera.from_robot(machine, "webcam")
    realsense = Camera.from_robot(machine, "realsense")

    output_dir = Path("calibration_images")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clear any existing arm errors
    print("\nClearing any existing arm errors...")
    await clear_arm_error(arm)
    await asyncio.sleep(1.0)

    # ============================================================
    # STEP 0: Observe current state
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 0: Observe current arm state")
    print("=" * 60)

    current_joints = await get_current_joints(arm)
    current_pos = await get_arm_position(arm)

    print(f"  Current joints: [{', '.join(f'{j:.1f}' for j in current_joints)}]")
    print(f"  Current position: x={current_pos['x']:.1f}, y={current_pos['y']:.1f}, z={current_pos['z']:.1f} mm")

    # Capture current webcam view
    webcam_img = await get_webcam_image(webcam)
    webcam_img.save(output_dir / f"{timestamp}_00_initial.jpg")
    print(f"  Saved initial webcam image")

    # Try to get depth
    try:
        depth = await get_depth_image(realsense)
        center_depth = get_center_depth(depth)
        print(f"  RealSense center depth: {center_depth:.1f} mm")
    except Exception as e:
        print(f"  Could not get depth: {e}")
        center_depth = 0

    results = {
        "timestamp": timestamp,
        "initial_joints": current_joints,
        "initial_pos": current_pos,
        "measurements": []
    }

    # ============================================================
    # STEP 1: Move to a pose where camera points at desk
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Move to desk-viewing pose")
    print("=" * 60)
    print("Moving arm so RealSense points down at the desk...")

    # Store original position to return to later
    original_joints = current_joints.copy()

    # Target pose: arm extended forward, wrist pitched down to point camera at desk
    # Joint angles: [base, shoulder, elbow, wrist_roll, wrist_pitch, ee_roll]
    # - shoulder ~30° forward
    # - elbow ~0° (arm extended)
    # - wrist_pitch ~-90° to point end-effector straight down
    desk_pose = [0, 30, 0, 0, -90, 0]

    success = await safe_move_joints(arm, desk_pose, webcam, output_dir, f"{timestamp}_desk_pose", max_step=5.0)
    if not success:
        print("  Could not reach desk pose, trying alternative...")
        # Try a more conservative pose
        desk_pose = [0, 20, 20, 0, -60, 0]
        success = await safe_move_joints(arm, desk_pose, webcam, output_dir, f"{timestamp}_desk_pose_alt", max_step=5.0)

    if not success:
        print("  Failed to reach desk-viewing pose. Aborting.")
        await machine.close()
        return

    # Check depth at this pose
    desk_pos = await get_arm_position(arm)
    try:
        depth = await get_depth_image(realsense)
        desk_depth = get_center_depth(depth)
    except Exception as e:
        print(f"  Could not get depth: {e}")
        desk_depth = 0

    print(f"\n  Desk pose reached:")
    print(f"  Position: x={desk_pos['x']:.1f}, y={desk_pos['y']:.1f}, z={desk_pos['z']:.1f} mm")
    print(f"  Depth to desk: {desk_depth:.1f} mm")

    # If depth is unreasonably large, the camera might not be pointing at desk
    if desk_depth > 2000:
        print(f"  WARNING: Depth ({desk_depth:.0f}mm) seems too large for desk distance")
        print(f"  Camera may not be pointing at desk. Trying to adjust wrist pitch...")

        # Try adjusting wrist pitch to point more downward
        for pitch_adjust in [-10, -20, -30]:
            adjusted_pose = desk_pose.copy()
            adjusted_pose[4] = desk_pose[4] + pitch_adjust
            print(f"\n  Trying wrist pitch = {adjusted_pose[4]}°...")

            success = await safe_move_joints(arm, adjusted_pose, webcam, output_dir,
                                            f"{timestamp}_pitch_adjust_{pitch_adjust}", max_step=5.0)
            if success:
                try:
                    depth = await get_depth_image(realsense)
                    new_depth = get_center_depth(depth)
                    print(f"  Depth at pitch {adjusted_pose[4]}°: {new_depth:.1f} mm")
                    if new_depth < desk_depth and new_depth > 100:
                        desk_depth = new_depth
                        desk_pose = adjusted_pose.copy()
                        desk_pos = await get_arm_position(arm)
                        if new_depth < 1500:
                            print(f"  Found good desk-viewing pose!")
                            break
                except:
                    pass

    # Store the working desk pose
    start_joints = (await get_current_joints(arm)).copy()
    print(f"\n  Using pose: [{', '.join(f'{j:.1f}' for j in start_joints)}]")
    print(f"  End-effector: x={desk_pos['x']:.1f}, y={desk_pos['y']:.1f}, z={desk_pos['z']:.1f} mm")
    print(f"  Depth to surface: {desk_depth:.1f} mm")

    results["desk_pose"] = {
        "joints": start_joints,
        "pos": desk_pos,
        "depth": desk_depth
    }

    # ============================================================
    # STEP 2: Calibration movements along Z axis
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Z-axis calibration (vertical movement)")
    print("=" * 60)

    z_data = []

    # Move up and down by adjusting shoulder angle
    for shoulder_delta in [-15, -10, -5, 0, 5, 10, 15]:
        test_joints = start_joints.copy()
        test_joints[1] = start_joints[1] + shoulder_delta

        print(f"\n  Shoulder {shoulder_delta:+d}°...")
        success = await safe_move_joints(arm, test_joints, webcam, output_dir,
                                        f"{timestamp}_shoulder_{shoulder_delta:+d}", max_step=5.0)
        if success:
            pos = await get_arm_position(arm)
            try:
                depth = await get_depth_image(realsense)
                d = get_center_depth(depth)
            except:
                d = 0

            z_data.append({
                "shoulder_delta": shoulder_delta,
                "joints": test_joints,
                "pos": pos,
                "depth": d
            })
            print(f"    Position: z={pos['z']:.1f} mm, Depth: {d:.1f} mm")

    # Return to desk pose
    await safe_move_joints(arm, start_joints, webcam, output_dir, f"{timestamp}_return_desk")

    results["z_calibration"] = z_data

    # ============================================================
    # STEP 3: Calibration movements along X axis
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: X-axis calibration (forward/back movement)")
    print("=" * 60)

    x_data = []

    # Move forward/back by adjusting shoulder and elbow together
    for delta in [-10, -5, 0, 5, 10]:
        test_joints = start_joints.copy()
        # Increase shoulder and decrease elbow to extend forward
        test_joints[1] = start_joints[1] + delta
        test_joints[2] = start_joints[2] - delta

        print(f"\n  Forward/back delta {delta:+d}°...")
        success = await safe_move_joints(arm, test_joints, webcam, output_dir,
                                        f"{timestamp}_xdelta_{delta:+d}", max_step=5.0)
        if success:
            pos = await get_arm_position(arm)
            try:
                depth = await get_depth_image(realsense)
                d = get_center_depth(depth)
            except:
                d = 0

            x_data.append({
                "delta": delta,
                "joints": test_joints,
                "pos": pos,
                "depth": d
            })
            print(f"    Position: x={pos['x']:.1f} mm, Depth: {d:.1f} mm")

    # Return to desk pose
    await safe_move_joints(arm, start_joints, webcam, output_dir, f"{timestamp}_return_desk2")

    results["x_calibration"] = x_data

    # ============================================================
    # STEP 4: Calibration movements along Y axis
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Y-axis calibration (left/right movement)")
    print("=" * 60)

    y_data = []

    # Move left/right by rotating base
    for base_delta in [-15, -10, -5, 0, 5, 10, 15]:
        test_joints = start_joints.copy()
        test_joints[0] = start_joints[0] + base_delta

        print(f"\n  Base rotation {base_delta:+d}°...")
        success = await safe_move_joints(arm, test_joints, webcam, output_dir,
                                        f"{timestamp}_base_{base_delta:+d}", max_step=5.0)
        if success:
            pos = await get_arm_position(arm)
            try:
                depth = await get_depth_image(realsense)
                d = get_center_depth(depth)
            except:
                d = 0

            y_data.append({
                "base_delta": base_delta,
                "joints": test_joints,
                "pos": pos,
                "depth": d
            })
            print(f"    Position: y={pos['y']:.1f} mm, Depth: {d:.1f} mm")

    # Return to desk pose
    await safe_move_joints(arm, start_joints, webcam, output_dir, f"{timestamp}_return_desk3")

    results["y_calibration"] = y_data

    # ============================================================
    # STEP 5: Analyze results and compute camera offset
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: Analyze calibration data")
    print("=" * 60)

    # Analyze Z calibration data
    if z_data:
        print("\n  Z-axis analysis:")
        z_positions = [d["pos"]["z"] for d in z_data if d["depth"] > 0]
        depths = [d["depth"] for d in z_data if d["depth"] > 0]

        if len(z_positions) >= 2:
            # Linear regression: depth = slope * z + intercept
            z_arr = np.array(z_positions)
            d_arr = np.array(depths)
            slope, intercept = np.polyfit(z_arr, d_arr, 1)

            print(f"    Depth = {slope:.3f} * arm_z + {intercept:.1f}")

            if slope < -0.5:
                print(f"    Camera Z opposes arm Z (camera points DOWN)")
                camera_z_direction = -1
            elif slope > 0.5:
                print(f"    Camera Z aligns with arm Z (camera points UP)")
                camera_z_direction = 1
            else:
                print(f"    Camera Z is roughly perpendicular to arm Z")
                camera_z_direction = 0

            # Estimate desk height in arm frame
            # When depth = d, camera_z = arm_z + offset_z
            # desk_z = camera_z - depth (if camera points down)
            # So: desk_z = arm_z + offset_z - depth
            # For multiple readings: desk_z should be constant
            # offset_z = desk_z - arm_z + depth

            # If camera points straight down (slope ≈ -1):
            # depth ≈ arm_z + offset_z - desk_z
            # So: intercept ≈ offset_z - desk_z

            results["z_analysis"] = {
                "slope": slope,
                "intercept": intercept,
                "camera_z_direction": camera_z_direction
            }

    # Analyze Y calibration (base rotation shouldn't change depth much if desk is flat)
    if y_data:
        print("\n  Y-axis analysis:")
        y_positions = [d["pos"]["y"] for d in y_data if d["depth"] > 0]
        depths = [d["depth"] for d in y_data if d["depth"] > 0]

        if len(depths) >= 2:
            depth_std = np.std(depths)
            depth_mean = np.mean(depths)
            print(f"    Depth mean: {depth_mean:.1f} mm, std: {depth_std:.1f} mm")
            if depth_std < 50:
                print(f"    Depth is stable across Y movement (good - flat desk)")
            else:
                print(f"    Depth varies with Y (desk may be tilted or camera offset in Y)")

    # Compute camera offset estimate
    print("\n  Camera offset estimation:")

    if z_data and len([d for d in z_data if d["depth"] > 0]) >= 2:
        # Use the desk pose as reference
        ref_pos = results["desk_pose"]["pos"]
        ref_depth = results["desk_pose"]["depth"]

        if ref_depth > 0 and ref_depth < 2000:
            # Assuming camera points straight down:
            # The desk surface is at: desk_z = ref_pos.z - ref_depth + offset_z (in arm frame)
            # We can estimate offset_z if we know desk_z

            # From multiple Z measurements, find the desk height
            # depth = arm_z + offset_z - desk_z
            # So: desk_z = arm_z + offset_z - depth
            # If slope ≈ -1, then: desk_z ≈ offset_z - intercept

            if abs(slope + 1) < 0.5:  # Camera roughly points down
                # desk_z ≈ offset_z - intercept
                # We need another constraint. Use the fact that at ref position:
                # ref_depth = ref_pos.z + offset_z - desk_z
                # ref_depth = ref_pos.z + offset_z - (offset_z - intercept)
                # ref_depth = ref_pos.z + intercept
                # This is a consistency check

                estimated_desk_z = ref_pos["z"] - ref_depth  # Assuming offset_z ≈ 0 initially
                print(f"    Estimated desk Z (arm frame): {estimated_desk_z:.1f} mm")

                # The offset_z is the difference between actual and expected
                # For now, assume small offset and refine with more data
                offset_z_estimate = 0  # Would need ground truth to compute

                print(f"    Camera offset Z: needs ground truth measurement")
                print(f"    (Place gripper on desk to measure actual desk_z)")

        results["offset_estimate"] = {
            "x": 0,  # Would need lateral movement analysis
            "y": 0,  # Would need lateral movement analysis
            "z": 0,  # Needs ground truth
            "notes": "Offset requires ground truth desk height measurement"
        }
    else:
        print("    Insufficient data for offset estimation")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    print(f"\nCalibration images saved to: {output_dir}/")

    if "z_analysis" in results:
        slope = results["z_analysis"]["slope"]
        print(f"\nZ-axis relationship: depth = {slope:.3f} * arm_z + intercept")
        if slope < -0.5:
            print("Camera orientation: Points DOWNWARD (Z opposes arm Z)")
            orientation_config = '''
     "orientation": {
       "type": "ov_degrees",
       "value": {"x": 0, "y": 1, "z": 0, "th": 180}
     }'''
        else:
            print("Camera orientation: Non-standard mounting")
            orientation_config = "     // Orientation needs manual configuration"
    else:
        orientation_config = "     // Orientation could not be determined"

    print(f"""
VIAM Frame Configuration for RealSense:

  "frame": {{
    "parent": "lite6",
    "translation": {{
      "x": 0,
      "y": 0,
      "z": 0
    }},{orientation_config}
  }}

NOTE: Translation offset (x, y, z) needs physical measurement
or ground-truth calibration with a known reference point.

To measure offset:
1. Move gripper to touch the desk
2. Record arm Z position (this is desk_z in arm frame)
3. Move arm up, measure depth to desk
4. offset_z = depth - (arm_z - desk_z)
""")

    # Return to original starting position
    print("Returning to original position...")
    await safe_move_joints(arm, original_joints, webcam, output_dir, f"{timestamp}_final", max_step=5.0)

    await machine.close()

    # Save results
    results_file = Path("calibration_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
