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
    # STEP 1: Small test movements from current position
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Test small movements from current position")
    print("=" * 60)
    print("Making small joint movements to verify arm responds correctly...")

    # Store starting position
    start_joints = current_joints.copy()

    # Test: move joint 1 (base) by +5 degrees
    print("\n--- Test: Rotate base +5° ---")
    test_joints = start_joints.copy()
    test_joints[0] = start_joints[0] + 5

    success = await safe_move_joints(arm, test_joints, webcam, output_dir, f"{timestamp}_test_base_plus5")
    if not success:
        print("  Base rotation test failed!")
    else:
        test_pos = await get_arm_position(arm)
        results["measurements"].append({
            "name": "base_plus5",
            "joints": test_joints,
            "pos": test_pos
        })

    # Return to start
    await safe_move_joints(arm, start_joints, webcam, output_dir, f"{timestamp}_return_start")

    # Test: move joint 2 (shoulder) by +5 degrees
    print("\n--- Test: Shoulder +5° ---")
    test_joints = start_joints.copy()
    test_joints[1] = start_joints[1] + 5

    success = await safe_move_joints(arm, test_joints, webcam, output_dir, f"{timestamp}_test_shoulder_plus5")
    if not success:
        print("  Shoulder test failed!")
    else:
        test_pos = await get_arm_position(arm)
        try:
            depth = await get_depth_image(realsense)
            test_depth = get_center_depth(depth)
        except:
            test_depth = 0
        results["measurements"].append({
            "name": "shoulder_plus5",
            "joints": test_joints,
            "pos": test_pos,
            "depth": test_depth
        })
        print(f"  Position after shoulder +5°: z={test_pos['z']:.1f} mm, depth={test_depth:.1f} mm")

    # Return to start
    await safe_move_joints(arm, start_joints, webcam, output_dir, f"{timestamp}_return_start2")

    # Test: move joint 2 (shoulder) by -5 degrees
    print("\n--- Test: Shoulder -5° ---")
    test_joints = start_joints.copy()
    test_joints[1] = start_joints[1] - 5

    success = await safe_move_joints(arm, test_joints, webcam, output_dir, f"{timestamp}_test_shoulder_minus5")
    if not success:
        print("  Shoulder test failed!")
    else:
        test_pos = await get_arm_position(arm)
        try:
            depth = await get_depth_image(realsense)
            test_depth = get_center_depth(depth)
        except:
            test_depth = 0
        results["measurements"].append({
            "name": "shoulder_minus5",
            "joints": test_joints,
            "pos": test_pos,
            "depth": test_depth
        })
        print(f"  Position after shoulder -5°: z={test_pos['z']:.1f} mm, depth={test_depth:.1f} mm")

    # Return to start
    await safe_move_joints(arm, start_joints, webcam, output_dir, f"{timestamp}_return_start3")

    # ============================================================
    # STEP 2: Analyze results
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Analyze calibration data")
    print("=" * 60)

    if len(results["measurements"]) >= 2:
        # Find shoulder measurements
        shoulder_plus = None
        shoulder_minus = None
        for m in results["measurements"]:
            if m["name"] == "shoulder_plus5":
                shoulder_plus = m
            elif m["name"] == "shoulder_minus5":
                shoulder_minus = m

        if shoulder_plus and shoulder_minus:
            z_diff = shoulder_plus["pos"]["z"] - shoulder_minus["pos"]["z"]
            print(f"\n  Shoulder +5° vs -5°:")
            print(f"    Z position change: {z_diff:.1f} mm")

            if "depth" in shoulder_plus and "depth" in shoulder_minus:
                depth_diff = shoulder_plus["depth"] - shoulder_minus["depth"]
                print(f"    Depth change: {depth_diff:.1f} mm")

                # If Z goes up and depth increases, camera points down
                if z_diff > 0 and depth_diff > 0:
                    print(f"    Camera Z axis opposes arm Z (camera points down)")
                elif z_diff > 0 and depth_diff < 0:
                    print(f"    Camera Z axis aligns with arm Z")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"""
Calibration images saved to: {output_dir}/

Based on the small test movements, you can now:
1. Review the saved images to verify arm movement directions
2. Use the depth measurements to compute camera offset
3. Update VIAM config with the frame parameters

For the RealSense frame config, set:
  "parent": "lite6"

The translation offset needs to be measured or computed from
larger movements once the safe operating range is confirmed.
""")

    # Return to starting position
    print("Returning to starting position...")
    await safe_move_joints(arm, start_joints, webcam, output_dir, f"{timestamp}_final")

    await machine.close()

    # Save results
    results_file = Path("calibration_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
