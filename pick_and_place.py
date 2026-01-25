"""
Pick and place script for dice rolling.

Uses VIAM motion service for collision-aware path planning.
Workspace obstacles defined in workspace_config.py.
"""

import asyncio
import json
from pathlib import Path

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame

from workspace_config import (
    create_world_state, get_home_pose, is_pose_in_work_area,
    print_workspace_info, APPROACH_HEIGHT, BOWLS
)
from vision_detect import (
    move_to_scan_position, find_cube_position, get_bowl_pose,
    CUBE_DETECTORS
)


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


async def move_to_pose_safe(motion: MotionClient, arm: Arm,
                            target: Pose, world_state,
                            description: str = "target"):
    """
    Move arm to target pose using motion planning with collision avoidance.
    """
    print(f"\nMoving to {description}...")
    print(f"  Target: x={target.x:.1f}, y={target.y:.1f}, z={target.z:.1f}")

    # Create PoseInFrame for the destination
    destination = PoseInFrame(
        reference_frame="world",
        pose=target
    )

    try:
        # Use motion service for collision-aware planning
        success = await motion.move(
            component_name=arm.name,
            destination=destination,
            world_state=world_state
        )

        if success:
            print(f"  Move successful!")
            # Verify final position
            final_pos = await arm.get_end_position()
            print(f"  Final: x={final_pos.x:.1f}, y={final_pos.y:.1f}, z={final_pos.z:.1f}")
        else:
            print(f"  Move failed - no valid path found")

        return success

    except Exception as e:
        print(f"  Move error: {e}")
        return False


async def pick_object(motion: MotionClient, arm: Arm, gripper: Gripper,
                      pick_pose: Pose, world_state,
                      approach_height: float = APPROACH_HEIGHT):
    """
    Pick an object at the given pose.

    1. Move to approach position (above pick pose)
    2. Open gripper
    3. Move down to pick pose
    4. Close gripper
    5. Move back up
    """
    print("\n" + "=" * 50)
    print("PICK SEQUENCE")
    print("=" * 50)

    # Approach position (above the pick pose)
    approach_pose = Pose(
        x=pick_pose.x,
        y=pick_pose.y,
        z=pick_pose.z + approach_height,
        o_x=pick_pose.o_x, o_y=pick_pose.o_y,
        o_z=pick_pose.o_z, theta=pick_pose.theta
    )

    # 1. Move to approach position
    success = await move_to_pose_safe(motion, arm, approach_pose,
                                       world_state, "approach position")
    if not success:
        return False

    # 2. Open gripper
    print("\n  Opening gripper...")
    await gripper.open()
    await asyncio.sleep(0.5)

    # 3. Move down to pick position
    success = await move_to_pose_safe(motion, arm, pick_pose,
                                       world_state, "pick position")
    if not success:
        return False

    # 4. Close gripper (grab)
    print("\n  Closing gripper...")
    grabbed = await gripper.grab()
    await asyncio.sleep(0.5)

    if grabbed:
        print("  Object grabbed!")
    else:
        print("  Gripper closed (may not have grabbed object)")

    # 5. Move back up
    success = await move_to_pose_safe(motion, arm, approach_pose,
                                       world_state, "lift position")

    return success


async def place_object(motion: MotionClient, arm: Arm, gripper: Gripper,
                       place_pose: Pose, world_state,
                       approach_height: float = APPROACH_HEIGHT):
    """
    Place an object at the given pose.

    1. Move to approach position (above place pose)
    2. Move down to place pose
    3. Open gripper
    4. Move back up
    """
    print("\n" + "=" * 50)
    print("PLACE SEQUENCE")
    print("=" * 50)

    # Approach position
    approach_pose = Pose(
        x=place_pose.x,
        y=place_pose.y,
        z=place_pose.z + approach_height,
        o_x=place_pose.o_x, o_y=place_pose.o_y,
        o_z=place_pose.o_z, theta=place_pose.theta
    )

    # 1. Move to approach position
    success = await move_to_pose_safe(motion, arm, approach_pose,
                                       world_state, "approach position")
    if not success:
        return False

    # 2. Move down to place position
    success = await move_to_pose_safe(motion, arm, place_pose,
                                       world_state, "place position")
    if not success:
        return False

    # 3. Open gripper
    print("\n  Opening gripper...")
    await gripper.open()
    await asyncio.sleep(0.5)

    # 4. Move back up
    success = await move_to_pose_safe(motion, arm, approach_pose,
                                       world_state, "retract position")

    return success


async def move_to_home(motion: MotionClient, arm: Arm, world_state):
    """Move arm to a safe home position."""
    home_pose = get_home_pose()
    return await move_to_pose_safe(motion, arm, home_pose,
                                    world_state, "home position")


async def roll_cube(machine, motion, arm, gripper, world_state,
                    cube_color: str, bowl_side: str):
    """
    Pick up a colored cube and roll it into a bowl using vision detection.

    Args:
        cube_color: "green", "blue", "red", or "yellow"
        bowl_side: "left" or "right"
    """
    print(f"\n{'=' * 60}")
    print(f"ROLLING {cube_color.upper()} CUBE INTO {bowl_side.upper()} BOWL")
    print("=" * 60)

    # Move to scanning position to see the workspace
    await move_to_scan_position(machine)
    await asyncio.sleep(0.5)  # Let camera stabilize

    # Use vision to find the cube
    print(f"\nLooking for {cube_color} cube...")
    pick_pose = await find_cube_position(machine, cube_color)

    if pick_pose is None:
        print(f"  Could not find {cube_color} cube!")
        return False

    print(f"\nDetected {cube_color} cube at:")
    print(f"  x={pick_pose.x:.1f}, y={pick_pose.y:.1f}, z={pick_pose.z:.1f}")

    # Get bowl position (hardcoded since bowls don't move)
    place_pose = get_bowl_pose(bowl_side)
    print(f"\nPlace pose ({bowl_side} bowl):")
    print(f"  x={place_pose.x:.1f}, y={place_pose.y:.1f}, z={place_pose.z:.1f}")

    # Pick up the cube
    success = await pick_object(motion, arm, gripper, pick_pose, world_state)

    if success:
        # Drop into bowl
        await place_object(motion, arm, gripper, place_pose, world_state)
    else:
        print(f"\nFailed to pick up {cube_color} cube")

    return success


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pick and place dice")
    parser.add_argument("--cube", "-c", choices=list(CUBE_DETECTORS.keys()),
                        default="yellow", help="Cube color to pick")
    parser.add_argument("--bowl", "-b", choices=list(BOWLS.keys()),
                        default="right", help="Bowl to drop into")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Roll all cubes (alternating bowls)")
    args = parser.parse_args()

    print("=" * 60)
    print("DICE PICK AND PLACE")
    print("=" * 60)

    print("\nConnecting to robot...")
    machine = await connect()
    print("Connected!")

    # Get components
    arm = Arm.from_robot(machine, "lite6")
    gripper = Gripper.from_robot(machine, "lite-gripper")

    # Get motion service (built-in)
    motion = MotionClient.from_robot(machine, "builtin")

    # Create world state with obstacles
    print("\nWorkspace configuration:")
    print_workspace_info()
    world_state = create_world_state()

    # Get current position
    current_pos = await arm.get_end_position()
    print(f"\nCurrent arm position:")
    print(f"  x={current_pos.x:.1f}, y={current_pos.y:.1f}, z={current_pos.z:.1f}")

    try:
        # Move to home first
        print("\n" + "=" * 60)
        print("STEP 1: Move to home position")
        print("=" * 60)
        await move_to_home(motion, arm, world_state)

        if args.all:
            # Roll all cubes, alternating between bowls
            bowls = ["left", "right"]
            for i, cube_color in enumerate(CUBE_DETECTORS.keys()):
                bowl_side = bowls[i % 2]
                await roll_cube(machine, motion, arm, gripper, world_state,
                               cube_color, bowl_side)
                # Return to home between picks
                await move_to_home(motion, arm, world_state)
        else:
            # Roll a single cube
            await roll_cube(machine, motion, arm, gripper, world_state,
                           args.cube, args.bowl)

        # Return home
        print("\n" + "=" * 60)
        print("FINAL: Return to home")
        print("=" * 60)
        await move_to_home(motion, arm, world_state)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        await machine.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
