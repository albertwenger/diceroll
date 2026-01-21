import asyncio
import json
from pathlib import Path

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.camera import Camera


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


async def main():
    machine = await connect()
    print("Connected to robot")

    # Get components
    arm = Arm.from_robot(machine, "lite6")
    gripper = Gripper.from_robot(machine, "lite-gripper")
    realsense = Camera.from_robot(machine, "realsense")
    webcam = Camera.from_robot(machine, "webcam")

    # Print current arm position
    pos = await arm.get_end_position()
    print(f"Arm position: x={pos.x:.1f}, y={pos.y:.1f}, z={pos.z:.1f}")

    # Print joint positions
    joints = await arm.get_joint_positions()
    print(f"Joint positions: {[f'{v:.1f}' for v in joints.values]}")

    await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
