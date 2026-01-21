import asyncio
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.camera import Camera


# TODO: Fill in your Viam credentials from the CONNECT tab in app.viam.com
ROBOT_ADDRESS = "<YOUR-ROBOT-ADDRESS>"
API_KEY = "<YOUR-API-KEY>"
API_KEY_ID = "<YOUR-API-KEY-ID>"


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=API_KEY,
        api_key_id=API_KEY_ID
    )
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)


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
