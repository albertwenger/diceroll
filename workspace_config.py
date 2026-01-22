"""
Workspace configuration for collision avoidance.

Defines the physical workspace boundaries and obstacles.
All measurements in millimeters relative to arm base frame.

Coordinate system:
  +X: Forward (away from arm base, toward back of desk)
  +Y: Left (from arm's perspective)
  +Z: Up

Adjust these values based on your actual workspace measurements.
"""

from viam.proto.common import (
    Pose, Vector3, GeometriesInFrame, Geometry,
    RectangularPrism, Sphere, Capsule, WorldState
)


# =============================================================================
# WORKSPACE DIMENSIONS
# =============================================================================

# Desk surface
DESK = {
    "center_x": 250,      # Center of desk relative to arm base
    "center_y": 0,        # Centered left-right
    "surface_z": -10,     # Desk surface level (slightly below arm base)
    "width_x": 600,       # Front-to-back dimension
    "width_y": 800,       # Left-to-right dimension
    "thickness": 30,      # Desk thickness (for collision box)
}

# Gray divider on the right side
RIGHT_DIVIDER = {
    "center_x": 300,      # Roughly centered front-to-back
    "center_y": -350,     # ~35cm to the right of arm
    "center_z": 150,      # Mid-height of divider
    "width_x": 400,       # Front-to-back span
    "width_y": 30,        # Thickness of divider
    "height_z": 300,      # Height of divider
}

# Gray divider at the back
BACK_DIVIDER = {
    "center_x": 450,      # ~45cm in front of arm
    "center_y": 0,        # Centered
    "center_z": 150,      # Mid-height
    "width_x": 30,        # Thickness
    "width_y": 600,       # Left-to-right span
    "height_z": 300,      # Height
}

# Equipment area behind the arm (RealSense mount, etc.)
EQUIPMENT_ZONE = {
    "center_x": -100,     # Behind arm base
    "center_y": 0,
    "center_z": 200,
    "width_x": 150,
    "width_y": 200,
    "height_z": 400,
}

# Safe working area boundaries
WORK_AREA = {
    "min_x": 50,          # Don't reach behind arm base
    "max_x": 400,         # Stay away from back divider
    "min_y": -300,        # Stay away from right divider
    "max_y": 300,         # Left boundary
    "min_z": 20,          # Stay above desk surface
    "max_z": 500,         # Upper limit
}

# Key positions
HOME_POSITION = {
    "x": 200,
    "y": 0,
    "z": 300,
    "gripper_down": True,  # Gripper pointing down
}

# Default approach height for pick/place operations
APPROACH_HEIGHT = 80  # mm above pick/place point


# =============================================================================
# GEOMETRY BUILDERS
# =============================================================================

def create_box_geometry(name: str, center_x: float, center_y: float, center_z: float,
                        width_x: float, width_y: float, height_z: float) -> Geometry:
    """Create a box geometry for collision avoidance."""
    return Geometry(
        center=Pose(
            x=center_x, y=center_y, z=center_z,
            o_x=0, o_y=0, o_z=1, theta=0
        ),
        box=RectangularPrism(
            dims_mm=Vector3(x=width_x, y=width_y, z=height_z)
        ),
        label=name
    )


def create_workspace_obstacles() -> GeometriesInFrame:
    """Create all workspace obstacles for collision avoidance."""
    geometries = []

    # Desk surface
    geometries.append(create_box_geometry(
        "desk_surface",
        DESK["center_x"], DESK["center_y"], DESK["surface_z"] - DESK["thickness"]/2,
        DESK["width_x"], DESK["width_y"], DESK["thickness"]
    ))

    # Right divider
    geometries.append(create_box_geometry(
        "right_divider",
        RIGHT_DIVIDER["center_x"], RIGHT_DIVIDER["center_y"], RIGHT_DIVIDER["center_z"],
        RIGHT_DIVIDER["width_x"], RIGHT_DIVIDER["width_y"], RIGHT_DIVIDER["height_z"]
    ))

    # Back divider
    geometries.append(create_box_geometry(
        "back_divider",
        BACK_DIVIDER["center_x"], BACK_DIVIDER["center_y"], BACK_DIVIDER["center_z"],
        BACK_DIVIDER["width_x"], BACK_DIVIDER["width_y"], BACK_DIVIDER["height_z"]
    ))

    # Equipment zone (optional - uncomment if needed)
    # geometries.append(create_box_geometry(
    #     "equipment_zone",
    #     EQUIPMENT_ZONE["center_x"], EQUIPMENT_ZONE["center_y"], EQUIPMENT_ZONE["center_z"],
    #     EQUIPMENT_ZONE["width_x"], EQUIPMENT_ZONE["width_y"], EQUIPMENT_ZONE["height_z"]
    # ))

    return GeometriesInFrame(
        reference_frame="world",
        geometries=geometries
    )


def create_world_state() -> WorldState:
    """Create WorldState with all obstacles."""
    return WorldState(obstacles=[create_workspace_obstacles()])


def get_home_pose() -> Pose:
    """Get the home position pose."""
    # Gripper pointing down: o_z=-1 or use orientation vector
    return Pose(
        x=HOME_POSITION["x"],
        y=HOME_POSITION["y"],
        z=HOME_POSITION["z"],
        o_x=0, o_y=0, o_z=-1, theta=0
    )


def is_pose_in_work_area(pose: Pose) -> bool:
    """Check if a pose is within the safe working area."""
    return (
        WORK_AREA["min_x"] <= pose.x <= WORK_AREA["max_x"] and
        WORK_AREA["min_y"] <= pose.y <= WORK_AREA["max_y"] and
        WORK_AREA["min_z"] <= pose.z <= WORK_AREA["max_z"]
    )


def print_workspace_info():
    """Print workspace configuration summary."""
    print("Workspace Configuration:")
    print(f"  Desk: {DESK['width_x']}x{DESK['width_y']}mm at z={DESK['surface_z']}mm")
    print(f"  Right divider: y={RIGHT_DIVIDER['center_y']}mm")
    print(f"  Back divider: x={BACK_DIVIDER['center_x']}mm")
    print(f"  Work area: x=[{WORK_AREA['min_x']}, {WORK_AREA['max_x']}], "
          f"y=[{WORK_AREA['min_y']}, {WORK_AREA['max_y']}], "
          f"z=[{WORK_AREA['min_z']}, {WORK_AREA['max_z']}]")


if __name__ == "__main__":
    print_workspace_info()
    print("\nObstacles created:")
    obstacles = create_workspace_obstacles()
    for geom in obstacles.geometries:
        print(f"  - {geom.label}: center=({geom.center.x}, {geom.center.y}, {geom.center.z})")
