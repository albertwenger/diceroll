"""
Workspace configuration for collision avoidance.

Setup: Green tray with colored cubes (dice) and black bowls
- Arm is positioned beside the tray
- Cubes are in the center of the tray on a white square
- Two black bowls on left and right sides of tray

All measurements in millimeters relative to arm base frame.

Coordinate system:
  +X: Forward (away from arm base, toward the tray)
  +Y: Left (from arm's perspective)
  +Z: Up

Adjust these values based on your actual workspace measurements.
"""

from viam.proto.common import (
    Pose, Vector3, GeometriesInFrame, Geometry,
    RectangularPrism, Sphere, Capsule, WorldState
)


# =============================================================================
# WORKSPACE DIMENSIONS - GREEN TRAY SETUP
# =============================================================================

# The green tray/bin that holds the workspace
# Estimated dimensions from webcam view
TRAY = {
    "center_x": 250,      # Center of tray relative to arm base
    "center_y": 50,       # Slightly to the left (tray is in front-left of arm)
    "surface_z": 0,       # Tray bottom surface level
    "width_x": 400,       # Front-to-back dimension (~40cm)
    "width_y": 600,       # Left-to-right dimension (~60cm)
    "wall_height": 80,    # Height of tray walls (~8cm)
    "wall_thickness": 10, # Thickness of tray walls
}

# Table/surface the tray sits on
TABLE = {
    "surface_z": -50,     # Table is below the tray
    "thickness": 30,
}

# =============================================================================
# OBJECT POSITIONS (approximate, will need vision for precise location)
# =============================================================================

# Colored cubes (dice) - in center of tray on white square
# The cubes appear to be ~25-30mm each
CUBE_SIZE = 25  # mm, approximate cube dimension

# Approximate cube positions (center of tray)
# From RealSense view: green(top-left), blue(top-right), red(bottom-left), yellow(bottom-right)
CUBES = {
    "green": {"x": 230, "y": 70, "z": CUBE_SIZE/2},
    "blue": {"x": 230, "y": 30, "z": CUBE_SIZE/2},
    "red": {"x": 270, "y": 70, "z": CUBE_SIZE/2},
    "yellow": {"x": 270, "y": 30, "z": CUBE_SIZE/2},
}

# Black bowls - on left and right sides of tray
# Bowls are approximately 100mm diameter, 40mm deep
BOWL_DIAMETER = 100
BOWL_DEPTH = 40

BOWLS = {
    "left": {"x": 250, "y": 200, "z": BOWL_DEPTH/2},   # Left bowl
    "right": {"x": 250, "y": -100, "z": BOWL_DEPTH/2}, # Right bowl
}

# =============================================================================
# SAFE WORKING AREA
# =============================================================================

WORK_AREA = {
    "min_x": 100,         # Don't reach too close to arm base
    "max_x": 400,         # Don't go past tray
    "min_y": -150,        # Right side limit (past right bowl)
    "max_y": 250,         # Left side limit (past left bowl)
    "min_z": 10,          # Stay above tray surface
    "max_z": 400,         # Upper limit
}

# =============================================================================
# KEY POSITIONS
# =============================================================================

HOME_POSITION = {
    "x": 200,
    "y": 50,
    "z": 250,
    "gripper_down": True,
}

# Height above pick point to approach from
APPROACH_HEIGHT = 60  # mm

# Height to lift object after picking
LIFT_HEIGHT = 100  # mm

# Height above bowl to release cube (for rolling/dropping)
DROP_HEIGHT = 80  # mm above bowl rim


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

    # Table surface (below the tray)
    geometries.append(create_box_geometry(
        "table_surface",
        TRAY["center_x"], TRAY["center_y"], TABLE["surface_z"] - TABLE["thickness"]/2,
        600, 800, TABLE["thickness"]
    ))

    # Note: Tray bottom obstacle removed - it was causing collision issues
    # with the arm's wrist link during picks. The table_surface obstacle
    # provides sufficient protection against going too low.

    # Tray walls (4 sides)
    wall_h = TRAY["wall_height"]
    wall_t = TRAY["wall_thickness"]
    cx, cy = TRAY["center_x"], TRAY["center_y"]
    wx, wy = TRAY["width_x"], TRAY["width_y"]

    # Front wall (closer to arm)
    geometries.append(create_box_geometry(
        "tray_wall_front",
        cx - wx/2 + wall_t/2, cy, wall_h/2,
        wall_t, wy, wall_h
    ))

    # Back wall (far from arm)
    geometries.append(create_box_geometry(
        "tray_wall_back",
        cx + wx/2 - wall_t/2, cy, wall_h/2,
        wall_t, wy, wall_h
    ))

    # Left wall
    geometries.append(create_box_geometry(
        "tray_wall_left",
        cx, cy + wy/2 - wall_t/2, wall_h/2,
        wx, wall_t, wall_h
    ))

    # Right wall
    geometries.append(create_box_geometry(
        "tray_wall_right",
        cx, cy - wy/2 + wall_t/2, wall_h/2,
        wx, wall_t, wall_h
    ))

    return GeometriesInFrame(
        reference_frame="world",
        geometries=geometries
    )


def create_world_state() -> WorldState:
    """Create WorldState with all obstacles."""
    return WorldState(obstacles=[create_workspace_obstacles()])


def get_home_pose() -> Pose:
    """Get the home position pose."""
    return Pose(
        x=HOME_POSITION["x"],
        y=HOME_POSITION["y"],
        z=HOME_POSITION["z"],
        o_x=0, o_y=0, o_z=-1, theta=0  # Gripper pointing down
    )


def get_cube_pose(color: str) -> Pose:
    """Get the pose for picking up a specific colored cube."""
    if color not in CUBES:
        raise ValueError(f"Unknown cube color: {color}. Available: {list(CUBES.keys())}")
    cube = CUBES[color]
    return Pose(
        x=cube["x"],
        y=cube["y"],
        z=cube["z"] + 5,  # Slightly above center for grip
        o_x=0, o_y=0, o_z=-1, theta=0
    )


def get_bowl_pose(side: str) -> Pose:
    """Get the pose for placing into a bowl."""
    if side not in BOWLS:
        raise ValueError(f"Unknown bowl: {side}. Available: {list(BOWLS.keys())}")
    bowl = BOWLS[side]
    return Pose(
        x=bowl["x"],
        y=bowl["y"],
        z=bowl["z"] + DROP_HEIGHT,  # Above the bowl to drop
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
    print("Workspace Configuration (Green Tray Setup):")
    print(f"  Tray: {TRAY['width_x']}x{TRAY['width_y']}mm at center ({TRAY['center_x']}, {TRAY['center_y']})")
    print(f"  Tray walls: {TRAY['wall_height']}mm high")
    print(f"  Cubes: {list(CUBES.keys())}")
    print(f"  Bowls: {list(BOWLS.keys())}")
    print(f"  Work area: x=[{WORK_AREA['min_x']}, {WORK_AREA['max_x']}], "
          f"y=[{WORK_AREA['min_y']}, {WORK_AREA['max_y']}], "
          f"z=[{WORK_AREA['min_z']}, {WORK_AREA['max_z']}]")


if __name__ == "__main__":
    print_workspace_info()
    print("\nCube positions:")
    for color, pos in CUBES.items():
        print(f"  {color}: ({pos['x']}, {pos['y']}, {pos['z']})")
    print("\nBowl positions:")
    for side, pos in BOWLS.items():
        print(f"  {side}: ({pos['x']}, {pos['y']}, {pos['z']})")
    print("\nObstacles created:")
    obstacles = create_workspace_obstacles()
    for geom in obstacles.geometries:
        print(f"  - {geom.label}")
