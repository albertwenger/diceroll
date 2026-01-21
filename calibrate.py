"""
Camera-to-arm calibration script.

Calibrates the RealSense camera position relative to the arm by:
1. Fitting a plane to the desk surface (gives camera height and tilt)
2. Detecting the arm base (gives x,y offset)

No arm movement required - just a single depth capture.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
from viam.robot.client import RobotClient
from viam.components.camera import Camera
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


def depth_to_points(depth_img: np.ndarray,
                    fx: float, fy: float,
                    cx: float, cy: float) -> np.ndarray:
    """
    Convert depth image to 3D point cloud.

    Returns Nx3 array of [x, y, z] points in camera frame (mm).
    Also returns the pixel coordinates as Nx2 array.
    """
    h, w = depth_img.shape[:2]

    # Create pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    # Get depth values
    z = depth_img.astype(np.float32)

    # Filter invalid depths - RealSense depth is typically in mm
    # Expand range to handle various setups (10cm to 10m)
    valid = (z > 100) & (z < 10000)

    # Convert to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack and filter
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    pixels = np.stack([u[valid], v[valid]], axis=-1)

    return points, pixels


def fit_plane_ransac(points: np.ndarray,
                     n_iterations: int = 1000,
                     distance_threshold: float = 10.0,
                     horizontal_only: bool = False) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Fit a plane to points using RANSAC.

    Args:
        points: Nx3 array of 3D points
        n_iterations: Number of RANSAC iterations
        distance_threshold: Inlier distance threshold in mm
        horizontal_only: If True, only accept planes that are roughly horizontal
                        (normal has large z component in camera frame)

    Returns:
        normal: Plane normal vector (3,)
        d: Plane offset (plane equation: normal · p + d = 0)
        inliers: Boolean mask of inlier points
    """
    best_inliers = None
    best_normal = None
    best_d = None
    best_count = 0

    n_points = len(points)

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # If horizontal_only, reject planes that aren't roughly horizontal
        # A horizontal desk viewed from above should have normal with large |z| component
        if horizontal_only:
            # Camera looks forward/down, desk normal should have significant z component
            # Allow for tilted camera - normal z should be > 0.5 (within ~60 deg of vertical)
            if abs(normal[2]) < 0.5:
                continue

        # Plane offset
        d = -np.dot(normal, p1)

        # Count inliers
        distances = np.abs(np.dot(points, normal) + d)
        inliers = distances < distance_threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if best_normal is None:
        return np.array([0, 0, -1]), 0.0, np.zeros(len(points), dtype=bool)

    # Ensure normal points "up" (toward camera, so negative z in camera frame)
    if best_normal[2] > 0:
        best_normal = -best_normal
        best_d = -best_d

    return best_normal, best_d, best_inliers


def find_arm_base(depth_img: np.ndarray,
                  points: np.ndarray,
                  pixels: np.ndarray,
                  desk_normal: np.ndarray,
                  desk_d: float,
                  desk_inliers: np.ndarray,
                  fx: float, fy: float,
                  cx: float, cy: float) -> tuple[np.ndarray | None, dict]:
    """
    Find the arm base in the depth image.

    Strategy: The arm base is a vertical structure rising from the desk.
    Look for points that are:
    1. NOT on the desk plane (elevated above it / closer to camera)
    2. Form a cluster
    3. Project to desk plane to find base position

    Returns:
        base_position: [x, y, z] in camera frame, or None if not found
        debug: Debug information
    """
    debug = {}

    # Distance from each point to the desk plane
    # Negative = point is "above" desk (closer to camera)
    distances_to_desk = np.dot(points, desk_normal) + desk_d

    debug["dist_to_desk_range"] = (float(distances_to_desk.min()), float(distances_to_desk.max()))

    # Find points above the desk (closer to camera = negative distance)
    # The arm should be significantly above desk - at least 50mm
    above_desk = distances_to_desk < -50

    elevated_points = points[above_desk]
    elevated_pixels = pixels[above_desk]
    elevated_distances = distances_to_desk[above_desk]

    debug["elevated_points"] = len(elevated_points)

    if len(elevated_points) < 50:
        return None, {"error": f"Not enough elevated points: {len(elevated_points)}"}

    # The arm is a distinct structure - find clusters of elevated points
    # The arm base should be where elevated points meet the desk
    # Look for points that are 50-500mm above desk (the lower part of the arm)
    base_region = (elevated_distances < -50) & (elevated_distances > -500)
    base_points = elevated_points[base_region]
    base_pixels = elevated_pixels[base_region]

    debug["base_region_points"] = len(base_points)

    if len(base_points) < 20:
        # Try wider range
        base_region = (elevated_distances < -30) & (elevated_distances > -1000)
        base_points = elevated_points[base_region]
        base_pixels = elevated_pixels[base_region]
        debug["base_region_points_expanded"] = len(base_points)

    if len(base_points) < 20:
        # Just use all elevated points and find the ones closest to desk
        sort_idx = np.argsort(elevated_distances)[-100:]  # Closest to desk
        base_points = elevated_points[sort_idx]
        base_pixels = elevated_pixels[sort_idx]
        debug["using_closest_to_desk"] = len(base_points)

    if len(base_points) < 10:
        return None, {"error": f"Not enough base region points after expansion"}

    # Find the centroid - use median for robustness
    base_centroid = np.median(base_points, axis=0)
    base_pixel = np.median(base_pixels, axis=0)

    debug["base_centroid_pixel"] = (int(base_pixel[0]), int(base_pixel[1]))
    debug["base_centroid_3d"] = [float(x) for x in base_centroid]

    # Project to desk plane to get the arm base origin
    dist_to_desk = np.dot(base_centroid, desk_normal) + desk_d
    base_on_desk = base_centroid - dist_to_desk * desk_normal

    debug["base_on_desk_3d"] = [float(x) for x in base_on_desk]
    debug["height_above_desk"] = float(-dist_to_desk)

    return base_on_desk, debug


def compute_camera_transform(desk_normal: np.ndarray,
                             desk_d: float,
                             arm_base_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the camera position in arm frame coordinates.

    Assumptions:
    - The desk is the z=0 plane in arm frame
    - The arm base is at the origin (0, 0, 0) in arm frame
    - The desk normal in arm frame is [0, 0, 1] (pointing up)
    - Camera +z points into the scene (away from camera)

    Args:
        desk_normal: Desk plane normal in camera frame (points toward camera)
        desk_d: Desk plane offset in camera frame (negative of distance to desk)
        arm_base_camera: Arm base position in camera frame [x, y, z]

    Returns:
        R: Rotation matrix (camera frame to arm frame)
        t: Translation vector (camera origin in arm frame)
    """
    # Camera coordinate system:
    # - Origin at camera
    # - +z points into scene (toward desk)
    # - +x typically points right, +y points down

    # Arm coordinate system:
    # - Origin at arm base
    # - +z points up
    # - +x points forward, +y points left (typical robot convention)

    # desk_normal in camera frame points toward camera (negative z if looking down)
    # In arm frame, desk normal is [0, 0, 1]

    # The camera sees the arm base at position arm_base_camera
    # We need to find where the camera is in arm frame

    # Step 1: Camera height above desk
    # Distance from camera to desk plane is |desk_d|
    camera_height = abs(desk_d)

    # Step 2: Camera's x,y position relative to arm base
    # arm_base_camera gives us the arm base position in camera frame
    # We need to transform this to arm frame

    # For a camera looking straight down (desk_normal ≈ [0, 0, -1]):
    # - Camera's +x corresponds to some direction in arm's x-y plane
    # - Camera's +y corresponds to another direction in arm's x-y plane
    # - Camera's +z corresponds to arm's -z (camera looks down)

    # Build rotation matrix from camera frame to arm frame
    # arm_z (up) = -desk_normal (flip because desk_normal points at camera)
    arm_z_in_cam = -desk_normal

    # For arm_x and arm_y, we need to determine the camera's yaw
    # Without additional information, assume camera x aligns with arm y
    # (camera is behind the arm, looking at it from the back-left based on webcam)

    # Create orthonormal basis
    if abs(arm_z_in_cam[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])

    arm_y_in_cam = np.cross(arm_z_in_cam, temp)
    arm_y_in_cam = arm_y_in_cam / np.linalg.norm(arm_y_in_cam)

    arm_x_in_cam = np.cross(arm_y_in_cam, arm_z_in_cam)
    arm_x_in_cam = arm_x_in_cam / np.linalg.norm(arm_x_in_cam)

    # R transforms from camera frame to arm frame
    # Rows are arm basis vectors expressed in camera frame
    R = np.array([arm_x_in_cam, arm_y_in_cam, arm_z_in_cam])

    # Camera position in arm frame:
    # The camera is at the origin in camera frame
    # The arm base is at arm_base_camera in camera frame
    # In arm frame, the arm base is at origin
    # So: camera_pos_arm = -R @ arm_base_camera (roughly)

    # But we also know camera is at height camera_height above desk (z=0)
    # So camera z in arm frame = camera_height

    # Transform arm base position to arm frame (should give ~[0,0,0])
    # Camera position = -R @ arm_base_camera would work if arm_base was on desk
    # But arm_base_camera is projected to desk, so we need to account for that

    # The arm base is at [0, 0, 0] in arm frame
    # The camera sees it at arm_base_camera in camera frame
    # Camera position in arm frame:
    camera_pos_arm = -R @ arm_base_camera

    # Adjust z to be the camera height (positive, above desk)
    camera_pos_arm[2] = camera_height

    return R, camera_pos_arm


def rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert rotation matrix to axis-angle representation."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if np.abs(angle) < 1e-6:
        return np.array([0, 0, 1]), 0.0

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))

    return axis / np.linalg.norm(axis), np.degrees(angle)


async def main():
    print("Connecting to robot...")
    machine = await connect()
    print("Connected!")

    realsense = Camera.from_robot(machine, "realsense")

    # Get camera intrinsics (approximate for RealSense D435)
    # These should ideally come from camera.get_properties()
    try:
        props = await realsense.get_properties()
        print(f"Camera properties: {props}")
        fx = props.intrinsic_parameters.focal_x_px
        fy = props.intrinsic_parameters.focal_y_px
        cx = props.intrinsic_parameters.center_x_px
        cy = props.intrinsic_parameters.center_y_px
        print(f"Using camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    except Exception as e:
        print(f"Could not get camera properties ({e}), using defaults")
        fx, fy = 380, 380
        cx, cy = 320, 240

    # Capture depth image
    print("\nCapturing depth image...")
    images, metadata = await realsense.get_images()

    depth_img = None
    color_img = None
    for img in images:
        print(f"  Found image: {img.name}, mime_type: {img.mime_type}")
        if "depth" in img.name.lower():
            # Depth images are raw 16-bit data
            if "depth" in img.mime_type or "raw" in img.mime_type or img.mime_type == "image/vnd.viam.dep":
                # Raw depth data - typically has a small header followed by 16-bit unsigned integers
                raw_data = img.data
                h, w = 720, 1280  # From camera properties
                expected_size = h * w * 2  # 2 bytes per pixel (uint16)

                # Check for header (data size > expected)
                header_size = len(raw_data) - expected_size
                if header_size > 0 and header_size < 100:
                    print(f"  Skipping {header_size} byte header")
                    raw_data = raw_data[header_size:]

                depth_data = np.frombuffer(raw_data, dtype=np.uint16)
                if len(depth_data) == h * w:
                    depth_img = depth_data.reshape((h, w))
                else:
                    print(f"  WARNING: Unexpected depth data size: {len(depth_data)} (expected {h*w})")
                    # Try to infer dimensions
                    total = len(depth_data)
                    for test_h in [480, 720, 1080]:
                        if total % test_h == 0:
                            test_w = total // test_h
                            print(f"  Trying shape: {test_h}x{test_w}")
                            depth_img = depth_data.reshape((test_h, test_w))
                            break
            else:
                try:
                    pil_img = viam_to_pil_image(img)
                    depth_img = np.array(pil_img)
                except Exception as e:
                    print(f"  Could not decode depth image: {e}")
        elif "color" in img.name.lower():
            try:
                pil_img = viam_to_pil_image(img)
                color_img = np.array(pil_img)
            except Exception as e:
                print(f"  Could not decode color image: {e}")

    if depth_img is None:
        print("ERROR: No depth image found!")
        await machine.close()
        return

    print(f"Depth image shape: {depth_img.shape}, dtype: {depth_img.dtype}")
    print(f"Depth range: {depth_img.min()} - {depth_img.max()}")

    # Analyze depth distribution
    nonzero_depth = depth_img[depth_img > 0]
    if len(nonzero_depth) > 0:
        print(f"Non-zero depth pixels: {len(nonzero_depth)} ({100*len(nonzero_depth)/depth_img.size:.1f}%)")
        print(f"Non-zero depth range: {nonzero_depth.min()} - {nonzero_depth.max()}")
        print(f"Non-zero depth median: {np.median(nonzero_depth):.0f}")
        print(f"Non-zero depth mean: {np.mean(nonzero_depth):.0f}")

        # Check depth values in plausible desk region (center of image)
        h, w = depth_img.shape
        center_region = depth_img[h//3:2*h//3, w//3:2*w//3]
        center_nonzero = center_region[center_region > 0]
        if len(center_nonzero) > 0:
            print(f"Center region depth: median={np.median(center_nonzero):.0f}, range={center_nonzero.min()}-{center_nonzero.max()}")

        # VIAM depth format seems to be in mm already based on typical RealSense
        # But values > 10000 suggest camera sees far background
        # The desk should be around 500-1500mm from camera
        desk_range = (nonzero_depth > 400) & (nonzero_depth < 2000)
        desk_pixels = np.sum(desk_range)
        print(f"Pixels in desk range (400-2000mm): {desk_pixels} ({100*desk_pixels/len(nonzero_depth):.1f}%)")
    else:
        print("WARNING: No non-zero depth values!")

    # Update intrinsics based on actual image size
    h, w = depth_img.shape[:2]
    if cx == 320 and w != 640:
        cx, cy = w / 2, h / 2
        print(f"Adjusted intrinsics for image size: cx={cx}, cy={cy}")

    # Convert to 3D points
    print("\nConverting to 3D points...")
    points, pixels = depth_to_points(depth_img, fx, fy, cx, cy)
    print(f"Generated {len(points)} valid 3D points")

    # Filter to desk-distance range (500-2500mm from camera)
    # The desk should be within this range based on typical setups
    desk_depth_mask = (points[:, 2] > 500) & (points[:, 2] < 2500)
    desk_candidate_points = points[desk_depth_mask]
    desk_candidate_pixels = pixels[desk_depth_mask]
    print(f"Points in desk depth range (500-2500mm): {len(desk_candidate_points)}")

    if len(desk_candidate_points) < 100:
        print("WARNING: Very few points in desk range, expanding search...")
        # Try wider range
        desk_depth_mask = (points[:, 2] > 300) & (points[:, 2] < 4000)
        desk_candidate_points = points[desk_depth_mask]
        desk_candidate_pixels = pixels[desk_depth_mask]
        print(f"Points in expanded range (300-4000mm): {len(desk_candidate_points)}")

    # Fit horizontal plane to desk using RANSAC
    print("\nFitting horizontal plane to desk...")
    desk_normal, desk_d, desk_inliers_local = fit_plane_ransac(
        desk_candidate_points, horizontal_only=True
    )

    # Map inliers back to full point set
    desk_inliers = np.zeros(len(points), dtype=bool)
    desk_inliers[desk_depth_mask] = desk_inliers_local

    inlier_pct = 100 * np.sum(desk_inliers) / len(points)
    print(f"Desk plane found:")
    print(f"  Normal: [{desk_normal[0]:.4f}, {desk_normal[1]:.4f}, {desk_normal[2]:.4f}]")
    print(f"  Distance from camera: {abs(desk_d):.1f} mm")
    print(f"  Inliers: {np.sum(desk_inliers)} ({inlier_pct:.1f}%)")

    # Camera height is distance to desk plane
    camera_height = abs(desk_d)
    print(f"\nCamera height above desk: {camera_height:.1f} mm")

    # Find arm base
    print("\nLooking for arm base...")
    arm_base_camera, debug = find_arm_base(
        depth_img, points, pixels,
        desk_normal, desk_d, desk_inliers,
        fx, fy, cx, cy
    )

    if arm_base_camera is None:
        print(f"ERROR: Could not find arm base - {debug}")
        await machine.close()
        return

    print(f"Arm base found in camera frame:")
    print(f"  Position: [{arm_base_camera[0]:.1f}, {arm_base_camera[1]:.1f}, {arm_base_camera[2]:.1f}] mm")
    print(f"  Debug: {debug}")

    # Compute transform
    print("\nComputing camera transform...")
    R, t = compute_camera_transform(desk_normal, desk_d, arm_base_camera)
    axis, angle = rotation_matrix_to_axis_angle(R)

    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)

    print(f"\nCamera position in arm frame:")
    print(f"  x: {t[0]:.1f} mm")
    print(f"  y: {t[1]:.1f} mm")
    print(f"  z: {t[2]:.1f} mm (height above desk)")

    print(f"\nCamera orientation (axis-angle):")
    print(f"  axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
    print(f"  angle: {angle:.1f} degrees")

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

    # Save results
    results = {
        "camera_position_mm": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
        "rotation_axis": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
        "rotation_angle_degrees": float(angle),
        "desk_plane": {
            "normal": desk_normal.tolist(),
            "distance_mm": float(abs(desk_d))
        },
        "arm_base_in_camera_frame": arm_base_camera.tolist()
    }

    with open("calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to calibration_results.json")

    await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
