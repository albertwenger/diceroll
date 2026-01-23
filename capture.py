#!/usr/bin/env python3
"""
Capture images from webcam and RealSense cameras.

Usage:
    python capture.py              # Capture from both cameras
    python capture.py webcam       # Capture from webcam only
    python capture.py realsense    # Capture from realsense only
    python capture.py --loop 5     # Capture every 5 seconds
"""

import asyncio
import argparse
import struct
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.media.utils.pil import viam_to_pil_image


# Output directory
IMAGES_DIR = Path(__file__).parent / "images"


def load_credentials():
    import json
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
    if raw_bytes[:8] != b"DEPTHMAP":
        raise ValueError("Not a DEPTHMAP format")

    # Try 32-bit width/height
    width = struct.unpack('<I', raw_bytes[8:12])[0]
    height = struct.unpack('<I', raw_bytes[12:16])[0]
    pixel_data = raw_bytes[16:]

    expected_size = width * height * 2
    if abs(expected_size - len(pixel_data)) < 100:
        depth = np.frombuffer(pixel_data[:expected_size], dtype=np.uint16)
        return depth.reshape((height, width))

    # Fallback: infer from common resolutions
    for header_size in [16, 24]:
        total_pixels = len(raw_bytes[header_size:]) // 2
        for w, h in [(1280, 720), (640, 480), (848, 480), (640, 360)]:
            if w * h == total_pixels:
                data = raw_bytes[header_size:header_size + w*h*2]
                return np.frombuffer(data, dtype=np.uint16).reshape((h, w))

    raise ValueError(f"Cannot parse depth image, size={len(raw_bytes)}")


def depth_to_colormap(depth: np.ndarray, min_depth: int = 0, max_depth: int = 5000) -> Image.Image:
    """Convert depth array to a colorized visualization."""
    # Normalize to 0-255
    depth_clipped = np.clip(depth, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # Apply colormap (blue=close, red=far)
    # Simple gradient: close=blue, far=red
    h, w = depth_normalized.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = depth_normalized  # Red channel = far
    rgb[:, :, 2] = 255 - depth_normalized  # Blue channel = close

    return Image.fromarray(rgb)


async def capture_webcam(camera: Camera, timestamp: str) -> str:
    """Capture and save webcam image."""
    images, metadata = await camera.get_images()
    if not images:
        raise ValueError("No image from webcam")

    img = viam_to_pil_image(images[0])
    filename = IMAGES_DIR / f"webcam_{timestamp}.jpg"
    img.save(filename)
    return str(filename)


async def capture_realsense(camera: Camera, timestamp: str) -> tuple[str, str]:
    """Capture and save RealSense RGB and depth images."""
    images, metadata = await camera.get_images()

    rgb_file = None
    depth_file = None

    for img in images:
        # Check for depth image
        if hasattr(img, 'data') and len(img.data) > 8 and img.data[:8] == b"DEPTHMAP":
            try:
                depth = parse_depth_image(img.data)

                # Save raw depth as numpy
                depth_raw_file = IMAGES_DIR / f"realsense_depth_{timestamp}.npy"
                np.save(depth_raw_file, depth)

                # Save colorized depth visualization
                depth_vis = depth_to_colormap(depth)
                depth_file = IMAGES_DIR / f"realsense_depth_{timestamp}.png"
                depth_vis.save(depth_file)
            except Exception as e:
                print(f"  Warning: Could not process depth: {e}")
        else:
            # Assume RGB image
            try:
                rgb_img = viam_to_pil_image(img)
                rgb_file = IMAGES_DIR / f"realsense_rgb_{timestamp}.jpg"
                rgb_img.save(rgb_file)
            except Exception as e:
                print(f"  Warning: Could not process RGB: {e}")

    return str(rgb_file) if rgb_file else None, str(depth_file) if depth_file else None


async def capture_once(machine, cameras: list[str], timestamp: str = None):
    """Capture images from specified cameras."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}

    if "webcam" in cameras:
        try:
            webcam = Camera.from_robot(machine, "webcam")
            filename = await capture_webcam(webcam, timestamp)
            results["webcam"] = filename
            print(f"  Webcam: {filename}")
        except Exception as e:
            print(f"  Webcam error: {e}")

    if "realsense" in cameras:
        try:
            realsense = Camera.from_robot(machine, "realsense")
            rgb_file, depth_file = await capture_realsense(realsense, timestamp)
            if rgb_file:
                results["realsense_rgb"] = rgb_file
                print(f"  RealSense RGB: {rgb_file}")
            if depth_file:
                results["realsense_depth"] = depth_file
                print(f"  RealSense Depth: {depth_file}")
        except Exception as e:
            print(f"  RealSense error: {e}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Capture camera images")
    parser.add_argument("camera", nargs="?", default="both",
                        choices=["webcam", "realsense", "both"],
                        help="Which camera to capture from")
    parser.add_argument("--loop", type=float, default=None,
                        help="Capture every N seconds (continuous mode)")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of captures in loop mode")
    args = parser.parse_args()

    # Determine which cameras to use
    if args.camera == "both":
        cameras = ["webcam", "realsense"]
    else:
        cameras = [args.camera]

    # Ensure output directory exists
    IMAGES_DIR.mkdir(exist_ok=True)

    print("Connecting to robot...")
    machine = await connect()
    print("Connected!\n")

    try:
        if args.loop:
            # Continuous capture mode
            count = 0
            print(f"Capturing every {args.loop} seconds. Press Ctrl+C to stop.\n")
            while True:
                count += 1
                if args.count and count > args.count:
                    break

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"Capture #{count} at {timestamp}:")
                await capture_once(machine, cameras, timestamp)
                print()

                if args.count and count >= args.count:
                    break

                await asyncio.sleep(args.loop)
        else:
            # Single capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Capturing at {timestamp}:")
            await capture_once(machine, cameras, timestamp)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        await machine.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
