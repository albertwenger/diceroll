# diceroll
Get a robot arm to pick up and roll dice

## Setup

1. Install dependencies:
   ```
   pip install viam-sdk numpy Pillow
   ```

2. Copy the credentials template and fill in your values:
   ```
   cp credentials.json.example credentials.json
   ```
   Get your credentials from [app.viam.com](https://app.viam.com) → your machine → CONNECT tab → Python SDK.

3. Run:
   ```
   python robot.py
   ```

## Calibration

To calibrate the camera-to-arm transform (required for accurate pick-and-place):

```
python calibrate.py
```

This moves the arm through multiple poses, detects the gripper position in the RealSense depth data, and computes the rigid transform. Results are saved to `calibration_results.json` and printed as VIAM frame configuration.

## Hardware

- UFactory Lite 6 arm with gripper
- Intel RealSense depth camera
- Webcam
