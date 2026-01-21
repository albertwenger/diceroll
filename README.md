# diceroll
Get a robot arm to pick up and roll dice

## Setup

1. Install dependencies:
   ```
   pip install viam-sdk
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

## Hardware

- UFactory Lite 6 arm with gripper
- Intel RealSense depth camera
- Webcam
