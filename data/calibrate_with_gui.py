#!/usr/bin/env python3
"""Interactive GUI calibration tool using MuJoCo's native control sliders.

This tool lets you control the robot joints using MuJoCo's built-in UI sliders.
Press SPACE to save current joint angles as keyframes.
"""

import json
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from env.mujoco_env import FrankaPickPlaceEnv
from env.controllers import KinematicsHelper


class CalibrationTool:
    def __init__(self):
        self.env = FrankaPickPlaceEnv()
        self.kin = KinematicsHelper(self.env.model, site_name="gripper")

        self.saved_keyframes = {}
        self.mode = "transport"  # Start with transport

        # Set initial position to transport default
        transport_default = np.array([0.4, 0.35, 0.0, -1.8, 0.1, 2.1, -0.55])
        self.env.data.qpos[self.env._joint_qpos_indices] = transport_default
        self.env.data.ctrl[:7] = transport_default

        print("\n" + "="*60)
        print("MuJoCo Interactive Calibration Tool")
        print("="*60)
        print("\nInstructions:")
        print("  1. Use the sliders in the RIGHT UI panel to control joints")
        print("  2. Position the robot over the bin for 'transport' pose")
        print("  3. Press SPACE to save as 'transport' keyframe")
        print("  4. Adjust for 'place' pose (lower, over bin center)")
        print("  5. Press SPACE again to save as 'place' keyframe")
        print("  6. Press ESC to exit and save")
        print("\nBin location: [0.55, 0.45, 0.08]")
        print("="*60 + "\n")

    def key_callback(self, keycode):
        """Handle keyboard input."""
        if keycode == 32:  # SPACE
            self.save_current_pose()
        elif keycode == 256:  # ESC
            self.save_and_exit()

    def save_current_pose(self):
        """Save current joint configuration as a keyframe."""
        # Get current joint positions from the simulation
        current_q = self.env.data.qpos[self.env._joint_qpos_indices].copy()

        # Compute end-effector position
        ee_pos, _ = self.kin.forward_kinematics(current_q)
        bin_pos = self.env.bin_position
        dist_to_bin = np.linalg.norm(ee_pos[:2] - bin_pos[:2])

        # Save keyframe
        self.saved_keyframes[self.mode] = {
            "joint_angles": current_q.tolist(),
            "ee_position": ee_pos.tolist(),
            "dist_to_bin_xy": float(dist_to_bin),
        }

        print(f"\n✓ Saved '{self.mode}' keyframe:")
        print(f"  Joint angles: {current_q}")
        print(f"  End-effector: {ee_pos}")
        print(f"  XY dist to bin: {dist_to_bin:.3f}m")
        print(f"  Height: {ee_pos[2]:.3f}m")

        # Switch mode
        if self.mode == "transport":
            self.mode = "place"
            print(f"\nNow adjust for '{self.mode}' pose and press SPACE")
        else:
            print("\nAll keyframes saved! Press ESC to exit and save to file.")

    def save_and_exit(self):
        """Save calibration to file."""
        if len(self.saved_keyframes) == 0:
            print("\n⚠ No keyframes saved. Exiting without saving.")
            return

        output_file = Path("data/calibrated_keyframes.json")

        calibration_data = {
            "bin_position": self.env.bin_position.tolist(),
            "keyframes": self.saved_keyframes,
        }

        with output_file.open("w") as f:
            json.dump(calibration_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ Saved calibration to {output_file}")
        print(f"{'='*60}")
        print("\nSummary:")
        for name, data in self.saved_keyframes.items():
            print(f"  {name}:")
            print(f"    Distance to bin: {data['dist_to_bin_xy']:.3f}m")
            print(f"    Height: {data['ee_position'][2]:.3f}m")

    def run(self):
        """Run the interactive calibration."""
        # Launch viewer with controls enabled
        with mujoco.viewer.launch_passive(
            self.env.model,
            self.env.data,
            key_callback=self.key_callback,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:

            print("GUI launched. Use sliders to control the robot.\n")

            # Keep viewer running until closed
            while viewer.is_running():
                # The ctrl values are automatically updated by the sliders
                # Copy ctrl to qpos to move the robot
                self.env.data.qpos[self.env._joint_qpos_indices] = self.env.data.ctrl[:7]

                # Update physics
                mujoco.mj_forward(self.env.model, self.env.data)

                # Sync viewer
                viewer.sync()

        # Save when viewer closes
        self.save_and_exit()


if __name__ == "__main__":
    tool = CalibrationTool()
    tool.run()
