"""Scripted demonstration collection for the Franka pick-and-place task.

This utility generates LeRobot-compatible episodes by running a simple
heuristic controller inside :class:`env.mujoco_env.FrankaPickPlaceEnv`.  The
resulting dataset can be used directly by ``train_bc.py`` and mirrors the
structure expected by the COMS4733 Milestone 1 baseline.

The script purposely keeps the policy trivial – it drives the gripper towards
the target object's site using a proportional controller and lifts it above the
table.  While this will not solve challenging scenes, it is sufficient for
smoke-testing the end-to-end data pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from env.mujoco_env import FrankaPickPlaceEnv
from env.controllers import KeyframeController, KinematicsHelper


def world_to_image_coords(obj_pos_3d: np.ndarray, workspace_bounds: tuple = None) -> np.ndarray:
    """Convert 3D world position to normalized 2D image coordinates [0, 1].
    
    Args:
        obj_pos_3d: 3D position in world coordinates (x, y, z)
        workspace_bounds: Tuple of ((x_min, x_max), (y_min, y_max)) or None for defaults
    
    Returns:
        Normalized 2D position (x_norm, y_norm) in [0, 1] range
    """
    if workspace_bounds is None:
        # Default workspace bounds for Franka pick-and-place task
        x_min, x_max = 0.40, 0.65  # X range: 40cm to 65cm (25cm width)
        y_min, y_max = -0.30, 0.30  # Y range: -30cm to 30cm (60cm depth)
    else:
        (x_min, x_max), (y_min, y_max) = workspace_bounds
    
    x_world, y_world, z_world = obj_pos_3d
    
    # Normalize to [0, 1] based on workspace bounds
    x_norm = (x_world - x_min) / (x_max - x_min)
    y_norm = (y_world - y_min) / (y_max - y_min)
    
    return np.array([np.clip(x_norm, 0.0, 1.0), np.clip(y_norm, 0.0, 1.0)], dtype=np.float32)


@dataclass(slots=True)
class EpisodeBuffer:
    """Stores trajectory information before writing to disk."""

    rgb_frames: List[np.ndarray]
    proprio: List[np.ndarray]
    actions: List[np.ndarray]
    timestamps: List[float]
    object_positions: List[np.ndarray]  # NEW: Ground truth object positions (normalized [0,1])
    instruction: str
    meta: Dict[str, object]

    def extend(self, obs: Dict[str, np.ndarray], action: np.ndarray, timestamp: float, object_pos_normalized: np.ndarray) -> None:
        """Add a step with ground truth object position."""
        self.rgb_frames.append((obs["rgb_static"] * 255).astype(np.uint8))
        self.proprio.append(obs["proprio"].astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.timestamps.append(float(timestamp))
        self.object_positions.append(object_pos_normalized.astype(np.float32))

    def save(self, root: Path, episode_id: int) -> None:
        episode_dir = root / f"episode_{episode_id:04d}"
        image_dir = episode_dir / "obs" / "rgb_static"
        episode_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(self.rgb_frames):
            Image.fromarray(frame).save(image_dir / f"{idx:06d}.png")

        np.save(episode_dir / "obs" / "proprio.npy", np.stack(self.proprio, axis=0))
        np.save(episode_dir / "actions.npy", np.stack(self.actions, axis=0))
        np.save(episode_dir / "timestamps.npy", np.asarray(self.timestamps, dtype=np.float32))
        
        # NEW: Save ground truth object positions
        np.save(episode_dir / "obs" / "object_positions.npy", np.stack(self.object_positions, axis=0))

        (episode_dir / "instruction.txt").write_text(self.instruction)
        (episode_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))


def compute_adaptive_keyframes(
    env: FrankaPickPlaceEnv,
    object_pos: np.ndarray,
    kin_helper: KinematicsHelper,
    noisy_object_pos: np.ndarray = None,
) -> dict[str, np.ndarray]:
    """Compute keyframes adapted to actual object position using IK.

    Args:
        env: The environment instance
        object_pos: 3D position of the target object (actual/ground truth)
        kin_helper: Kinematics helper for IK computation
        noisy_object_pos: Optional noisy position for diverse approach trajectories

    Returns:
        Dictionary mapping keyframe names to 7D joint configurations
    """
    # Base keyframes (proven for object at 0.5, 0, 0.03)
    base_keyframes = {
        "home": np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853]),
        "pre_grasp": np.array([0.1, 0.35, -0.1, -2.05, 0.0, 2.0, -0.5]),
        "grasp": np.array([0.0, 0.675, 0.0, -1.9, 0.0, 2.0, -0.5]),
    }

    # Use noisy position for pre_grasp (approach diversity), actual position for grasp
    approach_pos = noisy_object_pos if noisy_object_pos is not None else object_pos

    bin_y = 0.45
    ball_near_bin = abs(object_pos[1] - bin_y) < 0.35

    if ball_near_bin:
        pre_grasp_height = 0.15
        grasp_height = 0.010
        print(f"  ⚠ Ball near bin - using careful grasp")
    else:
        pre_grasp_height = 0.12
        # --- FIX 1: LOWER GRASP HEIGHT ---
        # 0.015 was grabbing the top. 0.01 grabs the "waist" of the ball.
        grasp_height = 0.010

    # Pre-grasp uses noisy position for diversity, grasp uses actual position
    target_positions = {
        "pre_grasp": approach_pos + np.array([0, 0, pre_grasp_height]),
        "grasp": object_pos + np.array([0, 0, grasp_height]),  # Always use actual position
    }
    
    downward_quat = np.array([0.0, 1.0, 0.0, 0.0])
    keyframes = {"home": base_keyframes["home"]}
    
    # Compute pre_grasp and grasp using IK
    for keyframe_name, target_pos in target_positions.items():
        # Use base keyframe as initial guess
        initial_q = base_keyframes[keyframe_name]
        
        use_ik = False
        
        for method in ["ccd", "standard", "staged"]:
            try:
                if method == "ccd":
                    result_q = kin_helper.inverse_kinematics_ccd(
                        target_pos=target_pos, target_quat=downward_quat, initial_q=initial_q,
                        max_iters=200, tol_pos=0.015, position_weight=20.0
                    )
                elif method == "standard":
                    result_q = kin_helper.inverse_kinematics(
                        target_pos=target_pos, target_quat=downward_quat, initial_q=initial_q,
                        max_iters=300, tol_pos=0.015, damping=5e-4
                    )
                elif method == "staged":
                    dist = np.linalg.norm(object_pos[:2])
                    result_q = kin_helper.inverse_kinematics_staged(
                        target_pos=target_pos, target_quat=downward_quat, initial_q=initial_q,
                        horizontal_distance=dist, max_iters=200
                    )
                
                check_pos, _ = kin_helper.forward_kinematics(result_q)
                if np.linalg.norm(check_pos - target_pos) < 0.025:
                    keyframes[keyframe_name] = result_q
                    use_ik = True
                    break
            except Exception:
                continue
        
        if not use_ik:
            print(f"  X IK Failed for {keyframe_name} - using fallback")
            fallback = initial_q.copy()
            offset = object_pos - np.array([0.5, 0.0, 0.03])
            fallback[0] += offset[1] * 2.0
            fallback[1] += offset[0] * 2.0
            keyframes[keyframe_name] = fallback
    
    keyframes["grasp_closed"] = keyframes["grasp"]
    keyframes["lift"] = keyframes["pre_grasp"]

    # Load calibrated keyframes if available, otherwise use defaults
    calibration_file = Path("data/calibrated_keyframes.json")

    if calibration_file.exists():
        # Use calibrated keyframes from GUI calibration
        with calibration_file.open("r") as f:
            calibration_data = json.load(f)

        transport_q = np.array(calibration_data["keyframes"]["transport"]["joint_angles"])
        place_q = np.array(calibration_data["keyframes"]["place"]["joint_angles"])

        # Verify positions
        transport_pos, _ = kin_helper.forward_kinematics(transport_q)
        place_pos, _ = kin_helper.forward_kinematics(place_q)
        bin_pos = env.bin_position

        print(f"  Using CALIBRATED keyframes:")
        print(f"    Transport: {transport_pos} (dist to bin: {np.linalg.norm(transport_pos[:2] - bin_pos[:2]):.3f}m)")
        print(f"    Place: {place_pos} (dist to bin: {np.linalg.norm(place_pos[:2] - bin_pos[:2]):.3f}m)")
    else:
        # Fallback to hardcoded defaults (these work for standard bin position)
        print(f"  ⚠ No calibration file found at {calibration_file}")
        print(f"  Using default keyframes (may not be accurate!)")
        print(f"  Run: mjpython data/calibrate_keyframes.py")

        transport_q = np.array([0.4, 0.35, 0.0, -1.8, 0.1, 2.1, -0.55])
        place_q = np.array([0.5, 0.3, 0.05, -1.7, 0.15, 2.0, -0.6])

    # Add small noise to transport only (if noise is enabled)
    if noisy_object_pos is not None:
        transport_noise = np.random.normal(0, 0.03, 7)  # Small noise
        transport_noise[4:] = 0  # Don't noise wrist
        keyframes["transport"] = transport_q + transport_noise
    else:
        keyframes["transport"] = transport_q

    # NO noise on place - always exact!
    keyframes["place"] = place_q
    keyframes["place_open"] = place_q

    return keyframes


def keyframe_policy(
    env: FrankaPickPlaceEnv,
    controller: KeyframeController,
    steps_at_keyframe: int,
    dwell_time: int = 20,
) -> tuple[np.ndarray, bool]:
    """Keyframe-based policy using P-Control (NO noise injection - diversity comes from position noise)."""
    # 1. Get targets
    keyframe_name, target_q = controller.get_current_target()

    # 2. Get current state
    current_q = env.data.qpos[env._joint_qpos_indices].copy()
    current_qvel = env.data.qvel[env._joint_dof_indices].copy()

    # NO NOISE INJECTION HERE - diversity comes from noisy object positions at keyframe computation
    # This ensures precise execution of the planned trajectory

    # --- P-CONTROLLER (Position Control with Smoothing) ---
    error = target_q - current_q

    kp = 0.25  # Smooth motion
    target_position = current_q + kp * error

    # --- CONVERGENCE CHECK ---
    dist = np.max(np.abs(target_q - current_q))
    vel_mag = np.max(np.abs(current_qvel))

    # --- FIX 2: DWELL TIME & CONVERGENCE ---
    if keyframe_name == "grasp_closed":
        required_dwell = 10  # Wait longer to ensure grip is solid
        is_converged = True  # We trust the time duration
    elif keyframe_name == "grasp":
        required_dwell = 5
        is_converged = dist < 0.10
    elif keyframe_name == "place":
        required_dwell = 10  # Give it more time to settle
        is_converged = dist < 0.20 or steps_at_keyframe >= 15  # More lenient threshold OR timeout
    elif keyframe_name == "place_open":
        required_dwell = 20  # Hold position after opening to let ball drop
        is_converged = True  # Trust time duration
    else:
        required_dwell = 1
        is_converged = dist < 0.20

    # Debug Prints
    if steps_at_keyframe % 10 == 0:
        print(f"  [DEBUG] {keyframe_name} | Err: {dist:.3f} | Vel: {vel_mag:.3f} | Step: {steps_at_keyframe}")

    if steps_at_keyframe >= required_dwell and is_converged:
        print(f"  ✓ Reached {keyframe_name}!")
        controller.advance_to_next_keyframe()

    # 3. Construct Action
    action = np.zeros(8)
    action[:7] = target_position

    # --- FIX 3: GRIPPER CONTROL ---
    # Strong grip during pick and transport, open for place
    if "closed" in keyframe_name or keyframe_name in ["lift", "transport", "place"]:
        action[7] = -0.01  # Closed: maximum grip
    elif keyframe_name in ["place_open"]:
        action[7] = 0.04  # Open: release ball
    else:
        action[7] = 0.04  # Default: open

    return action, controller.is_sequence_complete()


def collect_episode(env: FrankaPickPlaceEnv, hindered: bool, max_steps: int, add_noise: bool = True) -> EpisodeBuffer:
    obs, info = env.reset(hindered=hindered)
    buffer = EpisodeBuffer(
        rgb_frames=[],
        proprio=[],
        actions=[],
        timestamps=[],
        object_positions=[],
        instruction=info["instruction"],
        meta=info,
    )

    # Get target object position (actual/ground truth)
    target_color = env.target_color
    target_site_id = env._object_site_ids[target_color]
    object_pos = env.data.site_xpos[target_site_id].copy()

    # Add noise to object position for trajectory diversity
    # This creates different approach paths while still grasping at the actual location
    noisy_object_pos = None
    if add_noise:
        # Add XY noise for diverse approaches (not Z - keep height accurate)
        xy_noise_std = 0.02  # 2cm standard deviation in XY
        xy_noise = np.random.normal(0, xy_noise_std, 2)
        noisy_object_pos = object_pos.copy()
        noisy_object_pos[:2] += xy_noise
        print(f"Starting episode | Target: {target_color} at {object_pos} | Noisy approach: {noisy_object_pos} | Hindered: {hindered}")
    else:
        print(f"Starting episode | Target: {target_color} at {object_pos} | Hindered: {hindered}")

    # Create kinematics helper for adaptive IK
    kin_helper = KinematicsHelper(env.model, site_name="gripper")

    # Compute adaptive keyframes: noisy position for pre_grasp, actual for grasp
    keyframes = compute_adaptive_keyframes(env, object_pos, kin_helper, noisy_object_pos)
    
    # Initialize keyframe controller
    controller = KeyframeController(
        keyframes=keyframes,
        convergence_threshold=0.20,
        velocity_threshold=1.0,
    )
    
    # Skip 'home' to avoid getting stuck at start
    pick_place_sequence = [
        "pre_grasp", "grasp", "grasp_closed", "lift",
        "transport", "place", "place_open"
    ]
    controller.set_sequence(pick_place_sequence)

    timestamp = 0.0
    steps_at_keyframe = 0
    prev_keyframe_idx = 0
    
    for step_idx in range(max_steps):
        # Get action from keyframe policy
        action, sequence_complete = keyframe_policy(env, controller, steps_at_keyframe, dwell_time=20)
        
        # Get current object position and convert to normalized image coordinates
        current_obj_pos_3d = env.data.site_xpos[target_site_id].copy()
        obj_pos_normalized = world_to_image_coords(current_obj_pos_3d)
        
        # Record data with ground truth object position
        buffer.extend(obs, action, timestamp, obj_pos_normalized)
        
        # Step environment
        result = env.step(action)
        obs = result.observation
        timestamp += env.step_dt
        steps_at_keyframe += 1
        
        # Reset step counter when advancing to new keyframe
        current_keyframe_idx, _ = controller.get_progress()
        if current_keyframe_idx != prev_keyframe_idx:
            steps_at_keyframe = 0
            prev_keyframe_idx = current_keyframe_idx
        
        # Sync the viewer if GUI is enabled
        if env.viewer is not None:
            env.viewer.sync()
        
        if result.terminated or result.truncated:
            print(f"  Episode terminated at step {step_idx}")
            break
        
        # End when sequence complete (all keyframes reached)
        if sequence_complete:
            # Check if ball is actually in the box (success condition)
            obj_pos = env.data.site_xpos[target_site_id]
            horizontal_dist = np.linalg.norm(obj_pos[:2] - env.bin_position[:2])
            obj_in_bin = horizontal_dist < env.bin_radius and obj_pos[2] < 0.08

            if obj_in_bin:
                print(f"  ✓ Ball in box! Episode complete at step {step_idx}")
            else:
                print(f"  ⚠ Ball NOT in box (dist={horizontal_dist:.3f}, z={obj_pos[2]:.3f}) at step {step_idx}")
            break

    buffer.meta.update({"episode_length": len(buffer.actions), "sequence_complete": sequence_complete})
    return buffer


def write_metadata(dataset_root: Path, metadata: List[Dict[str, object]], train_fraction: float = 0.8) -> None:
    """Write dataset metadata with train/val splits.
    
    Args:
        dataset_root: Root directory for the dataset.
        metadata: List of episode metadata dictionaries.
        train_fraction: Fraction of episodes to use for training (default: 0.8).
    """
    num_episodes = len(metadata)
    num_train = int(train_fraction * num_episodes)
    
    # Create train/val splits
    train_episodes = [item["episode"] for item in metadata[:num_train]]
    val_episodes = [item["episode"] for item in metadata[num_train:]]
    
    payload = {
        "episodes": metadata,
        "num_static": sum(1 for item in metadata if not item.get("hindered", False)),
        "num_hindered": sum(1 for item in metadata if item.get("hindered", False)),
        "splits": {
            "train": train_episodes,
            "val": val_episodes,
        },
    }
    (dataset_root / "metadata.json").write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect LeRobot demonstrations using MuJoCo.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Output directory for episodes.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record.")
    parser.add_argument("--hindered-fraction", type=float, default=0.1, help="Fraction of episodes with hindered resets.")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Fraction of episodes to use for training (vs validation).")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum steps per episode for full pick-and-place.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--gui", action="store_true", help="Enable the interactive MuJoCo viewer.")
    parser.add_argument("--no-noise", action="store_true", help="Disable position noise for deterministic trajectories.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("env/mujoco_assets"),
        help="Directory containing franka_scene.xml and associated assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Check for existing episodes to determine starting point
    existing_episodes = []
    if dataset_root.exists():
        existing_episodes = [d for d in dataset_root.iterdir() 
                           if d.is_dir() and d.name.startswith('episode_')]
    
    start_episode = len(existing_episodes)
    print(f"Found {len(existing_episodes)} existing episodes. Starting from episode {start_episode}")
    
    # Load existing metadata if it exists
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        metadata = existing_metadata.get("episodes", [])
        print(f"Loaded existing metadata with {len(metadata)} episodes")
    else:
        metadata = []

    env = FrankaPickPlaceEnv(gui=args.gui, seed=args.seed, asset_root=args.asset_root)

    rng = np.random.default_rng(args.seed)
    hindered_fraction = float(np.clip(args.hindered_fraction, 0.0, 1.0))

    for _ in range(start_episode): rng.random()

    # Continue from where we left off
    for episode_idx in range(start_episode, args.episodes):
        hindered = rng.random() < hindered_fraction
        add_noise = not args.no_noise  # Add noise by default, unless --no-noise is specified
        buffer = collect_episode(env, hindered=hindered, max_steps=args.max_steps, add_noise=add_noise)
        buffer.save(dataset_root, episode_idx)
        metadata.append({
            "episode": f"episode_{episode_idx:04d}",
            "length": len(buffer.actions),
            "hindered": hindered,
            "instruction": buffer.instruction,
            "target_color": buffer.meta.get("target_color"),
        })
        print(f"Recorded episode {episode_idx:04d} | hindered={hindered} | steps={len(buffer.actions)}")

    write_metadata(dataset_root, metadata, train_fraction=args.train_fraction)
    env.close()
    print(f"Saved dataset with {len(metadata)} episodes to {dataset_root}")
    print(f"Train/val split: {int(args.train_fraction * len(metadata))}/{len(metadata) - int(args.train_fraction * len(metadata))} episodes")


if __name__ == "__main__":
    main()

