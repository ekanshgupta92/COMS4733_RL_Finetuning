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
from scipy.spatial.transform import Rotation as R


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
    """Compute keyframes adapted to actual object position using robust IK.

    Uses rotation matrix-based IK (from Simple-MuJoCo) for better stability.

    Args:
        env: The environment instance
        object_pos: 3D position of the target object (actual/ground truth)
        kin_helper: Kinematics helper for IK computation
        noisy_object_pos: Optional noisy position for diverse approach trajectories

    Returns:
        Dictionary mapping keyframe names to 7D joint configurations
    """
    # Define target downward orientation as rotation matrix (more stable than quaternions)
    # This is gripper pointing down with slight tilt
    z_deg = 0.0  # No z-rotation
    y_deg = 3.0  # Slight tilt
    x_deg = 180.0  # Pointing down
    
    rpy = np.array([np.deg2rad(z_deg), np.deg2rad(y_deg), np.deg2rad(x_deg)])
    target_R = R.from_euler('zyx', rpy).as_matrix()

    # Use noisy position for pre_grasp (approach diversity), actual position for grasp
    approach_pos = noisy_object_pos if noisy_object_pos is not None else object_pos

    # Adaptive heights based on cube position
    # Cubes are 5cm x 5cm x 5cm (size=0.025), center is at object_pos height
    bin_y = 0.45
    cube_near_bin = abs(object_pos[1] - bin_y) < 0.35

    if cube_near_bin:
        pre_grasp_height = 0.15  # High approach to avoid bin collision
        grasp_height = 0.000  # Grasp at cube center (0 offset from cube position)
        print(f"  ⚠ Cube near bin - using high approach")
    else:
        pre_grasp_height = 0.12  # Standard approach height
        grasp_height = 0.000  # Grasp at cube center (flat surfaces, no offset needed)
    
    # Base configuration for fallback
    base_home = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])
    
    keyframes = {}
    
    # 1. Home position
    keyframes["home"] = base_home
    
    # 2. Pre-grasp: Above the object
    pre_grasp_pos = approach_pos + np.array([0, 0, pre_grasp_height])
    try:
        pre_grasp_q = kin_helper.inverse_kinematics_robust(
            target_pos=pre_grasp_pos,
            target_R=target_R,
            initial_q=base_home,
            max_iters=300,
            tol=0.01,
            step_limit=np.radians(5.0)
        )
        # Verify solution
        check_pos, _ = kin_helper.forward_kinematics(pre_grasp_q)
        if np.linalg.norm(check_pos - pre_grasp_pos) < 0.03:
            keyframes["pre_grasp"] = pre_grasp_q
        else:
            print(f"  ⚠ Pre-grasp IK inaccurate, using fallback")
            keyframes["pre_grasp"] = base_home
    except Exception as e:
        print(f"  ✗ Pre-grasp IK failed: {e}")
        keyframes["pre_grasp"] = base_home
    
    # 3. Grasp: At the object (use actual position, not noisy) - MUST BE ACCURATE
    grasp_pos = object_pos + np.array([0, 0, grasp_height])
    try:
        grasp_q = kin_helper.inverse_kinematics_robust(
            target_pos=grasp_pos,
            target_R=target_R,
            initial_q=keyframes["pre_grasp"],  # Start from pre-grasp
            max_iters=500,  # More iterations for accuracy
            tol=0.005,  # Tighter tolerance (5mm instead of 8mm)
            step_limit=np.radians(2.5)  # Smaller steps for precision
        )
        # Verify solution - must be very accurate
        check_pos, _ = kin_helper.forward_kinematics(grasp_q)
        error = np.linalg.norm(check_pos - grasp_pos)
        if error < 0.010:  # Accept only if within 1cm (tighter than before)
            keyframes["grasp"] = grasp_q
            print(f"  ✓ Grasp IK converged (error: {error*1000:.1f}mm)")
        else:
            print(f"  ⚠ Grasp IK inaccurate ({error*1000:.1f}mm), using pre-grasp")
            # Fallback: copy pre-grasp (will descend from above)
            keyframes["grasp"] = keyframes["pre_grasp"].copy()
    except Exception as e:
        print(f"  ✗ Grasp IK failed: {e}")
        keyframes["grasp"] = keyframes["pre_grasp"].copy()
    
    # 4. Grasp closed (same position, gripper will close)
    keyframes["grasp_closed"] = keyframes["grasp"]
    
    # 5. Lift (back to pre-grasp height with object)
    keyframes["lift"] = keyframes["pre_grasp"]
    
    # 6. Transport and Place keyframes
    # Load calibrated keyframes if available
    calibration_file = Path("data/calibrated_keyframes.json")

    if calibration_file.exists():
        with calibration_file.open("r") as f:
            calibration_data = json.load(f)

        transport_q = np.array(calibration_data["keyframes"]["transport"]["joint_angles"])
        place_q = np.array(calibration_data["keyframes"]["place"]["joint_angles"])

        print(f"  ✓ Using CALIBRATED transport and place keyframes")
    else:
        print(f"  ⚠ No calibration file, using default transport/place")
        # Defaults (work for standard bin position at 0.55, 0.45)
        transport_q = np.array([0.4, 0.35, 0.0, -1.8, 0.1, 2.1, -0.55])
        place_q = np.array([0.5, 0.3, 0.05, -1.7, 0.15, 2.0, -0.6])

    keyframes["transport"] = transport_q
    keyframes["place"] = place_q
    keyframes["place_open"] = place_q

    return keyframes


def keyframe_policy(
    env: FrankaPickPlaceEnv,
    controller: KeyframeController,
    steps_at_keyframe: int,
    dwell_time: int = 20,
) -> tuple[np.ndarray, bool]:
    """Keyframe-based policy using smooth P-Control with robust convergence checking."""
    # 1. Get targets
    keyframe_name, target_q = controller.get_current_target()

    # 2. Get current state
    current_q = env.data.qpos[env._joint_qpos_indices].copy()
    current_qvel = env.data.qvel[env._joint_dof_indices].copy()

    # --- P-CONTROLLER with adaptive gains ---
    error = target_q - current_q
    
    # Adaptive gain based on error magnitude (slower when close)
    error_norm = np.linalg.norm(error)
    if error_norm > 0.5:
        kp = 0.5  # Faster when far (increased from 0.35)
    elif error_norm > 0.2:
        kp = 0.35  # Medium speed (increased from 0.25)
    else:
        kp = 0.25  # Slow and precise when close (increased from 0.15)
    
    target_position = current_q + kp * error

    # --- CONVERGENCE CHECK (more reliable) ---
    max_joint_error = np.max(np.abs(error))
    avg_joint_error = np.mean(np.abs(error))
    vel_mag = np.max(np.abs(current_qvel))

    # Keyframe-specific convergence criteria (relaxed for reliability)
    if keyframe_name == "home":
        required_dwell = 5
        is_converged = max_joint_error < 0.20 or steps_at_keyframe >= 30
    elif keyframe_name == "pre_grasp":
        required_dwell = 10  # Settle before descending
        is_converged = max_joint_error < 0.20 or steps_at_keyframe >= 40  # Relaxed + timeout
    elif keyframe_name == "grasp":
        required_dwell = 15  # Give time to settle at grasp position
        is_converged = max_joint_error < 0.12 or steps_at_keyframe >= 40  # Tighter threshold
    elif keyframe_name == "grasp_closed":
        required_dwell = 30  # Wait longer for secure grip (increased from 20)
        is_converged = True  # Time-based only
    elif keyframe_name == "lift":
        required_dwell = 10
        is_converged = max_joint_error < 0.20 or steps_at_keyframe >= 30
    elif keyframe_name == "transport":
        required_dwell = 10
        is_converged = max_joint_error < 0.20 or steps_at_keyframe >= 30
    elif keyframe_name == "place":
        required_dwell = 10  # Reduced from 15
        is_converged = max_joint_error < 0.20 or steps_at_keyframe >= 30  # Timeout fallback
    elif keyframe_name == "place_open":
        required_dwell = 25  # Hold after opening to let ball drop
        is_converged = True  # Time-based only
    else:
        required_dwell = 10
        is_converged = max_joint_error < 0.20 or steps_at_keyframe >= 30

    # Debug prints (less frequent)
    if steps_at_keyframe % 15 == 0:
        print(f"  [{keyframe_name:12s}] MaxErr:{max_joint_error:.3f} AvgErr:{avg_joint_error:.3f} Vel:{vel_mag:.3f} Step:{steps_at_keyframe}/{required_dwell}")

    # Advance when converged AND minimum dwell time met
    if steps_at_keyframe >= required_dwell and is_converged:
        if max_joint_error < 0.20:
            print(f"  ✓ Completed {keyframe_name}! (converged, error={max_joint_error:.3f})")
        else:
            print(f"  ⏱ Advancing {keyframe_name} (timeout at step {steps_at_keyframe})")
        controller.advance_to_next_keyframe()

    # 3. Construct Action
    action = np.zeros(8)
    action[:7] = target_position

    # --- GRIPPER CONTROL (improved) ---
    # Critical: keep gripper fully closed during transport
    if keyframe_name in ["grasp_closed", "lift", "transport", "place"]:
        action[7] = 0.0  # Fully closed (0.0 = closed, 0.04 = open)
    elif keyframe_name in ["place_open"]:
        action[7] = 0.04  # Fully open to release
    else:
        action[7] = 0.04  # Default: open for approach

    return action, controller.is_sequence_complete()


def collect_episode(env: FrankaPickPlaceEnv, hindered: bool, max_steps: int, add_noise: bool = True) -> tuple[EpisodeBuffer, bool]:
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

    # NOISE STRATEGY for trajectory diversity:
    # - Add noise to APPROACH path (pre_grasp) for varied trajectories
    # - Use ACTUAL position for GRASP to ensure accurate picking
    # - This gives us diverse data while maintaining success rate
    noisy_object_pos = None
    if add_noise:
        # Add XY noise for diverse approaches (not Z - keep height accurate)
        xy_noise_std = 0.02  # 2cm standard deviation in XY
        xy_noise = np.random.normal(0, xy_noise_std, 2)
        noisy_object_pos = object_pos.copy()
        noisy_object_pos[:2] += xy_noise
        print(f"Starting episode | Target: {target_color} at {object_pos[:2]}")
        print(f"  Approach noise: {xy_noise} → pre_grasp targets {noisy_object_pos[:2]}")
        print(f"  Grasp: Will use ACTUAL position (no noise) for accurate picking")
    else:
        print(f"Starting episode | Target: {target_color} at {object_pos} | NO NOISE (deterministic)")

    # Create kinematics helper for adaptive IK
    kin_helper = KinematicsHelper(env.model, site_name="gripper")

    # Compute adaptive keyframes: 
    # - noisy_object_pos used for pre_grasp (diverse approach)
    # - object_pos (actual) used for grasp (accurate picking)
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
                print(f"  ✓ Cube in box! Episode SUCCESSFUL at step {step_idx}")
            else:
                print(f"  ✗ Cube NOT in box (dist={horizontal_dist:.3f}m, z={obj_pos[2]:.3f}m) - FAILED")
            break
    
    # Final success check
    obj_pos = env.data.site_xpos[target_site_id]
    horizontal_dist = np.linalg.norm(obj_pos[:2] - env.bin_position[:2])
    success = horizontal_dist < env.bin_radius and obj_pos[2] < 0.08

    buffer.meta.update({
        "episode_length": len(buffer.actions), 
        "sequence_complete": bool(sequence_complete),  # Convert to native Python bool
        "success": bool(success)  # Convert to native Python bool
    })
    return buffer, bool(success)


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

    # Track success statistics
    successful_episodes = start_episode
    total_attempts = 0
    failed_attempts = 0
    
    print(f"\n{'='*60}")
    print(f"Collecting {args.episodes} SUCCESSFUL episodes")
    print(f"(Failed attempts will not be saved)")
    print(f"{'='*60}\n")

    # Continue collecting until we have enough successful episodes
    while successful_episodes < args.episodes:
        total_attempts += 1
        attempt_num = total_attempts - start_episode
        
        hindered = rng.random() < hindered_fraction
        add_noise = not args.no_noise  # Add noise by default, unless --no-noise is specified
        
        buffer, success = collect_episode(env, hindered=hindered, max_steps=args.max_steps, add_noise=add_noise)
        
        if success:
            # Save successful episode
            buffer.save(dataset_root, successful_episodes)
            metadata.append({
                "episode": f"episode_{successful_episodes:04d}",
                "length": len(buffer.actions),
                "hindered": hindered,
                "instruction": buffer.instruction,
                "target_color": buffer.meta.get("target_color"),
                "success": True,
            })
            print(f"✓ Saved episode {successful_episodes:04d} | steps={len(buffer.actions)} | hindered={hindered}")
            successful_episodes += 1
        else:
            # Don't save failed episode
            failed_attempts += 1
            print(f"✗ Failed attempt {attempt_num} - NOT saved (cube not in box)")
        
        # Print progress
        success_rate = (successful_episodes - start_episode) / attempt_num * 100 if attempt_num > 0 else 0
        print(f"Progress: {successful_episodes}/{args.episodes} successful ({failed_attempts} failed, {success_rate:.1f}% success rate)\n")

    write_metadata(dataset_root, metadata, train_fraction=args.train_fraction)
    env.close()
    
    print(f"\n{'='*60}")
    print(f"Dataset Collection Complete!")
    print(f"{'='*60}")
    print(f"Successful episodes: {len(metadata)}")
    print(f"Total attempts: {total_attempts}")
    print(f"Failed attempts: {failed_attempts}")
    print(f"Success rate: {(len(metadata) - start_episode) / (total_attempts - start_episode) * 100:.1f}%")
    print(f"Saved to: {dataset_root}")
    print(f"Train/val split: {int(args.train_fraction * len(metadata))}/{len(metadata) - int(args.train_fraction * len(metadata))} episodes")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

