"""Kinematics and control utilities for the Franka MuJoCo environment."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency for import-time checks
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None  # type: ignore


@dataclass(frozen=True)
class JointVelocityControllerConfig:
    kp: float = 150.0
    kd: float = 20.0
    max_velocity: float = 1.0


class JointVelocityController:
    """Simple PD controller for joint-velocity commands."""

    def __init__(self, config: Optional[JointVelocityControllerConfig] = None) -> None:
        self.config = config or JointVelocityControllerConfig()

    def __call__(self, target: np.ndarray, current: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        error = target - current
        damping = -self.config.kd * velocity
        command = self.config.kp * error + damping
        return np.clip(command, -self.config.max_velocity, self.config.max_velocity)


@dataclass(frozen=True)
class ActuatedJointInfo:
    """Indices describing the actuated arm joints."""

    joint_ids: np.ndarray
    qpos_indices: np.ndarray
    dof_indices: np.ndarray
    limits: np.ndarray


def infer_actuated_joints(model: "mujoco.MjModel", expected: int = 7) -> ActuatedJointInfo:
    """Infer indices for the actuated joints using the actuator mapping."""

    if mujoco is None:  # pragma: no cover - guard for documentation builds
        raise ImportError("MuJoCo is required to infer actuated joints")

    joint_ids: list[int] = []
    qpos_indices: list[int] = []
    dof_indices: list[int] = []

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        if joint_id < 0:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if joint_id in joint_ids:
            continue
        joint_ids.append(joint_id)
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
        if len(joint_ids) == expected:
            break

    if len(joint_ids) < expected:
        raise RuntimeError(
            f"Expected at least {expected} actuated joints, found {len(joint_ids)}."
        )

    limits = np.zeros((len(joint_ids), 2), dtype=np.float64)
    for idx, joint_id in enumerate(joint_ids):
        limited = int(model.jnt_limited[joint_id]) if hasattr(model, "jnt_limited") else 0
        if limited:
            limits[idx] = model.jnt_range[joint_id]
        else:
            limits[idx] = np.array([-np.inf, np.inf])

    return ActuatedJointInfo(
        joint_ids=np.asarray(joint_ids, dtype=np.int32),
        qpos_indices=np.asarray(qpos_indices, dtype=np.int32),
        dof_indices=np.asarray(dof_indices, dtype=np.int32),
        limits=limits,
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[1:] *= -1
    return result


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    angle = 2.0 * math.acos(np.clip(quat[0], -1.0, 1.0))
    s = math.sqrt(max(1.0 - quat[0] * quat[0], 0.0))
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = quat[1:] / s
    return axis * angle


class KinematicsHelper:
    """Provides FK/IK and Jacobian utilities for the Franka arm."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        *,
        site_name: str = "gripper",
        joint_info: Optional[ActuatedJointInfo] = None,
    ) -> None:
        if mujoco is None:  # pragma: no cover
            raise ImportError("MuJoCo is required for KinematicsHelper")

        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = model.qpos0
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in model")

        self.joint_info = joint_info or infer_actuated_joints(model)

    def forward_kinematics(self, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        self._apply_configuration(q)
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        # Convert rotation matrix to quaternion (MuJoCo 3.x API)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return pos, quat

    def analytic_jacobian(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        self._apply_configuration(q)
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        cols = self.joint_info.dof_indices
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def finite_difference_jacobian(self, q: Sequence[float], eps: float = 1e-6) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        base_pos, base_quat = self.forward_kinematics(q)
        num_joints = q.shape[0]
        jac = np.zeros((6, num_joints), dtype=np.float64)
        for j in range(num_joints):
            dq = np.zeros_like(q)
            dq[j] = eps
            pos, quat = self.forward_kinematics(q + dq)
            dpos = (pos - base_pos) / eps
            dquat = _quat_to_axis_angle(_quat_multiply(_quat_conjugate(base_quat), quat)) / eps
            jac[:3, j] = dpos
            jac[3:, j] = dquat
        return jac

    def manipulability(self, q: Sequence[float]) -> float:
        jac = self.analytic_jacobian(q)[:3, :]
        gram = jac @ jac.T
        value = float(np.linalg.det(gram))
        return max(value, 0.0)

    def verify_jacobian(self, q: Sequence[float], eps: float = 1e-6, tol: float = 1e-3) -> bool:
        """Verify analytic Jacobian against finite-difference approximation."""
        jac_analytic = self.analytic_jacobian(q)
        jac_fd = self.finite_difference_jacobian(q, eps=eps)
        error = np.linalg.norm(jac_analytic - jac_fd)
        return error < tol

    def inverse_kinematics(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        *,
        initial_q: Optional[Sequence[float]] = None,
        max_iters: int = 200,
        tol_pos: float = 1e-4,
        tol_ori: float = math.radians(2.0),
        damping: float = 1e-4,
        step_size: float = 1.0,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
    ) -> np.ndarray:
        """Damped least-squares IK with weighted position and orientation errors."""
        q = np.asarray(initial_q if initial_q is not None else self.model.qpos0[self.joint_info.qpos_indices], dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = _quat_normalize(np.asarray(target_quat, dtype=np.float64))

        for _ in range(max_iters):
            pos, quat = self.forward_kinematics(q)
            pos_error = target_pos - pos
            orient_error = _quat_to_axis_angle(_quat_multiply(_quat_conjugate(quat), target_quat))
            
            # Check convergence
            if np.linalg.norm(pos_error) < tol_pos and np.linalg.norm(orient_error) < tol_ori:
                return q
            
            # Weighted error for prioritization
            weighted_pos_error = position_weight * pos_error
            weighted_ori_error = orientation_weight * orient_error
            error = np.concatenate([weighted_pos_error, weighted_ori_error])
            
            jac = self.analytic_jacobian(q)
            # Apply weights to Jacobian rows as well
            weight_matrix = np.diag([position_weight] * 3 + [orientation_weight] * 3)
            weighted_jac = weight_matrix @ jac
            
            jtj = weighted_jac.T @ weighted_jac + damping * np.eye(jac.shape[1])
            dq = step_size * np.linalg.solve(jtj, weighted_jac.T @ error)
            q = q + dq
            q = self._clamp_to_limits(q)

        raise RuntimeError("IK failed to converge within the allotted iterations")
    
    def inverse_kinematics_ccd(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        *,
        initial_q: Optional[Sequence[float]] = None,
        max_iters: int = 100,
        tol_pos: float = 1e-3,
        tol_ori: float = math.radians(5.0),
        position_weight: float = 3.0,
        orientation_weight: float = 1.0,
    ) -> np.ndarray:
        """Cyclic Coordinate Descent IK - optimizes one joint at a time."""
        q = np.asarray(initial_q if initial_q is not None else self.model.qpos0[self.joint_info.qpos_indices], dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = _quat_normalize(np.asarray(target_quat, dtype=np.float64))
        
        best_q = q.copy()
        best_error = float('inf')
        
        for iteration in range(max_iters):
            # Cycle through all joints
            for joint_idx in range(len(q)):
                # Current end-effector pose
                pos, quat = self.forward_kinematics(q)
                pos_error = target_pos - pos
                orient_error = _quat_to_axis_angle(_quat_multiply(_quat_conjugate(quat), target_quat))
                
                # Weighted total error
                total_error = (position_weight * np.linalg.norm(pos_error) + 
                             orientation_weight * np.linalg.norm(orient_error))
                
                # Track best solution
                if total_error < best_error:
                    best_error = total_error
                    best_q = q.copy()
                
                # Check convergence
                if np.linalg.norm(pos_error) < tol_pos and np.linalg.norm(orient_error) < tol_ori:
                    return q
                
                # Optimize this joint using gradient
                current_val = q[joint_idx]
                delta = 0.01  # Small step for numerical gradient
                
                # Try small perturbations
                best_local_error = total_error
                best_local_dq = 0.0
                
                for dq in [-delta, delta]:
                    q[joint_idx] = current_val + dq
                    q[joint_idx] = self._clamp_to_limits(q)[joint_idx]
                    
                    pos_test, quat_test = self.forward_kinematics(q)
                    pos_err_test = target_pos - pos_test
                    ori_err_test = _quat_to_axis_angle(_quat_multiply(_quat_conjugate(quat_test), target_quat))
                    
                    error_test = (position_weight * np.linalg.norm(pos_err_test) + 
                                orientation_weight * np.linalg.norm(ori_err_test))
                    
                    if error_test < best_local_error:
                        best_local_error = error_test
                        best_local_dq = dq
                
                # Apply best local change
                if best_local_dq != 0.0:
                    q[joint_idx] = current_val + best_local_dq * 0.5  # Damped step
                else:
                    q[joint_idx] = current_val
                
                q = self._clamp_to_limits(q)
        
        # Return best found solution even if not converged
        return best_q

    def inverse_kinematics_staged(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        *,
        initial_q: Optional[Sequence[float]] = None,
        horizontal_distance: float = 0.0,
        max_iters: int = 100,
        damping: float = 0.01,
    ) -> np.ndarray:
        """Staged IK: prioritize position when far, balance when close."""
        # Adaptive weighting based on distance
        if horizontal_distance > 0.1:  # Far: 100% position priority
            pos_weight, ori_weight = 10.0, 0.1
        elif horizontal_distance > 0.06:  # Medium: gentle orientation
            pos_weight, ori_weight = 5.0, 0.5
        else:  # Close: balanced
            pos_weight, ori_weight = 1.0, 1.0
        
        return self.inverse_kinematics(
            target_pos, target_quat,
            initial_q=initial_q,
            max_iters=max_iters,
            damping=damping,
            position_weight=pos_weight,
            orientation_weight=ori_weight,
        )

    def ik_velocity_step(
        self,
        current_q: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        damping: float = 0.05,
        gain: float = 1.0,
    ) -> np.ndarray:
        """Single IK iteration for velocity control at high frequency."""
        pos, quat = self.forward_kinematics(current_q)
        pos_error = target_pos - pos
        orient_error = _quat_to_axis_angle(
            _quat_multiply(_quat_conjugate(quat), target_quat)
        )
        
        # Weighted 6D error
        weighted_error = np.concatenate([
            position_weight * pos_error,
            orientation_weight * orient_error
        ])
        
        # Damped least-squares Jacobian
        jac = self.analytic_jacobian(current_q)
        weight_matrix = np.diag([position_weight] * 3 + [orientation_weight] * 3)
        weighted_jac = weight_matrix @ jac
        jac_damped = weighted_jac.T @ np.linalg.inv(
            weighted_jac @ weighted_jac.T + damping * np.eye(6)
        )
        
        # Joint velocity command
        joint_vel = gain * jac_damped @ weighted_error
        return joint_vel

    def null_space_control(
        self,
        current_q: np.ndarray,
        jacobian: np.ndarray,
        home_config: np.ndarray,
        gains: np.ndarray,
    ) -> np.ndarray:
        """Compute null-space control for secondary objectives."""
        # Damped pseudo-inverse for null-space projection
        jac_damped = jacobian.T @ np.linalg.inv(
            jacobian @ jacobian.T + 0.05 * np.eye(6)
        )
        null_proj = np.eye(7) - jac_damped @ jacobian
        
        # Secondary objective: return to home configuration
        home_error = home_config - current_q
        null_cmd = null_proj @ (gains * home_error)
        return null_cmd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_configuration(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.joint_info.qpos_indices] = q

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        for i, (lower, upper) in enumerate(self.joint_info.limits):
            if np.isfinite(lower):
                q[i] = max(q[i], lower)
            if np.isfinite(upper):
                q[i] = min(q[i], upper)
        return q


class KeyframeController:
    """Controller for smooth interpolation between keyframe joint configurations."""
    
    def __init__(
        self,
        keyframes: dict[str, np.ndarray],
        convergence_threshold: float = 0.05,  # radians
        velocity_threshold: float = 0.1,  # rad/s
    ):
        """
        Initialize keyframe controller.

        Args:
            keyframes: Dictionary mapping keyframe names to joint configurations
            convergence_threshold: Joint position error threshold to consider converged
            velocity_threshold: Joint velocity threshold to consider stationary
        """
        self.keyframes = keyframes
        self.convergence_threshold = convergence_threshold
        self.velocity_threshold = velocity_threshold

        # Current state
        self.current_keyframe_idx = 0
        self.keyframe_sequence = []  # Will be set via set_sequence
        self.sequence_complete = False  # Track if final keyframe has been completed
        
    def set_sequence(self, sequence: list[str]) -> None:
        """Set the sequence of keyframes to execute."""
        self.keyframe_sequence = sequence
        self.current_keyframe_idx = 0
        self.sequence_complete = False
        
    def get_current_target(self) -> tuple[str, np.ndarray]:
        """Get the current target keyframe name and joint configuration."""
        if self.current_keyframe_idx >= len(self.keyframe_sequence):
            # Return last keyframe if sequence is complete
            name = self.keyframe_sequence[-1]
            return name, self.keyframes[name]
        
        name = self.keyframe_sequence[self.current_keyframe_idx]
        return name, self.keyframes[name]
    
    def check_convergence(
        self,
        current_q: np.ndarray,
        current_qvel: np.ndarray,
        target_q: np.ndarray,
    ) -> bool:
        """
        Check if robot has converged to target configuration.
        
        Args:
            current_q: Current joint positions
            current_qvel: Current joint velocities
            target_q: Target joint positions
            
        Returns:
            True if converged (position error small AND velocity small)
        """
        position_error = np.linalg.norm(target_q - current_q)
        velocity_norm = np.linalg.norm(current_qvel)
        
        position_converged = position_error < self.convergence_threshold
        velocity_converged = velocity_norm < self.velocity_threshold
        
        return position_converged and velocity_converged
    
    def advance_to_next_keyframe(self) -> bool:
        """
        Advance to the next keyframe in the sequence.

        Returns:
            True if advanced, False if already at last keyframe
        """
        if self.current_keyframe_idx < len(self.keyframe_sequence) - 1:
            self.current_keyframe_idx += 1
            return True
        else:
            # Attempted to advance from last keyframe - mark sequence complete
            self.sequence_complete = True
            return False
    
    def is_sequence_complete(self) -> bool:
        """Check if we've completed the entire keyframe sequence."""
        return self.sequence_complete
    
    def get_progress(self) -> tuple[int, int]:
        """Get current progress (current_idx, total_keyframes)."""
        return self.current_keyframe_idx, len(self.keyframe_sequence)


def compute_pick_place_keyframes(
    kin_helper: KinematicsHelper,
    object_pos: np.ndarray,
    bin_pos: np.ndarray,
    base_keyframes: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute adapted keyframes for a specific object and bin position.
    
    This uses base keyframes and adjusts them based on the actual object
    and bin positions to provide variety while maintaining reliability.
    
    Args:
        kin_helper: Kinematics helper for IK
        object_pos: [x, y, z] position of object to pick
        bin_pos: [x, y, z] position of bin center
        base_keyframes: Base keyframe configurations to adapt
        
    Returns:
        Dictionary of adapted keyframes
    """
    # For now, just return the base keyframes
    # In the future, we can add adaptive IK to adjust for object positions
    # while maintaining the proven joint space trajectories as a fallback
    return base_keyframes.copy()


__all__ = [
    "ActuatedJointInfo",
    "JointVelocityController",
    "JointVelocityControllerConfig",
    "KinematicsHelper",
    "KeyframeController",
    "compute_pick_place_keyframes",
    "infer_actuated_joints",
]

# Export quaternion utilities for orientation control
def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation."""
    return _quat_to_axis_angle(quat)

def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    return _quat_multiply(a, b)

def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate."""
    return _quat_conjugate(quat)
