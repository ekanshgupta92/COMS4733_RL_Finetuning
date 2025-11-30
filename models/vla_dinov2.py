"""Vision-Language-Action model built from DINOv2 and BERT backbones."""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class VLADinoV2Config:
    vision_encoder: str = "facebook/dinov2-base"
    language_encoder: str = "bert-base-uncased"
    fusion_hidden_dim: int = 512
    fusion_layers: int = 2
    action_dim: int = 7
    max_instruction_tokens: int = 64
    freeze_vision: bool = True
    freeze_language: bool = False
    dropout: float = 0.1
    history_length: int = 5  # Number of past actions to condition on
    use_object_detection: bool = True  # Multi-task learning with object detection
    object_detection_weight: float = 10.0  # Weight for object detection loss


class VLADinoV2Policy(nn.Module):
    """Minimal VLA policy combining pretrained vision/language encoders."""

    def __init__(self, config: VLADinoV2Config) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = AutoModel.from_pretrained(config.vision_encoder)
        self.language_encoder = AutoModel.from_pretrained(config.language_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_encoder)

        vision_dim = self.vision_encoder.config.hidden_size
        language_dim = self.language_encoder.config.hidden_size

        self.vision_projection = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, config.fusion_hidden_dim),
        )
        self.language_projection = nn.Sequential(
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, config.fusion_hidden_dim),
        )
        self.proprio_projection = nn.Sequential(
            nn.Linear(8, config.fusion_hidden_dim),  # 7 joints + 1 timestep
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
        )
        
        # Action history encoder for temporal context
        self.action_history_encoder = nn.Sequential(
            nn.Linear(config.action_dim * config.history_length, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.fusion_hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_hidden_dim,
            nhead=8,
            dim_feedforward=config.fusion_hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=config.fusion_layers)

        # self.head = nn.Sequential(
        #     nn.LayerNorm(config.fusion_hidden_dim),
        #     nn.Linear(config.fusion_hidden_dim, config.action_dim),
        # )

        # Arm Head: Predicts 7 joint positions
        self.arm_head = nn.Sequential(
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.Linear(config.fusion_hidden_dim, 7),
            # Optional: Tanh if you normalized actions to [-1, 1]
            # nn.Tanh() 
        )

        # Gripper Head: Predicts Logits for Binary Classification (Open/Close)
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.Linear(config.fusion_hidden_dim, 1), 
        )

        # Value head for RL (critic)
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.Linear(config.fusion_hidden_dim, 1),
        )

        # Object detection head for multi-task learning
        # Predicts 2D object position in image space (normalized [0,1])
        if config.use_object_detection:
            # Explicit color embedding to handle multi-object scenes
            # BERT embeddings are 99.7% similar across colors - too weak!
            self.color_names = ["red", "green", "blue", "yellow", "purple"]
            self.color_embedding = nn.Embedding(len(self.color_names), 64)
            
            self.object_detection_head = nn.Sequential(
                nn.LayerNorm(config.fusion_hidden_dim + 64),  # +64 for color embedding
                nn.Linear(config.fusion_hidden_dim + 64, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, 2),  # (x, y) in normalized image coordinates
                nn.Sigmoid(),  # Ensure output in [0, 1]
            )
        else:
            self.object_detection_head = None
            self.color_embedding = None

        if config.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if config.freeze_language:
            for param in self.language_encoder.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        rgb_static: torch.Tensor,
        instruction: List[str],
        proprio: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict joint velocities from multimodal inputs.

        Args:
            rgb_static: Batch of RGB images ``(B, C, H, W)`` normalized to ``[0, 1]``.
            instruction: Batch of natural-language strings.
            proprio: Batch of proprioceptive vectors ``(B, 8)`` - 7 joints + 1 timestep.
            action_history: Batch of past actions ``(B, history_length, action_dim)`` or None.
        """
        pooled = self._get_fused_features(rgb_static, instruction, proprio, action_history)
        # Predict separate components
        arm_preds = self.arm_head(pooled)       # (B, 7)
        gripper_logits = self.gripper_head(pooled) # (B, 1)
        
        # Return concatenated vector
        return torch.cat([arm_preds, gripper_logits], dim=1)
    
    # Helper to get actual actionable commands (inference time)
    def predict_action(self, rgb_static, instruction, proprio, action_history=None):
        out = self.forward(rgb_static, instruction, proprio, action_history)
        arm = out[:, :7]
        gripper_logit = out[:, 7:]
        
        # Convert logit to probability, then to action
        gripper_prob = torch.sigmoid(gripper_logit)
        # Threshold: if prob > 0.5 -> Open (0.04), else Closed (0.0)
        # 1 = Open, 0 = Closed.
        gripper_action = (gripper_prob > 0.5).float() * 0.04
        
        return torch.cat([arm, gripper_action], dim=1)

    def _get_fused_features(
        self,
        rgb_static: torch.Tensor,
        instruction: List[str],
        proprio: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get fused multimodal features for actor and critic."""
        device = rgb_static.device
        image_embedding = self._encode_image(rgb_static)
        text_embedding = self._encode_text(instruction, device=device)
        proprio_embedding = self.proprio_projection(proprio)

        # Encode action history if provided
        if action_history is not None:
            # Flatten history: (B, H, D) -> (B, H*D)
            batch_size = action_history.shape[0]
            history_flat = action_history.reshape(batch_size, -1)
            history_embedding = self.action_history_encoder(history_flat)

            # Fuse all modalities including history
            fusion_tokens = torch.stack([
                image_embedding,
                text_embedding,
                proprio_embedding,
                history_embedding
            ], dim=1)
        else:
            # Backward compatibility: no history
            fusion_tokens = torch.stack([image_embedding, text_embedding, proprio_embedding], dim=1)

        fused = self.fusion(fusion_tokens)
        pooled = fused.mean(dim=1)
        return pooled

    def get_value(
        self,
        rgb_static: torch.Tensor,
        instruction: List[str],
        proprio: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get state value prediction from critic.

        Args:
            rgb_static: Batch of RGB images ``(B, C, H, W)`` normalized to ``[0, 1]``.
            instruction: Batch of natural-language strings.
            proprio: Batch of proprioceptive vectors ``(B, 8)`` - 7 joints + 1 timestep.
            action_history: Batch of past actions ``(B, history_length, action_dim)`` or None.

        Returns:
            State values of shape ``(B, 1)``.
        """
        pooled = self._get_fused_features(rgb_static, instruction, proprio, action_history)
        return self.value_head(pooled)

    def predict_object_position(
        self,
        rgb_static: torch.Tensor,
        instruction: List[str],
        proprio: torch.Tensor,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict 2D object position from visual observations.

        Args:
            rgb_static: Batch of RGB images ``(B, C, H, W)`` normalized to ``[0, 1]``.
            instruction: Batch of natural-language strings.
            proprio: Batch of proprioceptive vectors ``(B, 8)`` - 7 joints + 1 timestep.
            action_history: Batch of past actions ``(B, history_length, action_dim)`` or None.

        Returns:
            Object positions: (B, 2) - (x, y) in normalized [0, 1] image coordinates
        """
        if self.object_detection_head is None:
            raise RuntimeError("Object detection head not initialized. Set use_object_detection=True in config.")
        pooled = self._get_fused_features(rgb_static, instruction, proprio, action_history)
        
        # Extract color from instruction and add explicit color embedding
        # This provides a STRONG signal since BERT embeddings are 99.7% similar across colors
        color_ids = self._extract_color_ids(instruction)
        color_embed = self.color_embedding(color_ids)  # (B, 64)
        
        # Concatenate fused features with color embedding
        detection_input = torch.cat([pooled, color_embed], dim=-1)  # (B, fusion_dim + 64)
        return self.object_detection_head(detection_input)

    def _extract_color_ids(self, instructions: List[str]) -> torch.Tensor:
        """Extract color IDs from instruction strings.
        
        Args:
            instructions: List of instruction strings (e.g., "Pick up the red sphere...")
            
        Returns:
            Tensor of color IDs (B,) - indices into self.color_names
        """
        color_ids = []
        for instruction in instructions:
            instruction_lower = instruction.lower()
            color_id = 0  # Default to red
            for idx, color in enumerate(self.color_names):
                if color in instruction_lower:
                    color_id = idx
                    break
            color_ids.append(color_id)
        return torch.tensor(color_ids, dtype=torch.long, device=next(self.parameters()).device)

    def get_action_and_value(
        self,
        rgb_static: torch.Tensor,
        instruction: List[str],
        proprio: torch.Tensor,
        action_history: torch.Tensor | None = None,
        action_std: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value for RL training.

        Args:
            rgb_static: Batch of RGB images ``(B, C, H, W)`` normalized to ``[0, 1]``.
            instruction: Batch of natural-language strings.
            proprio: Batch of proprioceptive vectors ``(B, 8)`` - 7 joints + 1 timestep.
            action_history: Batch of past actions ``(B, history_length, action_dim)`` or None.
            action_std: Standard deviation for action noise.

        Returns:
            Tuple of (action, log_prob, value) where:
                - action: Sampled actions of shape ``(B, action_dim)``
                - log_prob: Log probabilities of shape ``(B,)``
                - value: State values of shape ``(B, 1)``
        """
        pooled = self._get_fused_features(rgb_static, instruction, proprio, action_history)
        action_mean = self.head(pooled)
        value = self.value_head(pooled)

        # Sample action from Gaussian policy
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions

        return action, log_prob, value

    def evaluate_actions(
        self,
        rgb_static: torch.Tensor,
        instruction: List[str],
        proprio: torch.Tensor,
        actions: torch.Tensor,
        action_history: torch.Tensor | None = None,
        action_std: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Args:
            rgb_static: Batch of RGB images ``(B, C, H, W)`` normalized to ``[0, 1]``.
            instruction: Batch of natural-language strings.
            proprio: Batch of proprioceptive vectors ``(B, 8)`` - 7 joints + 1 timestep.
            actions: Actions to evaluate of shape ``(B, action_dim)``.
            action_history: Batch of past actions ``(B, history_length, action_dim)`` or None.
            action_std: Standard deviation for action noise.

        Returns:
            Tuple of (log_prob, value, entropy) where:
                - log_prob: Log probabilities of shape ``(B,)``
                - value: State values of shape ``(B, 1)``
                - entropy: Policy entropy of shape ``(B,)``
        """
        pooled = self._get_fused_features(rgb_static, instruction, proprio, action_history)
        action_mean = self.head(pooled)
        value = self.value_head(pooled)

        # Compute log prob and entropy under current policy
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)  # Sum over action dimensions
        entropy = dist.entropy().sum(dim=-1)  # Sum over action dimensions

        return log_prob, value, entropy

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("Images must have shape (B, C, H, W)")

        outputs = self.vision_encoder(pixel_values=images)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.vision_projection(features)

    def _encode_text(self, instructions: Iterable[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=self.config.max_instruction_tokens,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.language_encoder(**tokens)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.language_projection(features)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def to_checkpoint(self) -> Dict[str, Any]:
        return {
            "model_state": self.state_dict(),
            "config": dataclass_to_dict(self.config),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> "VLADinoV2Policy":
        config = dict_to_dataclass(VLADinoV2Config, checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state"])
        return model


def dataclass_to_dict(config: VLADinoV2Config) -> Dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}


def dict_to_dataclass(cls, values: Dict[str, Any]):
    return cls(**values)


__all__ = ["VLADinoV2Config", "VLADinoV2Policy"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utilities for the VLADinoV2Policy module")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a self-test forward pass with random dummy inputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size to use when executing the self-test.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image resolution to use for the dummy forward pass.",
    )
    return parser.parse_args()


def _run_self_test(batch_size: int, image_size: int) -> None:
    config = VLADinoV2Config()
    model = VLADinoV2Policy(config)
    model.eval()

    with torch.no_grad():
        dummy_rgb = torch.randn(batch_size, 3, image_size, image_size)
        dummy_proprio = torch.randn(batch_size, 7)
        dummy_instructions = ["test instruction"] * batch_size
        outputs = model(rgb_static=dummy_rgb, proprio=dummy_proprio, instruction=dummy_instructions)

    print("Self-test completed. Output shape:", tuple(outputs.shape))


def main() -> None:
    args = _parse_args()
    if args.test:
        _run_self_test(args.batch_size, args.image_size)
    else:
        print("No action specified. Use --test to run the built-in self-test.")


if __name__ == "__main__":  # pragma: no cover - CLI convenience wrapper
    main()
