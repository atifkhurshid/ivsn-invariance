"""Domain for IVSN invariance experiments."""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TransformSpec:
    rotation_deg: float = 0.0
    scale: float = 1.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    skew_x_deg: float = 0.0
    skew_y_deg: float = 0.0
    noise_std: float = 0.0
    blur_radius: float = 0.0
    sp_amount: float = 0.0
    sp_salt_ratio: float = 0.5


@dataclass
class BaseTrial:
    unique_id: int
    repeat_id: int
    trial_type: str
    target_category: str
    cue_path: str
    target_path: str
    distractor_paths: List[str]
    target_position: int
    distractor_rotations: List[float]


@dataclass
class Trial:
    condition_name: str
    condition_group: str
    condition_value: float
    unique_id: int
    repeat_id: int
    trial_type: str
    target_category: str
    cue_path: str
    target_path: str
    distractor_paths: List[str]
    target_position: int
    cue_transform: Dict[str, float]
    target_transform: Dict[str, float]
    distractor_transforms: List[Dict[str, float]]
