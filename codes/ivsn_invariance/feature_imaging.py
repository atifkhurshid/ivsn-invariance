"""Controlled image pairs used only by the feature-analysis experiment."""

from pathlib import Path
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from . import runtime
from .data import load_rgba
from .domain import TransformSpec
from .imaging import (
    alpha_paste_rgb,
    apply_transform_rgba,
    render_cue,
    rescale_image,
    shift_cutout_to_center,
)
from .runtime import IMAGE_SIZE


def jittered_position(
        position: Tuple[int, int],
        rng: random.Random,
    ) -> Tuple[int, int]:
    x, y = position
    x += rng.choice([-1, 1]) * runtime.EPSILON
    y += rng.choice([-1, 1]) * runtime.EPSILON
    return int(round(x)), int(round(y))


def apply_transform_with_fixed_noise(
        image: Image.Image,
        spec: TransformSpec,
        args,
        apply_smoothing: bool,
        noise_seed: int,
    ) -> Image.Image:
    """Use legacy transformation code with isolated deterministic NumPy state."""
    random_state = np.random.get_state()
    np.random.seed(noise_seed)
    try:
        return apply_transform_rgba(
            image,
            spec,
            args,
            apply_smoothing=apply_smoothing,
        )
    finally:
        np.random.set_state(random_state)


def render_paired_search_displays(
        target_path: Path,
        distractor_paths: List[Path],
        target_position: int,
        transformed_target_spec: TransformSpec,
        distractor_transforms: List[TransformSpec],
        args,
        position_rng: Optional[random.Random] = None,
        noise_seed: int = 0,
    ) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """Render displays differing only in the target transformation."""
    if len(distractor_paths) != len(distractor_transforms):
        raise ValueError('Each distractor path must have a matching transform.')
    if runtime.POSITIONS is None or runtime.N_POSITIONS is None:
        raise RuntimeError('Runtime geometry must be configured before rendering.')
    if not 0 <= target_position < runtime.N_POSITIONS:
        raise ValueError(
            f'Target position {target_position} is outside '
            f'0..{runtime.N_POSITIONS - 1}.'
        )
    expected_distractors = runtime.N_POSITIONS - 1
    if len(distractor_paths) != expected_distractors:
        raise ValueError(
            f'Expected {expected_distractors} distractors, got '
            f'{len(distractor_paths)}.'
        )

    if position_rng is None:
        position_rng = random.Random()
    centers = [
        jittered_position(position, position_rng)
        for position in runtime.POSITIONS
    ]
    baseline = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
    transformed = baseline.copy()

    target_rgba = rescale_image(shift_cutout_to_center(load_rgba(target_path)))
    target_original = apply_transform_rgba(
        target_rgba,
        TransformSpec(),
        args,
        apply_smoothing=args.smooth_target,
    )
    target_transformed = apply_transform_with_fixed_noise(
        target_rgba,
        transformed_target_spec,
        args,
        apply_smoothing=args.smooth_target,
        noise_seed=noise_seed,
    )
    target_center = centers[target_position]
    baseline = alpha_paste_rgb(baseline, target_original, target_center)
    transformed = alpha_paste_rgb(transformed, target_transformed, target_center)

    remaining_positions = [
        index
        for index in range(runtime.N_POSITIONS)
        if index != target_position
    ]
    for distractor_path, position_index, spec in zip(
            distractor_paths,
            remaining_positions,
            distractor_transforms,
        ):
        distractor_rgba = rescale_image(
            shift_cutout_to_center(load_rgba(distractor_path))
        )
        distractor = apply_transform_rgba(
            distractor_rgba,
            spec,
            args,
            apply_smoothing=args.smooth_distractors,
        )
        center = centers[position_index]
        baseline = alpha_paste_rgb(baseline, distractor, center)
        transformed = alpha_paste_rgb(transformed, distractor, center)

    return baseline, transformed, target_center


def render_feature_cue(cue_path: Path, args) -> Image.Image:
    """Render the untransformed cue exactly as the VGG search model does."""
    _, cue = render_cue(cue_path, args, TransformSpec())
    return cue
