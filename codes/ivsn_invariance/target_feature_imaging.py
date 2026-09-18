"""Rendering of isolated original/transformed target-image pairs."""

from pathlib import Path
from typing import Tuple

from PIL import Image

from .data import load_rgba
from .domain import TransformSpec
from .feature_imaging import apply_transform_with_fixed_noise
from .imaging import (
    alpha_paste_rgb,
    apply_transform_rgba,
    rescale_image,
    shift_cutout_to_center,
)
from .runtime import OBJ_SIZE


def render_paired_targets(
        target_path: Path,
        transformed_spec: TransformSpec,
        args,
        noise_seed: int,
    ) -> Tuple[Image.Image, Image.Image]:
    """Render target-only images differing solely by the transformation."""
    target_rgba = rescale_image(shift_cutout_to_center(load_rgba(target_path)))
    original_rgba = apply_transform_rgba(
        target_rgba,
        TransformSpec(),
        args,
        apply_smoothing=args.smooth_target,
    )
    transformed_rgba = apply_transform_with_fixed_noise(
        target_rgba,
        transformed_spec,
        args,
        apply_smoothing=args.smooth_target,
        noise_seed=noise_seed,
    )
    center = (OBJ_SIZE // 2, OBJ_SIZE // 2)
    original = Image.new('RGB', (OBJ_SIZE, OBJ_SIZE), (128, 128, 128))
    transformed = original.copy()
    original = alpha_paste_rgb(original, original_rgba, center)
    transformed = alpha_paste_rgb(transformed, transformed_rgba, center)
    return original, transformed
