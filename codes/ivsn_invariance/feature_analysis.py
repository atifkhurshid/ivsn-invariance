"""Feature representation and effective receptive field analysis for VGG16."""

from math import sqrt
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

import numpy as np
from PIL import Image
from torchvision import models


DEFAULT_LAYER_WINDOWS = {16: 5, 23: 3, 30: 1}


@dataclass(frozen=True)
class LayerRegion:
    """A square region centered on an object's feature-map location."""

    layer: int
    window_size: int
    center_row: int
    center_col: int
    coordinates: Tuple[Tuple[int, int], ...]


class VGG16FeatureProbe(nn.Module):
    """Frozen ImageNet VGG16 that exposes selected intermediate activations.

    Layer numbers denote the number of modules included from
    ``torchvision.models.vgg16(...).features``. This matches the existing IVSN
    backbone, whose final representation is ``features[:30]``.
    """

    def __init__(
            self,
            layer_numbers: Sequence[int],
            device: str = 'cpu',
            weights: models.VGG16_Weights = models.VGG16_Weights.IMAGENET1K_V1,
        ):
        super().__init__()
        layers = tuple(sorted(set(int(layer) for layer in layer_numbers)))
        if not layers:
            raise ValueError('At least one VGG layer must be requested.')

        vgg_features = models.vgg16(weights=weights).features
        if layers[0] < 1 or layers[-1] > len(vgg_features):
            raise ValueError(
                f'Layer numbers must be between 1 and {len(vgg_features)}, got {layers}.'
            )

        self.layer_numbers = layers
        self.features = vgg_features[:layers[-1]].eval()
        self.device = torch.device(device)
        self.to(self.device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        weights_transform = weights.transforms()
        self.normalization_mean = weights_transform.mean
        self.normalization_std = weights_transform.std

    def preprocess(
            self,
            image: Image.Image,
            requires_grad: bool = False,
            image_size: int = 224,
        ) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=self.normalization_mean,
                std=self.normalization_std,
            ),
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        tensor.requires_grad_(requires_grad)
        return tensor

    def forward(self, tensor: torch.Tensor) -> Dict[int, torch.Tensor]:
        activations = {}
        requested = set(self.layer_numbers)
        output = tensor
        for layer_number, layer in enumerate(self.features, start=1):
            output = layer(output)
            if layer_number in requested:
                activations[layer_number] = output
        return activations

    @torch.no_grad()
    def activations(
            self,
            image: Image.Image,
            image_size: int = 224,
        ) -> Dict[int, torch.Tensor]:
        return self(self.preprocess(
            image,
            requires_grad=False,
            image_size=image_size,
        ))

    def activations_with_input_gradient(
            self,
            image: Image.Image,
            image_size: int = 224,
        ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        tensor = self.preprocess(
            image,
            requires_grad=True,
            image_size=image_size,
        )
        return tensor, self(tensor)


def parse_layer_windows(
        values: Sequence[str],
        defaults: Mapping[int, int] = DEFAULT_LAYER_WINDOWS,
    ) -> Dict[int, int]:
    """Parse ``LAYER:WINDOW`` specifications and validate odd windows."""
    if not values:
        result = dict(defaults)
    else:
        result = {}
        for value in values:
            try:
                layer_text, window_text = value.split(':', maxsplit=1)
                layer = int(layer_text)
                window = int(window_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid layer window '{value}'; expected LAYER:ODD_WINDOW."
                ) from exc
            result[layer] = window

    for layer, window in result.items():
        if layer < 1:
            raise ValueError(f'Layer numbers must be positive, got {layer}.')
        if window < 1 or window % 2 == 0:
            raise ValueError(
                f'Window for layer {layer} must be a positive odd number, got {window}.'
            )
    return dict(sorted(result.items()))


def image_point_to_feature_cell(
        point_xy: Tuple[int, int],
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
    ) -> Tuple[int, int]:
    """Map a pixel-space point to its containing feature-map cell."""
    x, y = point_xy
    image_width, image_height = image_size
    feature_height, feature_width = feature_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f'Invalid image size: {image_size}.')
    if feature_width <= 0 or feature_height <= 0:
        raise ValueError(f'Invalid feature-map size: {feature_size}.')

    col = int(np.floor(float(x) * feature_width / image_width))
    row = int(np.floor(float(y) * feature_height / image_height))
    row = min(max(row, 0), feature_height - 1)
    col = min(max(col, 0), feature_width - 1)
    return row, col


def centered_region(
        layer: int,
        window_size: int,
        point_xy: Tuple[int, int],
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
    ) -> LayerRegion:
    """Return an in-bounds region, shifting it at edges instead of shrinking it."""
    feature_height, feature_width = feature_size
    if window_size > feature_height or window_size > feature_width:
        raise ValueError(
            f'{window_size}x{window_size} window does not fit feature map '
            f'{feature_height}x{feature_width} at layer {layer}.'
        )

    center_row, center_col = image_point_to_feature_cell(
        point_xy,
        image_size=image_size,
        feature_size=feature_size,
    )
    half = window_size // 2
    row_start = min(max(center_row - half, 0), feature_height - window_size)
    col_start = min(max(center_col - half, 0), feature_width - window_size)
    coordinates = tuple(
        (row, col)
        for row in range(row_start, row_start + window_size)
        for col in range(col_start, col_start + window_size)
    )
    return LayerRegion(
        layer=layer,
        window_size=window_size,
        center_row=center_row,
        center_col=center_col,
        coordinates=coordinates,
    )


def vector_comparison_metrics(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> dict:
    """Return scale-sensitive and scale-normalized vector comparisons."""
    if first.shape != second.shape:
        raise ValueError(
            f'Feature vector shapes differ: {tuple(first.shape)} vs '
            f'{tuple(second.shape)}.'
        )
    difference = second - first
    euclidean = torch.linalg.vector_norm(difference)
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)
    if first_norm <= 1e-8 and second_norm <= 1e-8:
        cosine_distance = torch.zeros((), device=first.device)
    elif first_norm <= 1e-8 or second_norm <= 1e-8:
        cosine_distance = torch.ones((), device=first.device)
    else:
        cosine_distance = 1.0 - F.cosine_similarity(
            first.unsqueeze(0),
            second.unsqueeze(0),
            dim=1,
            eps=1e-8,
        )[0]
    return {
        'euclidean_distance': float(euclidean.detach().cpu()),
        'rms_euclidean_distance': float(
            euclidean.detach().cpu() / sqrt(first.numel())
        ),
        'relative_euclidean_distance': float(
            (euclidean / first_norm.clamp_min(1e-8)).detach().cpu()
        ),
        'cosine_distance': float(cosine_distance.detach().cpu()),
        'cosine_similarity': float((1.0 - cosine_distance).detach().cpu()),
        'dot_product': float(torch.dot(first, second).detach().cpu()),
    }


def region_feature_vectors(
        activation: torch.Tensor,
        region: LayerRegion,
    ) -> torch.Tensor:
    """Return region features as ``[cells, channels]`` in row-major order."""
    return torch.stack([
        activation[0, :, row, col]
        for row, col in region.coordinates
    ])


def feature_distance_rows(
        original: torch.Tensor,
        transformed: torch.Tensor,
        region: LayerRegion,
    ) -> List[dict]:
    """Compute one-to-one vector distances at every cell in a region."""
    if original.shape != transformed.shape:
        raise ValueError(
            f'Activation shapes differ: {tuple(original.shape)} vs {tuple(transformed.shape)}.'
        )
    if original.ndim != 4 or original.shape[0] != 1:
        raise ValueError(f'Expected activation shape [1,C,H,W], got {tuple(original.shape)}.')

    rows = []
    for cell_index, (row, col) in enumerate(region.coordinates):
        original_vector = original[0, :, row, col]
        transformed_vector = transformed[0, :, row, col]
        rows.append({
            'cell_index': cell_index,
            'cell_row': row,
            'cell_col': col,
            'row_offset': row - region.center_row,
            'col_offset': col - region.center_col,
            **vector_comparison_metrics(original_vector, transformed_vector),
        })
    return rows


def summarize_cell_distances(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError('Cannot summarize an empty collection of cell distances.')
    result = {'n_cells': len(rows)}
    for metric in (
            'euclidean_distance',
            'rms_euclidean_distance',
            'cosine_distance',
        ):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[f'mean_{metric}'] = float(values.mean())
        result[f'std_{metric}'] = float(values.std(ddof=0))
    return result


def pooled_and_spatially_tolerant_metrics(
        original: torch.Tensor,
        transformed: torch.Tensor,
        region: LayerRegion,
    ) -> dict:
    """Compare pooled regions and features independent of cell ordering.

    The symmetric best-match distances act as a spatially tolerant complement
    to the strict one-to-one cell comparison. They measure whether similar
    features remain in the region even if a transformation moves them between
    neighboring cells.
    """
    original_vectors = region_feature_vectors(original, region)
    transformed_vectors = region_feature_vectors(transformed, region)
    result = {}
    for pooling_name, original_vector, transformed_vector in (
            ('mean', original_vectors.mean(dim=0), transformed_vectors.mean(dim=0)),
            ('max', original_vectors.max(dim=0).values,
             transformed_vectors.max(dim=0).values),
        ):
        metrics = vector_comparison_metrics(original_vector, transformed_vector)
        result.update({
            f'pooled_{pooling_name}_{name}': value
            for name, value in metrics.items()
        })

    channel_count = original_vectors.shape[1]
    pairwise_euclidean = torch.cdist(original_vectors, transformed_vectors, p=2)
    pairwise_rms = pairwise_euclidean / sqrt(channel_count)
    original_norms = torch.linalg.vector_norm(original_vectors, dim=1)
    transformed_norms = torch.linalg.vector_norm(transformed_vectors, dim=1)
    norm_products = original_norms[:, None] * transformed_norms[None, :]
    pairwise_similarity = (
        original_vectors @ transformed_vectors.T
    ) / norm_products.clamp_min(1e-8)
    pairwise_cosine = 1.0 - pairwise_similarity
    both_zero = (
        (original_norms[:, None] <= 1e-8)
        & (transformed_norms[None, :] <= 1e-8)
    )
    pairwise_cosine = torch.where(
        both_zero,
        torch.zeros_like(pairwise_cosine),
        pairwise_cosine,
    )

    def symmetric_nearest_mean(distances: torch.Tensor) -> float:
        forward = distances.min(dim=1).values.mean()
        backward = distances.min(dim=0).values.mean()
        return float((0.5 * (forward + backward)).detach().cpu())

    result.update({
        'best_match_euclidean_distance': symmetric_nearest_mean(pairwise_euclidean),
        'best_match_rms_euclidean_distance': symmetric_nearest_mean(pairwise_rms),
        'best_match_cosine_distance': symmetric_nearest_mean(pairwise_cosine),
    })
    return result


def cue_target_distance_rows(
        cue_activation: torch.Tensor,
        original_search: torch.Tensor,
        transformed_search: torch.Tensor,
        region: LayerRegion,
    ) -> List[dict]:
    """Compare a globally pooled cue with every target-region search cell."""
    cue_vector = F.adaptive_max_pool2d(cue_activation, output_size=1)[0, :, 0, 0]
    rows = []
    for cell_index, (row, col) in enumerate(region.coordinates):
        original_vector = original_search[0, :, row, col]
        transformed_vector = transformed_search[0, :, row, col]
        original_metrics = vector_comparison_metrics(cue_vector, original_vector)
        transformed_metrics = vector_comparison_metrics(cue_vector, transformed_vector)
        row_values = {
            'cell_index': cell_index,
            'cell_row': row,
            'cell_col': col,
            'row_offset': row - region.center_row,
            'col_offset': col - region.center_col,
        }
        for name, value in original_metrics.items():
            row_values[f'cue_original_{name}'] = value
        for name, value in transformed_metrics.items():
            row_values[f'cue_transformed_{name}'] = value
        row_values.update({
            'cue_similarity_drop': (
                original_metrics['cosine_similarity']
                - transformed_metrics['cosine_similarity']
            ),
            'cue_distance_increase': (
                transformed_metrics['cosine_distance']
                - original_metrics['cosine_distance']
            ),
            'cue_dot_product_change': (
                transformed_metrics['dot_product']
                - original_metrics['dot_product']
            ),
        })
        rows.append(row_values)
    return rows


def summarize_cue_target_rows(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError('Cannot summarize empty cue-target comparisons.')
    metrics = [
        key
        for key in rows[0]
        if key.startswith('cue_')
    ]
    result = {'n_cells': len(rows)}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[f'mean_{metric}'] = float(values.mean())
        result[f'std_{metric}'] = float(values.std(ddof=0))
    return result


def attention_position_metrics(
        cue_activation: torch.Tensor,
        search_activation: torch.Tensor,
        positions: Sequence[Tuple[int, int]],
        target_position: int,
        image_size: int,
        oracle_window: int,
    ) -> dict:
    """Reproduce the VGG IVSN attention score and target rank at layer 30."""
    cue_vector = F.adaptive_max_pool2d(cue_activation, output_size=1)
    attention = F.relu((search_activation * cue_vector).sum(dim=1, keepdim=True))
    attention = attention / (attention.max() + 1e-8)
    attention_up = F.interpolate(
        attention,
        size=(image_size, image_size),
        mode='bicubic',
        align_corners=False,
    )[0, 0]

    half = oracle_window // 2
    scores = []
    for x, y in positions:
        x1 = max(0, int(x) - half)
        x2 = min(image_size, int(x) + half)
        y1 = max(0, int(y) - half)
        y2 = min(image_size, int(y) + half)
        scores.append(attention_up[y1:y2, x1:x2].mean())
    scores_tensor = torch.stack(scores)
    target_score = scores_tensor[target_position]
    distractor_mask = torch.ones_like(scores_tensor, dtype=torch.bool)
    distractor_mask[target_position] = False
    distractor_scores = scores_tensor[distractor_mask]
    target_rank_zero_based = int((scores_tensor > target_score).sum().detach().cpu())
    probabilities = torch.softmax(scores_tensor, dim=0)
    return {
        'score_target': float(target_score.detach().cpu()),
        'score_max_distractor': float(distractor_scores.max().detach().cpu()),
        'score_mean_distractor': float(distractor_scores.mean().detach().cpu()),
        'score_margin': float(
            (target_score - distractor_scores.max()).detach().cpu()
        ),
        'target_rank': target_rank_zero_based,
        'is_top1': int(target_rank_zero_based == 0),
        'p_target': float(probabilities[target_position].detach().cpu()),
        'n_fixations': target_rank_zero_based + 1,
    }


def effective_receptive_field(
        input_tensor: torch.Tensor,
        activation: torch.Tensor,
        coordinates: Sequence[Tuple[int, int]],
        retain_graph: bool = False,
    ) -> np.ndarray:
    """Superimpose per-cell absolute input gradients across a spatial region."""
    if not coordinates:
        raise ValueError('At least one activation coordinate is required for an ERF.')

    cell_scores = torch.stack([
        activation[0, :, row, col].sum()
        for row, col in coordinates
    ])
    basis = torch.eye(
        len(coordinates),
        dtype=cell_scores.dtype,
        device=cell_scores.device,
    )

    try:
        gradients = torch.autograd.grad(
            outputs=cell_scores,
            inputs=input_tensor,
            grad_outputs=basis,
            is_grads_batched=True,
            retain_graph=retain_graph,
        )[0]
        # [cells, batch=1, channels, height, width]
        saliency = gradients.abs().sum(dim=(0, 1, 2))
    except (TypeError, RuntimeError):
        # Compatibility path for older PyTorch releases without batched VJPs.
        saliency = torch.zeros_like(input_tensor[0, 0])
        for index, score in enumerate(cell_scores):
            keep = retain_graph or index < len(cell_scores) - 1
            gradient = torch.autograd.grad(
                score,
                input_tensor,
                retain_graph=keep,
            )[0]
            saliency = saliency + gradient[0].abs().sum(dim=0)

    return saliency.detach().cpu().numpy().astype(np.float32, copy=False)


def erf_distance_metrics(original: np.ndarray, transformed: np.ndarray) -> dict:
    """Compare ERF mass distributions without depending on raw gradient scale."""
    if original.shape != transformed.shape:
        raise ValueError(f'ERF shapes differ: {original.shape} vs {transformed.shape}.')

    original_flat = np.asarray(original, dtype=np.float64).ravel()
    transformed_flat = np.asarray(transformed, dtype=np.float64).ravel()
    eps = np.finfo(np.float64).eps
    original_mass = original_flat / max(original_flat.sum(), eps)
    transformed_mass = transformed_flat / max(transformed_flat.sum(), eps)
    original_norm = np.linalg.norm(original_flat)
    transformed_norm = np.linalg.norm(transformed_flat)
    if original_norm <= eps and transformed_norm <= eps:
        cosine_distance = 0.0
    elif original_norm <= eps or transformed_norm <= eps:
        cosine_distance = 1.0
    else:
        cosine_similarity = np.dot(original_flat, transformed_flat) / (
            original_norm * transformed_norm
        )
        cosine_distance = float(np.clip(1.0 - cosine_similarity, 0.0, 2.0))
    height, width = original.shape
    y_coordinates, x_coordinates = np.mgrid[0:height, 0:width]

    def spatial_summary(mass: np.ndarray) -> dict:
        mass_map = mass.reshape(height, width)
        if mass_map.sum() <= eps:
            return {
                'center_x': (width - 1) / 2.0,
                'center_y': (height - 1) / 2.0,
                'spread': 0.0,
                'area_90_fraction': 0.0,
            }
        center_x = float((mass_map * x_coordinates).sum())
        center_y = float((mass_map * y_coordinates).sum())
        squared_radius = (
            (x_coordinates - center_x) ** 2
            + (y_coordinates - center_y) ** 2
        )
        spread = float(np.sqrt((mass_map * squared_radius).sum()))
        sorted_mass = np.sort(mass)[::-1]
        area_90 = int(np.searchsorted(np.cumsum(sorted_mass), 0.9) + 1)
        return {
            'center_x': center_x,
            'center_y': center_y,
            'spread': spread,
            'area_90_fraction': float(area_90 / mass.size),
        }

    original_spatial = spatial_summary(original_mass)
    transformed_spatial = spatial_summary(transformed_mass)
    centroid_shift = np.hypot(
        transformed_spatial['center_x'] - original_spatial['center_x'],
        transformed_spatial['center_y'] - original_spatial['center_y'],
    )
    return {
        'erf_cosine_distance': cosine_distance,
        'erf_total_variation_distance': float(
            0.5 * np.abs(original_mass - transformed_mass).sum()
        ),
        'erf_mass_overlap': float(np.minimum(original_mass, transformed_mass).sum()),
        'erf_original_centroid_x': original_spatial['center_x'],
        'erf_original_centroid_y': original_spatial['center_y'],
        'erf_transformed_centroid_x': transformed_spatial['center_x'],
        'erf_transformed_centroid_y': transformed_spatial['center_y'],
        'erf_centroid_shift_px': float(centroid_shift),
        'erf_original_spread_px': original_spatial['spread'],
        'erf_transformed_spread_px': transformed_spatial['spread'],
        'erf_spread_change_px': (
            transformed_spatial['spread'] - original_spatial['spread']
        ),
        'erf_original_area_90_fraction': original_spatial['area_90_fraction'],
        'erf_transformed_area_90_fraction': (
            transformed_spatial['area_90_fraction']
        ),
    }


def normalized_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    maximum = float(values.max()) if values.size else 0.0
    if maximum <= 0:
        return np.zeros_like(values)
    return values / maximum


def unit_mass_map(values: np.ndarray) -> np.ndarray:
    """Normalize a non-negative ERF so all spatial values sum to one."""
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    total = float(values.sum())
    if total <= np.finfo(np.float32).eps:
        return np.zeros_like(values)
    return values / total
