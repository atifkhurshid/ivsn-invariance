"""CSV summaries and figures for target-only VGG analyses."""

from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Patch

from .domain import TransformSpec
from .feature_analysis import erf_distance_metrics, normalized_map
from .feature_reporting import GROUPS, condition_x_label, confidence_interval_95
from .imaging import affine_from_spec
from .runtime import OBJ_SIZE


ACTIVATION_METRICS = (
    'mean_absolute_difference',
    'rms_difference',
    'relative_mean_absolute_difference',
    'cosine_distance',
)

POOLED_BLOCK_METRICS = (
    'mean_absolute_difference',
    'relative_mean_absolute_difference',
    'cosine_distance',
)

ALIGNED_ERF_METRICS = (
    'erf_cosine_distance',
    'aligned_erf_cosine_distance',
    'erf_mass_overlap',
    'aligned_erf_mass_overlap',
    'erf_alignment_gain_cosine_similarity',
    'erf_alignment_gain_mass_overlap',
)


def build_grouped_target_activation_summary(
        rows: Sequence[dict],
    ) -> List[dict]:
    """Aggregate trial-level activation differences with mean, SD and CI."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    keys = [
        'condition_name', 'condition_group', 'condition_value',
        'layer', 'channels', 'activation_height', 'activation_width',
    ]
    grouped_rows = []
    for key_values, condition_frame in frame.groupby(keys, sort=False):
        row = dict(zip(keys, key_values))
        for group_name, _ in GROUPS:
            if group_name == 'all':
                subset = condition_frame
            else:
                trial_type = group_name[len('target_'):]
                subset = condition_frame[
                    condition_frame['trial_type'] == trial_type
                ]
            row[f'n_{group_name}'] = int(len(subset))
            for metric in ACTIVATION_METRICS:
                values = subset[metric].to_numpy(dtype=np.float64)
                if len(values):
                    row[f'{metric}_{group_name}'] = float(values.mean())
                    row[f'std_{metric}_{group_name}'] = float(
                        values.std(ddof=1) if len(values) > 1 else 0.0
                    )
                    row[f'ci95_{metric}_{group_name}'] = (
                        confidence_interval_95(values)
                    )
                else:
                    row[f'{metric}_{group_name}'] = float('nan')
                    row[f'std_{metric}_{group_name}'] = float('nan')
                    row[f'ci95_{metric}_{group_name}'] = float('nan')
        grouped_rows.append(row)
    return sorted(
        grouped_rows,
        key=lambda row: (
            str(row['condition_group']),
            float(row['condition_value']),
            int(row['layer']),
        ),
    )


def save_grouped_target_activation_plots(
        grouped_rows: Sequence[dict],
        plot_dir: Path,
    ) -> None:
    """Plot per-layer transformation sweeps with trial SD error bars."""
    if not grouped_rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(grouped_rows)
    metric_labels = {
        'mean_absolute_difference': 'Mean absolute activation difference',
        'rms_difference': 'RMS activation difference',
        'relative_mean_absolute_difference': 'Relative mean absolute difference',
        'cosine_distance': 'Cosine distance',
    }
    for layer in sorted(frame['layer'].unique()):
        layer_frame = frame[frame['layer'] == layer].sort_values('condition_value')
        values = layer_frame['condition_value'].to_numpy(dtype=np.float64)
        condition_group = str(layer_frame['condition_group'].iloc[0])
        x = np.arange(len(values))
        width = 0.24
        for metric, label in metric_labels.items():
            fig, axis = plt.subplots(figsize=(8, 4.8))
            for group_index, (group_name, group_label) in enumerate(GROUPS):
                offset = (group_index - 1) * width
                axis.bar(
                    x + offset,
                    layer_frame[f'{metric}_{group_name}'],
                    width=width,
                    yerr=layer_frame[f'std_{metric}_{group_name}'],
                    capsize=3,
                    label=group_label,
                )
            axis.set_xticks(x, [f'{value:g}' for value in values])
            axis.set_xlabel(condition_x_label(condition_group))
            axis.set_ylabel(label)
            channels = int(layer_frame['channels'].iloc[0])
            height = int(layer_frame['activation_height'].iloc[0])
            width_value = int(layer_frame['activation_width'].iloc[0])
            axis.set_title(
                f'VGG layer {layer} ({channels}x{height}x{width_value})'
            )
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.legend()
            fig.tight_layout()
            fig.savefig(
                plot_dir / f'layer_{layer}_{metric}.png',
                dpi=300,
            )
            plt.close(fig)


def build_grouped_pooled_block_summary(
        rows: Sequence[dict],
    ) -> List[dict]:
    """Aggregate spatially pooled block differences by trial group."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    keys = [
        'condition_name', 'condition_group', 'condition_value',
        'block', 'layer', 'channels', 'activation_height', 'activation_width',
    ]
    grouped_rows = []
    for key_values, condition_frame in frame.groupby(keys, sort=False):
        row = dict(zip(keys, key_values))
        for group_name, _ in GROUPS:
            if group_name == 'all':
                subset = condition_frame
            else:
                trial_type = group_name[len('target_'):]
                subset = condition_frame[
                    condition_frame['trial_type'] == trial_type
                ]
            row[f'n_{group_name}'] = int(len(subset))
            for metric in POOLED_BLOCK_METRICS:
                values = subset[metric].to_numpy(dtype=np.float64)
                if len(values):
                    row[f'{metric}_{group_name}'] = float(values.mean())
                    row[f'std_{metric}_{group_name}'] = float(
                        values.std(ddof=1) if len(values) > 1 else 0.0
                    )
                    row[f'ci95_{metric}_{group_name}'] = (
                        confidence_interval_95(values)
                    )
                else:
                    row[f'{metric}_{group_name}'] = float('nan')
                    row[f'std_{metric}_{group_name}'] = float('nan')
                    row[f'ci95_{metric}_{group_name}'] = float('nan')
        grouped_rows.append(row)
    return sorted(
        grouped_rows,
        key=lambda row: (
            str(row['condition_group']),
            float(row['condition_value']),
            int(row['block']),
        ),
    )


def save_pooled_block_activation_plots(
        grouped_rows: Sequence[dict],
        plot_dir: Path,
    ) -> None:
    """Plot one connected layer profile per transformation condition."""
    if not grouped_rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(grouped_rows)
    metric_labels = {
        'mean_absolute_difference': (
            'Mean absolute difference across pooled channels'
        ),
        'relative_mean_absolute_difference': (
            'Relative mean absolute difference'
        ),
        'cosine_distance': 'Cosine distance between pooled channel vectors',
    }
    styles = {
        'all': ('o', '#1f77b4'),
        'target_identical': ('s', '#2ca02c'),
        'target_different': ('^', '#d62728'),
    }
    for condition_name in frame['condition_name'].drop_duplicates():
        condition_frame = frame[
            frame['condition_name'] == condition_name
        ].sort_values('block')
        x = np.arange(len(condition_frame))
        tick_labels = [
            f"Block {int(row.block)}\nLayer {int(row.layer)}"
            for row in condition_frame.itertuples()
        ]
        for metric, label in metric_labels.items():
            fig, axis = plt.subplots(figsize=(8.2, 5.0))
            for group_name, group_label in GROUPS:
                marker, color = styles[group_name]
                axis.errorbar(
                    x,
                    condition_frame[f'{metric}_{group_name}'],
                    yerr=condition_frame[f'std_{metric}_{group_name}'],
                    marker=marker,
                    linestyle=':',
                    linewidth=1.8,
                    markersize=6,
                    capsize=4,
                    color=color,
                    label=group_label,
                )
            axis.set_xticks(x, tick_labels)
            axis.set_xlabel('VGG16 max-pool output')
            axis.set_ylabel(label)
            axis.set_title(str(condition_name).replace('_', ' '))
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.grid(axis='y', alpha=0.22)
            axis.legend()
            fig.tight_layout()
            fig.savefig(
                plot_dir / f'{condition_name}_{metric}.png',
                dpi=300,
            )
            plt.close(fig)


def align_erf_to_original(
        transformed_erf: np.ndarray,
        transform_spec: TransformSpec,
    ) -> np.ndarray:
    """Warp a transformed-image ERF back into original target coordinates."""
    values = np.asarray(transformed_erf, dtype=np.float32)
    height, width = values.shape
    inverse_coefficients = affine_from_spec(transform_spec)
    inverse_matrix = np.asarray([
        inverse_coefficients[:3],
        inverse_coefficients[3:],
        (0.0, 0.0, 1.0),
    ], dtype=np.float64)
    forward_object = np.linalg.inv(inverse_matrix)

    # Convert the affine from the 156x156 stimulus coordinate system to the
    # ERF grid. Applying it directly avoids a lossy upsample/downsample cycle.
    object_from_erf = np.asarray([
        [(OBJ_SIZE - 1) / max(width - 1, 1), 0.0, 0.0],
        [0.0, (OBJ_SIZE - 1) / max(height - 1, 1), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    erf_from_object = np.linalg.inv(object_from_erf)
    forward_erf = erf_from_object @ forward_object @ object_from_erf
    forward_coefficients = tuple(forward_erf[:2, :].reshape(-1))
    aligned_image = Image.fromarray(values).transform(
        (width, height),
        Image.AFFINE,
        forward_coefficients,
        resample=Image.BILINEAR,
        fillcolor=0.0,
    )
    return np.asarray(aligned_image, dtype=np.float32)


def aligned_erf_metrics(
        original_erf: np.ndarray,
        transformed_erf: np.ndarray,
        aligned_erf: np.ndarray,
    ) -> dict:
    """Report alignment-aware ERF metrics and improvement over raw comparison."""
    unaligned = erf_distance_metrics(original_erf, transformed_erf)
    aligned = erf_distance_metrics(original_erf, aligned_erf)
    return {
        **unaligned,
        **{f'aligned_{key}': value for key, value in aligned.items()},
        'erf_alignment_gain_cosine_similarity': float(
            unaligned['erf_cosine_distance'] - aligned['erf_cosine_distance']
        ),
        'erf_alignment_gain_mass_overlap': float(
            aligned['erf_mass_overlap'] - unaligned['erf_mass_overlap']
        ),
    }


def _unit_mass(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    total = float(values.sum())
    if total <= np.finfo(np.float32).eps:
        return np.zeros_like(values)
    return values / total


def unit_mass_map(values: np.ndarray) -> np.ndarray:
    """Return a non-negative spatial map whose pixel values sum to one."""
    return _unit_mass(values)


def _signed_unit_mass_change_overlay(
        original_image: Image.Image,
        original_erf: np.ndarray,
        comparison_erf: np.ndarray,
    ) -> np.ndarray:
    """Show decreases in blue and increases in orange after mass normalization."""
    difference = _unit_mass(comparison_erf) - _unit_mass(original_erf)
    magnitude = np.abs(difference)
    maximum = float(magnitude.max()) if magnitude.size else 0.0
    size = (original_erf.shape[1], original_erf.shape[0])
    background = np.asarray(
        original_image.convert('L').convert('RGB').resize(size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    if maximum <= np.finfo(np.float32).eps:
        return background
    decreased = np.asarray([0.10, 0.45, 1.00], dtype=np.float32)
    increased = np.asarray([1.00, 0.40, 0.05], dtype=np.float32)
    color = np.where(
        (difference >= 0)[..., None],
        increased,
        decreased,
    )
    alpha = np.clip(magnitude / maximum, 0.0, 1.0) * 0.90
    return background * (1.0 - alpha[..., None]) + color * alpha[..., None]


def sensitivity_change_overlay(
        original_image: Image.Image,
        original_erf: np.ndarray,
        aligned_erf: np.ndarray,
    ) -> np.ndarray:
    """Overlay shared, decreased and increased normalized sensitivity."""
    original = _unit_mass(original_erf)
    aligned = _unit_mass(aligned_erf)
    shared = np.minimum(original, aligned)
    decreased = np.clip(original - aligned, 0.0, None)
    increased = np.clip(aligned - original, 0.0, None)
    green = np.asarray([0.20, 0.75, 0.35], dtype=np.float32)
    blue = np.asarray([0.10, 0.45, 1.00], dtype=np.float32)
    orange = np.asarray([1.00, 0.40, 0.05], dtype=np.float32)
    color = (
        shared[..., None] * green
        + decreased[..., None] * blue
        + increased[..., None] * orange
    )
    strength = shared + decreased + increased
    maximum = float(strength.max()) if strength.size else 0.0
    if maximum > 0:
        color = np.clip(color / maximum, 0.0, 1.0)
        alpha = np.clip(strength / maximum, 0.0, 1.0) * 0.88
    else:
        alpha = np.zeros_like(strength)
    size = (original_erf.shape[1], original_erf.shape[0])
    background = np.asarray(
        original_image.resize(size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return background * (1.0 - alpha[..., None]) + color * alpha[..., None]


def save_aligned_erf_comparison_figure(
        original_image: Image.Image,
        transformed_image: Image.Image,
        original_erfs: dict,
        transformed_erfs: dict,
        aligned_erfs: dict,
        layer_shapes: dict,
        title: str,
        path: Path,
    ) -> None:
    """Save a presentation-ready layer-by-layer aligned ERF comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted(original_erfs)
    fig, axes = plt.subplots(
        len(layers),
        3,
        figsize=(12.5, 3.55 * len(layers)),
        squeeze=False,
    )
    for row, layer in enumerate(layers):
        original_erf = original_erfs[layer]
        transformed_erf = transformed_erfs[layer]
        aligned_erf = aligned_erfs[layer]
        size = (original_erf.shape[1], original_erf.shape[0])
        original_rgb = np.asarray(
            original_image.resize(size, Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        transformed_rgb = np.asarray(
            transformed_image.resize(size, Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        axes[row, 0].imshow(original_rgb)
        axes[row, 0].imshow(
            normalized_map(original_erf),
            cmap='magma',
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        axes[row, 1].imshow(transformed_rgb)
        axes[row, 1].imshow(
            normalized_map(transformed_erf),
            cmap='magma',
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        axes[row, 2].imshow(sensitivity_change_overlay(
            original_image,
            original_erf,
            aligned_erf,
        ))
        metrics = aligned_erf_metrics(
            original_erf,
            transformed_erf,
            aligned_erf,
        )
        channels, height, width = layer_shapes[layer]
        axes[row, 0].set_ylabel(
            f'Layer {layer}\n{channels}x{height}x{width}',
            fontsize=12,
        )
        axes[row, 2].text(
            0.02,
            0.98,
            (
                f"aligned overlap = {metrics['aligned_erf_mass_overlap']:.2f}\n"
                f"alignment gain = "
                f"{metrics['erf_alignment_gain_mass_overlap']:+.2f}"
            ),
            transform=axes[row, 2].transAxes,
            va='top',
            ha='left',
            fontsize=9,
            color='white',
            bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 3},
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title('Original image + ERF')
    axes[0, 1].set_title('Transformed image + ERF')
    axes[0, 2].set_title('Aligned sensitivity change')
    fig.legend(
        handles=[
            Patch(color=(0.20, 0.75, 0.35), label='Shared'),
            Patch(color=(0.10, 0.45, 1.00), label='Decreased'),
            Patch(color=(1.00, 0.40, 0.05), label='Increased'),
        ],
        loc='lower center',
        ncol=3,
        frameon=False,
    )
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.965))
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def save_unit_mass_erf_comparison_figure(
        original_image: Image.Image,
        transformed_image: Image.Image,
        original_erfs: dict,
        transformed_erfs: dict,
        aligned_erfs: dict,
        layer_shapes: dict,
        title: str,
        path: Path,
    ) -> None:
    """Save ERFs normalized to unit mass, with a shared scale per layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted(original_erfs)
    fig, axes = plt.subplots(
        len(layers),
        4,
        figsize=(16.0, 3.45 * len(layers)),
        squeeze=False,
    )
    for row, layer in enumerate(layers):
        original_erf = original_erfs[layer]
        transformed_erf = transformed_erfs[layer]
        aligned_erf = aligned_erfs[layer]
        original_mass = _unit_mass(original_erf)
        transformed_mass = _unit_mass(transformed_erf)
        aligned_mass = _unit_mass(aligned_erf)
        shared_peak = max(
            float(original_mass.max()),
            float(transformed_mass.max()),
            float(aligned_mass.max()),
            np.finfo(np.float32).eps,
        )
        size = (original_erf.shape[1], original_erf.shape[0])
        original_rgb = np.asarray(
            original_image.resize(size, Image.BILINEAR), dtype=np.float32
        ) / 255.0
        transformed_rgb = np.asarray(
            transformed_image.resize(size, Image.BILINEAR), dtype=np.float32
        ) / 255.0

        axes[row, 0].imshow(original_rgb)
        axes[row, 0].imshow(
            original_mass, cmap='magma', alpha=0.60, vmin=0, vmax=shared_peak
        )
        axes[row, 1].imshow(transformed_rgb)
        axes[row, 1].imshow(
            transformed_mass, cmap='magma', alpha=0.60,
            vmin=0, vmax=shared_peak,
        )
        axes[row, 2].imshow(original_rgb)
        axes[row, 2].imshow(
            aligned_mass, cmap='magma', alpha=0.60,
            vmin=0, vmax=shared_peak,
        )
        axes[row, 3].imshow(_signed_unit_mass_change_overlay(
            original_image,
            original_erf,
            aligned_erf,
        ))

        channels, height, width = layer_shapes[layer]
        axes[row, 0].set_ylabel(
            f'Layer {layer}\n{channels}x{height}x{width}', fontsize=12
        )
        axes[row, 0].text(
            0.02, 0.98, f'raw sum = {float(original_erf.sum()):.2e}',
            transform=axes[row, 0].transAxes, va='top', color='white',
            fontsize=8, bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 2},
        )
        axes[row, 1].text(
            0.02, 0.98, f'raw sum = {float(transformed_erf.sum()):.2e}',
            transform=axes[row, 1].transAxes, va='top', color='white',
            fontsize=8, bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 2},
        )
        aligned_metrics = erf_distance_metrics(original_erf, aligned_erf)
        axes[row, 3].text(
            0.02, 0.98,
            f"TV distance = {aligned_metrics['erf_total_variation_distance']:.2f}",
            transform=axes[row, 3].transAxes, va='top', color='white',
            fontsize=8, bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 2},
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_title('Original: unit-mass ERF')
    axes[0, 1].set_title('Transformed: unit-mass ERF')
    axes[0, 2].set_title('Inverse-aligned: unit-mass ERF')
    axes[0, 3].set_title('Aligned change')
    fig.legend(
        handles=[
            Patch(color=(0.10, 0.45, 1.00), label='Decreased mass'),
            Patch(color=(1.00, 0.40, 0.05), label='Increased mass'),
        ],
        loc='lower center', ncol=2, frameon=False,
    )
    fig.suptitle(f'{title}\nEach ERF sums to 1', fontsize=15)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def save_unit_mass_direct_erf_figure(
        original_image: Image.Image,
        transformed_image: Image.Image,
        original_erfs: dict,
        transformed_erfs: dict,
        layer_shapes: dict,
        title: str,
        path: Path,
    ) -> None:
    """Reproduce the original ERF comparison using unit-mass maps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted(original_erfs)
    fig, axes = plt.subplots(
        len(layers),
        3,
        figsize=(12.5, 3.55 * len(layers)),
        squeeze=False,
    )
    for row, layer in enumerate(layers):
        original_erf = original_erfs[layer]
        transformed_erf = transformed_erfs[layer]
        original_mass = _unit_mass(original_erf)
        transformed_mass = _unit_mass(transformed_erf)
        absolute_difference = np.abs(original_mass - transformed_mass)
        shared_peak = max(
            float(original_mass.max()),
            float(transformed_mass.max()),
            np.finfo(np.float32).eps,
        )
        difference_peak = max(
            float(absolute_difference.max()),
            np.finfo(np.float32).eps,
        )
        size = (original_erf.shape[1], original_erf.shape[0])
        original_rgb = np.asarray(
            original_image.resize(size, Image.BILINEAR), dtype=np.float32
        ) / 255.0
        transformed_rgb = np.asarray(
            transformed_image.resize(size, Image.BILINEAR), dtype=np.float32
        ) / 255.0

        axes[row, 0].imshow(original_rgb)
        axes[row, 0].imshow(
            original_mass, cmap='magma', alpha=0.60,
            vmin=0, vmax=shared_peak,
        )
        axes[row, 1].imshow(transformed_rgb)
        axes[row, 1].imshow(
            transformed_mass, cmap='magma', alpha=0.60,
            vmin=0, vmax=shared_peak,
        )
        axes[row, 2].imshow(
            absolute_difference,
            cmap='viridis',
            vmin=0,
            vmax=difference_peak,
        )
        channels, height, width = layer_shapes[layer]
        axes[row, 0].set_ylabel(
            f'Layer {layer}\n{channels}x{height}x{width}', fontsize=12
        )
        direct_metrics = erf_distance_metrics(original_erf, transformed_erf)
        axes[row, 2].text(
            0.02, 0.98,
            f"TV distance = {direct_metrics['erf_total_variation_distance']:.2f}",
            transform=axes[row, 2].transAxes, va='top', color='white',
            fontsize=9, bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 3},
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_title('Original image + unit-mass ERF')
    axes[0, 1].set_title('Transformed image + unit-mass ERF')
    axes[0, 2].set_title('Absolute unit-mass difference')
    fig.suptitle(f'{title}\nDirect comparison; each ERF sums to 1', fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def build_grouped_aligned_erf_summary(rows: Sequence[dict]) -> List[dict]:
    """Aggregate alignment-aware ERF metrics across selected targets."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    keys = ['condition_name', 'condition_group', 'condition_value', 'layer']
    result = []
    for key_values, group in frame.groupby(keys, sort=False):
        row = dict(zip(keys, key_values))
        row['n'] = int(len(group))
        for metric in ALIGNED_ERF_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f'mean_{metric}'] = float(values.mean())
            row[f'std_{metric}'] = float(
                values.std(ddof=1) if len(values) > 1 else 0.0
            )
            row[f'ci95_{metric}'] = confidence_interval_95(values)
        result.append(row)
    return sorted(
        result,
        key=lambda row: (
            str(row['condition_group']),
            float(row['condition_value']),
            int(row['layer']),
        ),
    )


def save_aligned_erf_summary_plots(
        grouped_rows: Sequence[dict],
        plot_dir: Path,
    ) -> None:
    """Plot aligned versus unaligned ERF similarity for every layer."""
    if not grouped_rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(grouped_rows)
    for layer in sorted(frame['layer'].unique()):
        subset = frame[frame['layer'] == layer].sort_values('condition_value')
        values = subset['condition_value'].to_numpy(dtype=np.float64)
        unaligned = 1.0 - subset['mean_erf_cosine_distance'].to_numpy()
        aligned = 1.0 - subset['mean_aligned_erf_cosine_distance'].to_numpy()
        unaligned_error = subset['ci95_erf_cosine_distance'].to_numpy()
        aligned_error = subset['ci95_aligned_erf_cosine_distance'].to_numpy()
        fig, axis = plt.subplots(figsize=(7.4, 4.7))
        axis.errorbar(
            values,
            unaligned,
            yerr=unaligned_error,
            marker='o',
            linestyle='--',
            capsize=3,
            label='Unaligned',
        )
        axis.errorbar(
            values,
            aligned,
            yerr=aligned_error,
            marker='o',
            linewidth=2.2,
            capsize=3,
            label='Inverse-aligned',
        )
        axis.set_xlabel(condition_x_label(str(subset['condition_group'].iloc[0])))
        axis.set_ylabel('ERF cosine similarity')
        axis.set_ylim(0.0, 1.03)
        axis.set_title(f'VGG layer {layer}: geometric alignment')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f'layer_{layer}_aligned_similarity.png', dpi=300)
        plt.close(fig)


def save_target_erf_figure(
        original_image: Image.Image,
        transformed_image: Image.Image,
        original_erf: np.ndarray,
        transformed_erf: np.ndarray,
        title: str,
        path: Path,
    ) -> None:
    """Save original/transformed ERF overlays and normalized difference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    display_size = (original_erf.shape[1], original_erf.shape[0])
    original_rgb = np.asarray(
        original_image.resize(display_size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    transformed_rgb = np.asarray(
        transformed_image.resize(display_size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    original_normalized = normalized_map(original_erf)
    transformed_normalized = normalized_map(transformed_erf)
    difference = np.abs(original_normalized - transformed_normalized)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(original_rgb)
    axes[0].imshow(original_normalized, cmap='magma', alpha=0.55, vmin=0, vmax=1)
    axes[0].set_title('Original image + ERF')
    axes[1].imshow(transformed_rgb)
    axes[1].imshow(
        transformed_normalized,
        cmap='magma',
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title('Transformed image + ERF')
    axes[2].imshow(difference, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Absolute ERF difference')
    for axis in axes:
        axis.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def save_target_erf_arrays(
        original_erf: np.ndarray,
        transformed_erf: np.ndarray,
        aligned_erf: np.ndarray,
        path: Path,
    ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        original_erf=original_erf,
        transformed_erf=transformed_erf,
        aligned_transformed_erf=aligned_erf,
        original_erf_normalized=normalized_map(original_erf),
        transformed_erf_normalized=normalized_map(transformed_erf),
        aligned_transformed_erf_normalized=normalized_map(aligned_erf),
        original_erf_unit_mass=_unit_mass(original_erf),
        transformed_erf_unit_mass=_unit_mass(transformed_erf),
        aligned_transformed_erf_unit_mass=_unit_mass(aligned_erf),
        absolute_normalized_difference=np.abs(
            normalized_map(original_erf) - normalized_map(transformed_erf)
        ),
    )
