"""Tables and figures for VGG feature-representation analyses."""

import csv
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from .feature_analysis import erf_distance_metrics, normalized_map, unit_mass_map


FEATURE_METRICS = (
    'mean_euclidean_distance',
    'mean_rms_euclidean_distance',
    'mean_cosine_distance',
)

EXTENDED_FEATURE_METRICS = (
    'pooled_mean_rms_euclidean_distance',
    'pooled_mean_cosine_distance',
    'pooled_max_rms_euclidean_distance',
    'pooled_max_cosine_distance',
    'best_match_rms_euclidean_distance',
    'best_match_cosine_distance',
)

CUE_TARGET_METRICS = (
    'mean_cue_original_cosine_similarity',
    'mean_cue_transformed_cosine_similarity',
    'mean_cue_similarity_drop',
    'mean_cue_original_rms_euclidean_distance',
    'mean_cue_transformed_rms_euclidean_distance',
    'mean_cue_original_dot_product',
    'mean_cue_transformed_dot_product',
)

PERFORMANCE_METRICS = (
    'transformed_score_target',
    'transformed_score_margin',
    'transformed_target_rank',
    'transformed_p_target',
    'transformed_n_fixations',
    'target_score_drop',
    'score_margin_drop',
    'rank_increase',
)

GROUPS = (
    ('all', 'All'),
    ('target_identical', 'Target identical'),
    ('target_different', 'Target different'),
)


def write_rows_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def confidence_interval_95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))


def build_grouped_feature_summary(trial_rows: Sequence[dict]) -> List[dict]:
    """Group trial-level distances by condition/layer and trial relation."""
    if not trial_rows:
        return []

    frame = pd.DataFrame(trial_rows)
    grouped_rows = []
    keys = ['condition_name', 'condition_group', 'condition_value', 'layer', 'window_size']
    for key_values, condition_frame in frame.groupby(keys, sort=False):
        row = dict(zip(keys, key_values))
        for group_name, _ in GROUPS:
            if group_name == 'all':
                subset = condition_frame
            else:
                trial_type = group_name[len('target_'):]
                subset = condition_frame[condition_frame['trial_type'] == trial_type]
            row[f'n_{group_name}'] = int(len(subset))
            for metric in FEATURE_METRICS + EXTENDED_FEATURE_METRICS:
                values = subset[metric].to_numpy(dtype=np.float64)
                row[f'{metric}_{group_name}'] = (
                    float(values.mean()) if len(values) else float('nan')
                )
                row[f'ci95_{metric}_{group_name}'] = (
                    confidence_interval_95(values) if len(values) else float('nan')
                )
        grouped_rows.append(row)

    return sorted(
        grouped_rows,
        key=lambda row: (
            str(row['condition_group']),
            float(row['condition_value']),
            int(row['layer']),
        ),
    )


def build_grouped_metric_summary(
        rows: Sequence[dict],
        metrics: Sequence[str],
        include_layer: bool,
    ) -> List[dict]:
    """Build all/identical/different summaries for arbitrary trial metrics."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    keys = ['condition_name', 'condition_group', 'condition_value']
    if include_layer:
        keys.extend(['layer', 'window_size'])

    grouped_rows = []
    for key_values, condition_frame in frame.groupby(keys, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values))
        for group_name, _ in GROUPS:
            if group_name == 'all':
                subset = condition_frame
            else:
                trial_type = group_name[len('target_'):]
                subset = condition_frame[condition_frame['trial_type'] == trial_type]
            row[f'n_{group_name}'] = int(len(subset))
            for metric in metrics:
                values = subset[metric].to_numpy(dtype=np.float64)
                row[f'{metric}_{group_name}'] = (
                    float(values.mean()) if len(values) else float('nan')
                )
                row[f'ci95_{metric}_{group_name}'] = (
                    confidence_interval_95(values) if len(values) else float('nan')
                )
        grouped_rows.append(row)
    return sorted(
        grouped_rows,
        key=lambda row: (
            str(row['condition_group']),
            float(row['condition_value']),
            int(row.get('layer', 0)),
        ),
    )


def build_grouped_cue_target_summary(rows: Sequence[dict]) -> List[dict]:
    return build_grouped_metric_summary(
        rows,
        metrics=CUE_TARGET_METRICS,
        include_layer=True,
    )


def build_grouped_performance_summary(rows: Sequence[dict]) -> List[dict]:
    return build_grouped_metric_summary(
        rows,
        metrics=PERFORMANCE_METRICS,
        include_layer=False,
    )


def condition_x_label(condition_group: str) -> str:
    return {
        'rotation_deg': 'Rotation (degrees)',
        'scale': 'Scale factor',
        'shift_x': 'Horizontal shift (px)',
        'shift_y': 'Vertical shift (px)',
        'skew_x_deg': 'Horizontal skew (degrees)',
        'skew_y_deg': 'Vertical skew (degrees)',
        'noise_std': 'Gaussian noise std',
        'blur_radius': 'Blur radius',
        'mixed': 'Condition index',
        'random_rotation': 'Condition value',
    }.get(condition_group, 'Condition value')


def save_grouped_feature_plots(grouped_rows: Sequence[dict], plot_dir: Path) -> None:
    if not grouped_rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(grouped_rows)

    metric_labels = {
        'mean_euclidean_distance': 'Mean Euclidean distance',
        'mean_rms_euclidean_distance': 'Mean RMS Euclidean distance',
        'mean_cosine_distance': 'Mean cosine distance',
        'pooled_mean_cosine_distance': 'Mean-pooled cosine distance',
        'best_match_cosine_distance': 'Spatially tolerant cosine distance',
    }
    for layer in sorted(frame['layer'].unique()):
        layer_frame = frame[frame['layer'] == layer].sort_values('condition_value')
        condition_group = str(layer_frame['condition_group'].iloc[0])
        values = layer_frame['condition_value'].to_numpy()
        x = np.arange(len(values))
        width = 0.24

        for metric, y_label in metric_labels.items():
            fig, axis = plt.subplots(figsize=(8, 4.8))
            for group_index, (group_name, label) in enumerate(GROUPS):
                offset = (group_index - 1) * width
                axis.bar(
                    x + offset,
                    layer_frame[f'{metric}_{group_name}'],
                    width=width,
                    yerr=layer_frame[f'ci95_{metric}_{group_name}'],
                    capsize=3,
                    label=label,
                )
            axis.set_xticks(x, [f'{value:g}' for value in values])
            axis.set_xlabel(condition_x_label(condition_group))
            axis.set_ylabel(y_label)
            window_size = int(layer_frame['window_size'].iloc[0])
            axis.set_title(
                f'VGG layer {layer} ({window_size}x{window_size} target region)'
            )
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.legend()
            fig.tight_layout()
            metric_stem = (
                metric[len('mean_'):]
                if metric.startswith('mean_')
                else metric
            )
            fig.savefig(plot_dir / f'layer_{layer}_{metric_stem}.png', dpi=300)
            plt.close(fig)


def save_grouped_metric_plots(
        grouped_rows: Sequence[dict],
        plot_dir: Path,
        metric_labels: dict,
        include_layer: bool,
    ) -> None:
    if not grouped_rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(grouped_rows)
    layers = sorted(frame['layer'].unique()) if include_layer else [None]

    for layer in layers:
        layer_frame = frame if layer is None else frame[frame['layer'] == layer]
        layer_frame = layer_frame.sort_values('condition_value')
        condition_group = str(layer_frame['condition_group'].iloc[0])
        values = layer_frame['condition_value'].to_numpy(dtype=np.float64)
        x = np.arange(len(values))
        width = 0.24
        for metric, y_label in metric_labels.items():
            fig, axis = plt.subplots(figsize=(8, 4.8))
            for group_index, (group_name, label) in enumerate(GROUPS):
                offset = (group_index - 1) * width
                axis.bar(
                    x + offset,
                    layer_frame[f'{metric}_{group_name}'],
                    width=width,
                    yerr=layer_frame[f'ci95_{metric}_{group_name}'],
                    capsize=3,
                    label=label,
                )
            axis.set_xticks(x, [f'{value:g}' for value in values])
            axis.set_xlabel(condition_x_label(condition_group))
            axis.set_ylabel(y_label)
            title = y_label
            filename_prefix = ''
            if layer is not None:
                window_size = int(layer_frame['window_size'].iloc[0])
                title = (
                    f'VGG layer {layer} '
                    f'({window_size}x{window_size} target region)'
                )
                filename_prefix = f'layer_{layer}_'
            axis.set_title(title)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.legend()
            fig.tight_layout()
            fig.savefig(
                plot_dir / f'{filename_prefix}{metric}.png',
                dpi=300,
            )
            plt.close(fig)


def save_grouped_cue_target_plots(
        grouped_rows: Sequence[dict],
        plot_dir: Path,
    ) -> None:
    save_grouped_metric_plots(
        grouped_rows,
        plot_dir,
        metric_labels={
            'mean_cue_transformed_cosine_similarity': (
                'Cue-to-transformed-target cosine similarity'
            ),
            'mean_cue_similarity_drop': 'Cue-to-target similarity drop',
            'mean_cue_transformed_rms_euclidean_distance': (
                'Cue-to-transformed-target RMS distance'
            ),
        },
        include_layer=True,
    )


def save_grouped_performance_plots(
        grouped_rows: Sequence[dict],
        plot_dir: Path,
    ) -> None:
    save_grouped_metric_plots(
        grouped_rows,
        plot_dir,
        metric_labels={
            'transformed_score_target': 'Transformed target attention score',
            'transformed_score_margin': 'Target-minus-best-distractor margin',
            'transformed_target_rank': 'Target rank (0 is best)',
            'target_score_drop': 'Target attention-score drop',
        },
        include_layer=False,
    )


def save_cell_distance_matrix(
        distance_rows: Sequence[dict],
        window_size: int,
        title: str,
        path: Path,
    ) -> None:
    """Plot the colleague-proposed 1x1, 3x3, or 5x5 distance matrices."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (
        ('euclidean_distance', 'Euclidean distance'),
        ('rms_euclidean_distance', 'RMS Euclidean distance'),
        ('cosine_distance', 'Cosine distance'),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (metric, label) in zip(axes, metrics):
        matrix = np.asarray(
            [row[metric] for row in distance_rows],
            dtype=np.float64,
        ).reshape(window_size, window_size)
        image = axis.imshow(matrix, cmap='viridis')
        axis.set_title(label)
        axis.set_xlabel('Column offset')
        axis.set_ylabel('Row offset')
        offsets = np.arange(window_size) - window_size // 2
        axis.set_xticks(np.arange(window_size), offsets)
        axis.set_yticks(np.arange(window_size), offsets)
        for row in range(window_size):
            for col in range(window_size):
                axis.text(
                    col,
                    row,
                    f'{matrix[row, col]:.2f}',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white' if matrix[row, col] > matrix.max() / 2 else 'black',
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _correlation(values_x: np.ndarray, values_y: np.ndarray) -> dict:
    finite = np.isfinite(values_x) & np.isfinite(values_y)
    values_x = values_x[finite]
    values_y = values_y[finite]
    if (
            len(values_x) < 3
            or np.std(values_x) <= np.finfo(np.float64).eps
            or np.std(values_y) <= np.finfo(np.float64).eps
        ):
        return {
            'n': int(len(values_x)),
            'pearson_r': float('nan'),
            'pearson_p': float('nan'),
            'spearman_r': float('nan'),
            'spearman_p': float('nan'),
        }
    pearson = stats.pearsonr(values_x, values_y)
    spearman = stats.spearmanr(values_x, values_y)
    return {
        'n': int(len(values_x)),
        'pearson_r': float(pearson.statistic),
        'pearson_p': float(pearson.pvalue),
        'spearman_r': float(spearman.statistic),
        'spearman_p': float(spearman.pvalue),
    }


def build_feature_performance_correlations(
        feature_rows: Sequence[dict],
        cue_rows: Sequence[dict],
        performance_rows: Sequence[dict],
    ) -> List[dict]:
    """Correlate representation changes with IVSN-compatible target scores."""
    if not feature_rows or not cue_rows or not performance_rows:
        return []
    join_keys = [
        'condition_name', 'condition_group', 'condition_value',
        'unique_id', 'trial_type', 'layer', 'window_size',
    ]
    performance_keys = [
        'condition_name', 'condition_group', 'condition_value',
        'unique_id', 'trial_type',
    ]
    frame = pd.DataFrame(feature_rows).merge(
        pd.DataFrame(cue_rows),
        on=join_keys,
        how='inner',
        suffixes=('', '_cue'),
    ).merge(
        pd.DataFrame(performance_rows),
        on=performance_keys,
        how='inner',
        suffixes=('', '_performance'),
    )
    comparisons = (
        ('mean_cosine_distance', 'target_score_drop'),
        ('pooled_mean_cosine_distance', 'target_score_drop'),
        ('best_match_cosine_distance', 'target_score_drop'),
        ('mean_cue_transformed_cosine_similarity', 'transformed_score_target'),
        ('mean_cue_similarity_drop', 'target_score_drop'),
        ('mean_cue_transformed_cosine_similarity', 'transformed_score_margin'),
    )
    result = []
    group_keys = [
        'condition_name', 'condition_group', 'condition_value',
        'layer', 'window_size',
    ]
    for key_values, condition_frame in frame.groupby(group_keys, sort=False):
        metadata = dict(zip(group_keys, key_values))
        for group_name, _ in GROUPS:
            if group_name == 'all':
                subset = condition_frame
            else:
                trial_type = group_name[len('target_'):]
                subset = condition_frame[condition_frame['trial_type'] == trial_type]
            for predictor, outcome in comparisons:
                correlation = _correlation(
                    subset[predictor].to_numpy(dtype=np.float64),
                    subset[outcome].to_numpy(dtype=np.float64),
                )
                result.append({
                    **metadata,
                    'scope': 'within_condition',
                    'group': group_name,
                    'predictor': predictor,
                    'outcome': outcome,
                    **correlation,
                })
    across_keys = ['condition_group', 'layer', 'window_size']
    for key_values, layer_frame in frame.groupby(across_keys, sort=False):
        metadata = dict(zip(across_keys, key_values))
        metadata.update({
            'condition_name': 'all_conditions',
            'condition_value': float('nan'),
        })
        for group_name, _ in GROUPS:
            if group_name == 'all':
                subset = layer_frame
            else:
                trial_type = group_name[len('target_'):]
                subset = layer_frame[layer_frame['trial_type'] == trial_type]
            for predictor, outcome in comparisons:
                correlation = _correlation(
                    subset[predictor].to_numpy(dtype=np.float64),
                    subset[outcome].to_numpy(dtype=np.float64),
                )
                result.append({
                    **metadata,
                    'scope': 'across_conditions',
                    'group': group_name,
                    'predictor': predictor,
                    'outcome': outcome,
                    **correlation,
                })
    return result


def save_feature_performance_scatter_plots(
        feature_rows: Sequence[dict],
        cue_rows: Sequence[dict],
        performance_rows: Sequence[dict],
        plot_dir: Path,
    ) -> None:
    if not feature_rows or not cue_rows or not performance_rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    join_keys = [
        'condition_name', 'condition_group', 'condition_value',
        'unique_id', 'trial_type', 'layer', 'window_size',
    ]
    performance_keys = [
        'condition_name', 'condition_group', 'condition_value',
        'unique_id', 'trial_type',
    ]
    frame = pd.DataFrame(feature_rows).merge(
        pd.DataFrame(cue_rows),
        on=join_keys,
        how='inner',
    ).merge(
        pd.DataFrame(performance_rows),
        on=performance_keys,
        how='inner',
    )
    plots = (
        (
            'mean_cosine_distance',
            'target_score_drop',
            'Search-feature cosine distance',
            'Target attention-score drop',
            'feature_distance_vs_score_drop',
        ),
        (
            'mean_cue_transformed_cosine_similarity',
            'transformed_score_target',
            'Cue-to-target cosine similarity',
            'Transformed target attention score',
            'cue_similarity_vs_target_score',
        ),
    )
    for layer in sorted(frame['layer'].unique()):
        layer_frame = frame[frame['layer'] == layer]
        for x_metric, y_metric, x_label, y_label, stem in plots:
            fig, axis = plt.subplots(figsize=(6.5, 5.2))
            scatter = axis.scatter(
                layer_frame[x_metric],
                layer_frame[y_metric],
                c=layer_frame['condition_value'],
                cmap='viridis',
                alpha=0.35,
                s=18,
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(f'VGG layer {layer}')
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            colorbar = fig.colorbar(scatter, ax=axis)
            colorbar.set_label(condition_x_label(
                str(layer_frame['condition_group'].iloc[0])
            ))
            fig.tight_layout()
            fig.savefig(plot_dir / f'layer_{layer}_{stem}.png', dpi=300)
            plt.close(fig)


def save_erf_figure(
        original_image: Image.Image,
        transformed_image: Image.Image,
        original_erf: np.ndarray,
        transformed_erf: np.ndarray,
        title: str,
        path: Path,
    ) -> None:
    """Save paired inputs, ERF overlays, and their absolute differences."""
    path.parent.mkdir(parents=True, exist_ok=True)
    display_size = (original_erf.shape[1], original_erf.shape[0])
    original_small = original_image.resize(display_size, Image.BILINEAR)
    transformed_small = transformed_image.resize(display_size, Image.BILINEAR)
    original_rgb = np.asarray(original_small, dtype=np.float32) / 255.0
    transformed_rgb = np.asarray(transformed_small, dtype=np.float32) / 255.0
    pixel_difference = np.abs(original_rgb - transformed_rgb).mean(axis=2)

    original_normalized = normalized_map(original_erf)
    transformed_normalized = normalized_map(transformed_erf)
    erf_difference = np.abs(original_normalized - transformed_normalized)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = (
        (axes[0, 0], original_rgb, 'Original-target search image', None),
        (axes[0, 1], transformed_rgb, 'Transformed-target search image', None),
        (axes[0, 2], pixel_difference, 'Input absolute difference', 'gray'),
    )
    for axis, values, panel_title, cmap in panels:
        axis.imshow(values, cmap=cmap)
        axis.set_title(panel_title)
        axis.axis('off')

    axes[1, 0].imshow(original_rgb)
    axes[1, 0].imshow(original_normalized, cmap='magma', alpha=0.55, vmin=0, vmax=1)
    axes[1, 0].set_title('Original-target ERF')
    axes[1, 1].imshow(transformed_rgb)
    axes[1, 1].imshow(transformed_normalized, cmap='magma', alpha=0.55, vmin=0, vmax=1)
    axes[1, 1].set_title('Transformed-target ERF')
    axes[1, 2].imshow(erf_difference, cmap='viridis', vmin=0, vmax=1)
    axes[1, 2].set_title('ERF absolute difference')
    for axis in axes[1]:
        axis.axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_unit_mass_search_erf_figure(
        original_image: Image.Image,
        transformed_image: Image.Image,
        original_erf: np.ndarray,
        transformed_erf: np.ndarray,
        title: str,
        path: Path,
    ) -> None:
    """Save the original full-search ERF layout using unit-mass maps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    display_size = (original_erf.shape[1], original_erf.shape[0])
    original_rgb = np.asarray(
        original_image.resize(display_size, Image.BILINEAR), dtype=np.float32
    ) / 255.0
    transformed_rgb = np.asarray(
        transformed_image.resize(display_size, Image.BILINEAR), dtype=np.float32
    ) / 255.0
    pixel_difference = np.abs(original_rgb - transformed_rgb).mean(axis=2)
    original_mass = unit_mass_map(original_erf)
    transformed_mass = unit_mass_map(transformed_erf)
    erf_difference = np.abs(original_mass - transformed_mass)
    shared_peak = max(
        float(original_mass.max()),
        float(transformed_mass.max()),
        np.finfo(np.float32).eps,
    )
    difference_peak = max(
        float(erf_difference.max()),
        np.finfo(np.float32).eps,
    )
    metrics = erf_distance_metrics(original_erf, transformed_erf)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = (
        (axes[0, 0], original_rgb, 'Original-target search image', None),
        (axes[0, 1], transformed_rgb, 'Transformed-target search image', None),
        (axes[0, 2], pixel_difference, 'Input absolute difference', 'gray'),
    )
    for axis, values, panel_title, cmap in panels:
        axis.imshow(values, cmap=cmap)
        axis.set_title(panel_title)
        axis.axis('off')

    axes[1, 0].imshow(original_rgb)
    axes[1, 0].imshow(
        original_mass, cmap='magma', alpha=0.55,
        vmin=0, vmax=shared_peak,
    )
    axes[1, 0].set_title('Original-target unit-mass ERF')
    axes[1, 1].imshow(transformed_rgb)
    axes[1, 1].imshow(
        transformed_mass, cmap='magma', alpha=0.55,
        vmin=0, vmax=shared_peak,
    )
    axes[1, 1].set_title('Transformed-target unit-mass ERF')
    axes[1, 2].imshow(
        erf_difference, cmap='viridis', vmin=0, vmax=difference_peak
    )
    axes[1, 2].set_title('Absolute unit-mass difference')
    axes[1, 2].text(
        0.02, 0.98,
        f"TV distance = {metrics['erf_total_variation_distance']:.2f}",
        transform=axes[1, 2].transAxes, va='top', color='white',
        fontsize=9, bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 3},
    )
    for axis in axes[1]:
        axis.axis('off')

    fig.suptitle(f'{title}\nEach complete search-image ERF sums to 1')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_erf_arrays(
        original_erf: np.ndarray,
        transformed_erf: np.ndarray,
        path: Path,
    ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        original_erf=original_erf,
        transformed_erf=transformed_erf,
        original_erf_normalized=normalized_map(original_erf),
        transformed_erf_normalized=normalized_map(transformed_erf),
        original_erf_unit_mass=unit_mass_map(original_erf),
        transformed_erf_unit_mass=unit_mass_map(transformed_erf),
    )
