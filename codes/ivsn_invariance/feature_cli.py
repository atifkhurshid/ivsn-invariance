"""Command-line orchestration for VGG transformation feature analyses."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import List, Sequence, Set, Tuple

from . import runtime
from .data import load_base_manifest, load_dataset, save_base_manifest
from .domain import TransformSpec, Trial
from .feature_analysis import (
    VGG16FeatureProbe,
    attention_position_metrics,
    centered_region,
    cue_target_distance_rows,
    effective_receptive_field,
    erf_distance_metrics,
    feature_distance_rows,
    parse_layer_windows,
    pooled_and_spatially_tolerant_metrics,
    summarize_cell_distances,
    summarize_cue_target_rows,
)
from .feature_reporting import (
    build_feature_performance_correlations,
    build_grouped_cue_target_summary,
    build_grouped_feature_summary,
    build_grouped_performance_summary,
    save_cell_distance_matrix,
    save_erf_arrays,
    save_erf_figure,
    save_unit_mass_search_erf_figure,
    save_feature_performance_scatter_plots,
    save_grouped_cue_target_plots,
    save_grouped_feature_plots,
    save_grouped_performance_plots,
    write_rows_csv,
)
from .feature_imaging import render_feature_cue, render_paired_search_displays
from .runtime import (
    DEFAULT_N_DIFFERENT,
    DEFAULT_N_IDENTICAL,
    IMAGE_SIZE,
    ORACLE_WINDOW,
    SEED,
)
from .trials import build_trials_from_base, make_condition_specs, sample_base_trials


@dataclass
class TrialAnalysisOutput:
    feature_cell_rows: List[dict]
    feature_trial_rows: List[dict]
    cue_cell_rows: List[dict]
    cue_trial_rows: List[dict]
    erf_rows: List[dict]
    performance_row: dict


def parse_args(argv: Sequence[str] = None):
    parser = argparse.ArgumentParser(
        description=(
            'Compare VGG16 target-region features and effective receptive fields '
            'between original-target and transformed-target search displays.'
        )
    )
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n-identical', type=int, default=DEFAULT_N_IDENTICAL)
    parser.add_argument('--n-different', type=int, default=DEFAULT_N_DIFFERENT)
    parser.add_argument('--load-base-manifest', type=str, default=None)
    parser.add_argument('--no-save-base-manifest', action='store_true')

    parser.add_argument('--arrangement', choices=['grid', 'circle'], default='circle')
    parser.add_argument('--n-objects', choices=[6, 8], type=int, default=8)
    parser.add_argument('--n-matrix', choices=[2, 3, 4], type=int, default=3)
    parser.add_argument('--padding', type=int, default=30)
    parser.add_argument('--radius', type=int, default=None)
    parser.add_argument('--jitter', type=float, default=0.0)

    parser.add_argument(
        '--transform-mode',
        choices=[
            'original', 'rotation', 'scale', 'shift_x', 'shift_y',
            'skew_x', 'skew_y', 'noise', 'blur', 'mixed',
        ],
        required=True,
    )
    parser.add_argument(
        '--rotation-values', type=float, nargs='*',
        default=[0, 30, 60, 90, 120, 150, 180],
    )
    parser.add_argument(
        '--scale-values', type=float, nargs='*',
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
    )
    parser.add_argument(
        '--shift-values', type=float, nargs='*', default=[-30, -15, 0, 15, 30],
    )
    parser.add_argument(
        '--skew-values', type=float, nargs='*', default=[-20, -10, 0, 10, 20],
    )
    parser.add_argument(
        '--noise-values', type=float, nargs='*',
        default=[0.0, 0.03, 0.06, 0.09, 0.12],
    )
    parser.add_argument(
        '--blur-values', type=float, nargs='*', default=[0.0, 0.5, 1.0, 2.0, 3.0],
    )

    parser.add_argument('--smoothing-mode', choices=['none', 'alpha', 'cosine'], default='cosine')
    parser.add_argument('--alpha-soften-blur-radius', type=float, default=3.0)
    parser.add_argument('--edge-taper-width-px', type=float, default=5.0)
    parser.add_argument('--smooth-target', action='store_true')
    parser.add_argument('--smooth-distractors', action='store_true')
    parser.add_argument('--smooth-cue', action='store_true')
    parser.add_argument(
        '--cue-size',
        type=int,
        default=32,
        help='VGG cue input size; 32 matches the original IVSN VGG model.',
    )

    parser.add_argument(
        '--layer-windows',
        nargs='*',
        default=['16:5', '23:3', '30:1'],
        metavar='LAYER:WINDOW',
        help=(
            'VGG feature-module cut point and odd target-region width. '
            'Defaults: 16:5 23:3 30:1.'
        ),
    )
    parser.add_argument(
        '--erf-examples-per-group',
        type=int,
        default=1,
        help=(
            'ERF examples per condition for each of target-identical and '
            'target-different trials. Set to 0 to skip costly gradients.'
        ),
    )
    args = parser.parse_args(argv)

    if args.n_identical < 0 or args.n_different < 0:
        parser.error('Trial counts must be non-negative.')
    if args.n_identical + args.n_different == 0:
        parser.error('At least one trial is required.')
    if args.erf_examples_per_group < 0:
        parser.error('--erf-examples-per-group must be non-negative.')
    if args.cue_size < 32:
        parser.error('--cue-size must be at least 32 pixels for VGG layer 30.')
    try:
        args.layer_windows = parse_layer_windows(args.layer_windows)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def configure_runtime(args) -> int:
    if args.arrangement == 'circle':
        runtime.set_runtime_geometry_circle(args.n_objects, args.radius, args.jitter)
        return args.n_objects
    runtime.set_runtime_geometry_grid(args.n_matrix, args.padding, args.jitter)
    return args.n_matrix * args.n_matrix


def fixed_context_trials(base_trials, condition_spec) -> List[Trial]:
    """Build trials while holding every non-target transformation constant."""
    condition_name, condition_group, condition_value, sweep_spec = condition_spec
    trials = build_trials_from_base(
        base_trials,
        condition_name,
        condition_group,
        condition_value,
        sweep_spec,
        transform_cue_too=False,
        couple_cue_to_target=False,
    )
    by_id = {base_trial.unique_id: base_trial for base_trial in base_trials}
    for trial in trials:
        base_trial = by_id[trial.unique_id]
        trial.distractor_transforms = [
            asdict(TransformSpec(rotation_deg=angle))
            for angle in base_trial.distractor_rotations
        ]
    return trials


def select_erf_examples(trials: Sequence[Trial], per_group: int) -> Set[int]:
    selected = set()
    if per_group == 0:
        return selected
    for trial_type in ('identical', 'different'):
        group = [trial for trial in trials if trial.trial_type == trial_type]
        selected.update(trial.unique_id for trial in group[:per_group])
    return selected


def trial_metadata(trial: Trial, layer: int, window_size: int) -> dict:
    return {
        'condition_name': trial.condition_name,
        'condition_group': trial.condition_group,
        'condition_value': trial.condition_value,
        'unique_id': trial.unique_id,
        'repeat_id': trial.repeat_id,
        'trial_type': trial.trial_type,
        'target_category': trial.target_category,
        'target_position': trial.target_position,
        'cue_path': trial.cue_path,
        'target_path': trial.target_path,
        'layer': layer,
        'window_size': window_size,
    }


def analyze_trial(
        probe: VGG16FeatureProbe,
        trial: Trial,
        args,
        compute_erf: bool,
        erf_root: Path,
        matrix_root: Path,
    ) -> TrialAnalysisOutput:
    deterministic_seed = args.seed + 1_000_003 * int(trial.unique_id)
    original_image, transformed_image, target_center = render_paired_search_displays(
        target_path=Path(trial.target_path),
        distractor_paths=[Path(path) for path in trial.distractor_paths],
        target_position=trial.target_position,
        transformed_target_spec=TransformSpec(**trial.target_transform),
        distractor_transforms=[
            TransformSpec(**spec) for spec in trial.distractor_transforms
        ],
        args=args,
        position_rng=random.Random(deterministic_seed),
        noise_seed=deterministic_seed,
    )

    cue_image = render_feature_cue(Path(trial.cue_path), args)
    cue_activations = probe.activations(cue_image, image_size=args.cue_size)
    original_activations = probe.activations(original_image)
    transformed_activations = probe.activations(transformed_image)
    feature_cell_rows = []
    feature_trial_rows = []
    cue_cell_rows = []
    cue_trial_rows = []
    regions = {}

    for layer, window_size in args.layer_windows.items():
        original_activation = original_activations[layer]
        transformed_activation = transformed_activations[layer]
        feature_size = tuple(int(value) for value in original_activation.shape[-2:])
        region = centered_region(
            layer=layer,
            window_size=window_size,
            point_xy=target_center,
            image_size=original_image.size,
            feature_size=feature_size,
        )
        regions[layer] = region
        distances = feature_distance_rows(
            original_activation,
            transformed_activation,
            region,
        )
        metadata = trial_metadata(trial, layer, window_size)
        for distance in distances:
            feature_cell_rows.append({**metadata, **distance})
        feature_trial_rows.append({
            **metadata,
            'target_center_x': target_center[0],
            'target_center_y': target_center[1],
            'feature_center_row': region.center_row,
            'feature_center_col': region.center_col,
            **summarize_cell_distances(distances),
            **pooled_and_spatially_tolerant_metrics(
                original_activation,
                transformed_activation,
                region,
            ),
        })

        cue_distances = cue_target_distance_rows(
            cue_activations[layer],
            original_activation,
            transformed_activation,
            region,
        )
        for distance in cue_distances:
            cue_cell_rows.append({**metadata, **distance})
        cue_trial_rows.append({
            **metadata,
            **summarize_cue_target_rows(cue_distances),
        })

        if compute_erf:
            stem = f'trial_{trial.unique_id}_{trial.trial_type}_layer_{layer}'
            save_cell_distance_matrix(
                distances,
                window_size=window_size,
                title=(
                    f'{trial.condition_name} | {trial.trial_type} | '
                    f'VGG layer {layer}'
                ),
                path=matrix_root / trial.condition_name / f'{stem}.png',
            )

    original_performance = attention_position_metrics(
        cue_activations[30],
        original_activations[30],
        positions=runtime.POSITIONS,
        target_position=trial.target_position,
        image_size=IMAGE_SIZE,
        oracle_window=ORACLE_WINDOW,
    )
    transformed_performance = attention_position_metrics(
        cue_activations[30],
        transformed_activations[30],
        positions=runtime.POSITIONS,
        target_position=trial.target_position,
        image_size=IMAGE_SIZE,
        oracle_window=ORACLE_WINDOW,
    )
    performance_row = {
        'condition_name': trial.condition_name,
        'condition_group': trial.condition_group,
        'condition_value': trial.condition_value,
        'unique_id': trial.unique_id,
        'repeat_id': trial.repeat_id,
        'trial_type': trial.trial_type,
        'target_category': trial.target_category,
        'target_position': trial.target_position,
        'cue_path': trial.cue_path,
        'target_path': trial.target_path,
        'n_objects': len(runtime.POSITIONS),
    }
    performance_row.update({
        f'original_{name}': value
        for name, value in original_performance.items()
    })
    performance_row.update({
        f'transformed_{name}': value
        for name, value in transformed_performance.items()
    })
    performance_row.update({
        'target_score_drop': (
            original_performance['score_target']
            - transformed_performance['score_target']
        ),
        'score_margin_drop': (
            original_performance['score_margin']
            - transformed_performance['score_margin']
        ),
        'rank_increase': (
            transformed_performance['target_rank']
            - original_performance['target_rank']
        ),
        'fixation_increase': (
            transformed_performance['n_fixations']
            - original_performance['n_fixations']
        ),
    })

    erf_rows = []
    if compute_erf:
        original_input, original_gradient_activations = (
            probe.activations_with_input_gradient(original_image)
        )
        transformed_input, transformed_gradient_activations = (
            probe.activations_with_input_gradient(transformed_image)
        )
        layers = list(args.layer_windows)
        for layer_index, layer in enumerate(layers):
            keep_graph = layer_index < len(layers) - 1
            region = regions[layer]
            original_erf = effective_receptive_field(
                original_input,
                original_gradient_activations[layer],
                region.coordinates,
                retain_graph=keep_graph,
            )
            transformed_erf = effective_receptive_field(
                transformed_input,
                transformed_gradient_activations[layer],
                region.coordinates,
                retain_graph=keep_graph,
            )
            metadata = trial_metadata(trial, layer, region.window_size)
            erf_rows.append({
                **metadata,
                **erf_distance_metrics(original_erf, transformed_erf),
            })

            stem = f'trial_{trial.unique_id}_{trial.trial_type}_layer_{layer}'
            condition_dir = erf_root / trial.condition_name
            save_erf_figure(
                original_image,
                transformed_image,
                original_erf,
                transformed_erf,
                title=(
                    f'{trial.condition_name} | {trial.trial_type} | '
                    f'VGG layer {layer} | {region.window_size}x{region.window_size}'
                ),
                path=condition_dir / f'{stem}.png',
            )
            unit_mass_dir = (
                erf_root.parent / 'erf_unit_mass_examples'
                / trial.condition_name
            )
            save_unit_mass_search_erf_figure(
                original_image,
                transformed_image,
                original_erf,
                transformed_erf,
                title=(
                    f'{trial.condition_name} | {trial.trial_type} | '
                    f'VGG layer {layer} | '
                    f'{region.window_size}x{region.window_size}'
                ),
                path=unit_mass_dir / f'{stem}.png',
            )
            save_erf_arrays(
                original_erf,
                transformed_erf,
                path=condition_dir / f'{stem}.npz',
            )

    return TrialAnalysisOutput(
        feature_cell_rows=feature_cell_rows,
        feature_trial_rows=feature_trial_rows,
        cue_cell_rows=cue_cell_rows,
        cue_trial_rows=cue_trial_rows,
        erf_rows=erf_rows,
        performance_row=performance_row,
    )


def save_config(args, out_dir: Path, n_objects: int) -> None:
    config = vars(args).copy()
    config['data_root'] = str(Path(args.data_root).resolve())
    config['out_dir'] = str(out_dir.resolve())
    config['layer_windows'] = {
        str(layer): window for layer, window in args.layer_windows.items()
    }
    config['n_objects_effective'] = n_objects
    config['comparison_design'] = {
        'varied_pixels': 'target transformation only',
        'paired_jitter': True,
        'fixed_distractor_pixels_within_pair': True,
        'fixed_distractor_rotations_across_condition_values': True,
        'feature_cell_correspondence': 'same spatial cell before and after transformation',
        'spatially_tolerant_comparison': 'symmetric nearest-cell distance',
        'region_pooling': ['mean', 'max'],
        'cue_pooling': 'adaptive max pooling',
        'cue_target_comparison': True,
        'performance_layer': 30,
        'performance_metric_source': 'IVSN-compatible cue-search attention scores',
        'erf_aggregation': 'sum of per-cell absolute input-gradient maps',
    }
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)


def main(argv: Sequence[str] = None) -> None:
    args = parse_args(argv)
    if args.smoothing_mode != 'none' and not (
            args.smooth_target or args.smooth_distractors
        ):
        args.smooth_target = True

    runtime.set_seed(args.seed)
    n_objects = configure_runtime(args)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, out_dir, n_objects)

    print('Loading dataset...')
    dataset = load_dataset(data_root)
    if args.load_base_manifest:
        base_trials = load_base_manifest(Path(args.load_base_manifest))
    else:
        base_trials = sample_base_trials(
            dataset,
            args.n_identical,
            args.n_different,
        )
        if not args.no_save_base_manifest:
            save_base_manifest(base_trials, out_dir / 'base_trials_manifest.json')

    print(f'Loading VGG16 feature probe on {args.device}...')
    probe_layers = set(args.layer_windows)
    probe_layers.add(30)
    probe = VGG16FeatureProbe(probe_layers, device=args.device)
    condition_specs = make_condition_specs(args)
    all_feature_cell_rows = []
    all_feature_trial_rows = []
    all_cue_cell_rows = []
    all_cue_trial_rows = []
    all_erf_rows = []
    all_performance_rows = []

    for condition_index, condition_spec in enumerate(condition_specs, start=1):
        condition_name = condition_spec[0]
        trials = fixed_context_trials(base_trials, condition_spec)
        erf_ids = select_erf_examples(trials, args.erf_examples_per_group)
        print(
            f'Condition {condition_index}/{len(condition_specs)}: '
            f'{condition_name} ({len(trials)} trials)'
        )
        for trial_index, trial in enumerate(trials, start=1):
            output = analyze_trial(
                probe,
                trial,
                args,
                compute_erf=trial.unique_id in erf_ids,
                erf_root=out_dir / 'erf_examples',
                matrix_root=out_dir / 'distance_matrix_examples',
            )
            all_feature_cell_rows.extend(output.feature_cell_rows)
            all_feature_trial_rows.extend(output.feature_trial_rows)
            all_cue_cell_rows.extend(output.cue_cell_rows)
            all_cue_trial_rows.extend(output.cue_trial_rows)
            all_erf_rows.extend(output.erf_rows)
            all_performance_rows.append(output.performance_row)
            if trial_index % 25 == 0 or trial_index == len(trials):
                print(f'  completed {trial_index}/{len(trials)} trials')

    grouped_feature_rows = build_grouped_feature_summary(all_feature_trial_rows)
    grouped_cue_rows = build_grouped_cue_target_summary(all_cue_trial_rows)
    grouped_performance_rows = build_grouped_performance_summary(
        all_performance_rows
    )
    correlation_rows = build_feature_performance_correlations(
        all_feature_trial_rows,
        all_cue_trial_rows,
        all_performance_rows,
    )

    write_rows_csv(
        all_feature_cell_rows,
        out_dir / 'feature_cell_distances.csv',
    )
    write_rows_csv(
        all_feature_trial_rows,
        out_dir / 'feature_trial_distances.csv',
    )
    write_rows_csv(
        grouped_feature_rows,
        out_dir / 'grouped_feature_distances.csv',
    )
    write_rows_csv(all_cue_cell_rows, out_dir / 'cue_target_cell_metrics.csv')
    write_rows_csv(all_cue_trial_rows, out_dir / 'cue_target_trial_metrics.csv')
    write_rows_csv(grouped_cue_rows, out_dir / 'grouped_cue_target_metrics.csv')
    write_rows_csv(all_performance_rows, out_dir / 'search_performance.csv')
    write_rows_csv(
        grouped_performance_rows,
        out_dir / 'grouped_search_performance.csv',
    )
    write_rows_csv(
        correlation_rows,
        out_dir / 'feature_performance_correlations.csv',
    )
    write_rows_csv(all_erf_rows, out_dir / 'erf_example_metrics.csv')
    save_grouped_feature_plots(grouped_feature_rows, out_dir / 'plots')
    save_grouped_cue_target_plots(grouped_cue_rows, out_dir / 'cue_target_plots')
    save_grouped_performance_plots(
        grouped_performance_rows,
        out_dir / 'performance_plots',
    )
    save_feature_performance_scatter_plots(
        all_feature_trial_rows,
        all_cue_trial_rows,
        all_performance_rows,
        out_dir / 'correlation_plots',
    )

    print(f'Feature analysis complete. Results written to: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
