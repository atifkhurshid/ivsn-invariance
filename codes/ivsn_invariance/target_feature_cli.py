"""CLI for isolated-target VGG activation and ERF experiments."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Sequence

from . import runtime
from .data import (
    load_base_manifest,
    load_dataset,
    save_base_manifest,
)
from .domain import BaseTrial
from .feature_analysis import VGG16FeatureProbe
from .feature_reporting import write_rows_csv
from .target_feature_analysis import (
    activation_difference_metrics,
    full_layer_erf,
    pooled_activation_difference_metrics,
)
from .target_feature_imaging import render_paired_targets
from .target_feature_reporting import (
    align_erf_to_original,
    aligned_erf_metrics,
    build_grouped_aligned_erf_summary,
    build_grouped_pooled_block_summary,
    build_grouped_target_activation_summary,
    save_aligned_erf_comparison_figure,
    save_aligned_erf_summary_plots,
    save_grouped_target_activation_plots,
    save_pooled_block_activation_plots,
    save_target_erf_arrays,
    save_target_erf_figure,
    save_unit_mass_erf_comparison_figure,
    save_unit_mass_direct_erf_figure,
)
from .trials import make_condition_specs, sample_base_trials


@dataclass(frozen=True)
class TargetRecord:
    unique_id: int
    repeat_id: int
    trial_type: str
    target_category: str
    target_path: str


def parse_args(argv: Sequence[str] = None):
    parser = argparse.ArgumentParser(
        description=(
            'Compare VGG activations and effective receptive fields for '
            'isolated 32x32 original and transformed target images.'
        )
    )
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=runtime.SEED)
    parser.add_argument('--n-identical', type=int, default=runtime.DEFAULT_N_IDENTICAL)
    parser.add_argument('--n-different', type=int, default=runtime.DEFAULT_N_DIFFERENT)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        '--targets-csv',
        type=str,
        default=None,
        help=(
            'Existing per-trial feature CSV. Unique target rows are reused, '
            'allowing the exact same 300 targets as a previous experiment.'
        ),
    )
    source.add_argument('--load-base-manifest', type=str, default=None)
    parser.add_argument('--no-save-base-manifest', action='store_true')

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
        '--blur-values', type=float, nargs='*',
        default=[0.0, 0.5, 1.0, 2.0, 3.0],
    )

    parser.add_argument(
        '--smoothing-mode',
        choices=['none', 'alpha', 'cosine'],
        default='cosine',
    )
    parser.add_argument('--alpha-soften-blur-radius', type=float, default=3.0)
    parser.add_argument('--edge-taper-width-px', type=float, default=5.0)
    parser.add_argument('--smooth-target', action='store_true')
    parser.add_argument(
        '--input-size',
        type=int,
        default=32,
        help='Square VGG input size. The requested target-only experiment uses 32.',
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[16, 23, 30],
        help='torchvision VGG feature-module cut points to record.',
    )
    parser.add_argument(
        '--pooled-block-layers',
        type=int,
        nargs='+',
        default=[5, 10, 17, 24, 31],
        help=(
            'VGG cut points for spatially averaged block vectors. Defaults '
            'to the five max-pool outputs requested for the block analysis.'
        ),
    )
    parser.add_argument(
        '--erf-images-per-class',
        type=int,
        default=3,
        help='Unique targets per category used for qualitative ERF figures.',
    )
    args = parser.parse_args(argv)

    if args.n_identical < 0 or args.n_different < 0:
        parser.error('Trial counts must be non-negative.')
    if args.n_identical + args.n_different == 0 and not (
            args.targets_csv or args.load_base_manifest
        ):
        parser.error('At least one trial is required.')
    requested_layers = args.layers + args.pooled_block_layers
    if args.input_size < 32 and max(requested_layers) >= 30:
        parser.error('--input-size must be at least 32 when using deep VGG layers.')
    if any(layer < 1 or layer > 31 for layer in args.layers):
        parser.error('--layers must be between 1 and 31.')
    if any(layer < 1 or layer > 31 for layer in args.pooled_block_layers):
        parser.error('--pooled-block-layers must be between 1 and 31.')
    if args.erf_images_per_class < 0:
        parser.error('--erf-images-per-class must be non-negative.')
    args.layers = sorted(set(args.layers))
    args.pooled_block_layers = list(dict.fromkeys(args.pooled_block_layers))
    return args


def _record_from_base(trial: BaseTrial) -> TargetRecord:
    return TargetRecord(
        unique_id=int(trial.unique_id),
        repeat_id=int(trial.repeat_id),
        trial_type=str(trial.trial_type),
        target_category=str(trial.target_category),
        target_path=str(trial.target_path),
    )


def load_target_records_csv(path: Path) -> List[TargetRecord]:
    """Load one target per unique trial from an existing result CSV."""
    required = {
        'unique_id', 'repeat_id', 'trial_type', 'target_category', 'target_path',
    }
    records: Dict[int, TargetRecord] = {}
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f'{path} is missing required columns: {sorted(missing)}.'
            )
        for row in reader:
            record = TargetRecord(
                unique_id=int(row['unique_id']),
                repeat_id=int(row['repeat_id']),
                trial_type=str(row['trial_type']),
                target_category=str(row['target_category']),
                target_path=str(row['target_path']),
            )
            previous = records.get(record.unique_id)
            if previous is not None and previous != record:
                raise ValueError(
                    f'Conflicting metadata for unique_id={record.unique_id}.'
                )
            records[record.unique_id] = record
    if not records:
        raise ValueError(f'No target records found in {path}.')
    return sorted(records.values(), key=lambda record: record.unique_id)


def load_or_sample_targets(args, out_dir: Path) -> List[TargetRecord]:
    if args.targets_csv:
        return load_target_records_csv(Path(args.targets_csv))
    if args.load_base_manifest:
        return [
            _record_from_base(trial)
            for trial in load_base_manifest(Path(args.load_base_manifest))
        ]

    dataset = load_dataset(Path(args.data_root))
    base_trials = sample_base_trials(
        dataset,
        args.n_identical,
        args.n_different,
    )
    if not args.no_save_base_manifest:
        save_base_manifest(base_trials, out_dir / 'base_trials_manifest.json')
    return [_record_from_base(trial) for trial in base_trials]


def select_erf_records(
        records: Sequence[TargetRecord],
        per_class: int,
    ) -> List[TargetRecord]:
    """Choose deterministic unique target images within every category."""
    if per_class == 0:
        return []
    selected = []
    seen_by_category: Dict[str, set] = {}
    count_by_category: Dict[str, int] = {}
    for record in sorted(records, key=lambda item: item.unique_id):
        category = record.target_category
        seen = seen_by_category.setdefault(category, set())
        count = count_by_category.get(category, 0)
        if count >= per_class or record.target_path in seen:
            continue
        selected.append(record)
        seen.add(record.target_path)
        count_by_category[category] = count + 1
    return selected


def _metadata(record: TargetRecord, condition_spec, layer: int) -> dict:
    condition_name, condition_group, condition_value, _ = condition_spec
    return {
        'condition_name': condition_name,
        'condition_group': condition_group,
        'condition_value': condition_value,
        'unique_id': record.unique_id,
        'repeat_id': record.repeat_id,
        'trial_type': record.trial_type,
        'target_category': record.target_category,
        'target_path': record.target_path,
        'layer': layer,
    }


def save_config(args, out_dir: Path, n_targets: int, n_erf_targets: int) -> None:
    config = vars(args).copy()
    config['data_root'] = str(Path(args.data_root).resolve())
    config['out_dir'] = str(out_dir.resolve())
    config['n_targets_effective'] = n_targets
    config['n_erf_targets_effective'] = n_erf_targets
    config['activation_comparison'] = (
        'elementwise absolute difference between full original/transformed '
        'layer activation tensors'
    )
    config['pooled_block_activation_comparison'] = (
        'spatial mean over HxW at each max-pool output, followed by the '
        'absolute difference between corresponding channels and their mean'
    )
    config['erf_definition'] = (
        'sum of absolute input-gradient maps for every spatial cell; each '
        'cell score is the sum over channels'
    )
    config['erf_alignment'] = (
        'transformed ERFs are inverse-warped with the stimulus affine before '
        'aligned comparison; non-geometric degradations use identity alignment'
    )
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)


def main(argv: Sequence[str] = None) -> None:
    args = parse_args(argv)
    if args.smoothing_mode != 'none' and not args.smooth_target:
        args.smooth_target = True

    runtime.set_seed(args.seed)
    runtime.set_runtime_geometry_circle(8, radius=None, jitter=0.0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_or_sample_targets(args, out_dir)
    missing_paths = [record.target_path for record in records if not Path(record.target_path).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f'{len(missing_paths)} target paths do not exist; first missing path: '
            f'{missing_paths[0]}'
        )

    selected_erf_records = select_erf_records(
        records,
        args.erf_images_per_class,
    )
    selected_erf_ids = {record.unique_id for record in selected_erf_records}
    save_config(args, out_dir, len(records), len(selected_erf_records))
    write_rows_csv(
        [asdict(record) for record in records],
        out_dir / 'target_trials.csv',
    )
    write_rows_csv(
        [asdict(record) for record in selected_erf_records],
        out_dir / 'selected_erf_targets.csv',
    )

    print(
        f'Loading VGG16 feature probe on {args.device}; '
        f'{len(records)} target trials, {len(selected_erf_records)} ERF targets...'
    )
    probe_layers = sorted(set(args.layers).union(args.pooled_block_layers))
    probe = VGG16FeatureProbe(probe_layers, device=args.device)
    condition_specs = make_condition_specs(args)
    activation_rows = []
    pooled_block_rows = []
    erf_rows = []

    for condition_index, condition_spec in enumerate(condition_specs, start=1):
        condition_name, _, _, transformed_spec = condition_spec
        print(
            f'Condition {condition_index}/{len(condition_specs)}: '
            f'{condition_name}'
        )
        for record_index, record in enumerate(records, start=1):
            noise_seed = (
                args.seed
                + 1_000_003 * int(record.unique_id)
                + 10_007 * condition_index
            )
            original_image, transformed_image = render_paired_targets(
                Path(record.target_path),
                transformed_spec,
                args,
                noise_seed=noise_seed,
            )
            original_activations = probe.activations(
                original_image,
                image_size=args.input_size,
            )
            transformed_activations = probe.activations(
                transformed_image,
                image_size=args.input_size,
            )
            for layer in args.layers:
                original_activation = original_activations[layer]
                transformed_activation = transformed_activations[layer]
                activation_rows.append({
                    **_metadata(record, condition_spec, layer),
                    'input_size': args.input_size,
                    'channels': int(original_activation.shape[1]),
                    'activation_height': int(original_activation.shape[2]),
                    'activation_width': int(original_activation.shape[3]),
                    **activation_difference_metrics(
                        original_activation,
                        transformed_activation,
                    ),
                })

            for block, layer in enumerate(args.pooled_block_layers, start=1):
                original_activation = original_activations[layer]
                transformed_activation = transformed_activations[layer]
                pooled_block_rows.append({
                    **_metadata(record, condition_spec, layer),
                    'block': block,
                    'input_size': args.input_size,
                    'channels': int(original_activation.shape[1]),
                    'activation_height': int(original_activation.shape[2]),
                    'activation_width': int(original_activation.shape[3]),
                    **pooled_activation_difference_metrics(
                        original_activation,
                        transformed_activation,
                    ),
                })

            if record.unique_id in selected_erf_ids:
                original_input, original_gradient_activations = (
                    probe.activations_with_input_gradient(
                        original_image,
                        image_size=args.input_size,
                    )
                )
                transformed_input, transformed_gradient_activations = (
                    probe.activations_with_input_gradient(
                        transformed_image,
                        image_size=args.input_size,
                    )
                )
                example_original_erfs = {}
                example_transformed_erfs = {}
                example_aligned_erfs = {}
                example_layer_shapes = {}
                for layer_index, layer in enumerate(args.layers):
                    keep_graph = layer_index < len(args.layers) - 1
                    original_erf = full_layer_erf(
                        original_input,
                        original_gradient_activations[layer],
                        retain_graph=keep_graph,
                    )
                    transformed_erf = full_layer_erf(
                        transformed_input,
                        transformed_gradient_activations[layer],
                        retain_graph=keep_graph,
                    )
                    aligned_erf = align_erf_to_original(
                        transformed_erf,
                        transformed_spec,
                    )
                    activation = original_gradient_activations[layer]
                    spatial_cells = int(activation.shape[2] * activation.shape[3])
                    example_original_erfs[layer] = original_erf
                    example_transformed_erfs[layer] = transformed_erf
                    example_aligned_erfs[layer] = aligned_erf
                    example_layer_shapes[layer] = (
                        int(activation.shape[1]),
                        int(activation.shape[2]),
                        int(activation.shape[3]),
                    )
                    metadata = _metadata(record, condition_spec, layer)
                    erf_rows.append({
                        **metadata,
                        'input_size': args.input_size,
                        'channels': int(activation.shape[1]),
                        'activation_height': int(activation.shape[2]),
                        'activation_width': int(activation.shape[3]),
                        'n_superimposed_cell_maps': spatial_cells,
                        **aligned_erf_metrics(
                            original_erf,
                            transformed_erf,
                            aligned_erf,
                        ),
                    })
                    stem = (
                        f'trial_{record.unique_id}_'
                        f'{Path(record.target_path).stem}_layer_{layer}'
                    )
                    condition_dir = (
                        out_dir / 'erf_examples' / condition_name
                        / record.target_category
                    )
                    title = (
                        f'{condition_name} | {record.target_category} | '
                        f'VGG layer {layer} | '
                        f'{activation.shape[1]}x{activation.shape[2]}x{activation.shape[3]}'
                    )
                    save_target_erf_figure(
                        original_image,
                        transformed_image,
                        original_erf,
                        transformed_erf,
                        title=title,
                        path=condition_dir / f'{stem}.png',
                    )
                    save_target_erf_arrays(
                        original_erf,
                        transformed_erf,
                        aligned_erf,
                        path=condition_dir / f'{stem}.npz',
                    )
                aligned_dir = (
                    out_dir / 'erf_aligned_examples' / condition_name
                    / record.target_category
                )
                aligned_stem = (
                    f'trial_{record.unique_id}_{Path(record.target_path).stem}'
                )
                save_aligned_erf_comparison_figure(
                    original_image,
                    transformed_image,
                    example_original_erfs,
                    example_transformed_erfs,
                    example_aligned_erfs,
                    example_layer_shapes,
                    title=(
                        f'{condition_name} | {record.target_category} | '
                        'inverse-aligned ERF comparison'
                    ),
                    path=aligned_dir / f'{aligned_stem}.png',
                )
                unit_mass_dir = (
                    out_dir / 'erf_unit_mass_examples' / condition_name
                    / record.target_category
                )
                save_unit_mass_erf_comparison_figure(
                    original_image,
                    transformed_image,
                    example_original_erfs,
                    example_transformed_erfs,
                    example_aligned_erfs,
                    example_layer_shapes,
                    title=f'{condition_name} | {record.target_category}',
                    path=unit_mass_dir / f'{aligned_stem}.png',
                )
                direct_unit_mass_dir = (
                    out_dir / 'erf_unit_mass_direct_examples' / condition_name
                    / record.target_category
                )
                save_unit_mass_direct_erf_figure(
                    original_image,
                    transformed_image,
                    example_original_erfs,
                    example_transformed_erfs,
                    example_layer_shapes,
                    title=f'{condition_name} | {record.target_category}',
                    path=direct_unit_mass_dir / f'{aligned_stem}.png',
                )

            if record_index % 25 == 0 or record_index == len(records):
                print(f'  completed {record_index}/{len(records)} targets')

    grouped_rows = build_grouped_target_activation_summary(activation_rows)
    grouped_pooled_block_rows = build_grouped_pooled_block_summary(
        pooled_block_rows
    )
    write_rows_csv(
        activation_rows,
        out_dir / 'target_activation_trial_metrics.csv',
    )
    write_rows_csv(
        grouped_rows,
        out_dir / 'grouped_target_activation_metrics.csv',
    )
    write_rows_csv(
        pooled_block_rows,
        out_dir / 'pooled_block_activation_trial_metrics.csv',
    )
    write_rows_csv(
        grouped_pooled_block_rows,
        out_dir / 'grouped_pooled_block_activation_metrics.csv',
    )
    grouped_erf_rows = build_grouped_aligned_erf_summary(erf_rows)
    write_rows_csv(erf_rows, out_dir / 'target_erf_metrics.csv')
    write_rows_csv(
        grouped_erf_rows,
        out_dir / 'grouped_target_erf_metrics.csv',
    )
    save_grouped_target_activation_plots(
        grouped_rows,
        out_dir / 'activation_plots',
    )
    save_pooled_block_activation_plots(
        grouped_pooled_block_rows,
        out_dir / 'pooled_block_activation_plots',
    )
    save_aligned_erf_summary_plots(
        grouped_erf_rows,
        out_dir / 'erf_alignment_plots',
    )
    print(f'Target-only feature analysis complete: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
