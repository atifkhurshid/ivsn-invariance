"""Cli for IVSN invariance experiments."""

import json
import argparse
from pathlib import Path
from dataclasses import asdict

from . import runtime


def parse_args():
    parser = argparse.ArgumentParser(description='IVSN invariance sweep with selectable backbone')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=runtime.SEED)
    parser.add_argument('--save-examples', type=int, default=12)
    parser.add_argument('--n-identical', type=int, default=runtime.DEFAULT_N_IDENTICAL)
    parser.add_argument('--n-different', type=int, default=runtime.DEFAULT_N_DIFFERENT)
    parser.add_argument('--arrangement', type=str, choices=['grid', 'circle'], default='circle')
    parser.add_argument('--n-objects', choices=[6, 8], type=int, default=8)
    parser.add_argument('--n-matrix', choices=[2, 3, 4], type=int, default=3) 
    parser.add_argument('--padding', type=int, default=30)
    parser.add_argument('--radius', type=int, default=None)
    parser.add_argument('--jitter', type=float, default=0)
    parser.add_argument('--transform-mode', type=str, required=True,
                        choices=['original', 'rotation', 'scale', 'shift_x', 'shift_y', 'skew_x', 'skew_y',
                                 'noise', 'sp_noise', 'blur', 'mixed'])
    parser.add_argument('--rotation-values', type=float, nargs='*',
                        default=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
    parser.add_argument('--scale-values', type=float, nargs='*', default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument('--shift-values', type=float, nargs='*', default=[-30, -15, 0, 15, 30])
    parser.add_argument('--skew-values', type=float, nargs='*', default=[-20, -10, 0, 10, 20])
    parser.add_argument('--noise-values', type=float, nargs='*', default=[0.0, 0.03, 0.06, 0.09, 0.12])
    parser.add_argument('--sp-noise-values', type=float, nargs='*', default=[0.0, 0.05, 0.1, 0.15, 0.20, 0.3])
    parser.add_argument('--sp_salt_ratio', type=float, default=0.5)
    parser.add_argument('--blur-values', type=float, nargs='*', default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument('--transform-cue-too', action='store_true')
    parser.add_argument('--couple-cue-to-target', action='store_true', dest='couple_cue_to_target')
    parser.add_argument('--no-couple-cue-to-target', action='store_false', dest='couple_cue_to_target')
    parser.add_argument('--load-base-manifest', type=str, default=None)
    parser.add_argument('--no-save-base-manifest', action='store_true')
    parser.add_argument('--smoothing-mode', type=str, choices=['none', 'alpha', 'cosine'], default='cosine')
    parser.add_argument('--alpha-soften-blur-radius', type=float, default=3.0)
    parser.add_argument('--edge-taper-width-px', type=float, default=5.0)
    parser.add_argument('--smooth-target', action='store_true')
    parser.add_argument('--smooth-cue', action='store_true')
    parser.add_argument('--smooth-distractors', action='store_true')
    parser.add_argument('--no-dynamic-out-dir', action='store_true')
    parser.add_argument('--model-kind', type=str, default='vgg',
                        choices=['vgg', 'vgg_gist_pretrained', 'conv_gist', 'conv_gist_mlp', 'vgg_gist_imagenet64',
                                 'vgg_gist_new', 'vonenet-alexnet', 'vonenet-resnet50', 'vonenet-cornets'])
    parser.add_argument('--gist-image-size', type=int, default=224)
    parser.add_argument('--vgg-gist-checkpoint', type=str, default=None)
    parser.add_argument('--conv-gist-checkpoint', type=str, default=None)
    parser.add_argument('--conv-gist-mlp-checkpoint', type=str, default=None)
    parser.add_argument('--vgg-gist-imagenet64-checkpoint', type=str, default=None)
    parser.add_argument('--attention-padding', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Heavy ML dependencies are imported only for an actual experiment run. This
    # keeps package inspection and `--help` lightweight.
    from .data import load_base_manifest, load_dataset, save_base_manifest
    from .models import build_attention_model
    from .reporting import (
        build_dynamic_out_dir,
        build_grouped_summary,
        save_grouped_plots,
        summarize_subset,
        write_grouped_summary_csv,
        write_trial_csv,
    )
    from .runtime import (
        ensure_dir,
        set_runtime_geometry_circle,
        set_runtime_geometry_grid, set_seed
    )
    from .search import run_trial
    from .trials import (
        build_trials_from_base,
        make_condition_specs,
        sample_base_trials
    )
    from .visualization import save_examples

    if args.transform_mode == 'scale' and args.jitter > 0:
        print(f"WARNING: Overlaps or cutoffs may occur. If unwanted, ensure to use jitter = 0 when applying scale > 1 transformation.")

    if args.smoothing_mode != 'none':
        if not (args.smooth_target or args.smooth_cue or args.smooth_distractors):
            args.smooth_target = True

    set_seed(args.seed)

    n_objects = None
    if args.arrangement == 'circle':
        set_runtime_geometry_circle(args.n_objects, args.radius, args.jitter)
        n_objects = args.n_objects

    elif args.arrangement == 'grid':
        assert args.n_objects <= args.n_matrix * args.n_matrix, f"n_objects ({args.n_objects}) must be <= n_matrix^2 ({args.n_matrix ** 2}) for grid arrangement."
        set_runtime_geometry_grid(args.n_matrix, args.n_objects, args.padding, args.jitter)
        n_objects = args.n_objects

    data_root = Path(args.data_root)
    out_dir = build_dynamic_out_dir(Path(args.out_dir), args)
    ensure_dir(out_dir)
    print(f'Output directory: {out_dir}')

    n_unique_trials = args.n_identical + args.n_different
    n_total_trials = n_unique_trials

    print('Loading dataset...')
    dataset = load_dataset(data_root)

    print('Preparing condition specs...')
    condition_specs = make_condition_specs(args)

    print('Preparing base trial manifest...')
    if args.load_base_manifest:
        base_trials = load_base_manifest(Path(args.load_base_manifest))
    else:
        base_trials = sample_base_trials(dataset, args.n_identical, args.n_different)
        if not args.no_save_base_manifest:
            save_base_manifest(base_trials, out_dir / 'base_trials_manifest.json')

    print('Loading model...')
    model = build_attention_model(args)

    all_results = []
    for condition_name, condition_group, condition_value, sweep_spec in condition_specs:

        print(f'Running condition: {condition_name}')

        cond_dir = out_dir / condition_name
        ensure_dir(cond_dir)

        trials = build_trials_from_base(
            base_trials,
            condition_name,
            condition_group,
            condition_value,
            sweep_spec,
            args.transform_cue_too,
            args.couple_cue_to_target
        )

        cond_results = [run_trial(model, trial, args) for trial in trials]

        cond_summary = summarize_subset(cond_results)
        cond_summary.update({
            'n_unique_trials': n_unique_trials,
            'n_identical': args.n_identical,
            'n_different': args.n_different,
            'n_total_trials': n_total_trials,
            'smoothing_mode': args.smoothing_mode,
            'smooth_target': args.smooth_target,
            'smooth_cue': args.smooth_cue,
            'smooth_distractors': args.smooth_distractors,
            'alpha_soften_blur_radius': args.alpha_soften_blur_radius,
            'edge_taper_width_px': args.edge_taper_width_px,
            'model_kind': args.model_kind,
            'gist_image_size': args.gist_image_size,
            'n_objects': n_objects
        })

        with open(cond_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(cond_summary, f, indent=2)

        with open(cond_dir / 'trial_results.json', 'w', encoding='utf-8') as f:
            json.dump(cond_results, f, indent=2)

        with open(cond_dir / 'trial_manifest.json', 'w', encoding='utf-8') as f:
            json.dump([asdict(t) for t in trials], f, indent=2)

        write_trial_csv(cond_results, cond_dir / 'trial_results.csv')

        save_examples(model, trials, cond_dir, args.save_examples, args)

        all_results.extend(cond_results)

    overall = {
        'transform_mode': args.transform_mode,
        'model_kind': args.model_kind,
        'gist_image_size': args.gist_image_size,
        'n_conditions': len(condition_specs),
        'n_trials_total': len(all_results),
        'n_positions': runtime.N_POSITIONS,
        'n_categories': len(runtime.CATEGORIES),
        'n_unique_trials_per_condition': n_unique_trials,
        'n_identical_per_condition': args.n_identical,
        'n_different_per_condition': args.n_different,
        'n_total_trials_per_condition': n_total_trials,
        'early_success_fixations_threshold': runtime.EARLY_SUCCESS_FIXATIONS,
        'shared_base_trials_across_values': True,
        'grayscale_noise_only': True,
        'conditions': [x[0] for x in condition_specs],
        'smoothing_mode': args.smoothing_mode,
        'smooth_target': args.smooth_target,
        'smooth_cue': args.smooth_cue,
        'smooth_distractors': args.smooth_distractors,
        'alpha_soften_blur_radius': args.alpha_soften_blur_radius,
        'edge_taper_width_px': args.edge_taper_width_px,
        'n_objects': n_objects
    }

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(overall, f, indent=2)

    with open(out_dir / 'trial_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    write_trial_csv(all_results, out_dir / 'trial_results.csv')

    grouped = build_grouped_summary(all_results, n_objects)

    with open(out_dir / 'grouped_summary.json', 'w', encoding='utf-8') as f:
        json.dump(grouped, f, indent=2)

    write_grouped_summary_csv(grouped, out_dir / 'grouped_summary.csv')

    save_grouped_plots(grouped, out_dir, args.transform_mode)
    
    print(json.dumps(grouped, indent=2))
