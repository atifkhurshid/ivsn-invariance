"""Reporting for IVSN invariance experiments."""

from typing import List, Optional
from pathlib import Path

import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-deep')
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
})

from scipy import stats
from .runtime import ensure_dir


def margin_of_error(data: np.ndarray) -> float:
    """Calculate the margin of error for a given dataset."""
    sem = stats.sem(data)
    n = len(data)
    df = n - 1
    return stats.t.ppf(0.975, df) * sem


def summarize_subset(rows: List[dict]) -> dict:

    if not rows:
        return {
            'n_trials': 0,
            'n_objects': None,
            'mean_fixations': None,
            'std_fixations': None,
            'margin_of_error_fixations': None,
            'accuracy_by_n_fixations': None,
            'found_within_k_fixations_count': 0,
            'found_within_k_fixations_rate': None,
            'not_found_within_3_fixations_count': 0,
            'mean_score_target': None,
            'std_score_target': None,
            'margin_of_error_score_target': None,
            'mean_score_max_distractor': None,
            'std_score_max_distractor': None,
            'margin_of_error_score_max_distractor': None,
            'mean_score_mean_distractor': None,
            'std_score_mean_distractor': None,
            'margin_of_error_score_mean_distractor': None,
            'mean_score_margin': None,
            'std_score_margin': None,
            'margin_of_error_score_margin': None,
            'top1_rate': None,
            'mean_target_rank': None,
            'std_target_rank': None,
            'margin_of_error_target_rank': None,
            'mean_p_target': None,
            'std_p_target': None,
            'margin_of_error_p_target': None
        }
    
    fix = np.array([r['n_fixations'] for r in rows], dtype=np.float32)
    found = np.array([r['found'] for r in rows], dtype=np.float32)
    found3 = np.array([r['found_within_3_fixations'] for r in rows], dtype=np.float32)
    score_target = np.array([r['score_target'] for r in rows], dtype=np.float32)
    score_max_distractor = np.array([r['score_max_distractor'] for r in rows], dtype=np.float32)
    score_mean_distractor = np.array([r['score_mean_distractor'] for r in rows], dtype=np.float32)
    score_margin = np.array([r['score_margin'] for r in rows], dtype=np.float32)
    is_top1 = np.array([r['is_top1'] for r in rows], dtype=np.float32)
    target_rank = np.array([r['target_rank'] for r in rows], dtype=np.float32)
    p_target = np.array([r['p_target'] for r in rows], dtype=np.float32)

    n_objects = int(rows[0]['n_objects'])
    found_within_k_fixations_count = np.array([(fix <= k).sum() for k in range(1, n_objects + 1)])
    found_within_k_fixations_rate = np.round(found_within_k_fixations_count / len(rows), decimals=3)

    return {
        'n_trials': int(len(rows)),
        'n_objects': n_objects,
        'mean_fixations': float(fix.mean()),
        'std_fixations': float(fix.std(ddof=0)),
        'margin_of_error_fixations': float(margin_of_error(fix)),
        'accuracy_by_n_fixations': float(found.mean()),
        **{f"found_within_{k+1}_fixations_count": int(found_within_k_fixations_count[k]) for k in range(n_objects)},
        **{f"found_within_{k+1}_fixations_rate": float(found_within_k_fixations_rate[k]) for k in range(n_objects)},
        'not_found_within_3_fixations_count': int(len(rows) - found3.sum()),
        'mean_score_target': float(score_target.mean()),
        'std_score_target': float(score_target.std(ddof=0)),
        'margin_of_error_score_target': float(margin_of_error(score_target)),
        'mean_score_max_distractor': float(score_max_distractor.mean()),
        'std_score_max_distractor': float(score_max_distractor.std(ddof=0)),
        'margin_of_error_score_max_distractor': float(margin_of_error(score_max_distractor)),
        'mean_score_mean_distractor': float(score_mean_distractor.mean()),
        'std_score_mean_distractor': float(score_mean_distractor.std(ddof=0)),
        'margin_of_error_score_mean_distractor': float(margin_of_error(score_mean_distractor)),
        'mean_score_margin': float(score_margin.mean()),
        'std_score_margin': float(score_margin.std(ddof=0)),
        'margin_of_error_score_margin': float(margin_of_error(score_margin)),
        'top1_rate': float(is_top1.mean()),
        'mean_target_rank': float(target_rank.mean()),
        'std_target_rank': float(target_rank.std(ddof=0)),
        'margin_of_error_target_rank': float(margin_of_error(target_rank)),
        'mean_p_target': float(p_target.mean()),
        'std_p_target': float(p_target.std(ddof=0)),
        'margin_of_error_p_target': float(margin_of_error(p_target))
    }


def build_grouped_summary(results: List[dict], n_objects: int) -> List[dict]:

    grouped = []
    by_condition = {}

    for r in results:
        by_condition.setdefault(r['condition_name'], []).append(r)

    for cond_name, rows in sorted(by_condition.items(), key=lambda kv: kv[0]):
        overall = summarize_subset(rows)
        diff = summarize_subset([r for r in rows if r['trial_type'] == 'different'])
        ident = summarize_subset([r for r in rows if r['trial_type'] == 'identical'])

        condition_summary = {
            'condition_name': cond_name,
            'condition_group': rows[0]['condition_group'],
            'condition_value': rows[0]['condition_value'],
            'n_trials': overall['n_trials'],
            'n_objects': n_objects,
        }

        for key in overall:
            if key not in condition_summary:
                condition_summary[key + '_all'] = overall[key]
                condition_summary[key + '_target_different'] = diff[key]
                condition_summary[key + '_target_identical'] = ident[key]

        grouped.append(condition_summary)

    grouped.sort(key=lambda x: (str(x['condition_group']), float(x['condition_value'])))

    return grouped


def write_trial_csv(results: List[dict], out_path: Path):

    fieldnames = [
        'condition_name',
        'condition_group',
        'condition_value',
        'unique_id',
        'repeat_id',
        'trial_type',
        'target_category',
        'target_position',
        'n_fixations',
        'found',
        'found_within_3_fixations',
        'score_target',
        'score_max_distractor',
        'score_mean_distractor',
        'score_margin',
        'target_rank',
        'is_top1',
        'p_target',
        'cue_path',
        'target_path',
        'distractor_paths',
        'cue_transform',
        'target_transform',
        'distractor_transforms',
        'fixation_positions',
        'fixation_centers',
        'scores_initial',
        'score_history'
    ]

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = dict(r)
            for k in [
                'distractor_paths',
                'cue_transform',
                'target_transform',
                'distractor_transforms',
                'fixation_positions',
                'fixation_centers',
                'scores_initial',
                'score_history'
            ]:
                row[k] = json.dumps(row[k])
            writer.writerow(row)


def write_grouped_summary_csv(grouped: List[dict], out_path: Path):

    if not grouped:
        return
    
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(grouped[0].keys()), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(grouped)


def pick_x_label(condition_group: str) -> str:

    mapping = {
        'rotation_deg': 'Rotation (degrees)',
        'scale': 'Scale factor',
        'shift_x': 'Horizontal shift (px)',
        'shift_y': 'Vertical shift (px)', 
        'skew_x_deg': 'Horizontal skew (degrees)', 
        'skew_y_deg': 'Vertical skew (degrees)', 
        'noise_std': 'Gaussian noise std', 
        'blur_radius': 'Blur radius', 
        'mixed': 'Condition index'
    }

    return mapping.get(condition_group, 'Condition value')


def save_individual_cumulative_plot(
        df,
        index: int,
        max_fixations: int,
        title: str,
        x_label: str,
        y_label: str,
        out_path: Path
    ):
    plt.figure(figsize=(6, 6))

    x = np.arange(1, max_fixations + 1)

    series_all = np.array([
        df[f'found_within_{j+1}_fixations_rate_all'][index] for j in range(max_fixations)
    ])
    series_diff = np.array([
        df[f'found_within_{j+1}_fixations_rate_target_different'][index] for j in range(max_fixations)
    ])
    series_iden = np.array([
        df[f'found_within_{j+1}_fixations_rate_target_identical'][index] for j in range(max_fixations)
    ])
    plt.plot(x, series_all, marker='s', label="All")
    plt.plot(x, series_diff, marker='s', label="Target different")
    plt.plot(x, series_iden, marker='s', label="Target identical")

    plt.plot(x, np.cumsum([1/max_fixations]*len(x)),
             marker='s', label="Chance", linestyle='--', color='gray', markersize=4)

    plt.xticks(x, [f'{v:g}' for v in x])
    plt.yticks(np.arange(0, 1.1, 0.1), [f'{v:.1f}' for v in np.arange(0, 1.1, 0.1)])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(loc='lower right')

    plt.ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def save_grouped_cumulative_plot(
        df,
        max_fixations: int,
        labels: list,
        title: str,
        x_label: str,
        y_label: str,
        out_path: Path
    ):
    plt.figure(figsize=(6, 6))

    x = np.arange(1, max_fixations + 1)

    for i, label in enumerate(labels):
        series_all = np.array([
            df[f'found_within_{j+1}_fixations_rate_all'][i] for j in range(max_fixations)
        ])
        plt.plot(x, series_all, marker='s', label=label)

    plt.plot(x, np.cumsum([1/max_fixations]*len(x)),
             marker='s', label="Chance", linestyle='--', color='gray', markersize=4)

    plt.xticks(x, [f'{v:g}' for v in x])
    plt.yticks(np.arange(0, 1.1, 0.1), [f'{v:.1f}' for v in np.arange(0, 1.1, 0.1)])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(loc='lower right')

    plt.ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def save_grouped_bar_plot(
        df,
        x_col: str,
        y_cols: list,
        labels: list,
        title: str,
        x_label: str,
        y_label: str,
        out_path: Path,
        y_lim: Optional[tuple] = None,
        y_error_cols: Optional[list] = None,
        y_chance_level: Optional[float] = None
    ):
    x_vals = df[x_col].tolist()
    n_groups = len(x_vals)
    n_series = len(y_cols)
    x = np.arange(n_groups)
    width = 0.24 if n_series == 3 else 0.35

    plt.figure(figsize=(8, 4.8))

    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    for i, (y, label) in enumerate(zip(y_cols, labels)):
        yerr = df[y_error_cols[i]] if y_error_cols else None
        plt.bar(x + offsets[i], df[y], width=width, label=label,
                 yerr=yerr, capsize = 3, ecolor="black")

    if y_chance_level is not None:
        plt.axhline(y=y_chance_level, color='gray', linestyle='--', label='Chance')

    plt.xticks(x, [f'{v:g}' for v in x_vals])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if n_series > 1 or y_chance_level is not None:
        plt.legend()

    if y_lim is not None:
        plt.ylim(*y_lim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def save_grouped_plots(grouped: List[dict], out_dir: Path, transform_mode: str):

    if not grouped:
        return
    
    plot_dir = out_dir / 'plots'
    ensure_dir(plot_dir)

    df = pd.DataFrame(grouped).sort_values('condition_value')
    x_label = pick_x_label(str(df['condition_group'].iloc[0]))
    rank_ylim = (0, df['n_objects'].iloc[0])

    save_grouped_bar_plot(
        df,
        x_col='condition_value',
        y_cols=[
            'mean_fixations_all',
            'mean_fixations_target_different',
            'mean_fixations_target_identical'
        ],
        labels=['All', 'Target different', 'Target identical'],
        title=f'{transform_mode}: Mean fixations',
        x_label=x_label,
        y_label='Mean fixations',
        out_path=plot_dir / 'mean_fixations.png',
        y_lim=(0, df['n_objects'].iloc[0]),
        y_error_cols=[
            'margin_of_error_fixations_all',
            'margin_of_error_fixations_target_different',
            'margin_of_error_fixations_target_identical'
        ],
        y_chance_level = (rank_ylim[1] + 1) / 2,
    )

    save_grouped_bar_plot(
        df,
        x_col='condition_value',
        y_cols=[
            'found_within_3_fixations_rate_all',
            'found_within_3_fixations_rate_target_different',
            'found_within_3_fixations_rate_target_identical'
        ],
        labels=['All', 'Target different', 'Target identical'],
        title=f'{transform_mode}: Found within 3 fixations',
        x_label=x_label,
        y_label='Rate',
        out_path=plot_dir / 'found_within_3_fixations_rate.png',
        y_lim=(0, 1.1),
        y_chance_level = 1 - ((df['n_objects'].iloc[0] - 1) / df['n_objects'].iloc[0])**3,
    )

    save_grouped_bar_plot(
        df,
        x_col='condition_value',
        y_cols=[
            'top1_rate_all',
            'top1_rate_target_different',
            'top1_rate_target_identical'
        ],
        labels=['All', 'Target different', 'Target identical'],
        title=f'{transform_mode}: Top-1 accuracy',
        x_label=x_label,
        y_label='Rate',
        out_path=plot_dir / 'top1_rate.png',
        y_lim=(0, 1.1),
        y_chance_level = 1 - ((df['n_objects'].iloc[0] - 1) / df['n_objects'].iloc[0]),
    )

    save_grouped_cumulative_plot(
        df,
        max_fixations = df['n_objects'].iloc[0],
        labels = [f"{group} {value}" for group, value in zip(
            df["condition_group"].tolist(), df["condition_value"].tolist())],
        title = f'{transform_mode}: Cumulative fixations',
        x_label='Fixation number',
        y_label='Cumulative performance',
        out_path = plot_dir / 'cumulative_fixations_grouped.png',
    )

    for i in range(len(df)):
        name = f"{df['condition_group'].iloc[i]} {df['condition_value'].iloc[i]}"
        save_individual_cumulative_plot(
            df,
            index=i,
            max_fixations = df['n_objects'].iloc[0],
            title = f'{transform_mode}: Cumulative fixations ({name})',
            x_label='Fixation number',
            y_label='Cumulative performance',
            out_path = plot_dir / f'cumulative_fixations_{name}.png',
        )


def smoothing_suffix(args) -> str:

    if args.smoothing_mode == 'none':
        return 'unsmoothed'
    
    role_bits = []
    if args.smooth_target:
        role_bits.append('target')

    if args.smooth_cue:
        role_bits.append('cue')

    if args.smooth_distractors:
        role_bits.append('distractors')

    role_tag = '-'.join(role_bits) if role_bits else 'none'

    if args.smoothing_mode == 'alpha':
        return f'smoothed_alpha_{role_tag}_b{args.alpha_soften_blur_radius:g}'
    
    if args.smoothing_mode == 'cosine':
        return f'smoothed_cosine_{role_tag}_w{args.edge_taper_width_px:g}'
    
    raise ValueError(f'Unsupported smoothing mode: {args.smoothing_mode}')


def build_dynamic_out_dir(base_out_dir: Path, args) -> Path:

    if args.no_dynamic_out_dir:
        return base_out_dir
    
    suffix = smoothing_suffix(args)

    return base_out_dir / f'{base_out_dir.name}_{args.model_kind}_g{args.gist_image_size}_{suffix}'
