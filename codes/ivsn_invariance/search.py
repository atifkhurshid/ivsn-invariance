"""Search for IVSN invariance experiments."""

from typing import Optional
from pathlib import Path
import numpy as np
from . import runtime
from .models import BaseAttentionModel
from .domain import TransformSpec, Trial
from .imaging import render_cue, render_search_display


def ivsn_fixation_search(
        scores: np.ndarray,
        target_position: int,
        max_fixations: Optional[int]=None
    ):

    if max_fixations is None:
        max_fixations = runtime.MAX_FIXATIONS
    scores = scores.copy()

    fix_positions = []
    fix_centers = []
    score_history = []
    for _ in range(max_fixations):
        pos = int(np.argmax(scores))
        fix_positions.append(pos)
        fix_centers.append(runtime.POSITIONS[pos])
        score_history.append(scores.copy().tolist())
        if pos == target_position:
            break
        scores[pos] = -1000000000000.0

    return (fix_positions, fix_centers, fix_positions[-1] == target_position, score_history)


def render_trial_from_struct(trial: Trial, args):

    cue_spec = TransformSpec(**trial.cue_transform)
    target_spec = TransformSpec(**trial.target_transform)
    distractor_specs = [TransformSpec(**d) for d in trial.distractor_transforms]

    cue_orig, cue_t = render_cue(Path(trial.cue_path), args, cue_spec)

    search_img = render_search_display(
        Path(trial.target_path),
        [Path(p) for p in trial.distractor_paths],
        trial.target_position,
        target_spec,
        distractor_specs,
        args
    )

    return (cue_orig, cue_t, search_img)


def run_trial(model: BaseAttentionModel, trial: Trial, args) -> dict:

    cue_orig, cue_t, search_img = render_trial_from_struct(trial, args)
    attn_np, scores = model.position_scores(cue_t, search_img, runtime.POSITIONS)

    fix_positions, fix_centers, found, score_history = ivsn_fixation_search(
        scores, trial.target_position, runtime.MAX_FIXATIONS)

    found_within_3 = bool(found and len(fix_positions) <= runtime.EARLY_SUCCESS_FIXATIONS)
    score_target = float(scores[trial.target_position])
    distractor_indices = [i for i in range(len(scores)) if i != trial.target_position]
    distractor_scores = [float(scores[i]) for i in distractor_indices]
    score_max_distractor = float(max(distractor_scores))
    score_mean_distractor = float(np.mean(distractor_scores))
    score_margin = float(score_target - score_max_distractor)
    target_rank = int(np.sum(scores > score_target))
    is_top1 = bool(int(np.argmax(scores)) == trial.target_position)
    softmax_scores = np.exp(scores - np.max(scores))
    softmax_scores = softmax_scores / np.sum(softmax_scores)
    p_target = float(softmax_scores[trial.target_position])

    return {
        'condition_name': trial.condition_name,
        'condition_group': trial.condition_group,
        'condition_value': trial.condition_value,
        'unique_id': trial.unique_id,
        'repeat_id': trial.repeat_id,
        'trial_type': trial.trial_type,
        'target_category': trial.target_category,
        'target_position': trial.target_position,
        'n_objects': args.n_objects,
        'n_fixations': len(fix_positions),
        'found': bool(found),
        'found_within_3_fixations': found_within_3,
        'fixation_positions': fix_positions,
        'fixation_centers': fix_centers,
        'scores_initial': [float(x) for x in scores.tolist()],
        'score_history': score_history,
        'score_target': score_target,
        'score_max_distractor': score_max_distractor,
        'score_mean_distractor': score_mean_distractor,
        'score_margin': score_margin,
        'target_rank': target_rank,
        'is_top1': is_top1,
        'p_target': p_target,
        'cue_path': trial.cue_path,
        'target_path': trial.target_path,
        'distractor_paths': trial.distractor_paths,
        'cue_transform': trial.cue_transform,
        'target_transform': trial.target_transform,
        'distractor_transforms': trial.distractor_transforms
    }
