"""Trials for IVSN invariance experiments."""

import random

from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import asdict, replace

from . import runtime
from .domain import BaseTrial, TransformSpec, Trial


def make_condition_specs(args) -> List[Tuple[str, str, float, TransformSpec]]:
    specs = []
    if args.transform_mode == 'original':
        specs.append(('original', 'random_rotation', 0.0, TransformSpec()))
    elif args.transform_mode == 'rotation':
        for v in args.rotation_values:
            specs.append((f'rotation_{v:g}', 'rotation_deg', float(v), TransformSpec(rotation_deg=float(v))))
    elif args.transform_mode == 'scale':
        for v in args.scale_values:
            specs.append((f'scale_{v:g}', 'scale', float(v), TransformSpec(scale=float(v))))
    elif args.transform_mode == 'shift_x':
        for v in args.shift_values:
            specs.append((f'shift_x_{v:g}', 'shift_x', float(v), TransformSpec(shift_x=float(v))))
    elif args.transform_mode == 'shift_y':
        for v in args.shift_values:
            specs.append((f'shift_y_{v:g}', 'shift_y', float(v), TransformSpec(shift_y=float(v))))
    elif args.transform_mode == 'skew_x':
        for v in args.skew_values:
            specs.append((f'skew_x_{v:g}', 'skew_x_deg', float(v), TransformSpec(skew_x_deg=float(v))))
    elif args.transform_mode == 'skew_y':
        for v in args.skew_values:
            specs.append((f'skew_y_{v:g}', 'skew_y_deg', float(v), TransformSpec(skew_y_deg=float(v))))
    elif args.transform_mode == 'noise':
        for v in args.noise_values:
            specs.append((f'noise_{v:g}', 'noise_std', float(v), TransformSpec(noise_std=float(v))))
    elif args.transform_mode == 'sp_noise':
        for v in args.sp_noise_values:
            specs.append((f'sp_noise_{v:g}', 'sp_amount', float(v), TransformSpec(
                sp_amount=float(v), sp_salt_ratio=args.sp_salt_ratio)))
    elif args.transform_mode == 'blur':
        for v in args.blur_values:
            specs.append((f'blur_{v:g}', 'blur_radius', float(v), TransformSpec(blur_radius=float(v))))
    elif args.transform_mode == 'mixed':
        specs.extend([
            ('baseline', 'mixed', 0.0, TransformSpec()),
            ('rot30', 'mixed', 1.0, TransformSpec(rotation_deg=30)),
            ('scale1.25', 'mixed', 2.0, TransformSpec(scale=1.25)),
            ('shiftx20', 'mixed', 3.0, TransformSpec(shift_x=20)),
            ('skewx15', 'mixed', 4.0, TransformSpec(skew_x_deg=15)),
            ('noise0.09', 'mixed', 5.0, TransformSpec(noise_std=0.09)),
            ('sp_noise0.1', 'mixed', 5.0, TransformSpec(sp_amount=0.1, sp_salt_ratio=args.sp_salt_ratio)),
            ('blur2', 'mixed', 6.0, TransformSpec(blur_radius=2.0))
        ])
    else:
        raise ValueError(f'Unsupported transform mode: {args.transform_mode}')
    return specs


def sample_base_trials(dataset: Dict[str, List[Path]], n_identical: int, n_different: int) -> List[BaseTrial]:

    unique_trials = []
    uid = 0

    for _ in range(n_different):
        target_category = random.choice(runtime.CATEGORIES)
        target_pool = dataset[target_category]
        cue_path, target_path = random.sample(target_pool, 2)

        distractor_categories = [c for c in runtime.CATEGORIES if c != target_category]
        distractor_paths = [random.choice(dataset[c]) for c in distractor_categories]
        distractor_rotations = [random.uniform(0, 360) for _ in distractor_paths]

        target_position = random.randint(0, runtime.N_POSITIONS - 1)

        unique_trials.append(BaseTrial(
            uid, 0,
            'different',
            target_category,
            str(cue_path),
            str(target_path),
            [str(p) for p in distractor_paths],
            target_position,
            distractor_rotations
        ))
        uid += 1
    for _ in range(n_identical):
        target_category = random.choice(runtime.CATEGORIES)
        target_pool = dataset[target_category]
        target_path = random.choice(target_pool)
        cue_path = target_path

        distractor_categories = [c for c in runtime.CATEGORIES if c != target_category]
        distractor_paths = [random.choice(dataset[c]) for c in distractor_categories]
        distractor_rotations = [random.uniform(0, 360) for _ in distractor_paths]

        target_position = random.randint(0, runtime.N_POSITIONS - 1)

        unique_trials.append(BaseTrial(
            uid,
            0,
            'identical',
            target_category,
            str(cue_path),
            str(target_path),
            [str(p) for p in distractor_paths],
            target_position,
            distractor_rotations
        ))
        uid += 1

    random.shuffle(unique_trials)

    return unique_trials


def build_trials_from_base(
        base_trials: List[BaseTrial],
        condition_name: str,
        condition_group: str,
        condition_value: float,
        sweep_spec: TransformSpec,
        transform_cue_too: bool,
        couple_cue_to_target: bool
    ) -> List[Trial]:

    trials = []
    for bt in base_trials:

        if condition_name == 'original':
            sweep_spec = TransformSpec(rotation_deg=random.uniform(0, 360))

        target_transform = sweep_spec
        if bt.trial_type == 'identical' and couple_cue_to_target:
            cue_transform = target_transform

        elif transform_cue_too:
            cue_transform = sweep_spec

        else:
            cue_transform = TransformSpec()

        distractor_transforms = [asdict(replace(sweep_spec, rotation_deg=ang)) for ang in bt.distractor_rotations]

        trials.append(Trial(
            condition_name,
            condition_group,
            condition_value,
            bt.unique_id,
            bt.repeat_id,
            bt.trial_type,
            bt.target_category,
            bt.cue_path,
            bt.target_path,
            bt.distractor_paths,
            bt.target_position,
            asdict(cue_transform),
            asdict(target_transform),
            distractor_transforms
        ))
        
    return trials
