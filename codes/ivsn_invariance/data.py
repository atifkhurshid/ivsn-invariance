"""Data for IVSN invariance experiments."""

import json
from typing import Dict, List
from PIL import Image
from pathlib import Path
from dataclasses import asdict

from . import runtime
from .domain import BaseTrial


def resolve_category_dir(data_root: Path, category: str) -> Path:
    if category in runtime.ALT_CATEGORY_NAMES:
        for cand in runtime.ALT_CATEGORY_NAMES[category]:
            p = data_root / cand
            if p.exists():
                return p
    p = data_root / category
    if p.exists():
        return p
    raise RuntimeError(f"Missing category folder for '{category}' in {data_root}")


def load_dataset(data_root: Path) -> Dict[str, List[Path]]:
    dataset = {}
    for cat in runtime.CATEGORIES:
        cat_dir = resolve_category_dir(data_root, cat)
        files = sorted(list(cat_dir.glob('*.png')))
        if not files:
            raise RuntimeError(f'No PNG images found in {cat_dir}')
        dataset[cat] = files
    return dataset


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert('RGBA')


def load_base_manifest(path: Path) -> List[BaseTrial]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [BaseTrial(**item) for item in data]


def save_base_manifest(base_trials: List[BaseTrial], path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([asdict(t) for t in base_trials], f, indent=2)
