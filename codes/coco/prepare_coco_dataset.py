"""
prepare_coco_dataset.py
=====================================

Prepare segmented COCO object crops for a 16-category IVSN-style experiment,
saving them as RGBA PNGs with transparent background.

Categories:
    sheep, cattle (cow), cats, horses, teddy bears, kites, birds, dogs

Processing:
    1. Read COCO instance segmentation masks from instances_{train,val}2017.json
    2. Reject annotations whose object mask is split across multiple polygons
    3. Extract only the segmented object pixels
    4. Crop to the tight mask bounding box
    5. Resize so the object fits within 156x156 while preserving aspect ratio
    6. Convert object to grayscale
    7. Histogram-equalize the object luminance only
    8. Save as RGBA PNG with transparent background
    9. Save mask-to-source-image area ratios as CSV and histogram plots

Usage:
    python prepare_coco_dataset.py --coco_dir C:/path/to/coco

Requires:
    pip install pycocotools pillow numpy matplotlib
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


# 16 categories
CATEGORY_MAP: Dict[str, List[str]] = {
    "sheep": ["sheep"],
    "cattle": ["cow"],
    "cats": ["cat"],
    "horses": ["horse"],
    "teddybears": ["teddy bear"],
    "kites": ["kite"],
    "birds": ["bird"],
    "dogs": ["dog"],
}

OBJECT_SIZE = 156
IMAGES_PER_CATEGORY = 300


def histogram_equalize_gray_array(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Histogram-equalize grayscale values only where valid_mask is True."""
    result = arr.copy()
    values = arr[valid_mask]
    if values.size == 0:
        return result

    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return result

    cdf_min = nonzero.min()
    denom = values.size - cdf_min
    if denom <= 0:
        return result

    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.clip(round((cdf[i] - cdf_min) / denom * 255), 0, 255)

    result[valid_mask] = lut[arr[valid_mask]]
    return result


def coco_segmentation_to_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    """Convert COCO segmentation to a binary mask. Supports polygon and RLE."""
    seg = annotation["segmentation"]

    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def has_single_segmentation(annotation: dict) -> bool:
    """Return whether a COCO annotation describes one contiguous polygon.

    COCO stores polygon segmentations as a list with one entry per disconnected
    object part. Keeping only one-entry lists prevents crops that contain several
    separated fragments. RLE segmentations are retained because they do not
    expose separate polygon parts (and crowd RLEs are already excluded while
    loading annotations).
    """
    segmentation = annotation.get("segmentation")
    return not isinstance(segmentation, list) or len(segmentation) == 1


def tight_crop_from_mask(rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop RGB image and mask to the tight bounding box of the mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty mask.")
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return rgb[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def preprocess_segmented_object(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    target_size: int = OBJECT_SIZE
) -> Image.Image:
    """
    Create an RGBA image with transparent background:
    - object only
    - grayscale + histogram equalization on object pixels only
    - resized to fit within target_size x target_size
    """
    h, w = crop_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    rgb_img = Image.fromarray(crop_rgb, mode="RGB").resize((new_w, new_h), Image.LANCZOS)
    mask_img = Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L").resize((new_w, new_h), Image.NEAREST)

    alpha = np.array(mask_img, dtype=np.uint8) > 0
    gray = np.array(rgb_img.convert("L"), dtype=np.uint8)
    gray_eq = histogram_equalize_gray_array(gray, alpha)

    rgba_canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)

    x0 = (target_size - new_w) // 2
    y0 = (target_size - new_h) // 2

    rgba_canvas[y0:y0 + new_h, x0:x0 + new_w, 0] = gray_eq
    rgba_canvas[y0:y0 + new_h, x0:x0 + new_w, 1] = gray_eq
    rgba_canvas[y0:y0 + new_h, x0:x0 + new_w, 2] = gray_eq
    rgba_canvas[y0:y0 + new_h, x0:x0 + new_w, 3] = (alpha.astype(np.uint8) * 255)

    return Image.fromarray(rgba_canvas, mode="RGBA")


def make_montage_rgba(images: List[Image.Image], cols: int = 10, cell: int = OBJECT_SIZE) -> Image.Image:
    """Make a visualization montage on a gray background."""
    rows = math.ceil(len(images) / cols)
    montage = Image.new("RGB", (cols * cell, rows * cell), (128, 128, 128))
    for i, img in enumerate(images):
        c, r = i % cols, i // cols
        cell_bg = Image.new("RGBA", (cell, cell), (128, 128, 128, 255))
        thumb = img.resize((cell, cell), Image.LANCZOS)
        cell_bg.alpha_composite(thumb, (0, 0))
        montage.paste(cell_bg.convert("RGB"), (c * cell, r * cell))
    return montage


def remove_surplus_crops(category_dir: Path, expected_count: int) -> int:
    """Remove stale generated crops outside the current output index range."""
    expected_names = {f"img_{index:05d}.png" for index in range(expected_count)}
    surplus_paths = [
        path for path in category_dir.glob("img_*.png")
        if path.name not in expected_names
    ]
    for path in surplus_paths:
        path.unlink()
    return len(surplus_paths)


def save_segmentation_area_statistics(
    records_by_category: Dict[str, List[dict]],
    output_dir: Path,
    bins: int = 20,
) -> None:
    """Save per-crop area ratios and fixed-range histogram plots."""
    records = [
        record
        for category_records in records_by_category.values()
        for record in category_records
    ]
    if not records:
        return

    csv_path = output_dir / "segmentation_area_ratios.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    all_ratios = [record["segmentation_area_ratio"] for record in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_ratios, bins=bin_edges, edgecolor="black", linewidth=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Segmentation area / source image area")
    ax.set_ylabel("Image count")
    ax.set_title(f"Segmentation area ratios (all categories, n={len(records)})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "segmentation_area_histogram_all.png", dpi=150)
    plt.close(fig)

    columns = 4
    rows = math.ceil(len(records_by_category) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 4, rows * 3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, (category, category_records) in zip(
        axes.flat, records_by_category.items()
    ):
        ratios = [
            record["segmentation_area_ratio"] for record in category_records
        ]
        ax.hist(ratios, bins=bin_edges, edgecolor="black", linewidth=0.4)
        ax.set_xlim(0.0, 1.0)
        mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
        ax.set_title(f"{category} (n={len(ratios)}, mean={mean_ratio:.3f})")
        ax.grid(True, axis="y", alpha=0.3)

    for ax in list(axes.flat)[len(records_by_category):]:
        ax.set_visible(False)

    fig.supxlabel("Segmentation area / source image area")
    fig.supylabel("Image count")
    fig.suptitle("Segmentation area ratios by category")
    fig.tight_layout()
    fig.savefig(
        output_dir / "segmentation_area_histograms_by_category.png",
        dpi=150,
    )
    plt.close(fig)

    print(f"Area ratios : {csv_path}")
    print(f"Histogram   : {output_dir / 'segmentation_area_histogram_all.png'}")
    print(
        "By category: "
        f"{output_dir / 'segmentation_area_histograms_by_category.png'}"
    )


def load_coco_annotations(ann_file: str) -> Tuple[dict, dict, dict]:
    print(f"Loading {os.path.basename(ann_file)} ...", flush=True)
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    img_id_to_info = {img["id"]: img for img in data["images"]}

    ann_by_cat: Dict[int, list] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cid = ann["category_id"]
        ann_by_cat.setdefault(cid, []).append(ann)

    return cat_id_to_name, img_id_to_info, ann_by_cat


def crop_category(
    exp_cat_name: str,
    coco_names: List[str],
    cat_id_to_name: dict,
    img_id_to_info: dict,
    ann_by_cat: dict,
    images_dir: str,
    out_dir: Path,
    max_per_cat: int,
    min_area: int,
    split_name: str,
    existing_count: int = 0,
    seed: int = 0,
) -> Tuple[int, List[dict]]:
    matching_cat_ids = [
        cid for cid, name in cat_id_to_name.items()
        if name.lower() in [cn.lower() for cn in coco_names]
    ]
    if not matching_cat_ids:
        print(f"  No COCO match for {exp_cat_name} ({coco_names})")
        return 0, []

    anns = []
    for cid in matching_cat_ids:
        anns.extend(ann_by_cat.get(cid, []))

    anns = [
        a for a in anns
        if a.get("area", 0) >= min_area and has_single_segmentation(a)
    ]
    if not anns:
        print(f"  No valid annotations for {exp_cat_name}")
        return 0, []

    rng = random.Random(seed)
    rng.shuffle(anns)

    needed = max_per_cat - existing_count
    if needed <= 0:
        return 0, []

    saved = 0
    idx = existing_count
    area_records: List[dict] = []

    for ann in anns:
        if saved >= needed:
            break

        img_info = img_id_to_info.get(ann["image_id"])
        if img_info is None:
            continue

        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            rgb = np.array(img, dtype=np.uint8)

            mask = coco_segmentation_to_mask(ann, img_info["height"], img_info["width"])
            if mask.sum() == 0:
                continue

            crop_rgb, crop_mask = tight_crop_from_mask(rgb, mask)

            # reject very weak / fragmented silhouettes
            mask_area = int(crop_mask.sum())
            bbox_area = crop_mask.shape[0] * crop_mask.shape[1]
            fill_ratio = mask_area / max(1, bbox_area)
            if fill_ratio < 0.10:
                continue

            processed = preprocess_segmented_object(crop_rgb, crop_mask)
            out_path = out_dir / f"img_{idx:05d}.png"
            processed.save(out_path)
            source_image_area = img_info["height"] * img_info["width"]
            area_records.append({
                "category": exp_cat_name,
                "crop_file": str(out_path.relative_to(out_dir.parent)),
                "source_split": split_name,
                "source_image_file": img_info["file_name"],
                "image_id": ann["image_id"],
                "annotation_id": ann.get("id"),
                "segmentation_pixels": mask_area,
                "source_image_pixels": source_image_area,
                "segmentation_area_ratio": mask_area / source_image_area,
            })
            saved += 1
            idx += 1

        except Exception as exc:
            print(f"  Warning on annotation {ann.get('id')}: {exc}")
            continue

    return saved, area_records


def prepare(
    coco_dir: str,
    output_dir: str = "coco_crops_transparent_8cat",
    split: str = "both",
    max_per_cat: int = IMAGES_PER_CATEGORY,
    min_area: int = 2000,
    vis_samples: int = 1,
    seed: int = 42,
):
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)

    ann_dir = coco_dir / "annotations"
    if not ann_dir.exists():
        raise FileNotFoundError(f"Could not find annotations/ in {coco_dir}")

    splits_to_use = []
    if split in ("train2017", "both"):
        ann_file = ann_dir / "instances_train2017.json"
        img_dir = coco_dir / "train2017"
        if ann_file.exists() and img_dir.exists():
            splits_to_use.append(("train2017", str(ann_file), str(img_dir)))
    if split in ("val2017", "both"):
        ann_file = ann_dir / "instances_val2017.json"
        img_dir = coco_dir / "val2017"
        if ann_file.exists() and img_dir.exists():
            splits_to_use.append(("val2017", str(ann_file), str(img_dir)))

    if not splits_to_use:
        raise RuntimeError("No valid COCO splits found.")

    print("=" * 60)
    print("IVSN 16-category transparent segmented object preparation")
    print("=" * 60)
    print(f"COCO dir   : {coco_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Splits     : {[s[0] for s in splits_to_use]}")
    print(f"Max/cat    : {max_per_cat}")
    print(f"Min area   : {min_area}")
    print()

    for exp_cat in CATEGORY_MAP:
        (output_dir / exp_cat).mkdir(parents=True, exist_ok=True)

    counts = {cat: 0 for cat in CATEGORY_MAP}
    area_records_by_category: Dict[str, List[dict]] = {
        cat: [] for cat in CATEGORY_MAP
    }

    for split_name, ann_file, img_dir in splits_to_use:
        print(f"--- Split: {split_name}")
        cat_id_to_name, img_id_to_info, ann_by_cat = load_coco_annotations(ann_file)

        for exp_cat, coco_names in CATEGORY_MAP.items():
            if counts[exp_cat] >= max_per_cat:
                print(f"{exp_cat:12s}: already full")
                continue

            n_saved, area_records = crop_category(
                exp_cat_name=exp_cat,
                coco_names=coco_names,
                cat_id_to_name=cat_id_to_name,
                img_id_to_info=img_id_to_info,
                ann_by_cat=ann_by_cat,
                images_dir=img_dir,
                out_dir=output_dir / exp_cat,
                max_per_cat=max_per_cat,
                min_area=min_area,
                split_name=split_name,
                existing_count=counts[exp_cat],
                seed=seed + hash(split_name + exp_cat) % 10000,
            )
            counts[exp_cat] += n_saved
            area_records_by_category[exp_cat].extend(area_records)
            print(f"{exp_cat:12s}: +{n_saved:3d} -> total {counts[exp_cat]:3d}")
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    total = 0
    for cat, n in counts.items():
        total += n
        print(f"{cat:12s}: {n:4d}")
    print(f"{'TOTAL':12s}: {total:4d}")
    print()

    incomplete_categories = {
        cat: count for cat, count in counts.items() if count < max_per_cat
    }
    if incomplete_categories:
        details = ", ".join(
            f"{cat}={count}" for cat, count in incomplete_categories.items()
        )
        raise RuntimeError(
            f"Could not create {max_per_cat} valid images for every category: {details}"
        )

    for cat in CATEGORY_MAP:
        removed = remove_surplus_crops(output_dir / cat, max_per_cat)
        if removed:
            print(f"{cat:12s}: removed {removed} surplus crop(s)")

    print("Saving segmentation-area statistics...")
    save_segmentation_area_statistics(area_records_by_category, output_dir)

    if vis_samples > 0:
        print("Saving montage previews...")
        for cat in CATEGORY_MAP:
            paths = sorted((output_dir / cat).glob("*.png"))[:vis_samples * 20]
            if not paths:
                continue
            imgs = [Image.open(p).convert("RGBA") for p in paths]
            montage = make_montage_rgba(imgs, cols=min(10, len(imgs)))
            montage_path = output_dir / f"_montage_{cat}.png"
            montage.save(montage_path)
            print(f"{cat:12s}: {montage_path}")

    print()
    print("Done.")
    print("Next step: update the experiment scripts to 16 categories / positions.")
    print(f'Example output root:\n  "{output_dir.resolve()}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare transparent segmented COCO crops for a 16-category IVSN experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--coco_dir", type=str, required=True, help="Root COCO directory")
    parser.add_argument("--output_dir", type=str, default="coco_crops_transparent_8cat")
    parser.add_argument("--split", type=str, choices=["train2017", "val2017", "both"], default="both")
    parser.add_argument("--max_per_cat", type=int, default=IMAGES_PER_CATEGORY)
    parser.add_argument("--min_area", type=int, default=2000)
    parser.add_argument("--vis_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    prepare(
        coco_dir=args.coco_dir,
        output_dir=args.output_dir,
        split=args.split,
        max_per_cat=args.max_per_cat,
        min_area=args.min_area,
        vis_samples=args.vis_samples,
        seed=args.seed,
    )
