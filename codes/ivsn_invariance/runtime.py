"""Runtime for IVSN invariance experiments."""

import math
import random
import numpy as np
from pathlib import Path
from PIL import ImageFont


ALT_CATEGORY_NAMES = {'teddybears': ['teddybears', 'teddy_bears']}

ALL_CATEGORIES = [
    'sheep',
    'cattle',
    'cats',
    'horses',
    'teddybears',
    'kites',
    'birds',
    'dogs'
]


IMAGE_SIZE = 720


OBJ_SIZE = 156 


PADDING = None


CATEGORIES = None


N_POSITIONS = None


POSITIONS = None


MAX_FIXATIONS = None


JITTER = None  # percentage of container size -> determines how much center of cutoff can drift from center of container


EPSILON = None


MAX_EPSILON = int(round(0.15 * OBJ_SIZE))


CONTAINER_SIZE = OBJ_SIZE + 2 * MAX_EPSILON


RADIUS = None 


MAX_RADIUS = (IMAGE_SIZE - CONTAINER_SIZE) // 2


DEFAULT_N_IDENTICAL = 120


DEFAULT_N_DIFFERENT = 180


EARLY_SUCCESS_FIXATIONS = 3


ORACLE_WINDOW = 45


SEED = 0


DEFAULT_GIST_CONFIG = dict(
    in_channels = 1,
    mode = "dynamic",
    fmax = 0.35,
    fratio = 1.7,
    k = 0.52,
    n_scales = 4,
    n_orientations = 6,
    n_phases = 2,
    scale = 25,
    gaussian = True,
    gaussian_inverse = False,
    n_stds = 3,
    dc_compensate = True,
    stride = 4,
    energy = True,
    energy_mode = "substitute",
    divisive_norm = False,
    pooling = None,
    pool_size = 16,
    pool_stride = 16,
    flatten = False,
)

CODES_DIR = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS_DIR = CODES_DIR / 'model_weights'

DEFAULT_GIST_CHECKPOINTS = {
    'vgg_gist': MODEL_WEIGHTS_DIR / 'vgg_gist.pth',
}

def get_categories(n_objects: int):
    if n_objects <= 16:
        return ALL_CATEGORIES[:n_objects]
    raise ValueError(f'Unsupported n_objects: {n_objects}. Maximal 16 allowed.')


def circle_positions(radius: int):
    # 1. Determine epsilon 
    global EPSILON
    EPSILON = int(round(JITTER * MAX_EPSILON))  

    # 2. Constrain arrangement radius
    radius_container = CONTAINER_SIZE // 2
    min_placement_radius = int(round(radius_container / math.sin(math.pi / N_POSITIONS))) 
    if radius == None:
        placement_radius = MAX_RADIUS
    elif radius < min_placement_radius:
        placement_radius = min_placement_radius
        print(f"WARNING: Given radius '{radius}' is overwritten by minimal radius '{min_placement_radius}', because otherwise it may produce overlaps. Choose radius for {N_POSITIONS} objects within range [{min_placement_radius}, {MAX_RADIUS}].")
    elif radius > MAX_RADIUS:
        placement_radius = MAX_RADIUS
        print(f"WARNING: Given radius '{radius}' is overwritten by maximal radius '{MAX_RADIUS}', because otherwise it exceeds the image borders. Choose radius for {N_POSITIONS} within range [{min_placement_radius}, {MAX_RADIUS}].")
    else: 
        placement_radius = radius

    # 3. Determine center points 
    center = IMAGE_SIZE // 2
    pts = []
    angle_step = 2 * math.pi / N_POSITIONS
    for i in range(N_POSITIONS):
        x = center + int(round(math.cos(i * angle_step) * placement_radius)) 
        y = center + int(round(math.sin(i * angle_step) * placement_radius)) 
        pts.append((x, y))
    return pts

def grid_positions(n_matrix: int):
    """Determines the 2D center coordinates of the object to be pasted on the 
    search image in a nxn grid map arrangement. 

    Parameters: 
        n_matrix       (int): dimension of the nxn matrix 

    Returns: 
        []: list of object cutoff center coordinates in grid map fashion 
    """
    # 1. Constrain padding  
    global PADDING
    max_padding = (IMAGE_SIZE - n_matrix * CONTAINER_SIZE) // 2 
    if PADDING > max_padding:
        print(f"WARNING: Given padding '{PADDING}' is overwritten by maximal padding '{max_padding}', because otherwise it may produce overlaps. Choose padding for {N_POSITIONS} objects within range [0, {max_padding}].")
        PADDING = max_padding
    elif PADDING < 0:
        print(f"WARNING: Given padding '{PADDING}' is overwritten by minimal padding '{0}'. Choose padding for {N_POSITIONS} objects within range [0, {max_padding}].")
        PADDING = max_padding

    # 2. Determine margin 
    margin = (IMAGE_SIZE - 2 * PADDING - n_matrix * CONTAINER_SIZE) / (n_matrix - 1)

    # 3. Determine epsilon 
    global EPSILON
    EPSILON = JITTER * MAX_EPSILON 

    # 4. Determine center points 
    pts = []
    for row in range(n_matrix):
        y = PADDING + CONTAINER_SIZE * (float(row) + 0.5) + row * margin 
        for col in range(n_matrix):
            x = PADDING + CONTAINER_SIZE * (float(col) + 0.5) + col * margin 
            pts.append((int(round(x)), int(round(y))))
    return pts

def set_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_font(size=18):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except Exception:
        return ImageFont.load_default()


def set_runtime_geometry_circle(n_objects: int, radius: int, jitter: float):
    global POSITIONS, N_POSITIONS
    _set_runtime_geometry(n_objects, jitter)
    POSITIONS = circle_positions(radius)

def set_runtime_geometry_grid(n_matrix: int, n_objects: int, padding: int, jitter: float):
    global POSITIONS, PADDING, N_POSITIONS
    PADDING = padding
    _set_runtime_geometry(n_objects, jitter)
    POSITIONS = grid_positions(n_matrix)
    N_POSITIONS = len(POSITIONS)

def _set_runtime_geometry(n_objects: int, jitter: float):
    global CATEGORIES, MAX_FIXATIONS, JITTER, N_POSITIONS
    CATEGORIES = get_categories(n_objects)
    MAX_FIXATIONS = n_objects
    N_POSITIONS = n_objects
    JITTER = jitter

