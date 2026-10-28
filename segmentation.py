"""Section-5-inspired prototype; see README for differences from Poullis & You.
No neural networks; NumPy/SciPy graph cuts, seeded Gaussian colour models.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

@dataclass
class Config:
    n_angles: int = 8
    smoothness: float = 0.65
    orientation_weight: float = 1.0
    direction_boost: float = 1.5
    color_sigma: float = 0.15
    variance_floor: float = 0.015
    max_cycles: int = 4
    capacity_scale: int = 10000

    def __post_init__(self):
        for name, low, high in [('n_angles', 2, 32), ('max_cycles', 1, 20),
                                ('capacity_scale', 1, 1000000)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer between {low} and {high}')
        for name in ['smoothness', 'orientation_weight', 'direction_boost',
                     'color_sigma', 'variance_floor']:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
            if value < 0 or (name in ['color_sigma', 'variance_floor'] and value == 0):
                raise ValueError(f'invalid {name}')


def axial_distance(a, b):
    """Undirected angular difference, radians in [0, pi/2]."""
    return np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2)


def validate(data):
    image = np.asarray(data['image'])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image must be uint8 RGB, shape (H,W,3)')
    shape = image.shape[:2]
    for key in ['orientation_rad', 'orientation_confidence', 'road_seeds', 'background_seeds']:
        if np.asarray(data[key]).shape != shape:
            raise ValueError(f'{key} must have shape {shape}')
    conf = np.asarray(data['orientation_confidence'])
    angle = np.asarray(data['orientation_rad'])
    if not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
        raise ValueError('orientation_confidence must be finite and in [0,1]')
    if np.any((conf > 0) & (~np.isfinite(angle) | (angle < 0) | (angle >= np.pi))):
        raise ValueError('valid orientations must be finite tangent angles in [0,pi)')
    for key in ['road_seeds', 'background_seeds']:
        if np.asarray(data[key]).dtype != bool or np.count_nonzero(data[key]) < 5:
            raise ValueError(f'{key} must be boolean with at least 5 seed pixels')
    if np.any(data['road_seeds'] & data['background_seeds']):
        raise ValueError('road/background seeds overlap')


def build_energy(data, cfg, use_orientation=True):
    validate(data)
    if not (2 <= cfg.n_angles <= 32 and cfg.smoothness >= 0 and cfg.max_cycles >= 1
            and cfg.capacity_scale >= 1 and cfg.color_sigma > 0 and cfg.variance_floor > 0
            and cfg.orientation_weight >= 0 and cfg.direction_boost >= 0):
        raise ValueError('invalid Config parameters')
    image = data['image'].astype(float) / 255
    h, w = image.shape[:2]
    colors = image.reshape(-1, 3)
    # One diagonal Gaussian per class: intentionally simpler than GMM + MDL.
    costs = []
    for seed_key in ['background_seeds', 'road_seeds']:
        samples = image[data[seed_key]]
        mean = samples.mean(axis=0)
        var = np.maximum(samples.var(axis=0), cfg.variance_floor)
        costs.append(0.5 * (((colors-mean)**2/var).sum(axis=1) + np.log(var).sum()))
    color_cost = np.stack(costs, axis=1)
    color_cost -= color_cost.min(axis=1, keepdims=True)
    conf = data['orientation_confidence'].ravel()
    theta = np.where(conf > 0, data['orientation_rad'].ravel(), 0.0)
    k = cfg.n_angles if use_orientation else 1
    centers = np.arange(k) * np.pi / k
    unary = np.repeat(color_cost, k, axis=1)  # B0..B(k-1), R0..R(k-1)
    if use_orientation:
        angular = cfg.orientation_weight * conf[:, None] * (
            1 - np.cos(2 * (theta[:, None] - centers[None, :]))) / 2
        unary += np.tile(angular, (1, 2))
    ids = np.arange(h*w).reshape(h, w)
    src, dst, weights = [], [], []
    # Four offsets represent all undirected 8-neighbour edges, once each.
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ys = slice(0, h-dy)
        xs = slice(max(0, -dx), min(w, w-dx))
        yt = slice(dy, h)
        xt = slice(max(0, dx), min(w, w+dx))
        p, q = ids[ys, xs].ravel(), ids[yt, xt].ravel()
        color_diff = ((colors[p]-colors[q])**2).sum(axis=1)
        edge = np.exp(-color_diff/(2*cfg.color_sigma**2))
        if use_orientation:
            # Prototype adaptation: favour continuity along reliable curve tangents.
            phi = np.arctan2(dy, dx) % np.pi
            agreement = (1 + np.cos(2*(theta[p]-theta[q]))) / 2
            alignment = (np.cos(theta[p]-phi)**2 + np.cos(theta[q]-phi)**2) / 2
            evidence = np.minimum(conf[p], conf[q]) * agreement * alignment
            edge += cfg.direction_boost * evidence
        edge *= cfg.smoothness / np.hypot(dy, dx)
        src.append(p); dst.append(q); weights.append(edge)
    p, q, weight = np.concatenate(src), np.concatenate(dst), np.concatenate(weights)
    degree = np.bincount(np.r_[p, q], weights=np.r_[weight, weight], minlength=h*w)
    # Penalty exceeds any local unary/pairwise saving from violating a class seed.
    hard = unary.max(axis=1) + degree + 1
    unary[data['road_seeds'].ravel(), :k] = hard[data['road_seeds'].ravel(), None]
    unary[data['background_seeds'].ravel(), k:] = hard[data['background_seeds'].ravel(), None]
    return unary, p, q, weight, centers


def energy(labels, unary, p, q, weight):
    return float(unary[np.arange(len(labels)), labels].sum() +
                 weight[labels[p] != labels[q]].sum())


def binary_cut(d0, d1, p, q, pair, scale=10000):
    """Minimise sum D_i(x_i) + sum pair_ij [x_i != x_j].
    SciPy requires integer capacities. x=0 is source side, x=1 sink side.
    """
    n = len(d0)
    shift = np.minimum(d0, d1)
    d0, d1 = d0-shift, d1-shift
    vals = np.r_[d1, d0, pair, pair]
    if vals.min(initial=0) < -1e-7:
        raise ValueError('Non-submodular or invalid binary move')
    scaled = np.rint(np.maximum(vals, 0) * scale)
    if not np.isfinite(scaled).all() or scaled.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError('Capacity too large: lower capacity_scale')
    caps = scaled.astype(np.int64)
    rows = np.r_[np.full(n, n), np.arange(n), p, q]
    cols = np.r_[np.arange(n), np.full(n, n+1), q, p]
    graph = coo_matrix((caps, (rows, cols)), shape=(n+2, n+2)).tocsr()
    graph.eliminate_zeros()
    flow = maximum_flow(graph, n, n+1)
    residual = graph - flow.flow
    residual.data = (residual.data > 0).astype(np.int64)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    switch = np.ones(n, dtype=bool)
    switch[reachable[reachable < n]] = False
    return switch


def expansion_move(labels, alpha, unary, p, q, weight, scale):
    """Exact binary expansion subproblem for a weighted Potts energy, up to rounding."""
    d0 = unary[np.arange(len(labels)), labels].copy()
    d1 = unary[:, alpha].copy()
    e00 = weight * (labels[p] != labels[q])
    e01 = weight * (labels[p] != alpha)
    e10 = weight * (alpha != labels[q])
    pair = (e01 + e10 - e00) / 2  # E11 = 0
    np.add.at(d1, p, e10-e00-pair)
    np.add.at(d1, q, e01-e00-pair)
    result = labels.copy()
    result[binary_cut(d0, d1, p, q, pair, scale)] = alpha
    return result


def segment(data, cfg=None, use_orientation=True, progress=None):
    cfg = cfg or Config()
    unary, p, q, weight, centers = build_energy(data, cfg, use_orientation)
    labels = unary.argmin(axis=1)
    trace = [energy(labels, unary, p, q, weight)]
    converged = False
    cycles = 0
    for cycle in range(cfg.max_cycles):
        cycles += 1
        improved = False
        for alpha in range(unary.shape[1]):
            if progress:
                progress(cycle + 1, alpha + 1, unary.shape[1])
            proposal = expansion_move(labels, alpha, unary, p, q, weight, cfg.capacity_scale)
            candidate = energy(proposal, unary, p, q, weight)
            # Check original floating-point energy, not only rounded cut capacities.
            if candidate < trace[-1] - 1e-8:
                labels = proposal; trace.append(candidate); improved = True
        if not improved:
            converged = True
            break
    shape = data['image'].shape[:2]
    k = len(centers)
    labels = labels.reshape(shape)
    road = labels >= k
    assert road[data['road_seeds']].all() and not road[data['background_seeds']].any()
    angle = centers[labels % k] if use_orientation else np.full(shape, np.nan)
    return dict(road_mask=road, label_map=labels.astype(np.int16),
                orientation_rad=np.where(road, angle, np.nan).astype(np.float32),
                orientation_bin=np.where(road, labels % k, -1).astype(np.int16)
                if use_orientation else np.full(shape, -1, np.int16),
                energy_trace=trace, cycles=cycles, converged=converged,
                config=asdict(cfg), method='orientation_graphcut' if use_orientation else 'color_graphcut')
