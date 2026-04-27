import numpy as np
from numba import njit
from skimage.morphology import skeletonize
from skimage.draw import line as skimage_line
from scipy.ndimage import distance_transform_edt


# ── low-level kernels ────────────────────────────────────────────────────────

@njit(cache=True)
def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    H, W = mask.shape
    labels = np.zeros((H, W), np.int32)
    stack_r = np.empty(H * W, np.int32)
    stack_c = np.empty(H * W, np.int32)
    label = 0

    # pass 1: label all components
    for r in range(H):
        for c in range(W):
            if mask[r, c] == 1 and labels[r, c] == 0:
                label += 1
                top = 0
                stack_r[top] = r
                stack_c[top] = c
                top += 1
                labels[r, c] = label
                while top > 0:
                    top -= 1
                    cr, cc = stack_r[top], stack_c[top]
                    for dr in (-1, 0, 1):
                        nr = cr + dr
                        if nr < 0 or nr >= H:
                            continue
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nc = cc + dc
                            if nc < 0 or nc >= W:
                                continue
                            if mask[nr, nc] == 1 and labels[nr, nc] == 0:
                                labels[nr, nc] = label
                                stack_r[top] = nr
                                stack_c[top] = nc
                                top += 1

    # pass 2: count each component
    sizes = np.zeros(label + 1, np.int32)
    for r in range(H):
        for c in range(W):
            if labels[r, c] > 0:
                sizes[labels[r, c]] += 1

    # pass 3: keep only large enough components
    out = np.zeros((H, W), np.uint8)
    for r in range(H):
        for c in range(W):
            lbl = labels[r, c]
            if lbl > 0 and sizes[lbl] >= min_size:
                out[r, c] = 1
    return out


@njit(cache=True)
def _find_endpoints(skel: np.ndarray) -> np.ndarray:
    """Returns (N, 2) array of (row, col) endpoint positions."""
    H, W = skel.shape
    # worst case all pixels are endpoints
    pts = np.empty((H * W, 2), np.int32)
    n = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if skel[r, c] == 0:
                continue
            nbrs = (
                skel[r-1, c-1] + skel[r-1, c] + skel[r-1, c+1] +
                skel[r,   c-1]                 + skel[r,   c+1] +
                skel[r+1, c-1] + skel[r+1, c] + skel[r+1, c+1]
            )
            if nbrs == 1:
                pts[n, 0] = r
                pts[n, 1] = c
                n += 1
    return pts[:n]


@njit(cache=True)
def _connect_endpoints(
    mask: np.ndarray,
    endpoints: np.ndarray,   # (N, 2) in (row, col)
    max_gap: int,
    check_path_clear: bool,
) -> np.ndarray:
    result = mask.copy()
    H, W = result.shape
    n = endpoints.shape[0]
    max_gap_sq = max_gap * max_gap

    for i in range(n):
        r1, c1 = endpoints[i, 0], endpoints[i, 1]
        for j in range(i + 1, n):
            r2, c2 = endpoints[j, 0], endpoints[j, 1]

            dr = r2 - r1
            dc = c2 - c1
            if dr * dr + dc * dc > max_gap_sq:
                continue

            # Bresenham line between (r1,c1) and (r2,c2)
            steps = max(abs(dr), abs(dc))
            if steps == 0:
                continue

            if check_path_clear:
                # sample midpoint region — reject if it already overlaps a vessel
                mid = steps // 2
                lo = max(0, mid - 2)
                hi = min(steps, mid + 3)
                hit = False
                for t in range(lo, hi + 1):
                    r = r1 + int(round(dr * t / steps))
                    c = c1 + int(round(dc * t / steps))
                    if 0 <= r < H and 0 <= c < W and result[r, c] == 1:
                        hit = True
                        break
                if hit:
                    continue

            for t in range(steps + 1):
                r = r1 + int(round(dr * t / steps))
                c = c1 + int(round(dc * t / steps))
                if 0 <= r < H and 0 <= c < W:
                    result[r, c] = 1

    return result


# ── public API ───────────────────────────────────────────────────────────────

def remove_small_components(mask: np.ndarray, min_size: int = 700) -> np.ndarray:
    return _remove_small_components(mask.astype(np.uint8), min_size)


def bridge_gaps(mask: np.ndarray, max_gap: int = 22) -> np.ndarray:
    dist = distance_transform_edt(~mask.astype(bool))
    return (mask.astype(bool) | (dist <= max_gap / 2)).astype(np.uint8)


def connect_nearby_endpoints(
    mask: np.ndarray,
    skel: np.ndarray,          # precomputed on the *current* mask
    max_gap: int = 32,
    check_path_clear: bool = True,
) -> np.ndarray:
    endpoints = _find_endpoints(skel.astype(np.uint8))  # (N, 2) row/col
    if len(endpoints) == 0:
        return mask.copy()
    return _connect_endpoints(mask.astype(np.uint8), endpoints, max_gap, check_path_clear)


def postprocess_vessels(
    mask: np.ndarray,
    min_component_size: int = 700,
    max_bridge_gap: int = 50,
    max_endpoint_gap: int = 80,
    check_path_clear: bool = True,
) -> np.ndarray:
    mask = remove_small_components(mask, min_size=min_component_size)
    mask = bridge_gaps(mask, max_gap=max_bridge_gap)
    # recompute skeleton after bridging — endpoints may have changed
    skel = skeletonize(mask > 0).astype(np.uint8)
    mask = connect_nearby_endpoints(mask, skel, max_gap=max_endpoint_gap, check_path_clear=check_path_clear)
    mask = remove_small_components(mask, min_size=min_component_size)
    return mask