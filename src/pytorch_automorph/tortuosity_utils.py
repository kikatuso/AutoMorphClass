
import numpy as np
from numba import njit

def split_into_windows(image: np.ndarray, window_size: int, step: int = None):
    """
    Splits an image into windows of specified size, skipping background-only windows.

    :param image: Input image (H x W x C) or (H x W).
    :param window_size: Size of the square window (e.g., 128).
    :param step: Stride between windows. Defaults to window_size (non-overlapping).
    :return: List of valid windows, and their (row, col) top-left positions.
    """
    if step is None:
        step = window_size

    H, W = image.shape[:2]
    valid_windows = []
    positions = []

    for r in range(0, H - window_size + 1, step):
        for c in range(0, W - window_size + 1, step):
            window = image[r:r+window_size, c:c+window_size]

            # skip background-only windows (all black)
            if np.all(window == 0):
                continue

            valid_windows.append(window)
            positions.append((r, c))

    return valid_windows, positions


@njit(cache=True)
def _label_vessels(img, ignored_pixels):
    H, W = img.shape

    # pass 1: compute bifurcation mask from original img only
    bif = np.zeros((H, W), np.uint8)

    for x in range(ignored_pixels, H - ignored_pixels):
        for y in range(ignored_pixels, W - ignored_pixels):
            if img[x, y] == 1:
                active = 0
                for dx in (-1, 0, 1):
                    nx = x + dx
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny = y + dy
                        if img[nx, ny] == 1:
                            active += 1
                if active > 2:
                    bif[x, y] = 1

    # pass 2: remove all bifurcation pixels at once
    work = img.copy()
    for x in range(H):
        for y in range(W):
            if bif[x, y] == 1:
                work[x, y] = 0

    # pass 3: connected-component labeling
    labels = np.zeros((H, W), np.int32)
    stack_x = np.empty(H * W, np.int32)
    stack_y = np.empty(H * W, np.int32)
    label = 0

    for x in range(ignored_pixels, H - ignored_pixels):
        for y in range(ignored_pixels, W - ignored_pixels):
            if work[x, y] == 1 and labels[x, y] == 0:
                label += 1
                top = 0
                stack_x[top] = x
                stack_y[top] = y
                top += 1
                labels[x, y] = label

                while top > 0:
                    top -= 1
                    cx = stack_x[top]
                    cy = stack_y[top]

                    for dx in (-1, 0, 1):
                        nx = cx + dx
                        if nx < ignored_pixels or nx >= H - ignored_pixels:
                            continue
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            ny = cy + dy
                            if ny < ignored_pixels or ny >= W - ignored_pixels:
                                continue
                            if work[nx, ny] == 1 and labels[nx, ny] == 0:
                                labels[nx, ny] = label
                                stack_x[top] = nx
                                stack_y[top] = ny
                                top += 1

    return labels, label, bif



def _find_endpoints(skel: np.ndarray) -> np.ndarray:
    p = np.pad(skel, 1)

    neighbors = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2]                 + p[1:-1, 2:] +
        p[2:, :-2]  + p[2:, 1:-1]    + p[2:, 2:]
    )

    return np.argwhere((skel == 1) & (neighbors == 1))[:, ::-1]

def detect_vessel_border(arr: np.ndarray, ignored_pixels: int = 1, return_bifurcation_and_endpoints: bool = False):
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"detect_vessel_border expects a 2-D array, got shape {img.shape}")

    labels, nlab, bif = _label_vessels((img > 0).astype(np.uint8), ignored_pixels)

    flat = labels.ravel()
    nz = flat > 0
    idx = np.flatnonzero(nz)
    lab = flat[nz]

    order = np.argsort(lab, kind="mergesort")
    idx = idx[order]
    lab = lab[order]

    counts = np.bincount(lab, minlength=nlab + 1)

    vessels = []
    start = 0
    H, W = labels.shape
    for k in range(1, nlab + 1):
        n = counts[k]
        if n:
            sel = idx[start:start + n]
            xs = (sel // W).tolist()
            ys = (sel % W).tolist()
            vessels.append([xs, ys])
            start += n

    if return_bifurcation_and_endpoints:
        bifurcation_points = np.argwhere(bif)[:, ::-1]
        endpoint_points = _find_endpoints(img)
        return vessels, bifurcation_points, endpoint_points

    return vessels


@njit(cache=True)
def _order_path(points):
    n = points.shape[0]
    if n == 0:
        return np.empty(0, np.int32)

    # Build occupancy grid over bounding box
    xs = points[:, 0]
    ys = points[:, 1]
    min_x = xs.min()
    min_y = ys.min()
    max_x = xs.max()
    max_y = ys.max()

    H = max_x - min_x + 1
    W = max_y - min_y + 1

    occ = np.zeros((H, W), np.int32)
    for i in range(n):
        occ[points[i, 0] - min_x, points[i, 1] - min_y] = i + 1  # store index+1

    # Find an endpoint: pixel with exactly one 8-neighbor
    start = 0
    for i in range(n):
        x = points[i, 0] - min_x
        y = points[i, 1] - min_y
        deg = 0
        for dx in (-1, 0, 1):
            nx = x + dx
            if nx < 0 or nx >= H:
                continue
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                if ny < 0 or ny >= W:
                    continue
                if occ[nx, ny] != 0:
                    deg += 1
        if deg == 1:
            start = i
            break

    ordered = np.empty(n, np.int32)
    visited = np.zeros(n, np.uint8)

    cur = start
    prev = -1

    for t in range(n):
        ordered[t] = cur
        visited[cur] = 1

        x = points[cur, 0] - min_x
        y = points[cur, 1] - min_y

        nxt = -1
        for dx in (-1, 0, 1):
            nx = x + dx
            if nx < 0 or nx >= H:
                continue
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny = y + dy
                if ny < 0 or ny >= W:
                    continue
                j = occ[nx, ny] - 1
                if j >= 0 and j != prev and visited[j] == 0:
                    nxt = j
                    break
            if nxt != -1:
                break

        if nxt == -1:
            break

        prev = cur
        cur = nxt

    return ordered

def order_vessel_points(vessel, return_ordered_indices=False):
    points = np.column_stack(vessel).astype(np.int32)
    order = _order_path(points)

    if return_ordered_indices:
        return order.tolist()

    ordered = points[order]
    return [ordered[:, 0].tolist(), ordered[:, 1].tolist()]