import time
import torch 
import torch.nn as nn
from skimage import measure
from skimage.morphology import skeletonize
import numpy as np
import cv2


class AutoMorphNumpyWrapper:
    """
    Wraps a NumPy-based single-image feature extractor so it:
      - accepts batches
      - returns batched dictionaries
      - supports torch tensors or numpy arrays
    NOTE: NumPy path is NOT differentiable; input tensors are detached.
    """

    def __init__(self, single_extractor, num_channels=3,run_single_func=None,collate_disc=True):
        """
        single_extractor: an object with a method
            optic_cup_features(disc: np.ndarray(H,W), cup: np.ndarray(H,W)) -> dict[str, float]
        and/or a __call__ that expects a single image in numpy (H,W,C).
        """
        self.single = single_extractor
        self.num_channels = num_channels
        if run_single_func is None:
            self._run_single = self.single.__call__
        else:
            self._run_single = run_single_func.__get__(self.single)
        self.collate_disc = collate_disc
    def _to_numpy_hwc(self, x):
        """
        Convert a single image to numpy HxWxC (uint8 or float ok).
        Supports:
          - torch: (C,H,W) or (H,W,C)
          - numpy: (C,H,W) or (H,W,C)
        Uses self.num_channels as the expected channel count.
        """
        if torch.is_tensor(x):
            # detach; numpy path is non-differentiable anyway
            x = x.detach().cpu()
            if x.dim() == 3 and x.shape[0] == self.num_channels:
                x = x.permute(1, 2, 0)  # (H,W,C)
            x = x.numpy()
        else:
            # numpy
            if x.ndim == 3 and x.shape[0] == self.num_channels:
                x = np.transpose(x, (1, 2, 0))  # (H,W,C)
        assert x.ndim == 3 and x.shape[2] == self.num_channels, (
            f"Expect a single image as (H,W,{self.num_channels})"
        )
        return x

    def columns(self):
        return self.single.columns

    @staticmethod
    def _collate_dicts(dicts):
        """
        Turn a list[dict[key->scalar]] into dict[key->np.ndarray shape (N,)]
        """
        if not dicts:
            return {}
        keys = dicts[0].keys()
        out = {k: np.array([d[k] for d in dicts], dtype=np.float32) for k in keys}
        return out

    def __call__(self, img_or_batch):
        """
        Inputs supported:
          - torch.Tensor (N,C,H,W) or (C,H,W)
          - np.ndarray   (N,H,W,C) or (H,W,C) or (N,C,H,W)
        Returns:
          - If input is torch.Tensor -> dict[str, torch.Tensor] with shape (N,)
          - Else -> dict[str, np.ndarray] with shape (N,)
        """
        is_torch = torch.is_tensor(img_or_batch)
        device = img_or_batch.device if is_torch else None

        # Normalize to a Python list of single images
        imgs = []
        if is_torch:
            x = img_or_batch
            if x.dim() == 4:
                # (N,C,H,W)
                for i in range(x.shape[0]):
                    imgs.append(self._to_numpy_hwc(x[i]))
            elif x.dim() == 3:
                imgs = [self._to_numpy_hwc(x)]
            else:
                raise ValueError(f"Torch input must be (N,{self.num_channels},H,W) or ({self.num_channels},H,W)")
            imgs = [img * 255.0 for img in imgs]
        else:
            x = img_or_batch
            if isinstance(x, np.ndarray) and x.ndim == 4:
                # could be (N,H,W,C) or (N,C,H,W)
                if x.shape[1] == self.num_channels:  # (N,C,H,W)
                    x = np.transpose(x, (0, 2, 3, 1))
                assert x.shape[-1] == self.num_channels, "Expected channels-last if numpy batch"
                imgs = [x[i] for i in range(x.shape[0])]
            elif isinstance(x, np.ndarray) and x.ndim == 3:
                imgs = [self._to_numpy_hwc(x)]
            else:
                raise ValueError(f"NumPy input must be (N,H,W,{self.num_channels}), (N,{self.num_channels},H,W) or (H,W,{self.num_channels})")

        # Run per-sample
        results = [self._run_single(im) for im in imgs]
        if not self.collate_disc:
            return results  # list of dicts, one per image
        collated = self._collate_dicts(results)  # dict[str -> np.ndarray(N,)]

        # Convert back to torch if needed
        if is_torch:
            collated = {k: torch.from_numpy(v).to(device=device, dtype=torch.float32)
                        for k, v in collated.items()}
        return collated


class FeatureExtractor:
    def __init__(self):
        self.time_log = {}

    def _log_time(self,key,t1,t2):
        if key not in self.time_log:
            self.time_log[key] = 0.0
        self.time_log[key] += t2 - t1

    def _return_as_dict(self,**kwargs):
        return kwargs
    def __call__(self, img):
        raise NotImplementedError


class Optic_Disc_Cup_Features(FeatureExtractor):

    def __init__(self):
        """
        Submodule to calculate optic disc/cup features
        """
        super().__init__()

    @property
    def columns(self):
        return ["disc_width","disc_height","cup_width","cup_height","cdr_vertical","cdr_horizontal"]

    def _keep_largest_region(self, binary_mask):
        mask = measure.label(binary_mask)                  
        regions = measure.regionprops(mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions[1:]:
                mask[rg.coords[:,0], rg.coords[:,1]] = 0
        binary_mask[mask!=0] = 255
        return binary_mask

    def _calculate_width_height(self,mask):
        index = np.where(mask>0)
        if len(index[0]) == 0:
            return 0, 0, np.array([]), np.array([])

        index_width = index[1]
        index_height = index[0]
        width = np.max(index_width)-np.min(index_width)
        height = np.max(index_height)-np.min(index_height)
        return width, height, index_width, index_height
    
    def _conditions(self, 
                    cup, disc,
                    cup_horizontal_width, 
                    cup_vertical_height, 
                    disc_horizontal_width, 
                    disc_vertical_height,
                    cup_index_width, 
                    cup_index_height, 
                    disc_index_width, 
                    disc_index_height):

        if len(disc_index_width) == 0 or len(cup_index_width) == 0:
            return False

        cup_width_centre = np.mean(cup_index_width)
        cup_height_centre = np.mean(cup_index_height)

        ## 
        valid_size = disc_horizontal_width < (disc.shape[0]/3) and disc_vertical_height < (disc.shape[1]/3)

        cup_within_disc = (cup_width_centre <= np.max(disc_index_width) and 
                cup_width_centre >= np.min(disc_index_width) and 
                cup_height_centre <= np.max(disc_index_height) and 
                cup_height_centre >= np.min(disc_index_height))
        
        cup_smaller_than_disc = (cup_vertical_height < disc_vertical_height and cup_horizontal_width < disc_horizontal_width)

        return valid_size and cup_within_disc and cup_smaller_than_disc

    def optic_cup_features(self, disc, cup):
        """
        Calculates optic disc/cup features from segmentation masks

        input:
            disc: optic disc segmentation mask
            cup: optic cup segmentation mask
        output:
            dictionary of features
        """

        disc = self._keep_largest_region(disc)
        cup = self._keep_largest_region(cup)

        disc_width, disc_height, disc_index_width, disc_index_height = self._calculate_width_height(disc)
        cup_width, cup_height, cup_index_width, cup_index_height = self._calculate_width_height(cup)

        conditions_met = self._conditions(cup, disc,
                                        cup_width, cup_height,
                                        disc_width, disc_height,
                                        cup_index_width, cup_index_height,
                                        disc_index_width, disc_index_height)
        
        cdr_vertical = cup_height/disc_height if disc_height != 0 else -1
        cdr_horizontal = cup_width/disc_width if disc_width != 0 else -1

        if not conditions_met:
            disc_width, disc_height, cup_width, cup_height,cdr_vertical,cdr_horizontal = -1, -1, -1, -1, -1, -1


        return self._return_as_dict(
            disc_width = disc_width,
            disc_height = disc_height,
            cup_width = cup_width,
            cup_height = cup_height,
            cdr_vertical = cdr_vertical,
            cdr_horizontal = cdr_horizontal
        )
    
    def __call__(self, img):

        if img.shape[2] != 3:
            # assuming a pytorch tensor
            img = img.permute(1,2,0)
            img = img * 255.0
        img = img.numpy()

        assert(img.shape[2] == 3), "Input image must have 3 channels (vascular segmentation, optic disc, optic cup)"

        disc, cup = img[:,:,1], img[:,:,2]
        return self.optic_cup_features(disc, cup)


class Vessel_Features(FeatureExtractor):
    def __init__(self,return_skeleton=False, min_pixels_per_vessel=15, eps=1e-8):
        """
        Submodule to calculate vessel features
        """
        self.return_skeleton = return_skeleton
        self.min_pixels_per_vessel = min_pixels_per_vessel
        self.eps = eps
        super().__init__()
        
    @property
    def columns(self):
        return ['vessel_density','fractal_dimension','average_width','distance_tortuosity','squared_curvature_tortuosity','tortuosity_density']

    def _vessel_density(self,Z):

        assert(len(Z.shape) == 2)
        vessel_total_count = np.sum(Z==1)
        pixel_total_count = Z.shape[0]*Z.shape[1]

        return vessel_total_count/pixel_total_count

    def _fractal_dimension(self, Z):

        assert(len(Z.shape) == 2)


        def boxcount(Z, k):
            # Get image dimensions
            rows, cols = Z.shape
            # Indices where each k×k block starts
            row_starts = np.arange(0, rows, k)
            col_starts = np.arange(0, cols, k)
            # First pass: sum rows in blocks of size k
            row_sums = np.add.reduceat(Z, row_starts, axis=0)
            # Second pass: sum columns in blocks of size k
            block_sums = np.add.reduceat(row_sums, col_starts, axis=1)
            # Count boxes that are neither empty nor completely full
            partially_filled = (block_sums > 0) & (block_sums < k * k)

            return np.count_nonzero(partially_filled)


        p = min(Z.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        n = int(np.log(n)/np.log(2))
        sizes = 2**np.arange(n, 1, -1)
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))

        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def _vessel_width(self,Z,Z_skeleton=None):
        if Z_skeleton is None:
            Z_skeleton = skeletonize(Z)
        width = np.sum(Z)/np.sum(Z_skeleton)
        return width

    def _tortuosity_density(self, x, y):
        """
        Defined in "A Novel Method for the Automatic Grading of Retinal Vessel Tortuosity" by Grisan et al.
        DOI: 10.1109/IEMBS.2003.1279902

        :param x: the x points of the curve
        :param y: the y points of the curve
        :return: tortuosity density measure
        """
        curvature = self._curvature(x, y)
        inflections = self._detect_inflection_points(curvature)

        # segment boundaries
        indices = np.concatenate(([0], inflections, [len(x)-1]))

        n = len(indices) - 1
        if n < 2:
            return 0.0

        # total curve length
        Lc = self._curve_length(x, y)

        acc = 0.0
        for i in range(n):
            a, b = indices[i], indices[i+1]
            xs, ys = x[a:b+1], y[a:b+1]
            Lc_si = self._curve_length(xs, ys)
            Lx_si = self._chord_length(xs, ys)
            if Lx_si > 0:
                acc += (Lc_si / Lx_si) - 1

        return (n - 1) / n * acc / Lc

    def _curvature(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        dx = np.gradient(x)
        dy = np.gradient(y)

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = (dx * ddy - dy * ddx) / (dx*dx + dy*dy)**1.5
        return curvature

    def _squared_curvature_tortuosity(self, x, y):
        curvature = self._curvature(x, y)
        return np.trapezoid(curvature**2)
    
    def _distance_2p(self,x1, y1, x2, y2):
        """
        calculates the distance between two given points
        :param x1: starting x value
        :param y1: starting y value
        :param x2: ending x value
        :param y2: ending y value
        :return: the distance between [x1, y1] -> [x2, y2]
        """
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _curve_length(self, x, y):
        """
        calculates the length(distance) of the given curve, iterating from point to point.
        :param x: the x component of the curve
        :param y: the y component of the curve
        :return: the curve length
        """
        distance = 0
        for i in range(0, len(x) - 1):
            distance += self._distance_2p(x[i], y[i], x[i + 1], y[i + 1])
        return distance

    def _chord_length(self,x, y):
        """
        distance between starting and end point of the given curve

        :param x: the x component of the curve
        :param y: the y component of the curve
        :return: the chord length of the given curve
        """
        return self._distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])
    
    def _detect_inflection_points(self, curvature):
        # --- inflection points (exclude zeros) ---
        sign = np.sign(curvature)
        sign[sign == 0] = np.nan
        inflections = np.where(np.diff(np.signbit(sign)))[0] + 1
        return inflections

    def _distance_measure_tortuosity(self,x, y):
        """
        Distance measure tortuosity defined in:
        William E Hart, Michael Goldbaum, Brad Côté, Paul Kube, and Mark R Nelson. Measurement and
        classification of retinal vascular tortuosity. International journal of medical informatics,
        53(2):239–252, 1999.

        :param x: the list of x points of the curve
        :param y: the list of y points of the curve
        :return: the arc-chord tortuosity measure
        """
        if len(x) < 2:
            raise ValueError("Given curve must have at least 2 elements")

        return self._curve_length(x, y)/(self._chord_length(x, y) + self.eps)

    def normalize_curve(self,x, y):
        x=np.array(x)
        y=np.array(y)
        chord = ((x[-1]-x[0])**2 + (y[-1]-y[0])**2)**0.5
        x_n = (x - x[0]) / chord
        y_n = (y - y[0]) / chord
        return x_n, y_n

    def _tortuosity_per_window(self,window,len_weighted=True):
        H = window.shape[0]
        min_pixels_per_vessel = int(self.min_pixels_per_vessel *(H/912)) 

        t2, t4, td = 0, 0, 0
        vessel_count = 0
        vessels = detect_vessel_border(window) 

        for vessel_nonsort in vessels:
            vessel_x, vessel_y = order_vessel_points(vessel_nonsort)

            if len(vessel_x) > min_pixels_per_vessel:
                vessel_count_i = 1 if not len_weighted else self._curve_length(vessel_x, vessel_y)
                vessel_count += vessel_count_i
                
                t2 += self._distance_measure_tortuosity(vessel_x, vessel_y) * vessel_count_i
                
                t4 += self._squared_curvature_tortuosity(vessel_x, vessel_y) * vessel_count_i
                
                td += self._tortuosity_density(vessel_x, vessel_y) * vessel_count_i
                                        
        if vessel_count > 0:
            t2 = t2/vessel_count
            t4 = t4/vessel_count
            td = td/vessel_count
        return t2, t4, td

    def _tortuosities(self,Z,window_size = None):

        if window_size is None:
            window_size = Z.shape[0]  # process whole image as single window

        windows, _ = split_into_windows(Z, window_size)
        t2, t4, td = 0, 0, 0
        for win in windows:
            t2_win, t4_win, td_win = self._tortuosity_per_window(win)
            t2 += t2_win
            t4 += t4_win
            td += td_win

        return t2, t4, td
    
    def ungrouped_tortuosities(self,Z):
        Z = (Z>0).astype(np.uint8)
        if Z.ndim == 3:
            Z = Z[:,:,0]
        min_group_size = int(self.min_pixels_per_vessel *(Z.shape[0]/912))
        Z_skeleton = skeletonize(Z)
        vessels = detect_vessel_border(Z_skeleton)
        tortuosities = []
        for vessel_nonsort in vessels:
            vessel_x, vessel_y = order_vessel_points(vessel_nonsort)
            if len(vessel_x) > min_group_size:
                t2 = self._distance_measure_tortuosity(vessel_x, vessel_y)
                t4 = self._squared_curvature_tortuosity(vessel_x, vessel_y)
                td = self._tortuosity_density(vessel_x, vessel_y)
                tortuosities.append((t2,t4,td))
        return tortuosities

    def vessel_features(self, Z, Z_skeleton=None):
        """
        Calculates vessel features from segmentation masks

        input:
            Z: binary vessel segmentation mask
        output:
            dictionary of features
        """

        Z = (Z>0).astype(np.uint8)

        if Z_skeleton is None:
            Z_skeleton = skeletonize(Z)
        vessel_density = self._vessel_density(Z)
        fractal_dimension = self._fractal_dimension(Z)
        vessel_width = self._vessel_width(Z,Z_skeleton)
        distance_tortuosity, squared_curvature_tortuosity, tortuosity_density = self._tortuosities(Z_skeleton)

        out = self._return_as_dict(
            vessel_density = vessel_density,
            fractal_dimension = fractal_dimension,
            average_width = vessel_width,
            distance_tortuosity = distance_tortuosity,
            squared_curvature_tortuosity = squared_curvature_tortuosity,
            tortuosity_density = tortuosity_density)
        if self.return_skeleton:
            out['skeleton'] = Z_skeleton

        return out 

    def __call__(self, img, Z_skeleton=None):

        assert(img.shape[2] == 1), "Input image must have 1 channel (vascular segmentation)"

        vessel = img[:,:,0]
        return self.vessel_features(vessel, Z_skeleton=Z_skeleton)

class AutoMorph_Features(Optic_Disc_Cup_Features, Vessel_Features):

    def __init__(self,return_skeleton=False, min_pixels_per_vessel=15):
        """
        Calculates basic AutoMorph features from segmentation masks

        Vascular segmementation (M2/binary_vessel/binary_process/)
            - Vessel Density
            - Average width (requires skeleton)
            - Fractal Dimension 
            - Distance Tortuosity 
            - Squared Curvature Tortuosity
            - Tortuosity Density

        Optic Disc (M2/optic_disc_cup/resized/)
            - Disc Height
            - Disc Width
            - Cup Height
            - Cup Width
            - CDR Vertical
            - CDR Horizontal

        input:
            3-channel image (vascular segmentation, optic disc, optic cup)
        output:
            dictionary of features
        """
        Optic_Disc_Cup_Features.__init__(self)
        Vessel_Features.__init__(self,return_skeleton=return_skeleton, min_pixels_per_vessel=min_pixels_per_vessel)

    @property
    def columns(self):
        # call the property fget on self for each base
        cols_v = Vessel_Features.columns.fget(self)
        cols_o = Optic_Disc_Cup_Features.columns.fget(self)
        return cols_v + cols_o
    
    def __call__(self, img):
        """
        input:
            img: 3-channel image (vascular segmentation, optic disc, optic cup)
        output:
            dictionary of features
        """
        if img.shape[2] != 3:
            # assuming a pytorch tensor
            img = img.permute(1,2,0)
            img = img * 255.0
            img = img.numpy()

        assert(img.shape[2] == 3), "Input image must have 3 channels (vascular segmentation, optic disc, optic cup)"

        vessel = img[:,:,0]
        disc = img[:,:,1]
        cup = img[:,:,2]
        
        vessel_features = self.vessel_features(vessel)
        
        od_cup_features = self.optic_cup_features(disc, cup)
        
        features = {**vessel_features, **od_cup_features}

        return features
    


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


## functions for tortuosity calculation
from collections import deque



def detect_vessel_border(arr: np.ndarray, ignored_pixels: int = 1, return_bifurcation_points: bool = False):
    """
    Extract vessel border polylines from a 2-D binary/skeleton image (NumPy array).
    Returns: list of [xs, ys] where xs/ys are lists of coordinates (ints).
    """
    # Ensure 2-D and binary, and work on a copy (don't mutate caller's array)
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"detect_vessel_border expects a 2-D array, got shape {img.shape}")
    img = (img > 0).astype(np.uint8).copy()

    H, W = img.shape

    def neighbours(x: int, y: int):
        x_less = max(0, x - 1)
        y_less = max(0, y - 1)
        x_more = min(H - 1, x + 1)
        y_more = min(W - 1, y + 1)
        nbrs = []
        if img[x_less, y_less]: nbrs.append((x_less, y_less))
        if img[x_less, y     ]: nbrs.append((x_less, y     ))
        if img[x_less, y_more]: nbrs.append((x_less, y_more))
        if img[x     , y_less]: nbrs.append((x     , y_less))
        if img[x     , y_more]: nbrs.append((x     , y_more))
        if img[x_more, y_less]: nbrs.append((x_more, y_less))
        if img[x_more, y     ]: nbrs.append((x_more, y     ))
        if img[x_more, y_more]: nbrs.append((x_more, y_more))
        return nbrs

    def intersection_mask(base: np.ndarray, x: int, y: int):
        # Count 8-neighborhood actives from the ORIGINAL (pre-masked) binary image `base`
        x_less = max(0, x - 1)
        y_less = max(0, y - 1)
        x_more = min(H - 1, x + 1)
        y_more = min(W - 1, y + 1)
        active = int(base[x_less, y_less]) + int(base[x_less, y]) + int(base[x_less, y_more]) + \
                 int(base[x,      y_less]) +                             int(base[x,      y_more]) + \
                 int(base[x_more, y_less]) + int(base[x_more, y]) + int(base[x_more, y_more])
        return active

    bifurcation_points = []
    # Build an intersection mask: zero pixels whose neighborhood is very branchy
    base_bin = img.copy()
    mask = np.ones_like(img, dtype=np.uint8)  # stays uint8
    for x in range(ignored_pixels, H - ignored_pixels):
        for y in range(ignored_pixels, W - ignored_pixels):
            if base_bin[x, y]:
                active = intersection_mask(base_bin, x, y)
                if active > 2:
                    bifurcation_points.append((y, x))
                    cv2.circle(mask, (y, x), radius=1, color=0, thickness=-1)

    img &= mask  # apply mask in-place, still uint8 binary

    vessels = []

    # Extract connected polylines by flood-fill (BFS)
    for x in range(ignored_pixels, H - ignored_pixels):
        for y in range(ignored_pixels, W - ignored_pixels):
            if img[x, y]:
                q = deque([(x, y)])
                xs, ys = [], []
                while q:
                    cx, cy = q.popleft()
                    if img[cx, cy]:
                        img[cx, cy] = 0  # consume
                        xs.append(cx); ys.append(cy)
                        q.extend(neighbours(cx, cy))
                if xs:  # found one vessel
                    vessels.append([xs, ys])
    if return_bifurcation_points:
        bifurcation_points = np.array(bifurcation_points)
        return vessels, bifurcation_points
    else:
        return vessels

def derivative1_centered_h1(target, y):
    """
    Implements the taylor centered approach to calculate the first derivative.

    :param target: the position to be derived, must be len(y)-1 > target > 0
    :param y: an array with the values
    :return: the centered derivative of target
    """
    if len(y) - 1 <= target <= 0:
        raise(ValueError("Invalid target, array size {}, given {}".format(len(y), target)))
    return (y[target + 1] - y[target - 1])/2

def derivative2_centered_h1(target, y):
    """
    Implements the taylor centered approach to calculate the second derivative.

    :param target: the position to be derived,  must be len(y)-1 > target > 0
    :param y: an array with the values
    :return: the centered second derivative of target
    """
    if len(y) - 1 <= target <= 0:
        raise(ValueError("Invalid target, array size {}, given {}".format(len(y), target)))
    return (y[target + 1] - 2*y[target] + y[target - 1])/4


import sys
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
 
def order_vessel_points(vessel,return_ordered_indices=False):
    sys.setrecursionlimit(10000)  # or a higher value

    """
    Order vessel points using MST + DFS to avoid large jumps.
    Input: vessel = [[x1, x2, ...], [y1, y2, ...]]
    Returns: ordered_vessel = [[x_sorted], [y_sorted]]
    """
    points = np.column_stack(vessel)
    n = len(points)
 
    # Compute pairwise distances
    dists = squareform(pdist(points))
 
    # Compute Minimum Spanning Tree (MST)
    mst_sparse = minimum_spanning_tree(dists)
    mst = mst_sparse.toarray().astype(float)
 
    # Make it symmetric (since MST is undirected)
    mst = np.maximum(mst, mst.T)
 
    # Convert to adjacency list
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                graph[i].append(j)
 
    # Find a leaf node to start (degree == 1)
    degrees = [len(neighbors) for neighbors in graph.values()]
    start_index = degrees.index(1) if 1 in degrees else 0
 
    # DFS traversal to order points
    visited = np.zeros(n, dtype=bool)
    ordered_indices = []
 
    def dfs(u):
        visited[u] = True
        ordered_indices.append(u)
        for v in graph[u]:
            if not visited[v]:
                dfs(v)
 
    dfs(start_index)
    if return_ordered_indices:
        return ordered_indices
    ordered_points = points[ordered_indices]
    return [ordered_points[:, 0].tolist(), ordered_points[:, 1].tolist()]

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from tqdm import tqdm

    def open_image_as_numpy(path,mode='L'):
        img = Image.open(path).convert(mode)
        return np.expand_dims(np.array(img), axis=-1) if mode == 'L' else np.array(img) # keep as (H,W) for single channel, (H,W,C) for multi-channel

    conditioning_df = pd.read_csv('/gpfs3/well/papiez/shared/UKBB/tinnitus_cohort/conditioning_ready_data.csv')[['Imagename','hypertension']].rename(columns={'Imagename':'Name'})

    prefix = '/gpfs3/well/papiez/shared/UKBB/AutoMorph_segmentation/AutoMorph_21015_2/Results_1/M2/'
    img_path = prefix + 'binary_vessel/binary_process/'
    img_path_artery = prefix + 'artery_vein/artery_binary_process/'
    img_path_vein = prefix + 'artery_vein/vein_binary_process/'

    num_files = 1000
    files = os.listdir(img_path)

    keys = ['distance_tortuosity', 'squared_curvature_tortuosity', 'tortuosity_density']
    tort = {}

    for img_name in tqdm(files[:num_files], desc="Processing images"):
        img_np = open_image_as_numpy(os.path.join(img_path, img_name),mode='L')
        features = Vessel_Features(return_skeleton=False)(img_np)
        tort[img_name] = features
    
    # vessels
    for img_name in tqdm(files[:num_files], desc="Processing artery images"):
        img_np = open_image_as_numpy(os.path.join(img_path_artery, img_name),mode='L')
        features = Vessel_Features(return_skeleton=False)(img_np)
        features = {f"artery_{k}": v for k, v in features.items()}
        tort[img_name].update(features)

    # veins
    for img_name in tqdm(files[:num_files], desc="Processing vein images"):
        img_np = open_image_as_numpy(os.path.join(img_path_vein, img_name),mode='L')
        features = Vessel_Features(return_skeleton=False)(img_np)
        features = {f"vein_{k}": v for k, v in features.items()}
        tort[img_name].update(features)


    for key in keys:

        sorted_tort = sorted(tort.items(), key=lambda x: x[1][key], reverse=True)

        top_10 = sorted_tort[:10]
        bottom_10 = sorted_tort[-10:]

        pdf = PdfPages(f'tortuosity_visualization_{key}.pdf')
        for group_name, group in [('top', top_10), ('bottom', bottom_10)]:
            for img_name, tort_features in group:
                img_np = open_image_as_numpy(os.path.join(img_path, img_name),mode='L')
                features = Vessel_Features(return_skeleton=True)(img_np)

                vessel = img_np if img_np.ndim == 2 else img_np[:, :, 0]
                skeleton = features['skeleton']

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(vessel, cmap='gray')
                axes[0].set_title(img_name)
                axes[0].axis('off')

                axes[1].imshow(skeleton, cmap='gray')
                axes[1].set_title(f"{group_name}  {key}: {tort_features[key]:.4f}")
                axes[1].axis('off')

                pdf.savefig(fig)
                plt.close(fig)

        pdf.close()
    
    # link with conditioning_df
    tort_df = pd.DataFrame.from_dict(tort, orient='index')
    tort_df.index.name = 'Name'
    tort_df.reset_index(inplace=True)
    merged_df = pd.merge(tort_df, conditioning_df, on='Name', how='left')
    print('Merged df shape:', merged_df.shape)
    merged_df = merged_df.dropna()
    print('Merged df shape after dropping NA:', merged_df.shape)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    x_cols = [c for c in merged_df.columns if c not in ['Name','hypertension']]
    X = merged_df[x_cols].values
    y = merged_df['hypertension'].values


    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
    print(f"Cross-validated AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    pipe.fit(X, y)
    coefs = pd.Series(pipe.named_steps['model'].coef_[0], index=x_cols)
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index))
