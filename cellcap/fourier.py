"""Fourier tools for editing and reconstructing CAP residual signals.

Edges are compared with their fitted CAP arc, the normal residuals are
transformed in Fourier space, and reconstructed as vertex-to-vertex
pixel paths. The final rasterization keeps reconstructed edges
1-connected so they can be used like the original segmented edge
components.
"""

__author__ = "Tommaso Giacometti"
__email__ = "tommaso.giacometti5@unibo.it"

import numpy as np
import pandas as pd

def compute_fft(edge_proj):
    """Compute Fourier coefficients for a real edge projection."""
    # Store only the real-signal Fourier coefficients.
    return np.fft.rfft(edge_proj)

def compute_ifft(fft, n=None):
    """Reconstruct a real edge projection from Fourier coefficients."""
    # n preserves the original number of edge samples during reconstruction.
    return np.fft.irfft(fft, n=n)

def cap_residual_projection(edge, center, radius, endpoints=None):
    """Return normal residuals between edge pixels and a continuous CAP fit."""
    edge = np.asarray(edge, dtype=np.float64)
    if edge.ndim != 2 or edge.shape[0] < 2:
        return None

    endpoints = edge[[0, -1]] if endpoints is None else np.asarray(endpoints, dtype=np.float64)
    if endpoints.shape != (2, 2):
        return None

    midpoint = 0.5*(endpoints[0] + endpoints[1])
    tangent = endpoints[1] - endpoints[0]
    length = np.linalg.norm(tangent)
    if not np.isfinite(length) or length <= 0:
        return None
    tangent = tangent/length
    normal = tangent @ np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)

    local = edge - midpoint
    x = local @ tangent
    y_real = local @ normal

    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    if not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
        return y_real

    center_local = center - midpoint
    cx = center_local @ tangent
    cy = center_local @ normal
    root = np.sqrt(np.maximum(radius**2 - (x-cx)**2, 0.0))
    y_plus = cy + root
    y_minus = cy - root
    if np.mean((y_real-y_plus)**2) < np.mean((y_real-y_minus)**2):
        y_fit = y_plus
    else:
        y_fit = y_minus
    return y_real-y_fit

def edge_from_cap_residual(residual,
                           edge,
                           center,
                           radius,
                           endpoints=None):
    """Rebuild edge coordinates from a CAP fit and normal residuals."""
    edge = np.asarray(edge, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64).copy()
    if edge.ndim != 2 or edge.shape[0] < 2 or residual.ndim != 1:
        return edge

    endpoints = edge[[0, -1]] if endpoints is None else np.asarray(endpoints, dtype=np.float64)
    if endpoints.shape != (2, 2):
        return edge

    midpoint = 0.5*(endpoints[0] + endpoints[1])
    tangent = endpoints[1] - endpoints[0]
    length = np.linalg.norm(tangent)
    if not np.isfinite(length) or length <= 0:
        return edge
    tangent = tangent/length
    normal = tangent @ np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)

    local = edge - midpoint
    x = local @ tangent
    y_real = local @ normal
    if len(residual) != len(x):
        residual = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(residual)), residual)

    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    if not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
        y_fit = np.zeros_like(x)
    else:
        center_local = center - midpoint
        cx = center_local @ tangent
        cy = center_local @ normal
        root = np.sqrt(np.maximum(radius**2 - (x-cx)**2, 0.0))
        y_plus = cy + root
        y_minus = cy - root
        y_fit = y_plus if np.mean((y_real-y_plus)**2) < np.mean((y_real-y_minus)**2) else y_minus

    residual[0] = 0.0
    residual[-1] = 0.0
    points = midpoint + x[:, None]*tangent + (y_fit + residual)[:, None]*normal
    points[0] = endpoints[0]
    points[-1] = endpoints[1]
    return points

def add_signal_fft_analysis(df,
                            signal_column : str,
                            compute_threshold : int = 16,
                            prefix : str = 'cap_residual_fft',
                            valid_column : str | None = 'valid'):
    """Add Fourier columns for a 1D signal stored in a DataFrame column."""
    re, im = [], []
    for _, row in df.iterrows():
        if valid_column is not None and valid_column in df and not bool(row[valid_column]):
            re.append(None)
            im.append(None)
            continue

        signal = row[signal_column]
        if signal is None:
            re.append(None)
            im.append(None)
            continue
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1 or len(signal) < compute_threshold:
            re.append(None)
            im.append(None)
            continue

        fft = compute_fft(signal)
        re.append(np.real(fft))
        im.append(np.imag(fft))

    df[f'{prefix}_re'] = re
    df[f'{prefix}_im'] = im
    return df

def connected_pixel_path(points):
    """Rasterize float points into a 1-connected pixel path."""
    points = np.asarray(points, dtype=np.float64)
    path = []

    def add_pixel(pixel):
        # Move one axis at a time so consecutive pixels are 1-connected.
        pixel = np.asarray(pixel, dtype=int)
        if not path:
            path.append(pixel)
            return
        current = path[-1].copy()
        while not np.array_equal(current, pixel):
            delta = pixel-current
            axis = np.argmax(np.abs(delta))
            current = current.copy()
            current[axis] += int(np.sign(delta[axis]))
            path.append(current)

    for p0, p1 in zip(points[:-1], points[1:]):
        # Oversample each float segment before rounding to avoid pixel gaps.
        steps = int(np.ceil(np.max(np.abs(p1-p0))))
        if steps == 0:
            pixels = np.rint(p0)[None]
        else:
            t = np.linspace(0, 1, steps + 1, dtype=np.float64)
            pixels = np.rint(p0 + (p1-p0)*t[:,None])
        for pixel in pixels.astype(int):
            add_pixel(pixel)
    if not path:
        add_pixel(np.rint(points[0]))
    return np.asarray(path, dtype=int)

def _pixel_set(edge):
    edge = np.asarray(edge, dtype=int)
    if edge.size == 0:
        return set()
    edge = edge.reshape(-1, 2)
    return {tuple(pixel) for pixel in edge}

def _original_edge_occupancy(edges):
    occupied = {}
    for i, edge in enumerate(edges):
        for pixel in _pixel_set(edge):
            occupied.setdefault(pixel, set()).add(i)
    return occupied

def _intersects_other_edges(edge, edge_idx, original_occupied, accepted_pixels):
    for pixel in _pixel_set(edge):
        owners = original_occupied.get(pixel, set())
        if any(owner != edge_idx for owner in owners):
            return True
        if pixel in accepted_pixels:
            return True
    return False

def _stays_near_edge_cells(edge,
                           label_img=None,
                           cells=None,
                           original_edge=None,
                           max_edge_distance : float | None = None,
                           endpoint_margin : int = 0,
                           max_bad_fraction : float = 0.03):
    """Return True if an edge stays in its local cell pair and corridor."""
    edge = np.asarray(edge, dtype=int)
    if edge.size == 0:
        return False
    edge = edge.reshape(-1, 2)

    if original_edge is not None and max_edge_distance is not None:
        original_edge = np.asarray(original_edge, dtype=np.float64)
        if original_edge.size == 0:
            return False
        original_edge = original_edge.reshape(-1, 2)
        max_edge_distance = float(max_edge_distance)
        if not np.isfinite(max_edge_distance) or max_edge_distance < 0:
            return False
        max_sq = max_edge_distance**2
        for pixel in edge.astype(np.float64):
            dist_sq = np.sum((original_edge - pixel)**2, axis=1)
            if float(np.min(dist_sq)) > max_sq:
                return False

    if label_img is None or cells is None:
        return True

    cells = np.asarray(cells, dtype=int).ravel()
    if cells.size != 2 or cells[0] == cells[1]:
        return False

    h, w = label_img.shape
    if (
        np.any(edge[:,0] < 0) or np.any(edge[:,0] >= h) or
        np.any(edge[:,1] < 0) or np.any(edge[:,1] >= w)
    ):
        return False

    margin = min(max(int(endpoint_margin), 0), max((len(edge)-1)//2, 0))
    core = edge[margin:len(edge)-margin] if margin > 0 else edge
    if len(core) == 0:
        core = edge

    allowed = set(cells.tolist())
    bad = 0
    for r, c in core:
        r0, r1 = max(r-1, 0), min(r+2, h)
        c0, c1 = max(c-1, 0), min(c+2, w)
        labels = set(np.unique(label_img[r0:r1, c0:c1]).astype(int).tolist())
        labels.discard(0)
        if labels and not labels.issubset(allowed):
            bad += 1

    return bad == 0 or bad/len(core) <= max_bad_fraction

def reconstruct_edges_from_cap_residual_fft(df,
                                            compute_threshold : int = 16,
                                            pixel_column : str = 'pixels',
                                            prefix : str = 'cap_residual_fft',
                                            center_column : str = 'rho',
                                            radius_column : str = 'R',
                                            output_column : str = 'cap_residual_edges',
                                            avoid_intersections : bool = True,
                                            label_img=None,
                                            dump_factor : float = 1,
                                            max_edge_distance : float | None = None,
                                            max_attempts : int = 4,
                                            valid_column : str | None = 'valid'):
    """Reconstruct CAP edges from residual Fourier coefficients."""
    recon_edges = []
    original_edges = [np.asarray(edge, dtype=int) for edge in df[pixel_column]]
    original_occupied = _original_edge_occupancy(original_edges) if avoid_intersections else {}
    accepted_pixels = set()
    base_dump = dump_factor

    for edge_idx, (_, row) in enumerate(df.iterrows()):
        edge = original_edges[edge_idx]
        invalid = valid_column is not None and valid_column in df and not bool(row[valid_column])
        if invalid or len(edge) < compute_threshold or row[f'{prefix}_re'] is None or row[f'{prefix}_im'] is None:
            recon_edges.append(edge.tolist())
            accepted_pixels.update(_pixel_set(edge))
            continue

        center = np.asarray(row[center_column], dtype=np.float64)[::-1]
        radius = row[radius_column]
        fft = np.asarray(row[f'{prefix}_re']) + 1j*np.asarray(row[f'{prefix}_im'])
        accepted = None
        for attempt in range(max_attempts):
            factor = base_dump*(2**attempt)
            residual = compute_ifft(fft, n=len(edge)) / factor
            recon = connected_pixel_path(edge_from_cap_residual(residual,
                                                                edge,
                                                                center,
                                                                radius,
                                                                ))
            valid_geometry = (
                (not avoid_intersections or not _intersects_other_edges(recon, edge_idx, original_occupied, accepted_pixels)) and
                _stays_near_edge_cells(recon,
                                       label_img=label_img,
                                       cells=row.get('cells', None),
                                       original_edge=edge,
                                       max_edge_distance=max_edge_distance)
            )
            if valid_geometry:
                accepted = recon
                break

        if accepted is None:
            accepted = edge
        recon_edges.append(accepted.tolist())
        accepted_pixels.update(_pixel_set(accepted))

    df[output_column] = recon_edges
    return recon_edges


def load_fft_df(path : str,
                NUM_BINS : int = 5,
                MIN_LENGHT : int = 16,
                MAX_LENGHT : int = 200,
                MAX_CURVATURE : int = 1000,
                PERCENTILE_CLASSES : tuple[int, int, int, int] = (23, 46, 69, 92)):

    if NUM_BINS < 2:
        raise ValueError('NUM_BINS must be at least 2.')
    if len(PERCENTILE_CLASSES) != 4:
        raise ValueError('PERCENTILE_CLASSES must contain four percentile cutoffs.')

    df = pd.read_parquet(path).copy()
    BINS = np.linspace(0, MAX_LENGHT, num=NUM_BINS)

    if 'MSE' not in df:
        raise NotImplementedError('MSE must be precomputed :)')

    mse = df['MSE'].to_numpy(dtype=np.float64)
    percentiles = np.asarray(PERCENTILE_CLASSES, dtype=np.float64)
    if np.any(percentiles <= 0) or np.any(percentiles >= 100) or np.any(np.diff(percentiles) <= 0):
        raise ValueError('PERCENTILE_CLASSES must be strictly increasing percentiles between 0 and 100.')
    cutoffs = np.percentile(mse, q=percentiles)
    df['MSE_bin'] = np.searchsorted(cutoffs, mse, side='left').astype(np.uint8)

    mask = (
        (df.curvature<MAX_CURVATURE) &
        (df.edge_len<MAX_LENGHT) &
        (df.edge_len>=MIN_LENGHT)
    )
    binned_df = df[mask].copy()
    binned_df['len_bin'] = np.clip(np.digitize(binned_df['edge_len'], BINS)-1, 0, NUM_BINS-2).astype(np.uint8)
    return binned_df

def get_bin_number(ed_len, NUM_BINS : int = 5, MAX_LENGHT : int = 200):
    if NUM_BINS < 2:
        raise ValueError('NUM_BINS must be at least 2.')
    BINS = np.linspace(0,MAX_LENGHT, num=NUM_BINS)
    val = np.clip(np.digitize(ed_len, BINS)-1, 0, NUM_BINS-2)
    return val.item()

def sample_fft(bin, binned_df : pd.DataFrame, noise : int, rng=None):
    if len(binned_df) == 0:
        raise ValueError('samples_fft_df must contain at least one FFT sample.')

    noise = int(noise)
    len_bin = binned_df['len_bin'].astype(int)
    mse_bin = binned_df['MSE_bin'].astype(int)
    candidates = [
        binned_df[(len_bin == int(bin)) & (mse_bin == noise)],
        binned_df[mse_bin == noise],
        binned_df[(len_bin == int(bin)) & (mse_bin < 4)],
        binned_df[mse_bin < 4],
        binned_df,
    ]
    _df = next(candidate for candidate in candidates if len(candidate) > 0)
    if rng is None:
        idx = np.random.randint(low=0, high=len(_df))
    else:
        idx = rng.integers(low=0, high=len(_df))
    row = _df.iloc[idx]
    return row['fft_re'], row['fft_im']

def corrupt_edges_from_samples(df : pd.DataFrame,
                               samples_fft_df : pd.DataFrame,
                               output_prefix : str = 'corrupted_fft',
                               noise_amount : str | tuple = 'low',
                               seed : int | None = None,
                               min_edge_len : int = 16,
                               valid_column : str | None = 'valid'):
    if type(noise_amount) is tuple:
        if len(noise_amount)!=4 or sum(noise_amount)!=1:
            raise RuntimeError(f'noise amout must be a tuple of len 4 and sum 1, noise_amount: {noise_amount}')
        else:
            prop = noise_amount
    else:
        noise_profiles = {
            'very-low': [1., .0, .0, .0],
            'low': [.8, .15, .05, .0],
            'medium': [.5, .25, .15, .1],
            'normal': [.25, .25, .25, .25],
            'high': [.2, .2, .3, .3],
        }
        if noise_amount not in noise_profiles:            
            raise ValueError(f'noise_amount: {noise_amount}, not implemented')
        prop = noise_profiles[noise_amount]

    noisy_fft_re, noisy_fft_im = [], []
    rng = np.random.default_rng(seed)
    noise_classes = rng.choice(4, size=len(df), p=prop).astype(int)

    for (_, row), noise in zip(df.iterrows(), noise_classes):
        invalid = valid_column is not None and valid_column in df and not bool(row[valid_column])
        pix = row['pixels']
        ed_len = len(pix)
        if invalid or ed_len < int(min_edge_len):
            noisy_fft_re.append(None)
            noisy_fft_im.append(None)
            continue

        bin = get_bin_number(ed_len)
        re, im = sample_fft(bin, samples_fft_df, int(noise), rng=rng)
        noisy_fft_re.append(re)
        noisy_fft_im.append(im)
    df[f'{output_prefix}_re'] = noisy_fft_re
    df[f'{output_prefix}_im'] = noisy_fft_im
    return df
