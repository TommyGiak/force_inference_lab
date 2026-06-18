"""Synthetic cell-tissue objects built from dual graphs or masks.

TissueFromGraph converts a DualGraph into a rasterized generalized Voronoi
label image, extracts edges and vertices, and computes CAP circle
centers, radii, and tensions. TissueFromMask provides the same edge and
vertex extraction for an existing label image, without graph variables,
pressures, or tensions.
"""

__author__ = "Tommaso Giacometti"
__email__ = "tommaso.giacometti5@unibo.it"

import warnings
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from .DualGraph import DualGraph
from collections import defaultdict
from .plotting import plot_cell_tissue
from scipy import ndimage
from skimage import measure
from .fourier import add_signal_fft_analysis, cap_residual_projection, connected_pixel_path, reconstruct_edges_from_cap_residual_fft, corrupt_edges_from_samples
from .img_processing import detect_edges, detect_vertices, remove_vertices, remove_low_px_edges, make_corrupted_img as _make_corrupted_img
from .geometry import generalizedVoronoi, associate_edges_and_vertices, order_edge_pixels, cap_arc_points, compute_theoretical_edges_center_and_radius, compute_empirical_edges_center_and_radius, compute_tension, compute_cell_stress_tensors


class _BaseTissue:
  """Shared image, edge, vertex, and DataFrame utilities."""

  def detect_edges_and_vertices(self, img):
    edges = detect_edges(img)
    vertices, _ = detect_vertices(img)

    edges = remove_vertices(edges, vertices)
    edges = measure.label(edges, connectivity=1)
    edges = remove_low_px_edges(edges)

    edge_pxs_map = self.compute_label_pixels(edges)
    edge_pxs_map = {k: order_edge_pixels(v) for k, v in edge_pxs_map.items()}
    vertex_pxs_map = self.compute_label_pixels(vertices)
    vertex_coords_xy = self.compute_label_centroids_xy(vertices)

    edges_img_map, edges_vert_map = associate_edges_and_vertices(img, edges, vertices, edge_pxs_map)
    return edges, vertices, edge_pxs_map, vertex_pxs_map, vertex_coords_xy, edges_img_map, edges_vert_map

  @staticmethod
  def compute_label_pixels(labels : np.ndarray):
    """Return a mapping from label id to pixel coordinates [row, col]."""
    return {region.label: region.coords for region in measure.regionprops(labels)}

  @staticmethod
  def compute_label_centroids_xy(labels : np.ndarray):
    """Return label centroids in plotting coordinates [x, y]."""
    return {
      region.label: np.asarray([region.centroid[1], region.centroid[0]], dtype=float)
      for region in measure.regionprops(labels)
    }

  @staticmethod
  def compute_label_areas(labels : np.ndarray):
    """Return a mapping from label id to pixel area."""
    return {region.label: region.area for region in measure.regionprops(labels)}

  @staticmethod
  def compute_label_shape_tensors(labels : np.ndarray):
    """Return per-label shape tensors (area-normalized second moment of area).

    The shape tensor S = (1/A) integral (r - c)(r - c)^T dA is recovered from
    scikit-image's physical inertia tensor I via the 2D identity S = tr(I)*Id - I,
    then transposed from [row, col] to plotting coordinates [x, y]. The eigenvector
    of the largest eigenvalue of S is the cell major axis (elongation direction).
    """
    out = {}
    for region in measure.regionprops(labels):
      inertia = region.inertia_tensor              # [row, col], area-normalized
      shape_rc = np.trace(inertia) * np.eye(2) - inertia
      out[region.label] = shape_rc[::-1, ::-1]     # -> [x, y]
    return out

  @staticmethod
  def _labels_touching_zero_or_border(img : np.ndarray):
    """Return nonzero labels that touch the image boundary or label 0."""
    img = np.asarray(img)
    labels = np.unique(img)
    labels = labels[labels != 0].astype(int)
    invalid = set()
    if img.size == 0 or len(labels) == 0:
      return invalid

    border = np.concatenate((img[0], img[-1], img[:,0], img[:,-1]))
    invalid.update(border[border != 0].astype(int).tolist())

    zero_touch = ndimage.binary_dilation(img == 0, structure=np.ones((3, 3), dtype=bool)) & (img != 0)
    invalid.update(np.unique(img[zero_touch]).astype(int).tolist())
    return invalid

  def _cell_validity(self, img):
    """Return validity metadata for cell labels."""
    labels = set(np.unique(img).astype(int).tolist())
    labels.discard(0)
    invalid_cells = set(getattr(self, 'invalid_cells', set()))
    reasons = getattr(self, 'cell_invalid_reasons', {})

    valid = {}
    invalid_reason = {}
    for label in range(1, int(img.max()) + 1):
      if label not in labels:
        valid[label] = False
        invalid_reason[label] = 'missing_label'
      elif label in invalid_cells:
        valid[label] = False
        invalid_reason[label] = reasons.get(label, 'invalid_cell')
      else:
        valid[label] = True
        invalid_reason[label] = ''
    return valid, invalid_reason

  def _edge_validity(self, edge_labels):
    """Return validity metadata for edge labels."""
    invalid_edges = set(getattr(self, 'invalid_edges', set()))
    reasons = getattr(self, 'edge_invalid_reasons', {})
    valid = {}
    invalid_reason = {}
    for label in edge_labels:
      label = int(label)
      valid[label] = label not in invalid_edges
      invalid_reason[label] = reasons.get(label, '') if not valid[label] else ''
    return valid, invalid_reason

  def _set_border_validity(self, img, edges_img_map):
    """Invalidate cells and edges touching holes, background, or image borders."""
    self.invalid_cells = self._labels_touching_zero_or_border(img)
    self.cell_invalid_reasons = {
      label: 'touches_zero_or_border'
      for label in self.invalid_cells
    }

    self.invalid_edges = set()
    self.edge_invalid_reasons = {}
    for edge_lbl, cells in edges_img_map.items():
      cells = [int(cell) for cell in cells]
      if len(cells) < 2:
        self.invalid_edges.add(edge_lbl)
        self.edge_invalid_reasons[edge_lbl] = 'degenerate_cells'
      elif 0 in cells:
        self.invalid_edges.add(edge_lbl)
        self.edge_invalid_reasons[edge_lbl] = 'touches_zero'
      elif any(cell in self.invalid_cells for cell in cells):
        self.invalid_edges.add(edge_lbl)
        self.edge_invalid_reasons[edge_lbl] = 'touches_invalid_cell'

  def _set_default_validity(self, img, edges_img_map):
    """Mark detected cells and edges valid unless they touch the tissue boundary."""
    self._set_border_validity(img, edges_img_map)


  @staticmethod
  def _rasterize_xy_points(points_xy, shape):
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.size == 0:
      return []

    pixels = connected_pixel_path(points_xy[:, ::-1])
    pixels[:,0] = np.clip(pixels[:,0], 0, shape[0]-1)
    pixels[:,1] = np.clip(pixels[:,1], 0, shape[1]-1)
    return pixels.tolist()

  @staticmethod
  def _arc_points_from_endpoints(endpoints_xy, center_xy, radius, n_points):
    endpoints_xy = np.asarray(endpoints_xy, dtype=np.float64)
    center_xy = np.asarray(center_xy, dtype=np.float64)
    radius = float(radius)

    if not np.all(np.isfinite(center_xy)) or not np.isfinite(radius) or radius <= 0:
      return np.linspace(endpoints_xy[0], endpoints_xy[1], n_points)

    angles = np.arctan2(endpoints_xy[:,1]-center_xy[1], endpoints_xy[:,0]-center_xy[0])
    dtheta = (angles[1]-angles[0]) % (2*np.pi)
    if dtheta > np.pi:
      dtheta -= 2*np.pi

    theta = np.linspace(angles[0], angles[0]+dtheta, n_points)
    return center_xy + radius*np.column_stack((np.cos(theta), np.sin(theta)))

  def _add_cap_residual_fft_columns(self, edges_df, img, edges_vert_map, vertex_coords_xy, rho, R):
    fit_pixels = []
    cap_residuals = []
    for edge_lbl, row in tqdm(edges_df.iterrows(), desc='Computing fft w.r.t. CAP residuals', total=len(edges_df), mininterval=0.3):
      if 'valid' in edges_df and not bool(row['valid']):
        fit_pixels.append(None)
        cap_residuals.append(None)
        continue

      n_points = max(len(row['pixels']), 2)
      arc = cap_arc_points(edge_lbl, edges_vert_map, vertex_coords_xy, rho, R, n=n_points)
      if arc.size == 0:
        edge_pixels = np.asarray(row['pixels'], dtype=np.float64)
        endpoints_xy = edge_pixels[[0, -1], ::-1]
        arc = self._arc_points_from_endpoints(endpoints_xy, rho[edge_lbl], R[edge_lbl], n_points)

      fit_pixels.append(self._rasterize_xy_points(arc, img.shape))
      residual = cap_residual_projection(row['pixels'], np.asarray(rho[edge_lbl], dtype=np.float64)[::-1], R[edge_lbl])
      cap_residuals.append(None if residual is None else residual.tolist())

    edges_df['fit_pixels'] = fit_pixels
    edges_df['cap_residuals'] = cap_residuals
    return add_signal_fft_analysis(edges_df, signal_column='cap_residuals')

  def _add_edges_fft_columns(self, edges_df, img, edges_vert_map, vertex_coords_xy, rho, R):
    if rho is not None and R is not None:
      edges_df = self._add_cap_residual_fft_columns(edges_df, img, edges_vert_map, vertex_coords_xy, rho, R)
    return edges_df

  def reconstruct_edges_from_cap_residual_fft(self,
                                             edges_df=None,
                                             output_column : str = 'cap_residual_edges',
                                             compute_threshold : int = 16,
                                             avoid_intersections : bool = True,
                                             max_edge_distance : float | None = None,
                                             dump_factor : float = 1):
    """Reconstruct edge pixels from modified CAP residual FFT columns."""
    if edges_df is None:
      if not hasattr(self, 'edges_df'):
        self.convert_to_df()
      edges_df = self.edges_df

    reconstruct_edges_from_cap_residual_fft(edges_df,
                                            output_column=output_column,
                                            compute_threshold=compute_threshold,
                                            avoid_intersections=avoid_intersections,
                                            label_img=self.img,
                                            max_edge_distance=max_edge_distance,
                                            dump_factor=dump_factor)
    return edges_df

  def reconstruct_edges_from_sampled_cap_residual_fft(self,
                                                     samples_fft_df,
                                                     edges_df=None,
                                                     fft_prefix : str = 'sampled_cap_residual_fft',
                                                     output_column : str = 'sampled_cap_residual_edges',
                                                     compute_threshold : int = 16,
                                                     avoid_intersections : bool = True,
                                                     max_edge_distance : float | None = None,
                                                     noise_amount : str = 'normal',
                                                     dump_factor : float = 1,
                                                     seed : int | None = None):
    """Sample CAP residual FFTs and reconstruct new edge pixels."""
    if edges_df is None:
      if not hasattr(self, 'edges_df'):
        self.convert_to_df()
      edges_df = self.edges_df

    corrupt_edges_from_samples(edges_df,
                               samples_fft_df,
                               output_prefix=fft_prefix,
                               noise_amount=noise_amount,
                               seed=seed,
                               min_edge_len=compute_threshold)
    reconstruct_edges_from_cap_residual_fft(edges_df,
                                            prefix=fft_prefix,
                                            output_column=output_column,
                                            compute_threshold=compute_threshold,
                                            avoid_intersections=avoid_intersections,
                                            label_img=self.img,
                                            dump_factor=dump_factor,
                                            max_edge_distance=max_edge_distance)
    self._last_sampled_fft_corruption = {
      'samples_fft_df': samples_fft_df,
      'fft_prefix': fft_prefix,
      'output_column': output_column,
      'compute_threshold': compute_threshold,
      'avoid_intersections': avoid_intersections,
      'max_edge_distance': max_edge_distance,
      'noise_amount': noise_amount,
      'seed': seed,
    }
    return edges_df

  def make_corrupted_img(self,
                         edges_df=None,
                         vertices_df=None,
                         edge_column : str = 'sampled_cap_residual_edges',
                         expand_distance : int = 5,
                         vertex_preserve_radius : int = 3):
    """Create and store a label image from reconstructed noisy edges."""
    if edges_df is None or vertices_df is None:
      if not hasattr(self, 'edges_df') or not hasattr(self, 'vertices_df'):
        self.convert_to_df()
      edges_df = self.edges_df if edges_df is None else edges_df
      vertices_df = self.vertices_df if vertices_df is None else vertices_df

    corrupted = _make_corrupted_img(self.img,
                                    edges_df,
                                    vertices_df,
                                    edge_column=edge_column,
                                    expand_distance=expand_distance,
                                    vertex_preserve_radius=vertex_preserve_radius)

    self.corrupted_img = corrupted
    return self.corrupted_img

  def convert_to_df(self, stress_tensor : bool = False):
    """Convert the tissue geometry into DataFrames.

    Args:
      stress_tensor: When True, add a per-cell VMSI ``stress_tensor`` column to
        ``cells_df`` (only for tissues with pressures/tensions). Off by default.

    Returns:
      A tuple (cells_df, edges_df, vertices_df).
    """

    cache_attrs = ('cells_df', 'edges_df', 'vertices_df')
    if all(hasattr(self, attr) for attr in cache_attrs):
      if stress_tensor and 'stress_tensor' not in self.cells_df.columns:
        self._add_stress_tensor_column(self.cells_df)
      return tuple(getattr(self, attr) for attr in cache_attrs)

    img = self.img
    edges = self.edges
    edge_pxs_map = self.edge_pxs_map
    vertex_pxs_map = self.vertex_pxs_map
    vertex_coords_xy = self.vertex_coords_xy
    edges_img_map = self.edges_img_map
    edges_vert_map = self.edges_vert_map
    cells_df = self._cells_dataframe(img)
    edge_per_cell = defaultdict(list)
    vert_per_cell = defaultdict(list)
    for edge_lbl, cells in edges_img_map.items():
      if len(cells) < 2:
        continue
      c1, c2 = cells
      verts = edges_vert_map.get(edge_lbl, [])
      edge_per_cell[c1].append(edge_lbl)
      edge_per_cell[c2].append(edge_lbl)
      vert_per_cell[c1].extend(verts)
      vert_per_cell[c2].extend(verts)
    cells_df['edges'] = edge_per_cell
    cells_df['vertices'] = {k: np.unique(vert_per_cell[k]).tolist() for k in vert_per_cell}
    cell_valid, cell_invalid_reason = self._cell_validity(img)
    cells_df['valid'] = [cell_valid.get(int(label), False) for label in cells_df.index]
    cells_df['invalid_reason'] = [cell_invalid_reason.get(int(label), 'missing_label') for label in cells_df.index]
    shape_tensors = self.compute_label_shape_tensors(img)
    cells_df['shape_tensor'] = [shape_tensors.get(int(label)) for label in cells_df.index]

    edge_data = {'cells': edges_img_map, 'vertices': edges_vert_map}
    rho = getattr(self, 'rho', None)
    R = getattr(self, 'R', None)
    T = getattr(self, 'T', None)
    if rho is not None and R is not None:
      edge_data.update({'rho': rho, 'R': R})
    if T is not None:
      edge_data['tension'] = T
    edges_df = pd.DataFrame(edge_data)
    edge_pixels = {lb: edge_pxs_map[lb].tolist() for lb in range(1, edges.max()+1)}
    edges_df['pixels'] = pd.Series(edge_pixels)
    edge_valid, edge_invalid_reason = self._edge_validity(edges_df.index)
    edges_df['valid'] = [edge_valid.get(int(label), False) for label in edges_df.index]
    edges_df['invalid_reason'] = [edge_invalid_reason.get(int(label), 'missing_edge') for label in edges_df.index]
    edges_df = self._add_edges_fft_columns(edges_df, img, edges_vert_map, vertex_coords_xy, rho, R)

    v_ind = sorted(vertex_pxs_map)
    edge_per_vert = defaultdict(list)
    cell_per_vert = defaultdict(list)
    for edge_lbl, verts in edges_vert_map.items():
      cells = edges_img_map.get(edge_lbl, [])
      for vert_lbl in verts:
        edge_per_vert[vert_lbl].append(edge_lbl)
        cell_per_vert[vert_lbl].extend(cells)
    vertices_df = pd.DataFrame(
      {'edges': edge_per_vert,
       'cells': {k: np.unique(cell_per_vert[k]).tolist() for k in cell_per_vert},
       'pixels': [vertex_pxs_map[v].tolist() for v in v_ind],
       'coordinates_xy': [vertex_coords_xy[v].tolist() for v in v_ind],
      },
      index=np.arange(1, max(v_ind)+1) if v_ind else []
    )

    if stress_tensor:
      self._add_stress_tensor_column(cells_df)

    self.cells_df = cells_df
    self.edges_df = edges_df
    self.vertices_df = vertices_df
    return cells_df, edges_df, vertices_df

  def _add_stress_tensor_column(self, cells_df):
    """Attach the per-cell VMSI stress tensor column (no-op without pressures)."""
    p = getattr(self, 'p', None)
    T = getattr(self, 'T', None)
    if p is None or T is None:
      return
    edge_per_cell = defaultdict(list)
    for edge_lbl, cells in self.edges_img_map.items():
      if len(cells) < 2:
        continue
      for cell in cells:
        edge_per_cell[cell].append(edge_lbl)
    areas = self.compute_label_areas(self.img)
    stress = compute_cell_stress_tensors(edge_per_cell, self.edges_vert_map,
                                         self.edges_img_map, self.vertex_coords_xy,
                                         p, T, self.R, self.rho, areas)
    cells_df['stress_tensor'] = [stress.get(int(label)) for label in cells_df.index]


  def plot(self, mode : str | None = None, show : bool = True, ax=None):
    """Plot the tissue using the shared plotting helper."""
    return plot_cell_tissue(self, mode=mode, show=show, ax=ax)


class TissueFromGraph(_BaseTissue):
  """Rasterized synthetic CAP tissue derived from a DualGraph."""

  def __init__(self,
               dual : DualGraph,
               img_size : tuple | str = 'auto',
               ) -> None:
    """Build the raster labels and derived CAP quantities."""

    self.dual = dual
    if img_size == 'auto':
      img_size = self.auto_img_size(self.dual.q, self.dual.Nx, self.dual.Ny)
    else:
      self.warn_if_img_aspect_is_off(self.dual.q, img_size)

    assert len(img_size)==2 and img_size[0]>0 and img_size[1]>0
    self.img_size = img_size

    self.q, self.z2 = self._rescale_q_and_z_with_img_size(self.dual.q, self.dual.z2)
    self.p = self.dual.p

    self.img = generalizedVoronoi(self.img_size, self.q, self.p, self.z2, row_chunk=16)
    self.edges, self.vertices, self.edge_pxs_map, self.vertex_pxs_map, self.vertex_coords_xy, self.edges_img_map, self.edges_vert_map = self.detect_edges_and_vertices(self.img)
    self._set_default_validity(self.img, self.edges_img_map)

    self.rho, self.R = compute_theoretical_edges_center_and_radius(self.q, self.z2, self.p, self.edges, self.edges_img_map)
    self.T = compute_tension(self.p, self.edges, self.edges_img_map, self.R)

  def _rescale_q_and_z_with_img_size(self, q, z2):
    """Scale q and z2 from dual coordinates into image coordinates."""
    h, w = self.img_size
    q = np.asarray(q, dtype=float).copy()
    z2 = np.asarray(z2, dtype=float).copy()

    q_min = q.min(axis=0)
    q_span = q.max(axis=0) - q_min
    if np.any(q_span <= 0):
      raise ValueError('q must span both x and y directions')

    pad = 0.01*min(h,w)
    scale_q = min((w - 2*pad)/q_span[0], (h - 2*pad)/q_span[1])
    q = (q - q_min)*scale_q
    q[:,0] += (w - q_span[0]*scale_q)/2
    q[:,1] += (h - q_span[1]*scale_q)/2
    z2 *= scale_q**2

    return q, z2

  @staticmethod
  def auto_img_size(q, Nx, Ny):
    """Choose an image size that preserves the q aspect ratio."""
    long_side = max(Nx,Ny)*100
    q = np.asarray(q, dtype=float)
    span = q.max(axis=0) - q.min(axis=0)
    aspect = span[0]/span[1]
    if aspect >= 1:
      return (max(1, int(round(long_side/aspect))), long_side)
    return (long_side, max(1, int(round(long_side*aspect))))

  @staticmethod
  def warn_if_img_aspect_is_off(q, img_size, threshold : float = 0.1):
    """Warn when a manual image size noticeably distorts q geometry."""
    q = np.asarray(q, dtype=float)
    h, w = img_size
    span = q.max(axis=0) - q.min(axis=0)
    q_aspect = span[0]/span[1]
    img_aspect = w/h
    if abs(img_aspect/q_aspect - 1) > threshold:
      warnings.warn('img_size aspect ratio differs from q aspect ratio by more than 10%.', RuntimeWarning, stacklevel=2)
    return


  def _cells_dataframe(self, img):
    cells_df = pd.DataFrame({'qx': self.q[:,0], 'qy': self.q[:,1], 'p': self.p, 'z2': self.z2})
    cells_df.index = np.arange(1, img.max()+1)
    return cells_df


class TissueFromMask(_BaseTissue):
  """Cell-tissue geometry extracted from an existing label image."""

  def __init__(self, image : np.ndarray) -> None:
    """Build edge, vertex, and empirical CAP geometry from a label image."""
    assert image.ndim == 2, "The mask must be a grayscale image with ndim==2"

    self.img = np.asarray(image).copy()
    self.img_size = image.shape
    self.edges, self.vertices, self.edge_pxs_map, self.vertex_pxs_map, self.vertex_coords_xy, self.edges_img_map, self.edges_vert_map = self.detect_edges_and_vertices(self.img)
    self._set_mask_validity()
    self.rho, self.R = compute_empirical_edges_center_and_radius(self.edges, self.edges_vert_map, self.edge_pxs_map, self.vertex_coords_xy)

  def _set_mask_validity(self):
    """Invalidate cells and edges touching holes, background, or image borders."""
    self._set_border_validity(self.img, self.edges_img_map)

  def _cells_dataframe(self, img):
    return pd.DataFrame(index=np.arange(1, img.max()+1))
