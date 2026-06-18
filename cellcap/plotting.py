"""Plotting utilities for synthetic VMSI-style CAP tissues.

The main visualizations mirror the objects in Noll et al. (2020):
dual generator triangulations, pressure fields, circular arc polygon
(CAP) interfaces, and line tensions derived from the Young-Laplace
pressure-curvature relation.
"""

__author__ = "Tommaso Giacometti"
__email__ = "tommaso.giacometti5@unibo.it"

from .DualGraph import DualGraph
from .geometry import cap_arc_points
from .img_processing import detect_edges, detect_vertices, remove_vertices, remove_low_px_edges

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology


def _mask_figsize(shape, min_long : float = 8.0, max_long : float = 22.0, px_per_inch : float = 120.0):
  """Scale the figure size from the mask shape while preserving aspect."""
  h, w = shape
  long_side = max(h, w)
  long_inches = np.clip(long_side/px_per_inch, min_long, max_long)
  return (long_inches*w/long_side, long_inches*h/long_side)


def _cap_base_image(tissue):
  """Return a light mask image with detected edges and vertices visible."""
  base = np.ones((*tissue.img.shape, 3), dtype=np.float64)
  base[tissue.img != 0] = (0.92, 0.92, 0.90)
  base[tissue.edges != 0] = (0.35, 0.35, 0.35)
  base[tissue.vertices != 0] = (0.08, 0.08, 0.08)
  return base


def _cap_linewidth(shape):
  """Use slightly thicker CAP lines for larger masks."""
  return float(np.clip(max(shape)/1500, 1.0, 2.5))


def _edge_is_valid(tissue, edge_lbl):
  """Return False for edges marked invalid on the tissue or exported table."""
  edge_lbl = int(edge_lbl)
  if edge_lbl in set(getattr(tissue, 'invalid_edges', set())):
    return False

  edges_df = getattr(tissue, 'edges_df', None)
  if edges_df is not None and 'valid' in edges_df and edge_lbl in edges_df.index:
    return bool(edges_df.loc[edge_lbl, 'valid'])

  return True


def _show_label_img(ax, img):
  """Plot a label image with detected borders in the original style."""
  edges = detect_edges(img)
  vertices, _ = detect_vertices(img)
  edges = remove_vertices(edges, vertices)
  edges = measure.label(edges, connectivity=1)
  edges = remove_low_px_edges(edges)

  visible_cells = (img != 0) & ~(edges != 0) & ~(vertices != 0)
  ax.imshow(~visible_cells, cmap='gray')


def _pressure_image(tissue, img):
  """Map cell labels to pressure values, leaving background empty."""
  if not hasattr(tissue, 'p'):
    raise ValueError("No pressure values found for this tissue.")

  pressure = np.full(img.shape, np.nan, dtype=np.float64)
  cells = img != 0
  labels = img[cells].astype(int)
  valid = (labels > 0) & (labels <= len(tissue.p))
  values = np.full(labels.shape, np.nan, dtype=np.float64)
  values[valid] = tissue.p[labels[valid]-1]
  pressure[cells] = values
  return pressure


def plot_triangulation(dual_graph : DualGraph, ax=None, show : bool = True):
  """Plot the Delaunay triangulation of the dual generator points.

  Args:
    dual_graph: DualGraph instance after generate_graph().
    ax: Optional Matplotlib axes. If provided, draw into it.
    show: If True, display and close figures created by this function.
  """
  created_ax = ax is None
  if created_ax:
    fig, ax = plt.subplots(figsize=(8, 7))
  else:
    fig = ax.figure
  ax.set_aspect('equal')

  # Guard against missing data
  if dual_graph.q is None or dual_graph.triangles is None:
    return (fig, ax) if created_ax else ax

  # plot triangulation mesh
  ax.triplot(dual_graph.q[:, 0], dual_graph.q[:, 1], dual_graph.triangles, color="#9B0404", lw=0.8, alpha=0.9)

  # color nodes by p (if available) or z2, otherwise use a single color
  c = getattr(dual_graph, 'p', None)
  label = 'p'
  if c is None:
    c = getattr(dual_graph, 'z2', None)
    label = 'z2'

  if c is not None:
    sc = ax.scatter(dual_graph.q[:, 0], dual_graph.q[:, 1], c=c, cmap='plasma', s=50, edgecolors='k', linewidths=0.3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(label)
  else:
    ax.scatter(dual_graph.q[:, 0], dual_graph.q[:, 1], c='#1f77b4', s=40, edgecolors='k')

  ax.set_title('Delaunay triangulation of q (cell generators)', fontweight='bold')
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(False)
  if created_ax:
    plt.tight_layout()
    if show:
      plt.show()
      plt.close(fig)
    return fig, ax
  return ax


def plot_cell_tissue(tissue, mode : str | None = None, show : bool = True, ax=None):
  """Plot a TissueFromGraph or TissueFromMask in one of the supported visualization modes.

  Args:
    tissue: Tissue object containing labels, CAP arcs, and optional forces.
    mode: One of None/'none', 'corrupted', 'CAP', 'pressure',
      'corrupted_pressure', or 'tension'. 'corrupted' plots
      tissue.corrupted_img in the same style as the standard label image.
      'CAP' overlays circular arcs on the input mask; 'pressure' plots
      pressure when available; 'tension' colors the same arcs by
      Young-Laplace tension.
    ax: Optional Matplotlib axes. If provided, draw into it.
    show: If True, display and close the figure. If False, return the
      open figure and axes for further customization.

  Returns:
    Matplotlib (fig, ax) pair, or only ax when ax was provided.
  """
  created_ax = ax is None
  if created_ax:
    fig, ax = plt.subplots(figsize=_mask_figsize(tissue.img.shape), dpi=120)
  else:
    fig = ax.figure
  
  if mode is None or mode=='none' or mode == 'None':
    _show_label_img(ax, tissue.img)
    title = 'Tissue'

  elif mode == 'corrupted':
    if not hasattr(tissue, 'corrupted_img'):
      raise ValueError("No corrupted image found. Call tissue.make_corrupted_img(...) before plotting mode='corrupted'.")

    _show_label_img(ax, tissue.corrupted_img)
    title = 'Corrupted tissue'
  
  elif mode == 'CAP':
    ax.imshow(_cap_base_image(tissue), interpolation='none')
    linewidth = _cap_linewidth(tissue.img.shape)
    for edge_lbl in np.unique(tissue.edges)[1:]:
      edge_lbl = edge_lbl.item()
      if not _edge_is_valid(tissue, edge_lbl):
        continue
      edge_pixels = np.asarray(tissue.edge_pxs_map.get(edge_lbl, []), dtype=float)
      n_points = max(100, len(edge_pixels))
      reference_xy = edge_pixels[:, ::-1] if len(edge_pixels) else None
      arc = cap_arc_points(edge_lbl,
                           tissue.edges_vert_map,
                           tissue.vertex_coords_xy,
                           tissue.rho,
                           tissue.R,
                           n=n_points,
                           reference_points_xy=reference_xy)
      if arc.size == 0:
        continue
      ax.plot(arc[:,0], arc[:,1], color='#d62728', linewidth=linewidth, alpha=0.7, solid_capstyle='round')
    title = 'CAP fit'
  
  elif mode == 'pressure':
    im = ax.imshow(_pressure_image(tissue, tissue.img), cmap='plasma', interpolation='none')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pressure')
    title = 'Pressure'

  elif mode == 'corrupted_pressure':
    if not hasattr(tissue, 'corrupted_img'):
      raise ValueError("No corrupted image found. Call tissue.make_corrupted_img(...) before plotting mode='corrupted_pressure'.")

    im = ax.imshow(_pressure_image(tissue, tissue.corrupted_img), cmap='plasma', interpolation='none')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('G.T. Pressure')
    title = 'Corrupted Image'
  
  elif mode == 'tension':
    h, w = tissue.img.shape
    img = np.zeros_like(tissue.img).astype(np.float64)
    for edge_lbl in np.unique(tissue.edges)[1:]:
      edge_lbl = edge_lbl.item()
      if not _edge_is_valid(tissue, edge_lbl):
        continue
      arc = cap_arc_points(edge_lbl, tissue.edges_vert_map, tissue.vertex_coords_xy, tissue.rho, tissue.R)
      if arc.size == 0:
        continue
      arc[:,0] = np.clip(arc[:,0], 0, w-1)
      arc[:,1] = np.clip(arc[:,1], 0, h-1)
      arc = arc.astype(np.uint64)
      img[arc[:,1], arc[:,0]] = tissue.T[edge_lbl]
    img = morphology.dilation(img, footprint=morphology.disk(radius=2))
    img = np.where(img==0, np.nan, img)
    im = ax.imshow(img, cmap='hot', interpolation='none')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Tension')
    title = 'Tension CAP'
  
  else:
    if created_ax:
      plt.close(fig)
    raise NotImplementedError(mode)
  
  ax.set_aspect('equal')
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
  for spine in ax.spines.values():
    spine.set_visible(False)
  
  if show and created_ax:
    plt.show()
    plt.close(fig)
  
  return (fig, ax) if created_ax else ax
