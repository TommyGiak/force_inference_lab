"""Geometry routines for synthetic VMSI/CAP tissues.

This module implements the forward geometry used by the synthetic
pipeline. It follows the reduced variables of Noll, Streichan, and
Shraiman (2020): each cell alpha has q_alpha, p_alpha, and z_alpha^2,
and CAP edges are loci where p_alpha d_alpha^2 = p_beta d_beta^2.
The resulting circular arcs encode pressure differences and line
tensions through the Young-Laplace relation.
"""

__author__ = "Tommaso Giacometti"
__email__ = "tommaso.giacometti5@unibo.it"

import numpy as np

from skimage import morphology
from collections import Counter
from scipy.spatial import Delaunay


def triangulation(q : np.ndarray, threshold : int = 3):
  """Compute a filtered Delaunay triangulation of generator points.

  The triangulation is the dual graph of the CAP tiling. Very flat
  triangles are removed with a simple aspect-ratio filter to avoid
  unstable boundary faces.

  Args:
    q: Generator coordinates with shape (N, 2), stored as [x, y].
    threshold: Reserved for compatibility; the current filter uses the
      median triangle aspect ratio.

  Returns:
    Integer array of triangle vertex indices.
  """
  # Compute the Delaunay triangulation
  tri = Delaunay(q)

  # Extract triangle vertex coordinates
  triangles = q[tri.simplices]  # shape: (n_triangles, 3, 2)

  # Triangle vertices
  p0 = triangles[:, 0, :]
  p1 = triangles[:, 1, :]
  p2 = triangles[:, 2, :]

  # Compute side lengths
  a = np.linalg.norm(p1 - p0, axis=1)
  b = np.linalg.norm(p2 - p1, axis=1)
  c = np.linalg.norm(p0 - p2, axis=1)

  # Compute triangle areas using the 2D cross product
  area = 0.5 * np.abs(
      (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
      (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
  )

  # Compute normalized triangle quality
  # quality = 1 for an equilateral triangle
  # quality -> 0 for a very flat / degenerate triangle
  quality = (4 * np.sqrt(3) * area) / (a**2 + b**2 + c**2)

  # Compute aspect ratio
  # aspect_ratio = 1 for an equilateral triangle
  # aspect_ratio increases as the triangle becomes more stretched or flat
  aspect_ratio = 1 / quality

  median = np.median(aspect_ratio)
  mask = aspect_ratio<3*median
  triangles = tri.simplices[mask]
  
  return triangles


def generalizedVoronoi(img_size : tuple, q : np.ndarray, p : np.ndarray, z2 : np.ndarray, row_chunk = 16):
  """Rasterize the generalized Voronoi diagram of the VMSI variables.

  For every pixel r, the selected cell minimizes
  p_alpha * (|r - q_alpha|^2 + z_alpha^2). Boundaries between labels are
  synthetic CAP interfaces.

  Args:
    img_size: Image size as (height, width).
    q: Generator coordinates [x, y] in image coordinates.
    p: Positive pressure-like weights.
    z2: Non-negative squared height offsets.
    row_chunk: Number of image rows processed at a time.

  Returns:
    Label image with cell labels starting at 1.
  """
  img = np.empty(img_size, dtype=np.uint32)
  x = np.arange(img_size[1])
  y = np.arange(img_size[0])
  
  for yi in range(0,img_size[0], row_chunk):
    y_temp = y[yi:yi+row_chunk]
    xx, yy = np.meshgrid(x, y_temp)
    dist = distances(xx, yy, q, z2, p)
    img[yi:yi+row_chunk] = np.argmin(dist, axis=-1).astype(np.uint32)
    
  return img+1


def distances(xx, yy, q, z2, p):
  """Return weighted generalized squared distances to all generators."""
  qx, qy = q[:,0], q[:,1]
  d2 = (xx[..., None] - qx)**2 +(yy[..., None] - qy)**2
  d2 = d2 + z2
  return p*d2


def order_edge_pixels(pixels : np.ndarray):
  """Order a 1-connected edge from one endpoint to the other."""
  pixels = np.asarray(pixels, dtype=int)
  if len(pixels) <= 2:
    return pixels

  pixel_set = {tuple(p) for p in pixels}
  neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
  degrees = {
    p: sum((p[0]+dr, p[1]+dc) in pixel_set for dr, dc in neighbors)
    for p in pixel_set
  }
  endpoints = [p for p, degree in degrees.items() if degree == 1]
  if len(endpoints) != 2:
    return pixels

  path = []
  visited = set()
  current = endpoints[0]
  previous = None
  while current is not None:
    path.append(current)
    visited.add(current)
    next_pixels = [
      (current[0]+dr, current[1]+dc)
      for dr, dc in neighbors
      if (current[0]+dr, current[1]+dc) in pixel_set
      and (current[0]+dr, current[1]+dc) != previous
      and (current[0]+dr, current[1]+dc) not in visited
    ]
    previous = current
    current = next_pixels[0] if next_pixels else None

  if len(path) != len(pixels):
    return pixels
  return np.asarray(path, dtype=int)


def associate_edges_and_vertices(img : np.ndarray, edges : np.ndarray, vertices : np.ndarray, edge_pxs_map : dict):
  """Associate each edge label with neighboring cells and vertices.

  Args:
    img: Cell label image.
    edges: Edge label image.
    vertices: Vertex label image.
    edge_pxs_map: Mapping edge label -> edge pixel coordinates [row, col].

  Returns:
    A pair of dictionaries:
    edges_img_map maps edge label -> two adjacent cell labels.
    edges_vert_map maps edge label -> neighboring vertex labels.
  """
  edges_img_map = {}
  edges_vert_map = {}

  edge_lbl = np.unique(edges)[1:]

  for lb in edge_lbl:
    pxs = edge_pxs_map.get(lb.item())
    if pxs is None or pxs.size == 0:
      continue
    r0 = max(pxs[:,0].min() - 1, 0)
    r1 = min(pxs[:,0].max() + 2, edges.shape[0])
    c0 = max(pxs[:,1].min() - 1, 0)
    c1 = min(pxs[:,1].max() + 2, edges.shape[1])

    edge = edges[r0:r1, c0:c1] == lb
    edge = morphology.dilation(edge)
    
    neig_c = img[r0:r1, c0:c1][edge]
    count = Counter(neig_c).most_common(n=2)
    edges_img_map[lb.item()] = np.asarray([count[0][0],count[1][0]]).tolist()
    
    neig_v = np.unique(vertices[r0:r1, c0:c1][edge])[1:]
    if len(neig_v) >= 1:
      edges_vert_map[lb.item()] = neig_v.tolist()
  
  return edges_img_map, edges_vert_map


def cap_arc_points(edge_lbl : int,
                   edges_vert_map : dict,
                   vertex_coords_xy : dict,
                   rho : dict,
                   R : dict,
                   n : int = 100,
                   reference_points_xy=None):
  """Sample points along the CAP arc assigned to an edge.

  Edges with fewer than two vertices are skipped. If more than two
  vertices are associated with an edge, the two farthest vertices are
  used as endpoints. If reference edge pixels are passed, the arc branch
  closest to those pixels is selected.

  Args:
    edge_lbl: Edge label to draw.
    edges_vert_map: Mapping edge label -> vertex labels.
    vertex_coords_xy: Mapping vertex label -> centroid [x, y].
    rho: Mapping edge label -> circle center [x, y].
    R: Mapping edge label -> circle radius.
    n: Number of points sampled along the arc.
    reference_points_xy: Optional edge pixels in [x, y] coordinates.

  Returns:
    Array of arc points with shape (n, 2), or an empty array if the
    edge has no drawable CAP arc.
  """
  vert_lbls = edges_vert_map.get(edge_lbl, [])
  if len(vert_lbls) < 2:
    return np.empty((0, 2), dtype=float)

  vertices = []
  for vert_lbl in vert_lbls:
    if vert_lbl in vertex_coords_xy:
      vertices.append(vertex_coords_xy[vert_lbl])
  if len(vertices) < 2:
    return np.empty((0, 2), dtype=float)

  vertices = np.asarray(vertices)
  if len(vertices) > 2:
    dist2 = np.sum((vertices[:, None] - vertices[None, :])**2, axis=-1)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    vertices = vertices[[i, j]]

  center = np.asarray(rho[edge_lbl], dtype=float)
  radius = float(R[edge_lbl])
  if not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
    return np.linspace(vertices[0], vertices[1], n)

  angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])
  dtheta = (angles[1] - angles[0]) % (2*np.pi)
  if dtheta > np.pi:
    dtheta -= 2*np.pi

  def sample_arc(delta):
    theta = np.linspace(angles[0], angles[0] + delta, n)
    return center + radius*np.column_stack((np.cos(theta), np.sin(theta)))

  arc = sample_arc(dtheta)
  if reference_points_xy is None:
    return arc

  reference_points_xy = np.asarray(reference_points_xy, dtype=float)
  if reference_points_xy.ndim != 2 or reference_points_xy.shape[0] == 0 or reference_points_xy.shape[1] != 2:
    return arc

  other_dtheta = dtheta - np.sign(dtheta)*2*np.pi if dtheta != 0 else dtheta
  other_arc = sample_arc(other_dtheta)

  def mean_distance_to_reference(points):
    if len(points) > 500:
      points = points[np.linspace(0, len(points)-1, 500).astype(int)]
    ref = reference_points_xy
    if len(ref) > 500:
      ref = ref[np.linspace(0, len(ref)-1, 500).astype(int)]
    dist2 = np.sum((ref[:, None] - points[None, :])**2, axis=2)
    return np.min(dist2, axis=1).mean()

  if mean_distance_to_reference(other_arc) < mean_distance_to_reference(arc):
    return other_arc
  return arc


def compute_theoretical_edges_center_and_radius(q : np.ndarray, z2 : np.ndarray, p : np.ndarray, edges : np.ndarray, edges_img_map : dict):
  """Compute CAP circle centers and radii from VMSI dual variables.

  This implements the forward equations for rho_alpha_beta and
  R_alpha_beta from Noll et al. (2020), Eqs. (6)-(7), for every
  segmented edge.

  Args:
    q: Generator coordinates [x, y].
    z2: Squared height offsets.
    p: Cell pressures/weights.
    edges: Edge label image.
    edges_img_map: Mapping edge label -> two adjacent cell labels.

  Returns:
    Two dictionaries: rho maps edge label -> center [x, y], and R maps
    edge label -> radius. Straight edges are represented with infinite
    radius and NaN center.
  """
  rho, R = {}, {}

  lbls = np.unique(edges)[1:]  
  for lb in lbls:
    c1, c2 = edges_img_map[lb.item()]
    p1, p2 = p[c1-1], p[c2-1]
    q1, q2 = q[c1-1], q[c2-1]
    z_1, z_2 = z2[c1-1], z2[c2-1]
    
    if np.isclose(p1, p2):
      rho[lb.item()] = [np.nan, np.nan]
      R[lb.item()] = np.inf
    else:
      rho[lb.item()] = ((p1*q1 - p2*q2)/(p1 - p2)).tolist()
      R[lb.item()] = np.sqrt((p1*p2*np.sum((q1-q2)**2) - (p1 - p2)*(p1*z_1 - p2*z_2))/(p1-p2)**2).item()

  return rho, R


def compute_empirical_edges_center_and_radius(edges : np.ndarray, edges_vert_map : dict, edge_pxs_map : dict, vertex_coords_xy : dict):
  """Fit empirical CAP centers and radii from edge pixels and vertices.

  Straight or poorly fitted edges keep infinite radius and center.
  """
  rho, R = {}, {}

  rot90 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)

  for edge_lbl in np.unique(edges)[1:]:
    edge_lbl = edge_lbl.item()
    rho[edge_lbl] = [np.inf, np.inf]
    R[edge_lbl] = np.inf

    vert_lbls = edges_vert_map.get(edge_lbl, []) if hasattr(edges_vert_map, 'get') else edges_vert_map[edge_lbl]
    if len(vert_lbls) < 2:
      continue

    endpoints = []
    for vert_lbl in vert_lbls:
      if vert_lbl in vertex_coords_xy:
        endpoints.append(vertex_coords_xy[vert_lbl])
    if len(endpoints) < 2:
      continue

    endpoints = np.asarray(endpoints, dtype=float)
    if len(endpoints) > 2:
      dist2 = np.sum((endpoints[:, None] - endpoints[None, :])**2, axis=-1)
      i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
      endpoints = endpoints[[i, j]]

    r1, r2 = endpoints
    chord = r1-r2
    chord_len = np.linalg.norm(chord)
    if not np.isfinite(chord_len) or chord_len <= 0:
      continue

    edge_pixels = edge_pxs_map.get(edge_lbl)
    if edge_pixels is None or len(edge_pixels) < 3:
      continue
    edge_pixels = np.asarray(edge_pixels, dtype=float)[:, ::-1]

    normal = rot90 @ chord
    normal = normal/chord_len
    midpoint = 0.5*(r1+r2)
    delta = edge_pixels-midpoint
    normal_distance = delta @ normal
    line_energy = float(np.square(normal_distance).mean())
    if not np.isfinite(line_energy):
      continue

    half_chord = 0.5*chord_len
    A = 2.0*np.square(normal_distance).sum()
    if not np.isfinite(A) or A <= 0:
      continue

    B = ((np.square(delta).sum(axis=1) - half_chord**2)*normal_distance).sum()
    y = B/A
    if not np.isfinite(y):
      continue
    radius = np.sqrt(y**2 + half_chord**2)
    center = midpoint + y*normal

    circle_distance = np.sqrt(np.square(edge_pixels-center).sum(axis=1))
    circle_energy = float(np.square(circle_distance-radius).mean())
    if circle_energy < line_energy and np.isfinite(radius) and np.all(np.isfinite(center)):
      rho[edge_lbl] = center.tolist()
      R[edge_lbl] = radius.item()

  return rho, R


def compute_tension(p : np.ndarray, edges : np.ndarray, edges_img_map : dict, R : dict):
  """Compute edge tensions from pressure jumps and CAP radii.

  The Young-Laplace relation used by VMSI gives
  T_alpha_beta = |p_alpha - p_beta| * R_alpha_beta. Straight edges with
  equal pressures are assigned zero tension in this synthetic helper.

  Args:
    p: Cell pressures/weights.
    edges: Edge label image.
    edges_img_map: Mapping edge label -> two adjacent cell labels.
    R: Mapping edge label -> CAP radius.

  Returns:
    Dictionary mapping edge label -> scalar tension.
  """
  T = {}
  
  for lb in np.unique(edges)[1:]:
    c1, c2 = edges_img_map[lb.item()]
    p1, p2 = p[c1-1], p[c2-1]      
    if np.isfinite(R[lb.item()]):
      T[lb.item()] = (abs(p1-p2)*R[lb.item()]).item()
    else:
      T[lb.item()] = 0.

  return T


def compute_cell_stress_tensors(edge_per_cell : dict, edges_vert_map : dict, edges_img_map : dict,
                                vertex_coords_xy : dict, p : np.ndarray, T : dict, R : dict,
                                rho : dict, areas : dict):
  """Compute per-cell VMSI stress tensors (Noll et al., PRX 2020).

  For each cell alpha the stress tensor (2x2, in [x, y]) is

    sigma_alpha = (1/A_alpha) [ sum_b T_b l_b (t_b x t_b)
                                + 1/2 sum_b dp_b l_b (n_b x n_b) ]

  summed over the boundary edges b. Each edge contributes a single chord-based
  tangent t_b = (e1 - e0)/||e1 - e0|| (vertex to vertex) and normal n_b = [-t_y, t_x],
  both weighted by the edge length l_b (Batchelor stress); l_b is the CAP arc length
  R_b * dtheta_b (chord length for straight edges, R = inf). sigma is symmetric by
  construction. Edges touching background (label 0) or with missing endpoints are skipped.

  Args:
    edge_per_cell: Mapping cell label -> list of boundary edge labels.
    edges_vert_map: Mapping edge label -> two endpoint vertex labels.
    edges_img_map: Mapping edge label -> two adjacent cell labels.
    vertex_coords_xy: Mapping vertex label -> [x, y] coordinates.
    p: Cell pressures/weights (0-indexed; cell label L -> p[L-1]).
    T: Mapping edge label -> tension.
    R: Mapping edge label -> CAP radius.
    rho: Mapping edge label -> CAP center [x, y].
    areas: Mapping cell label -> pixel area.

  Returns:
    Dictionary mapping cell label -> 2x2 stress tensor.
  """
  rot = np.array([[0.0, -1.0], [1.0, 0.0]])
  out = {}
  for cell, edges in edge_per_cell.items():
    area = areas.get(int(cell))
    if not area:
      continue
    sigma = np.zeros((2, 2))
    for b in edges:
      verts = edges_vert_map.get(b)
      if verts is None or len(verts) < 2:
        continue
      e0 = np.asarray(vertex_coords_xy[verts[0]], dtype=float)
      e1 = np.asarray(vertex_coords_xy[verts[1]], dtype=float)
      chord = e1 - e0
      length = np.linalg.norm(chord)
      if not np.isfinite(length) or length <= 0:
        continue
      t = chord/length
      n = t @ rot
      Rb = R.get(b, np.inf)
      rb = np.asarray(rho.get(b, [np.nan, np.nan]), dtype=float)
      if np.isfinite(Rb) and np.all(np.isfinite(rb)):
        d0, d1 = e0 - rb, e1 - rb
        dtheta = np.arctan2(abs(d0[0]*d1[1] - d0[1]*d1[0]), d0 @ d1)
        ell = abs(Rb)*dtheta
      else:
        ell = length
      sigma += T.get(b, 0.0)*ell*np.outer(t, t)
      c1, c2 = edges_img_map.get(b, (0, 0))
      if c1 >= 1 and c2 >= 1:
        dp = abs(p[c1-1] - p[c2-1])
        sigma += 0.5*dp*ell*np.outer(n, n)
    out[int(cell)] = sigma/area
  return out
