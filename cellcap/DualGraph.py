"""Dual-graph generator for synthetic CAP tissues.

The VMSI construction of Noll et al. (2020) represents an equilibrium
CAP tiling through one dual generator per cell. The generator has an
in-plane coordinate q_alpha, a pressure p_alpha, and a height offset
z_alpha^2. This module samples those dual variables and builds their
Delaunay triangulation.
"""

__author__ = "Tommaso Giacometti"
__email__ = "tommaso.giacometti5@unibo.it"

import numpy as np

from .geometry import triangulation
from .sampling import generate_q, sample_p, sample_z2


class DualGraph:
  """Container for synthetic VMSI dual variables.

  Args:
    Nx: Number of generator points along x.
    Ny: Number of generator points along y.
    q_noise: Relative noise added to the triangular lattice positions.
    reference_p: Mean pressure used to sample p.
    p_noise: Additive pressure noise.
    z2_noise: Amplitude of z2 offsets.
    seed: NumPy random seed used for reproducible sampling.

  Attributes:
    q: Generator coordinates [x, y], populated by generate_graph().
    p: Positive cell pressures, populated by generate_graph().
    z2: Squared height offsets in the generalized Voronoi distance.
    triangles: Delaunay triangulation of q, filtered for very flat faces.
  """
  
  def __init__(self,
               Nx : int = 10,
               Ny : int = 10,
               q_noise : float = 0.08,
               reference_p : float = 2.,
               p_noise : float = 0.3,
               z2_noise : float = 0.3,
               seed : int = 709,
               ) -> None:
    
    assert Nx>0
    assert Ny>0
    
    self.Nx = Nx
    self.Ny = Ny
    self.N = Nx*Ny
    self.q_noise = q_noise
    
    self.reference_p = reference_p
    self.p_noise = p_noise
    self.z2_noise = z2_noise
    
    self.q = None
    
    np.random.seed(seed)
    pass


  def generate_graph(self):
    """Sample q, p, z2 and compute the Delaunay triangulation.

    The sampled values are the forward parameters later used by
    TissueFromGraph to rasterize the generalized Voronoi/CAP geometry.
    """
    self.q = generate_q(self.q_noise, self.Nx, self.Ny)
    self.triangles = triangulation(self.q)
    
    self.p = sample_p(self.reference_p, self.p_noise, self.N)
    self.z2 = sample_z2(self.z2_noise, self.N, self.typical_spacing(self.q))
    pass


  @staticmethod
  def typical_spacing(q : np.ndarray):
    """Return the median squared nearest-neighbor distance of q."""
    diff = q[:, None, :] - q[None, :, :]
    dist2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    return np.median(np.min(dist2, axis=1))


  def plot_triangulation(self, ax=None, show : bool = True):
    """Plot the dual Delaunay triangulation of the generator points."""
    from .plotting import plot_triangulation
    return plot_triangulation(self, ax=ax, show=show)
