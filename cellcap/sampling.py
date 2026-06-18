"""Sampling utilities for synthetic VMSI-style CAP tissues.

Noll et al. (2020) describe equilibrium circular-arc-polygon (CAP)
tilings through dual variables q_alpha, p_alpha, and z_alpha^2. This
module creates simple synthetic choices of those variables: a perturbed
triangular lattice for q, noisy positive pressures p, and non-negative
height offsets z2.
"""

__author__ = "Tommaso Giacometti"
__email__ = "tommaso.giacometti5@unibo.it"

import numpy as np

def generate_q(q_noise : float, Nx : int, Ny : int):
  """Return generator points on a noisy triangular lattice.

  The generated q points play the role of the dual generating points
  q_alpha in the generalized Voronoi construction of VMSI. Coordinates
  are min-max normalized after adding noise.

  Args:
    q_noise: Noise amplitude as a fraction of the lattice spacing.
    Nx: Number of generators along x.
    Ny: Number of generators along y.

  Returns:
    Array with shape (Nx*Ny, 2), stored as [x, y].
  """
    
    
  dx = 1/(Nx-1)
  x = np.arange(Nx) * dx
  y = np.arange(Ny) * dx * np.sqrt(3)/2
  xx, yy = np.meshgrid(x,y)
      
  xx[1::2] += dx/2
  
  q = np.asarray([[xi,yi] for xi,yi in zip(xx.ravel(),yy.ravel())])
  
  q_noise = dx*q_noise
  q += np.random.randn(*q.shape)*q_noise
  q = (q-q.min())/(q.max()-q.min()) # minmax normalization
  
  assert q.shape == (Nx*Ny,2)
  
  return q

def sample_p(reference_p: float, p_noise: float, N: int):
  """Sample positive cell pressures around a reference value.

  In the CAP model, pressure differences between adjacent cells control
  edge curvature through the Young-Laplace relation.

  Args:
      reference_p: Central pressure value.
      p_noise: Half-width of the additive uniform perturbation.
                Pressures are sampled in [reference_p - p_noise,
                reference_p + p_noise].
      N: Number of cells/generators.

  Returns:
      Positive pressure array with shape (N,).
  """
  p = reference_p * np.ones((N,))
  uniform_noise = np.random.uniform(-p_noise, p_noise, size=p.shape)

  lower_bound = max(0.01 * reference_p, reference_p - p_noise)
  upper_bound = reference_p + p_noise
  return np.clip(p + uniform_noise, lower_bound, upper_bound)

def sample_z2(z2_noise : float, N : int, spacing : float):
  """Sample non-negative z_alpha^2 offsets for the generalized distance.

  The VMSI generalized Voronoi distance is
  d_alpha^2(r) = |r - q_alpha|^2 + z_alpha^2. These offsets deform the
  CAP geometry while preserving the dual-variable parametrization.

  Args:
    z2_noise: Relative amplitude of the sampled offsets.
    N: Number of cells/generators.
    spacing: Typical nearest-neighbor spacing of q.

  Returns:
    Array of z2 values with shape (N,).
  """
  z2 = z2_noise * spacing * np.random.rand(N)
  return z2
