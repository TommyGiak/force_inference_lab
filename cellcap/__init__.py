"""Synthetic circular-arc-polygon (CAP) tissues inspired by VMSI.

The package follows the geometric construction described by Noll,
Streichan, and Shraiman, Phys. Rev. X 10, 011072 (2020),
DOI: 10.1103/PhysRevX.10.011072. TissueFromGraph generates synthetic
CAP-like tissues from dual variables (q, p, z2), while TissueFromMask
extracts edges, vertices, and empirical radii from existing label
images.

This is a synthetic data generator and visualization helper, not the
full variational inverse solver from the paper.
"""

__author__ = "Tommaso Giacometti"
__email__ = "tommaso.giacometti5@unibo.it"
__version__ = "0.4.4"

from .DualGraph import DualGraph
from .CellTissue import TissueFromGraph, TissueFromMask

__all__ = ['DualGraph', 'TissueFromGraph', 'TissueFromMask']
