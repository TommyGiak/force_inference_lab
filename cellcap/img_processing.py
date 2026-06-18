"""Image processing helpers for synthetic CAP label images.

The VMSI paper works from segmented epithelial images. In this package
we already have synthetic label images, so these utilities recover the
same basic geometric objects: interfaces between cells, tricellular
vertices, and cleaned edge labels suitable for CAP plotting.
"""

__author__ = "Tommaso Giacometti"
__email__ = "tommaso.giacometti5@unibo.it"

import numpy as np

from scipy import ndimage
from skimage import measure, segmentation


def detect_edges(img: np.ndarray):
  """Detect pixels where neighboring cell labels differ.

  Args:
    img: Cell label image with labels starting at 1.

  Returns:
    Boolean image marking cell-cell interfaces.
  """
  edge = np.zeros(img.shape, dtype=bool)

  right = img[:, :-1] != img[:, 1:]
  down = img[:-1, :] != img[1:, :]
  diag_down_right = img[:-1, :-1] != img[1:, 1:]

  edge[:, :-1] |= right
  edge[:-1, :] |= down
  edge[:-1, :-1] |= diag_down_right

  return edge


def detect_vertices(img : np.ndarray):
  """Detect junction pixels where at least three cell labels meet.

  Args:
    img: Cell label image with labels starting at 1.

  Returns:
    A labeled vertex image and the local nonzero label count used for
    detection.
  """
  if img.ndim != 2:
    raise ValueError("img must be a 2D label image.")

  edge = detect_edges(img)
  padded = np.pad(img, 1, mode='constant', constant_values=0)
  h, w = img.shape
  windows = np.stack([
    padded[dr:dr + h, dc:dc + w]
    for dr in range(3)
    for dc in range(3)
  ])
  windows.sort(axis=0)
  unique_nonzero = windows != 0
  unique_nonzero[1:] &= windows[1:] != windows[:-1]
  label_count = unique_nonzero.sum(axis=0).astype(np.uint8)

  out = (label_count >= 3) & edge
  vert = measure.label(out, connectivity=2)
  return vert, label_count


def remove_vertices(edges : np.ndarray, vertices : np.ndarray):
  """Remove vertex pixels from the edge image before edge labeling."""
  edges = np.where(
    vertices!=0, 
    np.zeros_like(edges), 
    edges,
    )
  return edges


def remove_low_px_edges(edges : np.ndarray):
  """Remove tiny edge components and relabel the remaining edges."""
  lbls = np.unique(edges)
  for lb in lbls:
    edge = edges==lb
    if edge.sum()<3:
      edges *= ~edge
  edges = measure.label(edges)
  return edges


def _label_set(img):
    labels = np.unique(img)
    return labels[labels != 0]


def _fix_disconnected_cells(img: np.ndarray) -> np.ndarray:
    """Merge smaller disconnected components of each cell into their nearest neighbours.

    When the Voronoi rasterisation produces cells with multiple connected
    components (a known artefact at high pressure-noise), those satellite
    fragments are cleared and their pixels are reassigned to the nearest
    remaining label.  The result has all cells represented by a single
    connected region.

    Only pixels that belong to satellite components are modified; the
    original background (label 0) is never touched.

    Args:
        img: Cell label image (background 0, cells ≥ 1).

    Returns:
        Copy of *img* with every cell's satellite components removed and
        absorbed into the surrounding cells.
    """
    fixed = img.copy()
    struct = ndimage.generate_binary_structure(2, 1)
    cleared = np.zeros(img.shape, dtype=bool)   # pixels we actually cleared

    for lbl in _label_set(img):
        labeled, n = ndimage.label(fixed == lbl, structure=struct)
        if n <= 1:
            continue
        sizes = [int(np.sum(labeled == c)) for c in range(1, n + 1)]
        main_comp = int(np.argmax(sizes)) + 1
        for comp in range(1, n + 1):
            if comp == main_comp:
                continue
            mask = labeled == comp
            fixed[mask] = 0
            cleared |= mask

    # Only fill the pixels we cleared (not the original background).
    if np.any(cleared):
        _, nearest = ndimage.distance_transform_edt(fixed == 0, return_indices=True)
        fixed[cleared] = fixed[nearest[0][cleared], nearest[1][cleared]]

    return fixed


def _local_corridor_watershed(
    img: np.ndarray,
    edges_df,
    corridor_half_width: int = 6,
    edge_column: str = "sampled_cap_residual_edges",
) -> np.ndarray:
    """Reconstruct cell boundaries using per-edge local corridor watersheds.

    Rather than running a single global watershed (which can produce a
    topology different from the original Voronoi rasterisation), this
    function modifies only the narrow corridor around each reconstructed
    edge.  Pixels outside every corridor are copied verbatim from *img*,
    preserving the original topology everywhere except near each edge.

    For each edge (A, B):

    1. Build a binary corridor around the union of the original and
       reconstructed edge pixels, dilated by *corridor_half_width*.
    2. Restrict the corridor to pixels currently labelled A or B.
    3. Run a flat watershed from A-seeds and B-seeds (pixels of each cell
       that lie outside the corridor) using the reconstructed edge as a
       hard barrier.
    4. Assign the reconstructed edge pixels their original label (avoids
       fill-nearest tie-breaking artefacts).
    5. Write the result back into the corridor zone.

    All edges that carry the *edge_column* are processed (both valid and
    invalid), provided the 'cells' entry contains exactly two labels.

    Args:
        img: Reference label image (all cells connected, background 0).
        edges_df: Edge DataFrame with at least a ``cells`` column and the
            *edge_column* column.
        corridor_half_width: Dilation radius (pixels) applied to the union
            of original and reconstructed edge pixels when building the
            corridor.
        edge_column: Column in *edges_df* containing the reconstructed edge
            pixel arrays.

    Returns:
        Label image with reconstructed boundaries.
    """
    corrupted = img.copy()
    H, W = img.shape
    struct = ndimage.generate_binary_structure(2, 1)

    # Process every edge that has a reconstructed pixel list.
    has_cells = "cells" in edges_df.columns
    has_orig  = "pixels" in edges_df.columns
    notna_mask = edges_df[edge_column].notna()

    for idx, row in edges_df[notna_mask].iterrows():
        if not has_cells:
            continue
        cells = list(row["cells"])
        if len(cells) != 2:
            continue
        A, B = int(cells[0]), int(cells[1])

        recon_pix = np.asarray(row[edge_column]).reshape(-1, 2).astype(int)
        if has_orig and row["pixels"] is not None:
            orig_pix = np.asarray(row["pixels"]).reshape(-1, 2).astype(int)
        else:
            orig_pix = recon_pix

        # --- build canvases for original and reconstructed edges ----------
        orig_canvas  = np.zeros((H, W), dtype=bool)
        recon_canvas = np.zeros((H, W), dtype=bool)
        for p in orig_pix:
            if 0 <= p[0] < H and 0 <= p[1] < W:
                orig_canvas[p[0], p[1]] = True
        for p in recon_pix:
            if 0 <= p[0] < H and 0 <= p[1] < W:
                recon_canvas[p[0], p[1]] = True

        # --- corridor: dilation of union of both edge sets ----------------
        corridor = ndimage.binary_dilation(
            orig_canvas | recon_canvas,
            iterations=corridor_half_width,
        )
        zone = corridor & ((corrupted == A) | (corrupted == B))
        if not np.any(zone):
            continue

        A_out = (corrupted == A) & ~zone
        B_out = (corrupted == B) & ~zone
        if not np.any(A_out) or not np.any(B_out):
            continue

        # --- local watershed with reconstructed edge as hard barrier ------
        markers_l = np.zeros((H, W), dtype=np.int32)
        markers_l[A_out] = A
        markers_l[B_out] = B
        mask_nb = (zone | (markers_l != 0)) & ~recon_canvas
        ws_l = segmentation.watershed(
            np.zeros(img.shape, dtype=np.uint8),
            markers=markers_l,
            mask=mask_nb,
        )

        # Assign reconstructed-edge pixels their *original* label (avoids
        # tie-breaking artefacts that would arise from fill-nearest when the
        # edge pixel is equidistant from both cells).
        barrier_in_zone = recon_canvas & zone
        if np.any(barrier_in_zone):
            ws_l[barrier_in_zone] = corrupted[barrier_in_zone]

        # Fill any remaining unassigned zone pixels with nearest label.
        zero_z = (ws_l == 0) & zone
        if np.any(zero_z):
            _, near = ndimage.distance_transform_edt(
                ws_l == 0, return_indices=True
            )
            ws_l[zero_z] = ws_l[near[0][zero_z], near[1][zero_z]]

        corrupted[zone] = ws_l[zone]

    return corrupted


def _fix_new_disconnections(corrupted: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Restore pixels from cells that became newly disconnected during reconstruction.

    After the corridor watershed, a cell that was connected in *img* might
    end up with multiple components in *corrupted* because separate corridors
    nibbled away at opposite sides of a thin cell.  For each such newly
    disconnected cell the smaller components have their pixels restored to
    the corresponding values in *img* (which re-establishes the original
    pixel assignments in those regions).

    Cells that were already disconnected in *img* (Voronoi rasterisation
    artefacts) are left unchanged.

    Args:
        corrupted: Reconstructed label image (modified in place).
        img: Reference label image.

    Returns:
        *corrupted* (modified in place and returned for convenience).
    """
    struct = ndimage.generate_binary_structure(2, 1)
    for lbl in _label_set(img):
        _, n_orig = ndimage.label(img == lbl, structure=struct)
        if n_orig > 1:
            continue   # pre-existing disconnection — skip

        labeled, n_new = ndimage.label(corrupted == lbl, structure=struct)
        if n_new <= 1:
            continue   # still connected — nothing to do

        # Keep the largest component; restore all others to original values.
        sizes = [int(np.sum(labeled == c)) for c in range(1, n_new + 1)]
        main_comp = int(np.argmax(sizes)) + 1
        for comp in range(1, n_new + 1):
            if comp == main_comp:
                continue
            mask = labeled == comp
            corrupted[mask] = img[mask]

    return corrupted


def make_corrupted_img(
    img: np.ndarray,
    edges_df,
    vertices_df,
    edge_column: str = "sampled_cap_residual_edges",
    expand_distance: int = 5,
    vertex_preserve_radius: int = 3,
    corridor_half_width: int = 6,
):
    """Create a corrupted/reconstructed label image from noisy reconstructed borders.

    The reconstruction keeps vertex positions stable while allowing the
    cell-boundary topology to change slightly:

    1. **Pre-fixing disconnected cells** — Voronoi rasterisation at high
       pressure-noise values can produce cells with multiple disconnected
       components.  Those satellite fragments are merged into neighbouring
       cells before reconstruction so that the reference topology is clean.

    2. **Local corridor watershed** — instead of a global nearest-seed flood,
       a separate local watershed is run for each edge inside a narrow
       corridor around the reconstructed edge pixels.  Pixels outside every
       corridor are copied verbatim from the (pre-fixed) reference image.

    3. **New-disconnection repair** — if the cumulative effect of multiple
       corridor operations disconnects a cell that was connected in the
       reference, the smaller fragment is restored to the reference values.

    4. **Vertex-neighbourhood restoration** — original labels are restored
       unconditionally in a small radius around each detected triple junction,
       keeping vertex positions stable for downstream arc fitting.

    Args:
        img: 2-D cell label image (background 0, cells ≥ 1).
        edges_df: Edge DataFrame; must contain *edge_column* and a ``cells``
            column (list of two cell labels per edge).
        vertices_df: Vertex DataFrame with a ``pixels`` column.
        edge_column: Column in *edges_df* holding reconstructed edge pixels.
        expand_distance: Accepted for backwards compatibility; not used by
            the corridor-based implementation.
        vertex_preserve_radius: Radius (pixels) of the vertex-restoration
            patch.  Set to 0 to disable vertex locking.
        corridor_half_width: Dilation radius (pixels) used when building the
            per-edge corridor.  Larger values tolerate bigger edge
            displacements but may affect more pixels per edge.

    Returns:
        Reconstructed label image with the same dtype as *img*.

    Raises:
        ValueError: If *img* is not 2-D, contains no labelled pixels, or
            *edge_column* is absent from *edges_df*.
    """

    if img.ndim != 2:
        raise ValueError("img must be a 2D label image.")

    if edge_column not in edges_df:
        raise ValueError(f"Edge column {edge_column!r} not found in edges_df.")

    # ------------------------------------------------------------------
    # 1. Pre-fix disconnected cells in the reference image.
    #    Voronoi rasterisation at high pressure-noise levels can produce
    #    cells with satellite fragments; these are merged into neighbours
    #    so that the reference topology is clean and all cells are connected.
    # ------------------------------------------------------------------
    img_ref = _fix_disconnected_cells(img)

    labels = _label_set(img_ref)

    if labels.size == 0:
        raise ValueError("img does not contain any positive cell label.")

    # ------------------------------------------------------------------
    # 2. Local corridor watershed.
    # ------------------------------------------------------------------
    corrupted = _local_corridor_watershed(
        img_ref,
        edges_df,
        corridor_half_width=int(corridor_half_width),
        edge_column=edge_column,
    )

    # ------------------------------------------------------------------
    # 3. Repair cells that became newly disconnected.
    # ------------------------------------------------------------------
    corrupted = _fix_new_disconnections(corrupted, img_ref)

    # ------------------------------------------------------------------
    # 4. Restore original labels near vertices (unconditional).
    #
    # Each patch is restored tentatively; if restoring it would disconnect
    # any cell label the patch is reverted for that vertex.
    # ------------------------------------------------------------------
    if vertex_preserve_radius is not None and vertex_preserve_radius > 0:
        r = int(vertex_preserve_radius)
        H, W = img_ref.shape
        struct = ndimage.generate_binary_structure(2, 1)
        for pix_list in vertices_df["pixels"]:
            for pix in pix_list:
                row, col = int(pix[0]), int(pix[1])
                r0 = max(0, row - r)
                r1 = min(H, row + r + 1)
                c0 = max(0, col - r)
                c1 = min(W, col + r + 1)
                ws_patch  = corrupted[r0:r1, c0:c1].copy()
                img_patch = img_ref[r0:r1, c0:c1]
                corrupted[r0:r1, c0:c1] = img_patch
                affected = np.unique(
                    np.concatenate([ws_patch.ravel(), img_patch.ravel()])
                )
                affected = affected[affected != 0]
                for lbl in affected:
                    _, n = ndimage.label(corrupted == lbl, struct)
                    if n > 1:
                        changed = ws_patch != img_patch
                        corrupted[r0:r1, c0:c1][changed] = ws_patch[changed]
                        break

    return corrupted.astype(img.dtype, copy=False)

