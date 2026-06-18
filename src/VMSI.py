import warnings
import nlopt
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree
from skimage import measure

from .grads_and_utils import *
from .grads_and_utils import (
    main_objective_core_safe,
    main_state_core_safe,
    theta_energy_core_safe,
    theta_radius_core_safe,
)
from .helper_functions import _compute_new_vertex_position


def _clone_nested_value(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, list):
        return [_clone_nested_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested_value(item) for item in value)
    return value


def _clone_selected_object_columns(df, object_columns):
    cloned = df.copy(deep=False)
    cloned.index = df.index.copy()
    cloned.columns = df.columns.copy()

    for column in object_columns:
        if column in cloned.columns:
            cloned[column] = cloned[column].map(_clone_nested_value)

    return cloned


def _edge_pixels_to_array(edge_pixels):
    if edge_pixels is None:
        return None

    try:
        pixels = np.asarray(edge_pixels, dtype=np.float64)
    except (TypeError, ValueError):
        try:
            pixels = np.vstack(edge_pixels).T.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            return None

    if pixels.ndim != 2:
        return None
    if pixels.shape[1] != 2:
        if pixels.shape[0] == 2:
            pixels = pixels.T
        else:
            return None
    if pixels.shape[0] == 0 or not np.isfinite(pixels).all():
        return None
    return pixels


def _regularized_sparse_least_squares(A, b, relative_damp=1e-3, atol=1e-12, btol=1e-12):
    if sparse.issparse(A):
        data = A.data
    else:
        data = np.asarray(A, dtype=np.float64).ravel()

    if data.size == 0:
        damp = relative_damp
    else:
        scale = float(np.sqrt(np.mean(data * data)))
        damp = relative_damp * max(scale, 1.0)

    return sparse_linalg.lsmr(A, b, damp=damp, atol=atol, btol=btol)[0]


_MAIN_DP_EPS = 1e-9
_MAIN_INVALID_PENALTY = 1e6
_NLOPT_CLEAN_RESULTS = {
    nlopt.SUCCESS,
    nlopt.STOPVAL_REACHED,
    nlopt.FTOL_REACHED,
    nlopt.XTOL_REACHED,
}
_NLOPT_RESULT_NAMES = {
    nlopt.SUCCESS: "SUCCESS",
    nlopt.STOPVAL_REACHED: "STOPVAL_REACHED",
    nlopt.FTOL_REACHED: "FTOL_REACHED",
    nlopt.XTOL_REACHED: "XTOL_REACHED",
    nlopt.MAXEVAL_REACHED: "MAXEVAL_REACHED",
    nlopt.MAXTIME_REACHED: "MAXTIME_REACHED",
    nlopt.FAILURE: "FAILURE",
    nlopt.INVALID_ARGS: "INVALID_ARGS",
    nlopt.OUT_OF_MEMORY: "OUT_OF_MEMORY",
    nlopt.ROUNDOFF_LIMITED: "ROUNDOFF_LIMITED",
    nlopt.FORCED_STOP: "FORCED_STOP",
}
_STATE_SCORE_PENALTY = 1e3
_TENSION_RADICAND_TOL = 1e-8


def _nlopt_clean_result(result):
    return result in _NLOPT_CLEAN_RESULTS


def _nlopt_result_name(result):
    return _NLOPT_RESULT_NAMES.get(result, f"UNKNOWN({result})")


def _positive_part(value):
    if not np.isfinite(value):
        return _MAIN_INVALID_PENALTY
    return max(float(value), 0.0)


class VMSI():

    def __init__(self, vertices, cells, edges, width, height, verbose, optimiser='nlopt'):
        self.vertices = _clone_selected_object_columns(vertices, ("coords", "ncells", "nverts", "edges"))
        self.cells = _clone_selected_object_columns(cells, ("nverts", "ncells", "edges"))
        self.edges = _clone_selected_object_columns(edges, ("pixels", "verts", "cells"))
        self.width = width
        self.height = height
        self.verbose = verbose
        self.optimiser = optimiser

        # Mark fourfold vertices
        self.vertices['fourfold'] = self.vertices['nverts'].map(len).to_numpy() != 3

        # Initialize new columns
        n_edges = len(self.edges)
        n_cells = len(self.cells)
        self.edges['radius'] = np.zeros(n_edges, dtype=float)
        self.edges['rho'] = [(0.0, 0.0)] * n_edges
        self.edges['fitenergy'] = np.zeros(n_edges, dtype=float)
        self.edges['tension'] = np.zeros(n_edges, dtype=float)
        self.cells['pressure'] = np.zeros(n_cells, dtype=float)
        self.cells['qx'] = np.zeros(n_cells, dtype=float)
        self.cells['qy'] = np.zeros(n_cells, dtype=float)
        self.cells['theta'] = np.zeros(n_cells, dtype=float)
        self.cells['stress'] = [(0.0, 0.0, 0.0)] * n_cells

        # Initialize attributes
        self.dV = None
        self.dC = None
        self.involved_cells = None
        self.involved_vertices = None
        self.involved_edges = None
        self.bulk_cells = None
        self.bulk_vertices = None
        self.ext_cells = None
        self.ext_vertices = None
        self.cell_pairs = None
        self.edgearc_x = None
        self.edgearc_y = None
        self.avg_edge_length = None
        self.mapping_dict = None
        self.optimisation_status = {}

        if self.optimiser == 'nlopt':
            import nlopt
        else:
            raise ValueError(f"Unsupported optimiser '{self.optimiser}'. Use 'nlopt'.")


    def map_index(self, seg_mask, labelled_mask):
        # Avoid upsampling the full mask: centroid coordinates on a 2x nearest-neighbor
        # rescaled mask are equivalent to (2 * centroid + 0.5).
        seg_props = measure.regionprops_table(seg_mask, properties=('label', 'centroid'))
        if len(seg_props['label']) == 0:
            return {}

        img_centroids = np.column_stack(((2.0 * seg_props['centroid-1']) + 0.5,(2.0 * seg_props['centroid-0']) + 0.5))
        img_labels = np.asarray(seg_props['label'])

        labelled_props = measure.regionprops_table(labelled_mask, properties=('centroid',))
        if len(labelled_props['centroid-0']) == 0:
            return {}

        labelled_centroids = np.column_stack((labelled_props['centroid-1'],labelled_props['centroid-0']))

        # Nearest-neighbor lookup scales better than a dense pairwise distance matrix.
        tree = cKDTree(img_centroids)
        indices = np.asarray(tree.query(labelled_centroids, k=1)[1]).ravel()
        matched_labels = np.asarray(img_labels).ravel()[indices]
        return {i: int(label) for i, label in enumerate(matched_labels)}


    def _repair_multicell_edges(self, labelled_mask):
        if labelled_mask is None or "label" not in self.cells.columns:
            return

        labels = np.asarray(self.cells["label"], dtype=int)
        label_to_cell = {int(label): int(cell) for cell, label in enumerate(labels)}
        areas = np.asarray(self.cells["area"], dtype=float) if "area" in self.cells.columns else np.ones(len(self.cells))

        edge_cells = self.edges["cells"].tolist()
        edge_pixels = self.edges["pixels"].tolist()
        repaired = 0
        skipped = 0

        for edge_idx, cells in enumerate(edge_cells):
            raw_cells = np.asarray(cells, dtype=int).ravel()
            candidates = []
            seen = set()
            for cell in raw_cells:
                cell = int(cell)
                if 0 <= cell < len(self.cells) and cell not in seen:
                    candidates.append(cell)
                    seen.add(cell)

            if len(candidates) <= 2:
                continue

            candidate_to_pos = {cell: pos for pos, cell in enumerate(candidates)}
            counts = np.zeros(len(candidates), dtype=np.int64)
            pixels = _edge_pixels_to_array(edge_pixels[edge_idx])

            if pixels is not None:
                coords = np.rint(pixels).astype(int, copy=False)
                height, width = labelled_mask.shape
                for x, y in coords:
                    if x < 0 or y < 0 or x >= width or y >= height:
                        continue
                    y0 = max(y - 1, 0)
                    y1 = min(y + 2, height)
                    x0 = max(x - 1, 0)
                    x1 = min(x + 2, width)
                    local_labels, local_counts = np.unique(labelled_mask[y0:y1, x0:x1], return_counts=True)
                    for label, count in zip(local_labels, local_counts):
                        if label == 0:
                            continue
                        cell = label_to_cell.get(int(label))
                        pos = candidate_to_pos.get(cell)
                        if pos is not None:
                            counts[pos] += int(count)

            if np.count_nonzero(counts) >= 2:
                ranked = np.argsort(-counts, kind="stable")
                chosen = [candidates[pos] for pos in ranked[:2]]
            else:
                ranked = np.argsort(-areas[candidates], kind="stable")
                chosen = [candidates[pos] for pos in ranked[:2]]

            if len(chosen) == 2 and chosen[0] != chosen[1]:
                edge_cells[edge_idx] = np.asarray(sorted(chosen), dtype=int)
                repaired += 1
            else:
                skipped += 1

        self.edges["cells"] = edge_cells
        self.optimisation_status["multicell_edges_repaired"] = repaired
        self.optimisation_status["multicell_edges_skipped"] = skipped

        if self.verbose and repaired:
            print(f"     Repaired {repaired} multicell edge(s)")


    def fit_circle(self):
        """
        Fit circle to each edge
        If edge is too flat, fit line instead
        """
        n_edges = len(self.edges)
        coords = self.vertices["coords"].to_numpy()
        edge_verts = self.edges["verts"].to_numpy()
        edge_pixels_all = self.edges["pixels"].to_numpy()
        radius_out = np.full(n_edges, np.inf, dtype=np.float64)
        rho_out = np.full((n_edges, 2), np.inf, dtype=np.float64)
        fitenergy_out = np.full(n_edges, np.inf, dtype=np.float64)
        rot90 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        n_vertices = len(coords)

        for i in range(n_edges):
            verts = np.asarray(edge_verts[i], dtype=int).ravel()
            if verts.size != 2:
                continue

            v0, v1 = verts
            if v0 < 0 or v1 < 0 or v0 >= n_vertices or v1 >= n_vertices or v0 == v1:
                continue

            r1 = np.asarray(coords[v0], dtype=np.float64)
            r2 = np.asarray(coords[v1], dtype=np.float64)
            if r1.shape != (2,) or r2.shape != (2,) or not np.isfinite(r1).all() or not np.isfinite(r2).all():
                continue

            edge_pixels = _edge_pixels_to_array(edge_pixels_all[i])
            if edge_pixels is None:
                continue

            nB = np.matmul(rot90, r1 - r2)
            D = np.linalg.norm(nB)
            if not np.isfinite(D) or D <= 0:
                continue

            nB = nB / D
            x0 = 0.5 * (r1 + r2)
            delta = np.subtract(edge_pixels, x0)
            IP = (delta[:, 0] * nB[0]) + (delta[:, 1] * nB[1])
            linedistance = float(np.square(IP).mean())
            if not np.isfinite(linedistance):
                continue

            L0 = D * 0.5
            A = 2.0 * np.square(IP).sum()
            if not np.isfinite(A) or A <= 0:
                fitenergy_out[i] = linedistance
                continue

            B = ((np.square(delta).sum(axis=1) - np.square(L0)) * IP).sum()
            y0 = B / A

            def energyfunc(x):
                x = float(np.asarray(x, dtype=np.float64).reshape(-1)[0])
                shifted = delta - (x * nB)
                return float(np.mean(
                    np.square(
                        np.sqrt(np.square(shifted).sum(axis=1))
                        - np.sqrt(np.square(x) + np.square(L0))
                    )
                ))

            y_init = y0 if np.isfinite(y0) else 0.0
            res = minimize(energyfunc, y_init, tol=1e-8)
            if not np.isfinite(res.fun):
                fitenergy_out[i] = linedistance
                continue

            y = np.asarray(res.x, dtype=np.float64).reshape(-1)[0]
            E = float(res.fun)

            if E < linedistance and edge_pixels.shape[0] > 3 and np.isfinite(y):
                radius_out[i] = float(np.sqrt(np.square(y) + np.square(L0)))
                rho_out[i] = x0 + (y * nB)
                fitenergy_out[i] = E
            else:
                fitenergy_out[i] = linedistance

        self.edges["radius"] = radius_out
        self.edges["rho"] = [rho_out[i] for i in range(n_edges)]
        self.edges["fitenergy"] = fitenergy_out


    def remove_fourfold(self):
        """
        Recursively removes fourfold (or greater) vertices by splitting vertex apart
        in direction of greatest variance.
        """
        self.invalid_cells = np.array([], dtype=int)

        vertices = self.vertices
        edges = self.edges
        cells = self.cells
        v_coords = vertices["coords"].tolist()
        v_nverts = vertices["nverts"].tolist()
        v_ncells = vertices["ncells"].tolist()
        v_edges = vertices["edges"].tolist()
        v_fourfold = vertices["fourfold"].tolist()
        e_pixels = edges["pixels"].tolist()
        e_verts = edges["verts"].tolist()
        e_cells = edges["cells"].tolist()
        e_radius = edges["radius"].tolist()
        e_rho = edges["rho"].tolist()
        e_fitenergy = edges["fitenergy"].tolist()
        e_tension = edges["tension"].tolist()
        c_nverts = cells["nverts"].tolist()
        c_ncells = cells["ncells"].tolist()
        c_numv = cells["numv"].tolist()

        n_vertices_initial = len(v_coords)

        for v in range(n_vertices_initial):
            ncells = np.array([], dtype=int)
            try:
                if v >= len(v_nverts):
                    continue

                nverts = np.asarray(v_nverts[v], dtype=int)
                if nverts.size > 1:
                    nverts = nverts[np.sort(np.unique(nverts, return_index=True)[1])]
                v_nverts[v] = nverts

                nedges = np.asarray(v_edges[v], dtype=int)
                if nedges.size > 1:
                    nedges = nedges[np.sort(np.unique(nedges, return_index=True)[1])]
                v_edges[v] = nedges

                ncells = np.asarray(v_ncells[v], dtype=int)
                if ncells.size > 1:
                    ncells = ncells[np.sort(np.unique(ncells, return_index=True)[1])]
                v_ncells[v] = ncells

                if len(v_nverts[v]) > 3 and not (0 in v_ncells[v]):
                    while len(v_nverts[v]) > 3:
                        num_v = len(v_coords)
                        num_e = len(e_verts)

                        nverts = np.asarray(v_nverts[v], dtype=int)
                        nedges = np.asarray(v_edges[v], dtype=int)
                        ncells = np.asarray(v_ncells[v], dtype=int)

                        if nverts.size > 1:
                            nverts = nverts[np.sort(np.unique(nverts, return_index=True)[1])]
                            v_nverts[v] = nverts
                        if nedges.size > 1:
                            nedges = nedges[np.sort(np.unique(nedges, return_index=True)[1])]
                            v_edges[v] = nedges
                        if ncells.size > 1:
                            ncells = ncells[np.sort(np.unique(ncells, return_index=True)[1])]
                            v_ncells[v] = ncells

                        if self.verbose and self.mapping_dict is not None:
                            print(f"        Processing vertex {v} at cells {[self.mapping_dict[cell] for cell in ncells]}")

                        if len(ncells) < 4:
                            if self.verbose:
                                print("Check cells. Unlikely to be true fourfold")
                            v_fourfold[v] = False
                            break

                        R = np.asarray([v_coords[vert] for vert in nverts], dtype=np.float64)
                        rV = np.asarray(v_coords[v], dtype=np.float64)
                        R = R - np.mean(R, axis=0)
                        I = R.T @ R
                        W, V = np.linalg.eig(I)
                        direction = V[:, np.argmax(W)]

                        # create two new vertices
                        rV1 = rV + 0.5 * direction
                        rV2 = rV - 0.5 * direction

                        proj = R @ direction
                        indices = np.argsort(proj)[-2:]
                        pos_mask = np.zeros(len(nverts), dtype=bool)
                        pos_mask[indices] = True
                        neg_mask = ~pos_mask
                        pos_nverts = nverts[pos_mask]
                        neg_nverts = nverts[neg_mask]

                        # update current vertex -> negative vertex
                        v_coords[v] = rV2.tolist()
                        v_nverts[v] = np.concatenate((neg_nverts, np.array([num_v], dtype=int)))
                        v_fourfold[v] = len(v_nverts[v]) > 3

                        neg_cells_list = []
                        neg_nverts_set = set(neg_nverts.tolist())
                        for cell in ncells:
                            cell_verts_set = set(np.asarray(c_nverts[cell], dtype=int).tolist())
                            if sum((nv in cell_verts_set) for nv in neg_nverts_set) == 2:
                                neg_cells_list.append(cell)
                        neg_cells = np.asarray(neg_cells_list, dtype=int)

                        # add positive vertex
                        v_coords.append(rV1.tolist())
                        v_ncells.append(np.array([], dtype=int))
                        v_nverts.append(np.concatenate((pos_nverts, np.array([v], dtype=int))))
                        v_edges.append(np.array([], dtype=int))
                        v_fourfold.append(False)

                        pos_cells_list = []
                        pos_nverts_set = set(pos_nverts.tolist())
                        for cell in ncells:
                            cell_verts_set = set(np.asarray(c_nverts[cell], dtype=int).tolist())
                            if sum((nv in cell_verts_set) for nv in pos_nverts_set) == 2:
                                pos_cells_list.append(cell)
                        pos_cell = np.asarray(pos_cells_list, dtype=int)

                        # update neighbouring vertices
                        for vert in pos_nverts:
                            arr = np.asarray(v_nverts[vert], dtype=int).copy()
                            arr[arr == v] = num_v
                            v_nverts[vert] = arr

                        excluded_cells = np.concatenate((pos_cell.ravel(), neg_cells.ravel()))
                        joint_cells = ncells[~np.isin(ncells, excluded_cells)]

                        v_ncells[v] = np.concatenate((joint_cells, neg_cells))
                        v_ncells[num_v] = np.concatenate((joint_cells, pos_cell))

                        # update edges
                        neg_edges = nedges[neg_mask]
                        pos_edges = nedges[pos_mask]

                        for edge_idx in pos_edges:
                            arr = np.asarray(e_verts[edge_idx], dtype=int).copy()
                            arr[arr == v] = num_v
                            e_verts[edge_idx] = arr

                        # add new edge
                        e_pixels.append(np.array([], dtype=int))
                        e_verts.append(np.array([v, num_v], dtype=int))
                        e_cells.append(joint_cells)
                        e_radius.append(np.inf)
                        e_rho.append(np.array([np.inf, np.inf], dtype=float))
                        e_fitenergy.append(np.inf)
                        e_tension.append(float(0))

                        # update edges of vertices
                        v_edges[v] = np.concatenate((neg_edges, np.array([num_e], dtype=int)))
                        v_edges[num_v] = np.concatenate((pos_edges, np.array([num_e], dtype=int)))

                        # update cells
                        for cell in pos_cell:
                            cell_nverts = np.asarray(c_nverts[cell], dtype=int).copy()
                            cell_nverts[cell_nverts == v] = num_v
                            c_nverts[cell] = cell_nverts

                            cell_ncells = np.asarray(c_ncells[cell], dtype=int)
                            c_ncells[cell] = cell_ncells[~np.isin(cell_ncells, neg_cells)]

                        for cell in neg_cells:
                            cell_ncells = np.asarray(c_ncells[cell], dtype=int)
                            c_ncells[cell] = cell_ncells[~np.isin(cell_ncells, pos_cell)]

                        for cell in joint_cells:
                            c_nverts[cell] = np.concatenate((np.asarray(c_nverts[cell], dtype=int), np.array([num_v], dtype=int)))
                            c_numv[cell] = c_numv[cell] + 1

            except Exception:
                raise ValueError(f"     Error processing vertex {v} at {[cell + 1 for cell in ncells]}")

        # Expand frames if new vertices/edges were created during splitting.
        if len(self.vertices) != len(v_coords):
            self.vertices = self.vertices.reindex(range(len(v_coords)))
        if len(self.edges) != len(e_verts):
            self.edges = self.edges.reindex(range(len(e_verts)))

        # write back once
        self.vertices["coords"] = v_coords
        self.vertices["nverts"] = v_nverts
        self.vertices["ncells"] = v_ncells
        self.vertices["edges"] = v_edges
        self.vertices["fourfold"] = v_fourfold

        self.edges["pixels"] = e_pixels
        self.edges["verts"] = e_verts
        self.edges["cells"] = e_cells
        self.edges["radius"] = e_radius
        self.edges["rho"] = e_rho
        self.edges["fitenergy"] = e_fitenergy
        self.edges["tension"] = e_tension

        self.cells["nverts"] = c_nverts
        self.cells["ncells"] = c_ncells
        self.cells["numv"] = c_numv

        return


    def make_convex(self):
        """
        remove concave vertices by moving vertex to ensure all angles < pi
        """
        holes = self.cells["holes"].to_numpy()  # (Nc,)
        boundary_cells = np.flatnonzero(holes)  # (Nb_cells,)

        if boundary_cells.size > 0:
            boundary_vert_lists = [
                np.asarray(nverts, dtype=np.int64).ravel()
                for nverts in self.cells.loc[boundary_cells, "nverts"].tolist()
            ]
            boundary_verts = np.unique(np.concatenate(boundary_vert_lists)).astype(np.int64, copy=False)  # (Nb_verts,)
        else:
            boundary_verts = np.array([], dtype=int)  # (0,)

        n_vertices = len(self.vertices)

        boundary_mask = np.zeros(n_vertices, dtype=bool)  # (Nv,)
        boundary_mask[boundary_verts] = True

        coords_col = self.vertices["coords"].tolist()   # list of (2,)
        nverts_col = self.vertices["nverts"].tolist()   # list of (deg_v,)

        for v in range(n_vertices):
            if boundary_mask[v]:
                continue
            if len(nverts_col[v]) != 3:
                continue

            rv = np.asarray(coords_col[v], dtype=np.float64)           # (2,)
            nverts = np.asarray(nverts_col[v], dtype=np.int64)         # (3,)
            n = np.asarray([coords_col[nverts[0]],
                            coords_col[nverts[1]],
                            coords_col[nverts[2]]], dtype=np.float64)  # (3, 2)

            nrv = _compute_new_vertex_position(rv, n)                  # (2,)
            coords_col[v] = nrv.tolist()

        self.vertices["coords"] = coords_col


    def prepare_data(self):
        """
        Prepare data for tension inference:
        remove fourfold vertices, fit circles, make vertices convex.
        """
        if self.verbose:
            print("     Fitting circular arcs to edges...")
        try:
            self.fit_circle()
        except Exception as e:
            raise ValueError("      Fitting circular arcs failed") from e
        if self.verbose:
            print("     Fitting circular arcs success\n")

            print("     Removing fourfold vertices...")
        try:
            self.remove_fourfold()
        except Exception as e:
            raise ValueError("      Fourfold vertices removal failed") from e

        if len(self.invalid_cells) != 0:
            raise ValueError("      Fourfold vertices removal failed")
        if self.verbose:
            print("     Fourfold vertices removal success\n")
            
            print("     Making vertices convex...")
        try:
            self.make_convex()
        except Exception as e:
            raise ValueError("      Making convex failed") from e
        if self.verbose:
            print("     Making convex success\n")


    def classify_cells(self):
        """
        Determine which cells are involved in tension inference.
        Initialize q as cell centroids.
        """
        cells = self.cells
        vertices = self.vertices

        n_cells = len(cells)
        c_ncells = cells["ncells"].tolist()         # list, each element shape (k_i,)
        c_nverts = cells["nverts"].tolist()         # list, each element shape (m_i,)
        c_centroids = cells["centroids"].tolist()   # list, each element shape (2,)
        holes = cells["holes"].to_numpy()           # (Nc,)
        v_ncells = vertices["ncells"].tolist()      # list, each element shape (k_i,)
        v_nverts = vertices["nverts"].tolist()      # list, each element shape (m_i,)

        bulk_cells = np.arange(n_cells, dtype=int) # bulk_cells: (Nc,)

        boundary_from_zero = np.asarray(c_ncells[0], dtype=int)                 # (k0,)
        boundary_from_holes = np.flatnonzero(holes[1:]).astype(int) + 1         # shift back to the original cell indices after skipping cell 0
        boundary_cells = np.unique(np.concatenate((boundary_from_zero, boundary_from_holes)))  # (Nb,)

        bulk_mask = np.ones(n_cells, dtype=bool)                                # (Nc,)
        bulk_mask[boundary_cells] = False
        bulk_cells = bulk_cells[bulk_mask]                                      # (Nb0,)

        bulk_set = set(bulk_cells.tolist())
        bad_cells_list = []
        for cell in bulk_cells:
            neigh = c_ncells[cell]
            has_bulk_neighbor = False
            for nb in neigh:
                if nb in bulk_set:
                    has_bulk_neighbor = True
                    break
            if not has_bulk_neighbor:
                bad_cells_list.append(cell)

        if bad_cells_list:
            bad_cells = np.asarray(bad_cells_list, dtype=int)                   # (Nbad,)
            keep_mask = ~np.isin(bulk_cells, bad_cells)
            bulk_cells = bulk_cells[keep_mask]

        self.bulk_cells = bulk_cells

        if bulk_cells.size > 0:
            self.bulk_vertices = np.unique(np.concatenate([np.asarray(c_nverts[cell], dtype=int) for cell in bulk_cells]))
        else:
            self.bulk_vertices = np.array([], dtype=int)

        if self.bulk_vertices.size > 0:
            involved_cells = np.unique(np.concatenate([np.asarray(v_ncells[vert], dtype=int) for vert in self.bulk_vertices]))
        else:
            involved_cells = np.array([], dtype=int)

        ext_cells = involved_cells[~np.isin(involved_cells, self.bulk_cells)]   # (Next_cells,)
        self.ext_cells = ext_cells
        self.involved_cells = np.concatenate((self.bulk_cells, ext_cells))      # (Ninvolved_cells,)

        if self.bulk_vertices.size > 0:
            involved_vertices = np.unique(np.concatenate([np.asarray(v_nverts[vert], dtype=int) for vert in self.bulk_vertices]))
        else:
            involved_vertices = np.array([], dtype=int)

        ext_vertices = involved_vertices[~np.isin(involved_vertices, self.bulk_vertices)]  # (Next_vertices,)
        self.ext_vertices = ext_vertices
        self.involved_vertices = np.concatenate((self.bulk_vertices, ext_vertices))         # (Ninvolved_vertices,)

        n_involved = len(self.involved_cells)
        x0 = np.zeros((n_involved, 3), dtype=float)                                # (Ninvolved_cells, 3)
        if n_involved > 0:
            x0[:, :2] = np.asarray([c_centroids[cell] for cell in self.involved_cells], dtype=float)
        return x0


    def build_diff_operators(self):
        """
        Compute difference operators to enable vectorized operations.
        """
        involved_cells = np.asarray(self.involved_cells, dtype=int)                  # (Nc,)
        involved_vertices = np.asarray(self.involved_vertices, dtype=int)            # (Nv,)
        n_involved_cells = involved_cells.size
        n_involved_vertices = involved_vertices.size

        cells = self.cells
        edges = self.edges

        c_ncells = cells["ncells"].tolist()                                          # list, each ~ (k_i,)
        c_nverts = cells["nverts"].tolist()                                          # list, each ~ (m_i,)

        cell_to_idx = {cell: i for i, cell in enumerate(involved_cells)}
        vert_to_idx = {vert: i for i, vert in enumerate(involved_vertices)}

        edge_pairs = set()
        for ec in edges["cells"].tolist():
            ec = np.asarray(ec, dtype=int)
            if ec.size == 2:
                a, b = int(ec[0]), int(ec[1])
                if a != b:
                    if a < b:
                        edge_pairs.add((a, b))
                    else:
                        edge_pairs.add((b, a))

        cell_pairs_list = []
        for i, cell in enumerate(involved_cells):
            row_pairs = set()
            try:
                for ncell in c_ncells[cell]:
                    j = cell_to_idx.get(int(ncell))
                    if j is None or j <= i:
                        continue

                    a, b = (int(cell), int(ncell)) if cell < ncell else (int(ncell), int(cell))
                    if (a, b) in edge_pairs:
                        row_pairs.add((i, j))
            except Exception:
                print(
                    f"failed to build difference operator at cell {cell+1}. ncells: {ncell+1} "
                    f"(mask labels. cell = {[self.mapping_dict[cell]]}. ncells: {[self.mapping_dict[ncell]]})"
                )
                continue

            # Match the legacy VMSI.py ordering: scan neighbours in ascending
            # local-cell index after deduplication.
            cell_pairs_list.extend(sorted(row_pairs, key=lambda pair: pair[1]))

        valid_cell_pairs = []
        edge_vertex_pairs = []
        for i, j in cell_pairs_list:
            verts_i = c_nverts[involved_cells[i]]
            verts_j = c_nverts[involved_cells[j]]

            # Match VMSI.py sign conventions: np.intersect1d returns sorted
            # vertex indices, and the first shared vertex gets the +1 sign.
            shared = np.intersect1d(np.asarray(verts_i, dtype=int), np.asarray(verts_j, dtype=int))
            if shared.size != 2:
                continue

            v0 = vert_to_idx.get(int(shared[0]))
            v1 = vert_to_idx.get(int(shared[1]))
            if v0 is None or v1 is None:
                continue

            valid_cell_pairs.append((i, j))
            edge_vertex_pairs.append((v0, v1))

        num_edges = len(valid_cell_pairs)
        if num_edges == 0:
            self.involved_vertices = np.array([], dtype=int)
            self.dC = sparse.csr_matrix((0, n_involved_cells), dtype=np.float64)
            self.dV = sparse.csr_matrix((0, 0), dtype=np.float64)
            self.cell_pairs = np.empty((0, 2), dtype=int)
            return

        cell_pairs = np.asarray(valid_cell_pairs, dtype=int)                          # (Ne, 2)
        edge_vertex_pairs = np.asarray(edge_vertex_pairs, dtype=int)                  # (Ne, 2)

        used_vertex_mask = np.zeros(n_involved_vertices, dtype=bool)
        used_vertex_mask[edge_vertex_pairs[:, 0]] = True
        used_vertex_mask[edge_vertex_pairs[:, 1]] = True
        kept_vertices = np.flatnonzero(used_vertex_mask)
        vertex_remap = -np.ones(n_involved_vertices, dtype=int)
        vertex_remap[kept_vertices] = np.arange(kept_vertices.size, dtype=int)
        edge_vertex_pairs = vertex_remap[edge_vertex_pairs]

        rows = np.repeat(np.arange(num_edges, dtype=int), 2)
        data = np.tile(np.array([1.0, -1.0], dtype=np.float64), num_edges)

        self.involved_vertices = involved_vertices[kept_vertices]
        self.dC = sparse.csr_matrix((data, (rows, cell_pairs.ravel())), shape=(num_edges, n_involved_cells), dtype=np.float64)
        self.dV = sparse.csr_matrix((data, (rows, edge_vertex_pairs.ravel())), shape=(num_edges, kept_vertices.size), dtype=np.float64)
        self.cell_pairs = cell_pairs
        return


    @staticmethod
    def _extract_signed_columns(operator):
        operator = operator.tocsr()
        n_rows = operator.shape[0]
        pos_idx = -np.ones(n_rows, dtype=int)
        neg_idx = -np.ones(n_rows, dtype=int)

        for row in range(n_rows):
            start = operator.indptr[row]
            end = operator.indptr[row + 1]
            row_indices = operator.indices[start:end]
            row_data = operator.data[start:end]
            if row_indices.size != 2:
                continue

            pos = row_indices[row_data > 0]
            neg = row_indices[row_data < 0]
            if pos.size == 1 and neg.size == 1:
                pos_idx[row] = int(pos[0])
                neg_idx[row] = int(neg[0])

        return pos_idx, neg_idx


    def estimate_tau(self):
        """
        Estimate tension vector tau, which is tangent to the edge.
        """
        self.build_diff_operators()

        dC = self.dC                                                          # (Ne, Nc)
        dV = self.dV.tocsr()                                                  # (Ne, Nv)
        involved_cells = np.asarray(self.involved_cells, dtype=int)           # (Nc,)
        involved_vertices = np.asarray(self.involved_vertices, dtype=int)     # (Nv,)
        n_diff_edges = dC.shape[0]

        edge_cells_list = self.edges["cells"].tolist()                        # list, each ~ (2,)
        edge_radius_list = self.edges["radius"].tolist()                      # list
        edge_rho_list = self.edges["rho"].tolist()                            # list, each (2,)
        v_coords = np.asarray(self.vertices["coords"][involved_vertices].tolist(), dtype=float)  # (Nv, 2)
        cell_to_local = {cell: i for i, cell in enumerate(involved_cells)}

        pair_to_diff = {}
        e_cells = np.array(self.cell_pairs, copy=True)                        # (Ne, 2)
        for e, (i, j) in enumerate(e_cells):
            pair_to_diff[(int(i), int(j))] = e
            pair_to_diff[(int(j), int(i))] = e

        involved_edges = -np.ones(n_diff_edges, dtype=int)                    # (Ne,)
        for i, ec in enumerate(edge_cells_list):
            ec = np.asarray(ec, dtype=int)
            if ec.size != 2:
                continue
            a = cell_to_local.get(int(ec[0]))
            b = cell_to_local.get(int(ec[1]))
            if a is None or b is None:
                continue
            idx = pair_to_diff.get((a, b))
            if idx is not None:
                involved_edges[idx] = i
        self.involved_edges = involved_edges

        tau_1 = np.zeros((n_diff_edges, 2), dtype=float)                      # (Ne, 2)
        tau_2 = np.zeros((n_diff_edges, 2), dtype=float)                      # (Ne, 2)
        r1 = np.zeros((n_diff_edges, 2), dtype=float)                         # (Ne, 2)
        r2 = np.zeros((n_diff_edges, 2), dtype=float)                         # (Ne, 2)

        e_chord = dV @ v_coords                                               # (Ne, 2)
        pos_idx, neg_idx = self._extract_signed_columns(dV)

        for e in range(n_diff_edges):
            v0 = pos_idx[e]
            v1 = neg_idx[e]
            if v0 < 0 or v1 < 0:
                continue
            r1e = v_coords[v0]                                                # (2,)
            r2e = v_coords[v1]                                                # (2,)
            r1[e, :] = r1e
            r2[e, :] = r2e
            edge_idx = involved_edges[e]

            use_straight_edge = True
            # curved edge
            if edge_idx >= 0 and edge_radius_list[edge_idx] < np.inf:
                rho = np.asarray(edge_rho_list[edge_idx], dtype=float)        # (2,)

                t1 = r1e - rho                                                # (2,)
                t2 = r2e - rho                                                # (2,)
                n1 = np.sqrt(t1[0] * t1[0] + t1[1] * t1[1])
                n2 = np.sqrt(t2[0] * t2[0] + t2[1] * t2[1])
                if np.isfinite(n1) and np.isfinite(n2) and n1 > _MAIN_DP_EPS and n2 > _MAIN_DP_EPS:
                    t1 = t1 / n1
                    t2 = t2 / n2
                    det_t = t1[0] * t2[1] - t1[1] * t2[0]

                    if det_t > 0.0:
                        tau_1[e, 0] = -t1[1]
                        tau_1[e, 1] =  t1[0]
                        tau_2[e, 0] =  t2[1]
                        tau_2[e, 1] = -t2[0]
                    else:
                        # Match VMSI.py exactly: both determinant branches apply
                        # the same 90-degree rotation to the radius vectors.
                        tau_1[e, 0] = -t1[1]
                        tau_1[e, 1] =  t1[0]

                        tau_2[e, 0] =  t2[1]
                        tau_2[e, 1] = -t2[0]
                    use_straight_edge = False

            if use_straight_edge:
                # straight edge
                chord = e_chord[e]                                            # (2,)
                n = np.sqrt(chord[0] * chord[0] + chord[1] * chord[1])
                if not np.isfinite(n) or n <= _MAIN_DP_EPS:
                    continue
                u = chord / n
                tau_1[e, :] = -u
                tau_2[e, :] =  u
        return e_cells, tau_1, tau_2, r1, r2


    def estimate_pressure(self, q, e_cells, tau_1, tau_2, r1, r2):
        """
        Initial estimate of pressure.
        This provides the initial values for the initial minimisation step
        :param q: (numpy array) generating points q as defined under the VMSI formulation
        :param e_cells: (numpy array) indexes of cell pairs at each edge
        :param tau_1: (numpy array) unit tension vectors at vertex 1 for each edge
        :param tau_2: (numpy array) unit tension vectors at vertex 2 for each edge
        :param r1: (numpy array) co-ordinates of vertex 1 for each edge
        :param r2: (numpy array) co-ordinates of vertex 2 for each edge
        """
        q = np.asarray(q, dtype=float)                                    # (Nc, 2)
        e_cells = np.asarray(e_cells, dtype=int)                          # (Ne, 2)
        tau_1 = np.asarray(tau_1, dtype=float)                            # (Ne, 2)
        tau_2 = np.asarray(tau_2, dtype=float)                            # (Ne, 2)
        r1 = np.asarray(r1, dtype=float)                                  # (Ne, 2)
        r2 = np.asarray(r2, dtype=float)                                  # (Ne, 2)
        n_edges = e_cells.shape[0]                                        # ()
        n_cells = q.shape[0]                                              # ()
        c0 = e_cells[:, 0]                                                # (Ne,)
        c1 = e_cells[:, 1]                                                # (Ne,)
        l1_a = np.sum((q[c0] - r1) * tau_1, axis=1)                       # (Ne,)
        l1_b = -np.sum((q[c1] - r1) * tau_1, axis=1)                      # (Ne,)
        l2_a = np.sum((q[c0] - r2) * tau_2, axis=1)                       # (Ne,)
        l2_b = -np.sum((q[c1] - r2) * tau_2, axis=1)                      # (Ne,)

        scale = np.mean(np.linalg.norm(q, axis=1))                        # ()
        b = np.zeros(2 * n_edges + 1, dtype=float)                        # (2*Ne+1,)
        b[-1] = scale

        dense_bytes = ((4 * n_edges * n_cells) + n_cells) * np.dtype(np.float64).itemsize
        if dense_bytes <= 64 * 1024 * 1024:
            L1 = np.zeros((n_edges, n_cells), dtype=float)                # (Ne, Nc)
            L2 = np.zeros((n_edges, n_cells), dtype=float)                # (Ne, Nc)
            rows = np.arange(n_edges)                                     # (Ne,)
            L1[rows, c0] = l1_a
            L1[rows, c1] = l1_b
            L2[rows, c0] = l2_a
            L2[rows, c1] = l2_b
            L = np.vstack((L1, L2, np.full((1, n_cells), 1.0 / n_cells, dtype=float)))  # (2*Ne+1, Nc)
            p = np.linalg.lstsq(L, b, rcond=None)[0]                      # (Nc,)
        else:
            rows = np.concatenate((
                np.arange(n_edges, dtype=int),
                np.arange(n_edges, dtype=int),
                n_edges + np.arange(n_edges, dtype=int),
                n_edges + np.arange(n_edges, dtype=int),
                np.full(n_cells, 2 * n_edges, dtype=int),
            ))
            cols = np.concatenate((c0, c1, c0, c1, np.arange(n_cells, dtype=int)))
            data = np.concatenate((l1_a, l1_b, l2_a, l2_b, np.full(n_cells, 1.0 / n_cells, dtype=float)))
            L = sparse.coo_matrix((data, (rows, cols)), shape=(2 * n_edges + 1, n_cells), dtype=np.float64).tocsr()
            p = _regularized_sparse_least_squares(L, b)                   # (Nc,)

        p = np.nan_to_num(p, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
        mean_p = float(np.mean(p))
        if not np.isfinite(mean_p) or abs(mean_p) <= _MAIN_DP_EPS:
            mean_p = 1.0
        p = p / mean_p                                                    # (Nc,)
        return p


    def generate_circular_arcs(self):
        """
            Construct circular arc for each edge and use these instead of raw segmented edges for minimization
        """
        def edge_pixel_count(edge_pixels):
            if isinstance(edge_pixels, tuple):
                return len(edge_pixels[0])
            pixels = np.asarray(edge_pixels)
            if pixels.ndim == 2 and pixels.shape[0] == 2:
                return pixels.shape[1]
            return len(edge_pixels)

        involved_edges = np.asarray(self.involved_edges, dtype=int)
        e_pixels = self.edges["pixels"].tolist()
        e_verts = self.edges["verts"].tolist()
        e_radius = self.edges["radius"].tolist()
        e_rho = self.edges["rho"].tolist()
        v_coords = self.vertices["coords"].tolist()

        #self.avg_edge_length = int(np.median([edge_pixel_count(e_pixels[e]) for e in involved_edges]))
        self.avg_edge_length = 16
        K = self.avg_edge_length
        s = np.linspace(0.0, 1.0, K)

        self.edgearc_x = np.zeros((len(involved_edges), K), dtype=float)
        self.edgearc_y = np.zeros((len(involved_edges), K), dtype=float)

        for i, e in enumerate(involved_edges):
            v0, v1 = np.asarray(e_verts[e], dtype=int)
            r0 = np.asarray(v_coords[v0], dtype=float)
            r1 = np.asarray(v_coords[v1], dtype=float)

            if e_radius[e] < np.inf:
                rho = np.asarray(e_rho[e], dtype=float)
                a = r0 - rho
                b = r1 - rho

                denom = np.linalg.norm(a) * np.linalg.norm(b)
                if np.isfinite(denom) and denom > _MAIN_DP_EPS:
                    c = np.dot(a, b) / denom
                    theta = np.arccos(np.clip(c, -1.0, 1.0))

                    if a[0] * b[1] - a[1] * b[0] < 0.0:
                        a, b = b, a

                    th = theta * s
                    cs = np.cos(th)
                    sn = np.sin(th)

                    self.edgearc_x[i] = rho[0] + a[0] * cs - a[1] * sn
                    self.edgearc_y[i] = rho[1] + a[0] * sn + a[1] * cs
                else:
                    d = r1 - r0
                    self.edgearc_x[i] = r0[0] + s * d[0]
                    self.edgearc_y[i] = r0[1] + s * d[1]
            else:
                d = r1 - r0
                self.edgearc_x[i] = r0[0] + s * d[0]
                self.edgearc_y[i] = r0[1] + s * d[1]


    def estimate_theta(self, x):
        """
        Initialize theta, defined as p_a * z^2_a for each cell a

        :param x: (numpy array) with dimensions [num_cells x 3] containing
                q in x[:, 0:2] and p in x[:, 2]
        :return: (numpy array) with dimensions [num_cells] containing
                initialized values of theta
        """
        x = np.asarray(x, dtype=float)                                         # (Nc, 3)
        q = x[:, :2]                                                           # (Nc, 2)
        p = x[:, 2]                                                            # (Nc,)

        dC = self.dC                                                           # (Ne, Nc)
        cell_pairs = self.cell_pairs                                           # (Ne, 2)
        involved_edges = np.asarray(self.involved_edges, dtype=int)            # (Ne,)
        c0 = cell_pairs[:, 0]                                                  # (Ne,)
        c1 = cell_pairs[:, 1]                                                  # (Ne,)
        dq = dC @ q                                                            # (Ne, 2)
        q_sq = np.sum(dq * dq, axis=1)                                         # (Ne,)
        pq = p[:, None] * q                                                    # (Nc, 2)
        dP = dC @ p                                                            # (Ne,)
        valid = np.isfinite(dP) & (np.abs(dP) > _MAIN_DP_EPS) & np.isfinite(q_sq)
        rho = np.zeros((dP.size, 2), dtype=float)                               # (Ne, 2)
        if np.any(valid):
            rho[valid] = (dC[valid] @ pq) / dP[valid, None]
        e_verts = self.edges["verts"].tolist()                                 # list, each (2,)
        v_coords = self.vertices["coords"].tolist()                            # list, each (2,)
        r = np.zeros(len(involved_edges), dtype=float)                         # (Ne,)

        for i, edge in enumerate(involved_edges):
            if not valid[i]:
                continue
            verts = np.asarray(e_verts[edge], dtype=int)                       # (2,)
            r1 = np.asarray(v_coords[verts[0]], dtype=float)                   # (2,)
            r2 = np.asarray(v_coords[verts[1]], dtype=float)                   # (2,)
            d1 = r1 - rho[i]                                                   # (2,)
            d2 = r2 - rho[i]                                                   # (2,)
            # mean([||d1||^2, ||d2||^2])
            r[i] = 0.5 * ((d1[0] * d1[0] + d1[1] * d1[1]) +
                        (d2[0] * d2[0] + d2[1] * d2[1]))                     # ()
            if not np.isfinite(r[i]):
                valid[i] = False

        p1 = p[c0]                                                             # (Ne,)
        p2 = p[c1]                                                             # (Ne,)
        r_flat = p1 * p2 * q_sq                                                # (Ne,)
        r = r * (dP * dP)                                                      # (Ne,)
        valid = valid & np.isfinite(r_flat) & np.isfinite(r)
        if not np.any(valid):
            return np.zeros(q.shape[0], dtype=float)
        A = sparse.diags(dP[valid]) @ dC[valid]                                # (Nvalid, Nc)
        b = r_flat[valid] - r[valid]                                           # (Nvalid,)

        mean_row = sparse.csr_matrix(np.full((1, A.shape[1]), 1.0, dtype=float))
        A_aug = sparse.vstack((A, mean_row), format="csr")                     # (Ne+1, Nc)
        b_aug = np.concatenate((b, np.array([0.0], dtype=float)))              # (Ne+1,)
        theta = _regularized_sparse_least_squares(A_aug, b_aug)                # (Nc,)
        return theta


    def initial_minimization(self):
        """
        Initial minimisation step to ensure that vector t (between q_a and r_i
        for vertex i at cell a) is orthogonal to tension vector tau.

        This prevents the main minimisation step converging on the trivial
        solution q_a=q_b, p_a=p_b
        """
        x0 = self.classify_cells()                                                  # (Nc, 3)
        e_cells, tau_1, tau_2, r1, r2 = self.estimate_tau()                        # (Ne, 2)
        x0[:, 2] = self.estimate_pressure(x0[:, :2], e_cells, tau_1, tau_2, r1, r2)

        q0, p0 = x0[:, :2], x0[:, 2]                                               # (Nc, 2), (Nc,)
        dC, cell_pairs = self.dC, self.cell_pairs                                  # (Ne, Nc), (Ne, 2)
        c0, c1 = cell_pairs[:, 0], cell_pairs[:, 1]                                # (Ne,), (Ne,)
        n_edges, n_cells, n_vars = dC.shape[0], dC.shape[1], x0.size

        rows6 = np.concatenate([c0, c0 + n_cells, c0 + 2 * n_cells,
                                c1, c1 + n_cells, c1 + 2 * n_cells])               # (6*Ne,)
        rows2 = np.concatenate([c0, c1])                                           # (2*Ne,)

        qp0 = q0 * p0[:, None]                                                     # (Nc, 2)
        b0 = dC @ qp0                                                               # (Ne, 2)
        delta_p0 = dC @ p0                                                          # (Ne,)
        t1_0 = b0 - r1 * delta_p0[:, None]                                          # (Ne, 2)
        t2_0 = b0 - r2 * delta_p0[:, None]                                          # (Ne, 2)

        if self.optimiser == "nlopt":
            scale = 0.5 * (np.mean(np.linalg.norm(t1_0, axis=1)) +
                        np.mean(np.linalg.norm(t2_0, axis=1)))

            last_x = {"value": np.array(x0.ravel(order="F"), copy=True)}
            best_x_state = {"value": None, "score": np.inf, "E": np.inf, "valid": False}
            last_theta = {"value": None}
            best_theta_state = {"value": None, "score": np.inf, "E": np.inf, "valid": False}

            Aeq_p = np.concatenate((np.zeros(2 * n_cells), np.ones(n_cells)))      # (3*Nc,)
            p0_sum = np.mean(p0) * n_cells

            r1x, r1y = r1[:, 0], r1[:, 1]                                           # (Ne,), (Ne,)
            r2x, r2y = r2[:, 0], r2[:, 1]                                           # (Ne,), (Ne,)

            def _reshape_x(x):
                return np.asarray(x).reshape(x0.shape, order="F")                   # (Nc, 3)

            def _state_from_x(x):
                xM = _reshape_x(x)                                                  # (Nc, 3)
                q, p = xM[:, :2], xM[:, 2]                                          # (Nc, 2), (Nc,)
                qp = q * p[:, None]                                                 # (Nc, 2)
                b = dC @ qp                                                         # (Ne, 2)
                delta_p = dC @ p                                                    # (Ne,)
                v1 = b - r1 * delta_p[:, None]                                      # (Ne, 2)
                v2 = b - r2 * delta_p[:, None]                                      # (Ne, 2)
                return q, p, b, delta_p, v1, v2

            def energy(x, grad=np.array([])):
                last_x["value"] = np.array(x, copy=True)
                q, p, _, _, v1, v2 = _state_from_x(x)
                l1, l2, t1, t2, ip1, ip2, E = tension_projections_and_energy(v1, v2, tau_1, tau_2)

                scale_violation = abs(0.5 * (np.mean(l1) + np.mean(l2)) - scale)
                pressure_violation = abs(np.dot(Aeq_p, x) - p0_sum) / max(n_cells, 1)
                score = E + _STATE_SCORE_PENALTY * (scale_violation * scale_violation + pressure_violation * pressure_violation)
                if (
                    np.isfinite(score)
                    and np.isfinite(x).all()
                    and np.isfinite(p).all()
                    and np.all(p > 0.0)
                    and score < best_x_state["score"]
                ):
                    best_x_state["value"] = np.array(x, copy=True)
                    best_x_state["score"] = float(score)
                    best_x_state["E"] = float(E)
                    best_x_state["valid"] = True

                if grad.size > 0:
                    pc0, pc1 = p[c0], p[c1]
                    q0x, q0y = q[c0, 0], q[c0, 1]
                    q1x, q1y = q[c1, 0], q[c1, 1]

                    drX1 = sx_grad(pc0, pc1, q0x, q0y, q1x, q1y, r1x, r1y)          # (Ne, 6)
                    drX2 = sx_grad(pc0, pc1, q0x, q0y, q1x, q1y, r2x, r2y)
                    drY1 = sy_grad(pc0, pc1, q0x, q0y, q1x, q1y, r1x, r1y)
                    drY2 = sy_grad(pc0, pc1, q0x, q0y, q1x, q1y, r2x, r2y)

                    dE = ((ip1 * tau_1[:, 0])[:, None] * drX1 +
                        (ip1 * tau_1[:, 1])[:, None] * drY1 +
                        (ip2 * tau_2[:, 0])[:, None] * drX2 +
                        (ip2 * tau_2[:, 1])[:, None] * drY2) / n_edges            # (Ne, 6)

                    grad[:] = np.bincount(rows6, weights=dE.ravel(order="F"), minlength=n_vars)

                if self.verbose:
                    print(f'  Energy: {E:.6f}', end='\r', flush=True)
                return float(E)

            def nonlinear_con(x, grad=np.array([])):
                q, p, _, _, v1, v2 = _state_from_x(x)
                l1, l2, t1, t2, _, _, _ = tension_projections_and_energy(v1, v2, tau_1, tau_2)
                E = 0.5 * (np.mean(l1) + np.mean(l2)) - scale

                if grad.size > 0:
                    pc0, pc1 = p[c0], p[c1]
                    q0x, q0y = q[c0, 0], q[c0, 1]
                    q1x, q1y = q[c1, 0], q[c1, 1]

                    drX1 = rx_grad(pc0, pc1, q0x, q1x, r1x)                          # (Ne, 6)
                    drX2 = rx_grad(pc0, pc1, q0x, q1x, r2x)
                    drY1 = ry_grad(pc0, pc1, q0y, q1y, r1y)
                    drY2 = ry_grad(pc0, pc1, q0y, q1y, r2y)

                    dE = 0.5 * (t1[:, 0][:, None] * drX1 +
                                t1[:, 1][:, None] * drY1 +
                                t2[:, 0][:, None] * drX2 +
                                t2[:, 1][:, None] * drY2) / n_edges                 # (Ne, 6)

                    grad[:] = np.bincount(rows6, weights=dE.ravel(order="F"), minlength=n_vars)

                return float(E)

            def linear_con(x, grad=np.array([])):
                E = np.dot(Aeq_p, x) - p0_sum
                if grad.size > 0:
                    grad[:] = Aeq_p
                return float(E)

            local_opt = nlopt.opt(nlopt.LD_LBFGS, n_vars)
            init_opt = nlopt.opt(nlopt.AUGLAG, n_vars)
            init_opt.set_local_optimizer(local_opt)
            init_opt.set_min_objective(energy)

            lb = np.concatenate((-np.inf * np.ones(n_cells),
                                -np.inf * np.ones(n_cells),
                                0.001 * np.ones(n_cells)))                         # (3*Nc,)
            ub = np.concatenate(( np.inf * np.ones(n_cells),
                                np.inf * np.ones(n_cells),
                            1000.0 * np.ones(n_cells)))                           # (3*Nc,)

            init_opt.set_lower_bounds(lb)
            init_opt.set_upper_bounds(ub)
            # NLopt is more stable with the legacy scale constraint as a lower
            # bound.
            init_opt.add_inequality_constraint(nonlinear_con, 1e-6)
            init_opt.add_equality_constraint(linear_con, 1e-6)
            init_opt.set_maxeval(2000)
            init_exc = None
            try:
                optimized_x = init_opt.optimize(np.clip(x0.ravel(order="F"), lb, ub))
                last_x["value"] = np.asarray(optimized_x, dtype=np.float64)
            except Exception as exc:
                init_exc = exc
            print('')

            init_result = init_opt.last_optimize_result()
            self.optimisation_status["init_result"] = _nlopt_result_name(init_result)

            if init_exc is None and _nlopt_clean_result(init_result) and np.isfinite(last_x["value"]).all():
                x_source = last_x["value"]
            elif best_x_state["valid"]:
                issue = str(init_exc) if init_exc is not None else _nlopt_result_name(init_result)
                warnings.warn(
                    f"Initial minimization did not converge cleanly ({issue}); using the best finite iterate found.",
                    RuntimeWarning,
                )
                x_source = best_x_state["value"]
                self.optimisation_status["init_used_best_iterate"] = True
            else:
                issue = str(init_exc) if init_exc is not None else _nlopt_result_name(init_result)
                raise RuntimeError(f"Initial minimization failed before finding a finite iterate ({issue}).")

            x = x_source.reshape(x0.shape, order="F")                                # (Nc, 3)
            q, p = x[:, :2], x[:, 2]                                                 # (Nc, 2), (Nc,)

            self.generate_circular_arcs()
            theta0 = self.estimate_theta(x)                                          # (Nc,)
            theta0 = np.asarray(theta0, dtype=np.float64)
            theta0 = np.nan_to_num(theta0, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            edgearc_x, edgearc_y = self.edgearc_x, self.edgearc_y                    # (Ne, K), (Ne, K)

            # Keep the default theta path close to the old fast formulation, and
            # fall back to the stricter safe path only if the original vectorized
            # state produces non-finite values.
            pc0, pc1 = p[c0], p[c1]
            dP = pc0 - pc1
            dQ = q[c0, :] - q[c1, :]
            QL = np.sum(dQ * dQ, axis=1)
            pq = p[:, None] * q
            _, _, _, _, _, _, rho_safe, _, _, theta_invalid_base = main_state_core_safe(
                q, p, np.zeros_like(p), c0, c1, _MAIN_DP_EPS
            )
            dP_valid = np.isfinite(dP) & (np.abs(dP) > _MAIN_DP_EPS)
            rho_fast = np.array(rho_safe, copy=True)
            if np.any(dP_valid):
                rho_fast[dP_valid] = (dC[dP_valid] @ pq) / dP[dP_valid, None]
            theta_fast_available = (
                bool(np.all(dP_valid))
                and np.isfinite(rho_fast).all()
                and not bool(np.any(theta_invalid_base))
            )
            theta_rows = np.arange(n_edges, dtype=int)                               # (Ne,)
            b_theta = pc0 * pc1 * QL                                                 # (Ne,)

            q0x, q0y = q[c0, 0], q[c0, 1]
            q1x, q1y = q[c1, 0], q[c1, 1]

            def record_theta_state(theta, E, invalid_count=0):
                theta = np.asarray(theta, dtype=np.float64)
                if not np.isfinite(theta).all() or not np.isfinite(E):
                    return

                ineq = dP * (theta[c0] - theta[c1]) - b_theta
                max_ineq = _positive_part(np.nanmax(ineq) if ineq.size else 0.0)
                eq_violation = abs(np.sum(theta)) / max(theta.size, 1)
                score = (
                    float(E)
                    + _STATE_SCORE_PENALTY * (max_ineq * max_ineq + eq_violation * eq_violation)
                    + _MAIN_INVALID_PENALTY * int(invalid_count)
                )
                if np.isfinite(score) and score < best_theta_state["score"]:
                    best_theta_state["value"] = np.array(theta, copy=True)
                    best_theta_state["score"] = float(score)
                    best_theta_state["E"] = float(E)
                    best_theta_state["valid"] = True
                    best_theta_state["max_constraint"] = float(max_ineq)
                    best_theta_state["mean_sum"] = float(eq_violation)
                    best_theta_state["invalid_count"] = int(invalid_count)

            def theta_energy(theta, grad=np.array([])):
                last_theta["value"] = np.array(theta, copy=True)

                dT = theta[c0] - theta[c1]                                           # (Ne,)
                r_sq = (b_theta - dP * dT) / (dP * dP)
                ind_z = r_sq < 0
                r_sq = np.maximum(r_sq, 0.0)
                r = np.sqrt(r_sq)
                avg_d, E = theta_energy_core(rho_fast, edgearc_x, edgearc_y, r)
                record_theta_state(theta, E, int(np.count_nonzero(ind_z)))

                if grad.size > 0:
                    dR = radius_grad_theta(pc0, pc1, q0x, q0y, q1x, q1y,
                                        theta[c0], theta[c1])                     # (Ne, 2)
                    dR[ind_z, :] = 0.0
                    dE = -(avg_d[:, None] * dR) / n_edges                            # (Ne, 2)
                    grad[:] = np.bincount(rows2, weights=dE.ravel(order="F"), minlength=theta0.size)

                if self.verbose:
                    print(f'  Theta energy: {E:.4f}', end='\r', flush=True)
                return float(E)

            def theta_neqlincon(result, theta, grad=np.array([])):
                result[:] = dP * (theta[c0] - theta[c1]) - b_theta
                if grad.size > 0:
                    grad.fill(0.0)
                    grad[theta_rows, c0] = dP
                    grad[theta_rows, c1] = -dP

            def safe_theta_energy(theta, grad=np.array([])):
                last_theta["value"] = np.array(theta, copy=True)

                dT = theta[c0] - theta[c1]                                           # (Ne,)
                r, ind_z, invalid = theta_radius_core_safe(dP, dT, b_theta, theta_invalid_base)
                avg_d, E, invalid_count = theta_energy_core_safe(rho_safe, edgearc_x, edgearc_y, r, invalid)

                if not np.isfinite(E):
                    invalid_count = n_edges
                    E = float(_MAIN_INVALID_PENALTY)

                safe_E = float(E)

                record_theta_state(theta, safe_E, invalid_count)

                if grad.size > 0:
                    dR = radius_grad_theta(pc0, pc1, q0x, q0y, q1x, q1y,
                                        theta[c0], theta[c1])                     # (Ne, 2)
                    invalid_rows = invalid.copy()
                    invalid_rows[ind_z] = True
                    dR[invalid_rows, :] = 0.0
                    dE = -(avg_d[:, None] * dR) / n_edges                            # (Ne, 2)
                    dE = np.nan_to_num(dE, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    grad[:] = np.bincount(rows2, weights=dE.ravel(order="F"), minlength=theta0.size)

                if self.verbose:
                    if invalid_count > 0:
                        print(f'  Theta energy: {safe_E:.4f} ({invalid_count} invalid edges)', end='\r', flush=True)
                    else:
                        print(f'  Theta energy: {safe_E:.4f}', end='\r', flush=True)
                return safe_E

            def safe_theta_neqlincon(result, theta, grad=np.array([])):
                if not np.isfinite(theta).all():
                    result[:] = _MAIN_INVALID_PENALTY
                    if grad.size > 0:
                        grad.fill(0.0)
                    return

                result[:] = dP * (theta[c0] - theta[c1]) - b_theta
                if not np.isfinite(result).all():
                    result[:] = _MAIN_INVALID_PENALTY
                if grad.size > 0:
                    grad.fill(0.0)
                    grad[theta_rows, c0] = dP
                    grad[theta_rows, c1] = -dP

            if theta_fast_available and theta_energy(np.zeros_like(theta0)) < theta_energy(theta0):
                theta0 = np.zeros_like(theta0)

            theta_exc = None
            if theta_fast_available:
                theta_local_opt = nlopt.opt(nlopt.LD_LBFGS, theta0.size)
                theta_opt = nlopt.opt(nlopt.AUGLAG, theta0.size)
                theta_opt.set_local_optimizer(theta_local_opt)
                theta_opt.set_ftol_abs(1e-5)
                theta_opt.set_min_objective(theta_energy)
                theta_opt.add_inequality_mconstraint(theta_neqlincon, 1e-5 * np.ones(n_edges))
                # Keep the theta gauge free in NLopt. The main minimization fixes
                # the mean theta and pressure scale from this initialization.
                theta_opt.set_maxeval(2000)
                try:
                    optimized_theta = theta_opt.optimize(theta0)
                    last_theta["value"] = np.array(optimized_theta, copy=True)
                except Exception as exc:
                    theta_exc = exc
                theta_result = theta_opt.last_optimize_result()
            else:
                theta_exc = RuntimeError("Skipping fast theta optimization because near-zero pressure differences would create non-finite radii.")
                theta_result = nlopt.FORCED_STOP
            print('')
            self.optimisation_status["theta_fast_result"] = _nlopt_result_name(theta_result)
            theta_fast_ok = (
                theta_exc is None
                and _nlopt_clean_result(theta_result)
                and last_theta["value"] is not None
                and np.isfinite(last_theta["value"]).all()
            )

            if theta_fast_ok:
                theta = np.array(last_theta["value"], copy=True)
            else:
                issue = str(theta_exc) if theta_exc is not None else _nlopt_result_name(theta_result)
                warnings.warn(
                    f"Theta optimization did not converge cleanly ({issue}); retrying with the safer objective.",
                    RuntimeWarning,
                )

                theta_start = np.array(last_theta["value"] if last_theta["value"] is not None else theta0, copy=True)
                theta_start = np.nan_to_num(theta_start, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                best_theta_state["value"] = None
                best_theta_state["score"] = np.inf
                best_theta_state["E"] = np.inf
                best_theta_state["valid"] = False

                theta_local_opt = nlopt.opt(nlopt.LD_LBFGS, theta0.size)
                theta_opt = nlopt.opt(nlopt.AUGLAG, theta0.size)
                theta_opt.set_local_optimizer(theta_local_opt)
                theta_opt.set_ftol_abs(1e-5)
                theta_opt.set_min_objective(safe_theta_energy)
                theta_opt.add_inequality_mconstraint(safe_theta_neqlincon, 1e-5 * np.ones(n_edges))
                # Keep the theta gauge free in NLopt, as in the fast path.
                theta_opt.set_maxeval(2000)
                safe_theta_energy(theta_start, np.array([]))

                theta_exc = None
                try:
                    optimized_theta = theta_opt.optimize(theta_start)
                    last_theta["value"] = np.array(optimized_theta, copy=True)
                except Exception as exc:
                    theta_exc = exc
                print('')

                theta_result = theta_opt.last_optimize_result()
                self.optimisation_status["theta_safe_result"] = _nlopt_result_name(theta_result)
                theta_safe_ok = (
                    theta_exc is None
                    and _nlopt_clean_result(theta_result)
                    and last_theta["value"] is not None
                    and np.isfinite(last_theta["value"]).all()
                )
                if not theta_safe_ok:
                    issue = str(theta_exc) if theta_exc is not None else _nlopt_result_name(theta_result)
                    warnings.warn(
                        f"Theta optimization did not converge cleanly ({issue}); using the best finite iterate found.",
                        RuntimeWarning,
                    )

                if theta_safe_ok:
                    theta = np.array(last_theta["value"], copy=True)
                elif best_theta_state["valid"]:
                    theta = np.array(best_theta_state["value"], copy=True)
                    self.optimisation_status["theta_used_best_iterate"] = True
                else:
                    raise RuntimeError("Theta optimization failed before finding a finite iterate.")

        return q, p, theta

    
    def fit(self, seg_mask, labelled_mask):
        """
        Perform the minimization of equation 5 with respect to the variables (q, z, p)
        """
        print("Fitting data...")
        self.optimisation_status = {}
        self.mapping_dict = self.map_index(seg_mask, labelled_mask)

        print("  Preparing data...")
        self.prepare_data()
        self._repair_multicell_edges(labelled_mask)

        print("  Initial minimization...")
        q0, p0, theta0 = self.initial_minimization()

        X0 = np.ascontiguousarray(np.column_stack((q0, theta0.squeeze(), p0.squeeze())), dtype=np.float64)  # (Nc, 4)
        print("  Initial minimization complete\n")
        
        print("Main minimization...")
        if self.optimiser == "nlopt":
            dC = self.dC                                                                # (Ne, Nc)
            cell_pairs = np.ascontiguousarray(self.cell_pairs, dtype=np.int64)         # (Ne, 2)
            edgearc_x = np.ascontiguousarray(self.edgearc_x, dtype=np.float64)         # (Ne, K)
            edgearc_y = np.ascontiguousarray(self.edgearc_y, dtype=np.float64)         # (Ne, K)

            c0, c1 = cell_pairs[:, 0], cell_pairs[:, 1]                                # (Ne,), (Ne,)
            n_edges, n_cells = dC.shape
            n_vars = X0.size

            rows8 = np.concatenate([c0,c0 + n_cells,c0 + 2 * n_cells,c0 + 3 * n_cells,c1,c1 + n_cells,c1 + 2 * n_cells,c1 + 3 * n_cells]) # (8*Ne,)
            rows_edges = np.arange(n_edges, dtype=int)                                  # (Ne,)

            Aeq = np.vstack((
                np.concatenate((np.zeros(3 * n_cells), np.ones(n_cells) / n_cells)),
                np.concatenate((np.zeros(2 * n_cells), np.ones(n_cells) / n_cells, np.zeros(n_cells))),
            ))                                                                         # (2, 4*Nc)
            beq = np.array([np.mean(X0[:, 3]), np.mean(X0[:, 2])], dtype=np.float64)  # (2,)

            lb = np.concatenate((-np.inf * np.ones(3 * n_cells),0.001 * np.ones(n_cells))) # (4*Nc,)
            ub = np.concatenate((np.inf * np.ones(3 * n_cells),1000.0 * np.ones(n_cells))) # (4*Nc,)

            last_X = np.ascontiguousarray(X0.ravel(order="F").copy())                  # (4*Nc,)
            best_main_state = {"X": last_X.copy(), "score": np.inf, "E": np.inf, "valid": False}

            def _reshape_X(X):
                return np.asarray(X, dtype=np.float64).reshape(X0.shape, order="F")    # (Nc, 4)

            def record_main_state(X, E, invalid_count=0):
                X = np.asarray(X, dtype=np.float64)
                if not np.isfinite(X).all() or not np.isfinite(E):
                    return

                XM = _reshape_X(X)
                q = XM[:, :2]
                theta = XM[:, 2]
                p = XM[:, 3]
                if not np.isfinite(p).all() or np.any(p <= 0.0):
                    return

                pc0 = p[c0]
                pc1 = p[c1]
                dP = pc0 - pc1
                dT = theta[c0] - theta[c1]
                dQ = q[c0, :] - q[c1, :]
                QL = np.sum(dQ * dQ, axis=1)
                ineq = nonlinear_constraint_core(pc0, pc1, dP, dT, QL)
                max_ineq = _positive_part(np.nanmax(ineq) if ineq.size else 0.0)
                eq_violation = float(np.nanmax(np.abs(Aeq @ X - beq)))
                if not np.isfinite(eq_violation):
                    return

                score = (
                    float(E)
                    + _STATE_SCORE_PENALTY * (max_ineq * max_ineq + eq_violation * eq_violation)
                    + _MAIN_INVALID_PENALTY * int(invalid_count)
                )
                if np.isfinite(score) and score < best_main_state["score"]:
                    best_main_state["X"] = np.array(X, copy=True)
                    best_main_state["score"] = float(score)
                    best_main_state["E"] = float(E)
                    best_main_state["valid"] = True
                    best_main_state["max_constraint"] = float(max_ineq)
                    best_main_state["max_eq"] = eq_violation
                    best_main_state["invalid_count"] = int(invalid_count)

            def objective(X, grad=np.array([])):
                last_X[:] = X
                XM = _reshape_X(X)
                q = XM[:, :2]                                                           # (Nc, 2)
                theta = XM[:, 2]                                                        # (Nc,)
                p = XM[:, 3]                                                            # (Nc,)

                pc0, pc1, dP, dT, dQ, QL, rho, r, ind_z = main_state_core(
                    q, p, theta, c0, c1
                )
                avg_d, dNormX, dNormY, E = main_objective_core(
                    rho, edgearc_x, edgearc_y, r
                )
                record_main_state(X, E, int(np.count_nonzero(ind_z)))

                if grad.size > 0:
                    q0x, q0y = q[c0, 0], q[c0, 1]
                    q1x, q1y = q[c1, 0], q[c1, 1]
                    th0, th1 = theta[c0], theta[c1]

                    dRhoX = rho_x_grad(pc0, pc1, q0x, q1x)                              # (Ne, 8)
                    dRhoY = rho_y_grad(pc0, pc1, q0y, q1y)                              # (Ne, 8)
                    dR = radius_grad(pc0, pc1, q0x, q0y, q1x, q1y, th0, th1)           # (Ne, 8)
                    dR[ind_z, :] = 0.0

                    dE = (dNormX[:, None] * dRhoX + dNormY[:, None] * dRhoY - avg_d[:, None] * dR) / n_edges # (Ne, 8)
                    grad[:] = np.bincount(rows8, weights=dE.ravel(order="F"), minlength=n_vars)

                if self.verbose:
                    print(f"Main loss: {E:.6f}", end="\r", flush=True)
                return float(E)

            def nonlinear_con(result, X, grad=np.array([])):
                XM = _reshape_X(X)
                q = XM[:, :2]
                theta = XM[:, 2]
                p = XM[:, 3]

                pc0 = p[c0]
                pc1 = p[c1]
                dP = pc0 - pc1
                dT = theta[c0] - theta[c1]
                dQ = q[c0, :] - q[c1, :]
                QL = np.sum(dQ * dQ, axis=1)
                result[:] = nonlinear_constraint_core(pc0, pc1, dP, dT, QL)

                if grad.size > 0:
                    coeff_xy = -2.0 * pc0 * pc1
                    coeff_p = QL * pc0 * pc1

                    grad.fill(0.0)
                    grad[rows_edges, c0] = coeff_xy * dQ[:, 0]
                    grad[rows_edges, c1] = -coeff_xy * dQ[:, 0]
                    grad[rows_edges, n_cells + c0] = coeff_xy * dQ[:, 1]
                    grad[rows_edges, n_cells + c1] = -coeff_xy * dQ[:, 1]
                    grad[rows_edges, 2 * n_cells + c0] = dP
                    grad[rows_edges, 2 * n_cells + c1] = -dP
                    grad[rows_edges, 3 * n_cells + c0] = dT - (coeff_p / p[c0])
                    grad[rows_edges, 3 * n_cells + c1] = -dT - (coeff_p / p[c1])

            def linear_con(result, X, grad=np.array([])):
                result[:] = Aeq @ X - beq
                if grad.size > 0:
                    grad[:] = Aeq

            def safe_objective(X, grad=np.array([])):
                last_X[:] = X
                XM = _reshape_X(X)
                q = XM[:, :2]                                                           # (Nc, 2)
                theta = XM[:, 2]                                                        # (Nc,)
                p = XM[:, 3]                                                            # (Nc,)

                pc0, pc1, dP, dT, dQ, QL, rho, r, ind_z, invalid = main_state_core_safe(
                    q, p, theta, c0, c1, _MAIN_DP_EPS
                )
                avg_d, dNormX, dNormY, E, invalid_count = main_objective_core_safe(
                    rho, edgearc_x, edgearc_y, r, invalid
                )

                if not np.isfinite(E):
                    invalid_count = n_edges
                    E = float(_MAIN_INVALID_PENALTY)

                safe_E = float(E)
                record_main_state(X, safe_E, invalid_count)

                if grad.size > 0:
                    q0x, q0y = q[c0, 0], q[c0, 1]
                    q1x, q1y = q[c1, 0], q[c1, 1]
                    th0, th1 = theta[c0], theta[c1]

                    dRhoX = rho_x_grad(pc0, pc1, q0x, q1x)                              # (Ne, 8)
                    dRhoY = rho_y_grad(pc0, pc1, q0y, q1y)                              # (Ne, 8)
                    dR = radius_grad(pc0, pc1, q0x, q0y, q1x, q1y, th0, th1)           # (Ne, 8)
                    invalid_rows = invalid.copy()
                    invalid_rows[ind_z] = True

                    dRhoX[invalid_rows, :] = 0.0
                    dRhoY[invalid_rows, :] = 0.0
                    dR[invalid_rows, :] = 0.0

                    dE = (dNormX[:, None] * dRhoX + dNormY[:, None] * dRhoY - avg_d[:, None] * dR) / n_edges # (Ne, 8)
                    dE = np.nan_to_num(dE, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    grad[:] = np.bincount(rows8, weights=dE.ravel(order="F"), minlength=n_vars)

                if self.verbose:
                    if invalid_count > 0:
                        print(f"Main loss: {safe_E:.6f} ({invalid_count} invalid edges)", end="\r", flush=True)
                    else:
                        print(f"Main loss: {safe_E:.6f}", end="\r", flush=True)
                return safe_E

            def safe_nonlinear_con(result, X, grad=np.array([])):
                if not np.isfinite(X).all():
                    result[:] = _MAIN_INVALID_PENALTY
                    if grad.size > 0:
                        grad.fill(0.0)
                    return

                XM = _reshape_X(X)
                q = XM[:, :2]
                theta = XM[:, 2]
                p = XM[:, 3]

                pc0 = p[c0]
                pc1 = p[c1]
                dP = pc0 - pc1
                dT = theta[c0] - theta[c1]
                dQ = q[c0, :] - q[c1, :]
                QL = np.sum(dQ * dQ, axis=1)
                result[:] = nonlinear_constraint_core(pc0, pc1, dP, dT, QL)

                if not np.isfinite(result).all():
                    result[:] = _MAIN_INVALID_PENALTY

                if grad.size > 0:
                    coeff_xy = -2.0 * pc0 * pc1
                    coeff_p = QL * pc0 * pc1

                    grad.fill(0.0)
                    grad[rows_edges, c0] = coeff_xy * dQ[:, 0]
                    grad[rows_edges, c1] = -coeff_xy * dQ[:, 0]
                    grad[rows_edges, n_cells + c0] = coeff_xy * dQ[:, 1]
                    grad[rows_edges, n_cells + c1] = -coeff_xy * dQ[:, 1]
                    grad[rows_edges, 2 * n_cells + c0] = dP
                    grad[rows_edges, 2 * n_cells + c1] = -dP
                    grad[rows_edges, 3 * n_cells + c0] = dT - (coeff_p / p[c0])
                    grad[rows_edges, 3 * n_cells + c1] = -dT - (coeff_p / p[c1])
                    np.nan_to_num(grad, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            def safe_linear_con(result, X, grad=np.array([])):
                if not np.isfinite(X).all():
                    result[:] = _MAIN_INVALID_PENALTY
                    if grad.size > 0:
                        grad.fill(0.0)
                    return

                result[:] = Aeq @ X - beq
                if not np.isfinite(result).all():
                    result[:] = _MAIN_INVALID_PENALTY
                if grad.size > 0:
                    grad[:] = Aeq

            main_exc = None
            initial_XM = _reshape_X(np.clip(last_X, lb, ub))
            *_, initial_invalid = main_state_core_safe(
                initial_XM[:, :2],
                initial_XM[:, 3],
                initial_XM[:, 2],
                c0,
                c1,
                _MAIN_DP_EPS,
            )
            main_fast_available = not bool(np.any(initial_invalid))

            if main_fast_available:
                local_opt = nlopt.opt(nlopt.LD_LBFGS, n_vars)
                main_opt = nlopt.opt(nlopt.AUGLAG, n_vars)
                main_opt.set_local_optimizer(local_opt)
                main_opt.set_min_objective(objective)
                main_opt.set_lower_bounds(lb)
                main_opt.set_upper_bounds(ub)
                main_opt.add_inequality_mconstraint(nonlinear_con, 1e-6 * np.ones(n_edges))
                main_opt.add_equality_mconstraint(linear_con, 1e-6 * np.ones(2))
                main_opt.set_maxeval(2000)

                try:
                    optimized_X = main_opt.optimize(np.clip(last_X, lb, ub))
                    last_X[:] = np.asarray(optimized_X, dtype=np.float64)
                except Exception as exc:
                    main_exc = exc

                main_result = main_opt.last_optimize_result()
            else:
                main_exc = RuntimeError("Skipping fast main optimization because near-zero pressure differences would create non-finite geometry.")
                main_result = nlopt.FORCED_STOP

            print('')
            self.optimisation_status["main_fast_result"] = _nlopt_result_name(main_result)
            main_fast_ok = (
                main_exc is None
                and _nlopt_clean_result(main_result)
                and np.isfinite(last_X).all()
            )

            if main_fast_ok:
                X = last_X.reshape(X0.shape, order="F")
            else:
                issue = str(main_exc) if main_exc is not None else _nlopt_result_name(main_result)
                warnings.warn(
                    f"Main minimization did not converge cleanly ({issue}); retrying with the safer objective.",
                    RuntimeWarning,
                )

                safe_source = best_main_state["X"] if best_main_state["valid"] else last_X
                safe_start = np.clip(np.nan_to_num(safe_source.copy(), nan=0.0, posinf=0.0, neginf=0.0), lb, ub)
                best_main_state["X"] = safe_start.copy()
                best_main_state["score"] = np.inf
                best_main_state["E"] = np.inf
                best_main_state["valid"] = False

                local_opt = nlopt.opt(nlopt.LD_LBFGS, n_vars)
                main_opt = nlopt.opt(nlopt.AUGLAG, n_vars)
                main_opt.set_local_optimizer(local_opt)
                main_opt.set_min_objective(safe_objective)
                main_opt.set_lower_bounds(lb)
                main_opt.set_upper_bounds(ub)
                main_opt.add_inequality_mconstraint(safe_nonlinear_con, 1e-6 * np.ones(n_edges))
                main_opt.add_equality_mconstraint(safe_linear_con, 1e-6 * np.ones(2))
                main_opt.set_maxeval(2000)
                safe_objective(safe_start, np.array([]))

                main_exc = None
                try:
                    optimized_X = main_opt.optimize(safe_start)
                    last_X[:] = np.asarray(optimized_X, dtype=np.float64)
                except Exception as exc:
                    main_exc = exc

                print('')
                main_result = main_opt.last_optimize_result()
                self.optimisation_status["main_safe_result"] = _nlopt_result_name(main_result)
                main_safe_ok = (
                    main_exc is None
                    and _nlopt_clean_result(main_result)
                    and np.isfinite(last_X).all()
                )
                if not main_safe_ok:
                    issue = str(main_exc) if main_exc is not None else _nlopt_result_name(main_result)
                    warnings.warn(
                        f"Main minimization did not converge cleanly ({issue}); using the best finite iterate found.",
                        RuntimeWarning,
                    )

                if main_safe_ok:
                    X = last_X.reshape(X0.shape, order="F")
                elif best_main_state["valid"]:
                    X = best_main_state["X"].reshape(X0.shape, order="F")
                    self.optimisation_status["main_used_best_iterate"] = True
                else:
                    raise RuntimeError("Main minimization failed before finding a finite iterate.")

        q = X[:, :2]
        theta = X[:, 2]
        p = X[:, 3]

        T = self.get_tensions(q, p, theta)
        self.upload_mechanics(p, T, q, theta)

        if self.verbose:
            print("Main minimization complete\n")
        print("Data fitted!\n")
        return


    def get_tensions(self, q, p, theta):

        """

        applies the Young-Laplace law to obtain the tensions at every edge

        """
        dq = self.dC @ q
        radicand = np.sum(dq * dq, axis=1)
        pc0 = p[self.cell_pairs[:, 0]]
        pc1 = p[self.cell_pairs[:, 1]]
        radicand = radicand * np.abs(pc0 * pc1)
        radicand = radicand - (self.dC @ p) * (self.dC @ theta)

        finite = np.isfinite(radicand)
        min_radicand = float(np.min(radicand[finite])) if np.any(finite) else np.nan
        scale = float(np.max(np.abs(radicand[finite]))) if np.any(finite) else 1.0
        bad = (~finite) | (radicand < -_TENSION_RADICAND_TOL * max(scale, 1.0))
        if np.any(bad):
            warnings.warn(
                f"{int(np.count_nonzero(bad))} inferred tensions had invalid radicands; clipping them to zero.",
                RuntimeWarning,
            )

        self.optimisation_status["min_tension_radicand"] = min_radicand
        self.optimisation_status["invalid_tension_radicands"] = int(np.count_nonzero(bad))

        radicand = np.nan_to_num(radicand, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        T = np.sqrt(np.maximum(radicand, 0.0))
        return T


    def upload_mechanics(self, p, T, q, theta):
        """

        Match tensions and pressures to edges and cells

        """

        # Use .loc writes to avoid pandas Copy-on-Write dropping updates.
        sorted_cells = np.sort(self.involved_cells)
        sorted_index = np.argsort(self.involved_cells)
        self.cells.loc[sorted_cells, 'pressure'] = p[sorted_index]
        self.cells.loc[sorted_cells, 'qx'] = q[sorted_index, 0]
        self.cells.loc[sorted_cells, 'qy'] = q[sorted_index, 1]
        self.cells.loc[sorted_cells, 'theta'] = theta[sorted_index]
        
        for diff_idx, edge_idx in enumerate(np.asarray(self.involved_edges, dtype=int)):
            if edge_idx >= 0:
                self.edges.at[int(edge_idx), 'tension'] = float(T[diff_idx])
        return
    
    
    def compute_stresstensor(self):
        """
        Compute stress tensor for each cell from tensions and pressures.
        This uses a small-angle approximation assuming edge curvatures are small.
        """
        print(" Computing stress tensor...")

        involved_cells = np.asarray(self.involved_cells, dtype=int)                    # (Nc,)
        involved_vertices = np.asarray(self.involved_vertices, dtype=int)              # (Nv,)
        dC = self.dC.tocsr()                                                           # (Ne, Nc)
        dV = self.dV.tocsr()                                                           # (Ne, Nv)
        cells = self.cells
        edges = self.edges
        vertices = self.vertices
        cell_pressure = cells["pressure"].tolist()                                     # list
        cell_area = np.asarray(cells["area"].to_list(), dtype=float)                   # (Ncells_total,)
        edge_verts_list = edges["verts"].tolist()                                      # list, each ~ (2,)
        edge_tension_list = edges["tension"].tolist()                                  # list
        vertex_coords = vertices["coords"].tolist()                                    # list, each (2,)
        p = np.asarray([cell_pressure[cell] for cell in involved_cells], dtype=float)  # (Nc,)
        n_edges = dC.shape[0]

        edge_map = {}
        for idx, verts in enumerate(edge_verts_list):
            verts = np.asarray(verts, dtype=int)
            if verts.size == 2:
                a, b = int(verts[0]), int(verts[1])
                if a > b:
                    a, b = b, a
                edge_map[(a, b)] = idx

        pos_idx, neg_idx = self._extract_signed_columns(dV)
        T = np.ones(n_edges, dtype=float)                                              # (Ne,)
        i1 = -np.ones(n_edges, dtype=int)                                              # (Ne,)

        for e in range(n_edges):
            pv = pos_idx[e]
            nv = neg_idx[e]
            if pv < 0 or nv < 0:
                continue
            v0 = int(involved_vertices[pv])
            v1 = int(involved_vertices[nv])
            if v0 > v1:
                v0, v1 = v1, v0
            idx = edge_map.get((v0, v1))
            if idx is None:
                continue
            i1[e] = idx
            tension = edge_tension_list[idx]
            if np.size(tension) > 0:
                T[e] = float(tension)
            else:
                T[e] = 1.0

        rv = np.asarray([vertex_coords[v] for v in involved_vertices], dtype=float)     # (Nv, 2)
        rb = dV @ rv                                                                    # (Ne, 2)
        D = np.sqrt(np.sum(rb * rb, axis=1))                                            # (Ne,)
        D_safe = D.copy()
        D_safe[D_safe == 0.0] = 1.0
        rb = rb / D_safe[:, None]                                                       # (Ne, 2)

        nb = np.empty_like(rb)                                                          # (Ne, 2)
        nb[:, 0] = -rb[:, 1]
        nb[:, 1] =  rb[:, 0]

        dP = dC @ p                                                                     # (Ne,)

        # sigmaB: [xx, xy, yy]
        sigmaB = np.empty((n_edges, 3), dtype=float)                                    # (Ne, 3)
        sigmaB[:, 0] = T * rb[:, 0] * rb[:, 0]
        sigmaB[:, 1] = T * rb[:, 0] * rb[:, 1]
        sigmaB[:, 2] = T * rb[:, 1] * rb[:, 1]

        # sigmaP: [xx, xy, yy]
        dPD = dP * D_safe                                                               # (Ne,)
        sigmaP = np.empty((n_edges, 3), dtype=float)                                    # (Ne, 3)
        sigmaP[:, 0] = dPD * nb[:, 0] * nb[:, 0]
        sigmaP[:, 1] = dPD * nb[:, 0] * nb[:, 1]
        sigmaP[:, 2] = dPD * nb[:, 1] * nb[:, 1]

        dC_abs = dC.copy()
        dC_abs.data = np.abs(dC_abs.data)
        sigma = (dC_abs.T @ sigmaB) + 0.5 * (dC.T @ sigmaP)                             # (Nc, 3)

        A = cell_area[involved_cells]                                                   # (Nc,)
        sigma = sigma / A[:, None]                                                      # (Nc, 3)

        stress_col = cells["stress"].tolist()
        for idx, cell in enumerate(involved_cells):
            stress_col[cell] = sigma[idx]

        self.cells["stress"] = stress_col
        print(" Computing stress tensor success")
        return


    def plot(self, options='', mask=np.array([]), line_thickness=5, size=80, file=None, dpi=300):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import cm, patches, colors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        """

        Plots results of stress inference.

        :param mask: (numpy array) image on which to overlay plotted objects. If plotting pressure, must be labelled, segmented image.
        :param options: (list) list of options for plotting. Available options are: 'stress', 'pressure', 'tension', 'CAP'.
        :param line_thickness: (int) thickness of lines used for tension and CAP plotting. Default: 5.
        :param size: (int) text size for legends. Default: 10.
        :param file: (str) filename to save plot to. If none provided, outputs plot to console.
        """
        if isinstance(options, str):
            options = [options.lower()]
        else:
            options = [option.lower() for option in options]

        # Pressure first as requires remapping cell area colours
        if mask.size > 0:
            # size = mask.shape[0]/(2*dpi)
            if np.isin('pressure', options):
                img = np.zeros_like(mask, dtype=float)
                colourmap = matplotlib.colormaps['plasma'].copy()
                pressures = self.cells.pressure.to_numpy()[self.involved_cells]
                pressure_min = np.min(pressures)
                maxP = np.percentile(pressures, 90)
                if maxP <= pressure_min:
                    maxP = np.max(pressures)
                if maxP <= pressure_min:
                    maxP = pressure_min + 1

                cell_labels = self.cells.loc[self.involved_cells, 'label'].to_numpy(dtype=int)
                valid_labels = (cell_labels > 0) & np.isin(cell_labels, np.unique(mask))

                if not np.all(valid_labels):
                    props = measure.regionprops(mask)
                    centroids = np.array(self.cells.loc[self.involved_cells, 'centroids'].tolist())
                    img_centroids = np.array([np.flip(regionprops.centroid) for regionprops in props])
                    indices = np.argmin(cdist(centroids, img_centroids), axis=1)
                    cell_labels = np.array([props[index].label for index in indices], dtype=int)

                involvedcells_labels = cell_labels
                clipped_pressures = np.clip(pressures, pressure_min, maxP)

                for label, pressure in zip(involvedcells_labels, clipped_pressures):
                    img[mask == label] = pressure

                img = np.ma.masked_where(~np.isin(mask, involvedcells_labels), img)
                colourmap.set_bad(color='white')
            else:
                img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255
        else:
            if np.isin('pressure', options):
                return("Error: 'pressure' plotting specified without providing labelled, segmented image.")
            img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255

        # Set up figure
        fig, ax = plt.subplots(1,1,figsize=(np.divide(mask.shape,dpi)), dpi=dpi)
        ax.imshow(img)
        ax.set_axis_off()
        divider = make_axes_locatable(ax)
        fig.tight_layout()

        # Now we add colourbar for pressure if it was specified
        if np.isin('pressure', options):
            cax1 = divider.append_axes("right", size="5%", pad=0.5)
            pressure_norm = colors.Normalize(pressure_min, maxP)
            ax.images[0].set_norm(pressure_norm)
            ax.images[0].set_cmap(colourmap)
            p_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=pressure_norm, cmap=colourmap), cax=cax1)
            p_cb.set_label('Pressure (a.u.)', size=size)
            p_cb.ax.tick_params(labelsize=size)

        if all(np.isin(options, ['stress', 'pressure', 'tension', 'cap'])):
            for option in options:
                if option == 'stress':
                    stress = np.array([self.cells.at[cell, 'area'] * np.array([[self.cells.at[cell, 'stress'][0], self.cells.at[cell, 'stress'][1]],[self.cells.at[cell, 'stress'][1], self.cells.at[cell, 'stress'][2]]]) for cell in range(len(self.cells))])
                    eigvals, eigvects = np.linalg.eig(stress)
                    scalefct = np.sqrt(np.median(np.multiply(eigvals[:,0], eigvals[:,1])))

                    for i in range(len(self.involved_cells)):
                        cell = self.involved_cells[i]
                        if np.max(stress[i] > 0):
                            centroid = self.cells.at[cell, 'centroids']
                            eigval = eigvals[i,:]
                            eigvect = eigvects[i,:,:]
                            idx = np.flip(np.argsort(eigval))
                            eigval = eigval[idx]
                            eigvect = eigvect[:,idx]

                            # scale eigenvalues
                            eigval = np.divide(eigval, scalefct)
                            eigval[eigval>3] = 3
                            eigval[eigval<0] = 0
                            eigval = eigval * 0.4 * np.mean(np.sqrt(np.divide(np.array(self.cells.area.to_list())[1:], np.pi)))

                            # calculate angle of rotation
                            theta = np.arctan2(eigvect[1,0], eigvect[0,0])
                            if theta < 0:
                                theta = theta + 2*np.pi
                            theta = np.degrees(theta)
                            stress_ellipse = patches.Ellipse(centroid, eigval[0], eigval[1], angle=theta, fill=False, color='red', lw=line_thickness)
                            ax.add_patch(stress_ellipse)
                elif option == 'tension':
                    maxT = np.percentile(self.edges['tension'], 95)
                    minT = np.percentile(self.edges['tension'], 1)
                    colourmap = cm.get_cmap('hot')

                    for e in range(len(self.edges)):
                        if self.edges.at[e, 'tension'] > 0:
                            radius = self.edges.at[e, 'radius']
                            tension_norm = np.divide(self.edges.at[e, 'tension'], maxT-minT)
                            if tension_norm > 1:
                                tension_norm = 1
                            colour = colourmap(tension_norm)

                            if np.isinf(radius):

                                v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                                points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                                   np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                                ax.plot(points[:,0], points[:,1], lw=line_thickness, color=colour)
                            else:
                                v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                                rho = self.edges.at[e, 'rho']

                                theta = np.arctan2(v[:,1]-rho[1], v[:,0]-rho[0])
                                theta[theta<0] = theta[theta<0] + 2*np.pi
                                theta = np.sort(theta)

                                if theta[1] - theta[0] > np.pi:
                                    theta[1] = theta[1] - 2*np.pi

                                theta_range = np.linspace(theta[0], theta[1], len(self.edges.at[e, 'pixels']))
                                points = np.array([rho[0] + radius*np.cos(theta_range),
                                                   rho[1] + radius*np.sin(theta_range)]).T
                                ax.plot(points[:,0], points[:,1], lw=line_thickness, color=colour)
                    cax2 = divider.append_axes("right", size="5%", pad=0.5)
                    t_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=colors.Normalize(minT, maxT), cmap=colourmap), cax=cax2)
                    t_cb.set_label('Tension (a.u.)', size=size)
                    t_cb.ax.tick_params(labelsize=size)
                elif option == 'cap':
                    for e in range(len(self.edges)):
                        radius = self.edges.at[e, 'radius']

                        if np.isinf(radius):

                            v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                            points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                               np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                            ax.plot(points[:,0], points[:,1], lw=line_thickness, color='b')
                        else:
                            v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                            rho = self.edges.at[e, 'rho']

                            theta = np.arctan2(v[:,1]-rho[1], v[:,0]-rho[0])
                            theta[theta<0] = theta[theta<0] + 2*np.pi
                            theta = np.sort(theta)

                            if theta[1] - theta[0] > np.pi:
                                theta[1] = theta[1] - 2*np.pi

                            theta_range = np.linspace(theta[0], theta[1], len(self.edges.at[e, 'pixels']))
                            points = np.array([rho[0] + radius*np.cos(theta_range),
                                               rho[1] + radius*np.sin(theta_range)]).T
                            ax.plot(points[:,0], points[:,1], lw=line_thickness, color='b')
        else:
            raise ValueError("invalid options")
        if file is not None:
            ax.set_facecolor((1,1,1))
            ax.set_alpha(1.0)
            fig.tight_layout()
            plt.savefig(file, facecolor=ax.get_facecolor())
        else:
            plt.show()
        return


    def output_results(self, metrics=['centroids','pressure','stress','inertia','perimeter','polygon_perimeter','feret_d','moments_hu','bbox', 'area','label'], neighbours=False):
        """
        Outputs force inference/morphometric quantities and cell adjacency matrix as Pandas Dataframes.

        :param metrics: (list) list of metrics to output. Default: all metrics.
        :param neighbours: (bool) whether to output adjacency matrix. Nonzero values in matrix represent junction tension. Default: False.
        """

        # Assign actual id's to each cell instead of just using labels
        cell_ids = {label:id for label, id in zip(self.involved_cells, ['cell_' + str(number+1) for number in range(len(self.involved_cells))])}

        if metrics is not None:
            out_df = self.cells.loc[self.involved_cells, metrics]
            if 'centroids' in out_df.columns:
                out_df['centroid_x'] = np.array(out_df['centroids'].tolist())[:,0]
                out_df['centroid_y'] = np.array(out_df['centroids'].tolist())[:,1]
                out_df.drop('centroids', axis=1, inplace=True)
            if 'stress' in out_df.columns:
                # We want each value to make sense on its own
                # Therefore, compute eigenvalues, orientation and anisotropy of stress tensor
                stress = np.array([np.array([[out_df.at[cell, 'stress'][0], out_df.at[cell, 'stress'][1]],[out_df.at[cell, 'stress'][1], out_df.at[cell, 'stress'][2]]]) for cell in out_df.index])
                eigvals, eigvects = np.linalg.eig(stress)
                eigvals = np.abs(eigvals)
                idx = np.flip(np.argsort(eigvals, axis=1), axis=1)
                eigvals = eigvals[np.arange(eigvals.shape[0])[:,None], idx]
                eigvects = eigvects[np.arange(eigvects.shape[0])[:,None], idx]
                out_df['stresstensor_eigval1'] = eigvals[:,0]
                out_df['stresstensor_eigval2'] = eigvals[:,1]
                # define principal stresses as max and min eigenvalues
                out_df["principal_stress1"] = out_df[["stresstensor_eigval1", "stresstensor_eigval2"]].max(axis=1)
                out_df["principal_stress2"] = out_df[["stresstensor_eigval1", "stresstensor_eigval2"]].min(axis=1)

                # define isotropic stress as half of principal stresses
                out_df["isotropic_stress"] = (out_df["stresstensor_eigval1"] + out_df["stresstensor_eigval2"]) / 2
                
                # define anisotropic stress as difference between principal stresses
                out_df["anisotropic_stress"] = (out_df["stresstensor_eigval1"] - out_df["stresstensor_eigval2"]).abs()
                
                # define orientation as absolute angle between largest eigenvector and x-axis
                out_df['stresstensor_orientation'] = np.arctan2(np.abs(eigvects[:,0,:])[:,1], np.abs(eigvects[:,0,:])[:,0])
                # define anisotropy using definition from Hartkamp et al., J. Chem. Phys. 2012
                out_df['stresstensor_anisotropy'] = np.divide(eigvals[:,0] - eigvals[:,1], (eigvals[:,0] + eigvals[:,1]))
                out_df.drop('stress', axis=1, inplace=True)
            if 'inertia' in out_df.columns:
                # We want each value to make sense on its own
                # Therefore, compute eigenvalues, orientation and anisotropy of inertia tensor
                inertia = np.array([np.array([[out_df.at[cell, 'inertia'][0], out_df.at[cell, 'inertia'][1]],[out_df.at[cell, 'inertia'][1], out_df.at[cell, 'inertia'][2]]]) for cell in out_df.index])
                eigvals, eigvects = np.linalg.eig(inertia)
                eigvals = np.abs(eigvals)
                idx = np.flip(np.argsort(eigvals, axis=1), axis=1)
                eigvals = eigvals[np.arange(eigvals.shape[0])[:,None], idx]
                eigvects = eigvects[np.arange(eigvects.shape[0])[:,None], idx]
                out_df['inertiatensor_eigval1'] = eigvals[:,0]
                out_df['inertiatensor_eigval2'] = eigvals[:,1]
                # define orientation as absolute angle between largest eigenvector and x-axis
                out_df['inertiatensor_orientation'] = np.arctan2(np.abs(eigvects[:,0,:])[:,1], np.abs(eigvects[:,0,:])[:,0])
                out_df['inertiatensor_anisotropy'] = np.divide(eigvals[:,0] - eigvals[:,1], (eigvals[:,0] + eigvals[:,1]))
                out_df.drop('inertia', axis=1, inplace=True)
            if 'moments_hu' in out_df.columns:
                # Hu moments 1 and 3 (2 is omitted as it is equivalent to inertia tensor anisotropy)
                out_df['moments_hu_1'] = np.array(out_df['moments_hu'].tolist())[:,0]
                out_df['moments_hu_3'] = np.array(out_df['moments_hu'].tolist())[:,2]
                out_df.drop('moments_hu', axis=1, inplace=True)
            if 'bbox' in out_df.columns:
                out_df['bbox_x'] = np.array(out_df['bbox'].tolist())[:,0]
                out_df['bbox_y'] = np.array(out_df['bbox'].tolist())[:,1]
                out_df.drop('bbox', axis=1, inplace=True)
            out_df.rename(index=cell_ids, inplace=True)

        if neighbours:
            # Generate adjacency matrix.
            # Magnitude of non-zero values represents the tension (in arbitrary units) along the junction.
            adj_mat = pd.DataFrame(np.zeros([len(self.involved_cells), len(self.involved_cells)]), index=self.involved_cells, columns=self.involved_cells)

            for cell in self.involved_cells:
                adj_cells = self.cells.at[cell, 'ncells']
                for adj_cell in adj_cells[np.isin(adj_cells, self.involved_cells)]:
                    # adj_mat.loc[cell, adj_cell] = self.edges[self.edges.cells.apply(tuple) == tuple(np.sort([cell, adj_cell]))]['tension'].iloc[0]
                    matched_edge = self.edges[self.edges.cells.apply(tuple) == tuple(np.sort([cell, adj_cell]))]
        
                    if not matched_edge.empty:
                        adj_mat.loc[cell, adj_cell] = matched_edge['tension'].iloc[0]
                    else:
                        print(f"Warning: No edge found between {cell + 1} and {adj_cell + 1}. Likely 1px long edge")
                        adj_mat.loc[cell, adj_cell] = np.nan  # or 0
            adj_mat.rename(index=cell_ids, columns=cell_ids, inplace=True)

        if neighbours and metrics is not None:
            return out_df, adj_mat
        elif metrics is None:
            return adj_mat
        elif not neighbours:
            return out_df
        else:
            return False
        

def run_VMSI(img, VMSI_obj, labelled_mask, seg_mask, verbose=False, optimiser="nlopt"):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = VMSI(vertices=VMSI_obj.V_df,cells=VMSI_obj.C_df,edges=VMSI_obj.E_df,height=img.shape[0],width=img.shape[1],verbose=verbose,optimiser=optimiser)

        try:
            model.fit(seg_mask, labelled_mask)
        except Exception as e:
            raise ValueError("Fitting failure") from e

        try:
            model.compute_stresstensor()
        except Exception as e:
            raise ValueError("Computing stress tensor failed") from e

    print("VMSI complete")
    return model

