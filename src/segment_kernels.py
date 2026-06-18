import numpy as np
from numba import njit


@njit
def clear_low_diversity_edges(tmp1, padded_tmp3):
    h, w = tmp1.shape

    for i in range(h):
        for j in range(w):
            if tmp1[i, j] != 0:
                continue

            seen = np.empty(9, dtype=np.int64)
            n_seen = 0

            for di in range(3):
                for dj in range(3):
                    value = int(padded_tmp3[i + di, j + dj])
                    is_new = True

                    for k in range(n_seen):
                        if seen[k] == value:
                            is_new = False
                            break

                    if is_new:
                        seen[n_seen] = value
                        n_seen += 1
                        if n_seen >= 3:
                            break

                if n_seen >= 3:
                    break

            if n_seen < 3:
                tmp1[i, j] = 1


@njit
def build_adj_from_pairs(ncells_arr, pairs):
    matched_pairs = np.empty((pairs.shape[0], 2), dtype=np.int32)
    matched_count = 0

    for idx in range(pairs.shape[0]):
        i = pairs[idx, 0]
        j = pairs[idx, 1]
        count = 0

        for a in ncells_arr[i]:
            if a < 0:
                break
            for b in ncells_arr[j]:
                if b < 0:
                    break
                if a == b:
                    count += 1
                    if count >= 2:
                        matched_pairs[matched_count, 0] = i
                        matched_pairs[matched_count, 1] = j
                        matched_count += 1
                        break
                if count >= 2:
                    break

    return matched_pairs[:matched_count]


@njit
def _intersect_first(a, b):
    for i in range(len(a)):
        if a[i] < 0:
            break
        for j in range(len(b)):
            if b[j] < 0:
                break
            if a[i] == b[j]:
                return a[i]
    return -1


@njit
def compute_cell_edges(V_coords, V_nverts, V_edges, C_nverts, C0_flag):
    n_cells = C_nverts.shape[0]
    # A cell can have more incident edges than the maximum degree of any single
    # vertex (typically 3). Size the output by the maximum number of vertices per cell
    # to avoid truncating polygonal cells.
    max_edges = C_nverts.shape[1]

    C_edges = -np.ones((n_cells, max_edges), dtype=np.int32)

    for c in range(1, n_cells):
        nverts_c = 0
        for i in range(C_nverts.shape[1]):
            if C_nverts[c, i] < 0:
                break
            nverts_c += 1

        if nverts_c <= 1:
            continue

        if C0_flag[c]:
            continue

        verts = np.empty(nverts_c, dtype=np.int32)
        for i in range(nverts_c):
            verts[i] = C_nverts[c, i]

        coords = np.empty((nverts_c, 2), dtype=np.float64)
        mx = 0.0
        my = 0.0
        for i in range(nverts_c):
            v = verts[i]
            x = V_coords[v, 0]
            y = V_coords[v, 1]
            coords[i, 0] = x
            coords[i, 1] = y
            mx += x
            my += y

        mx /= nverts_c
        my /= nverts_c

        for i in range(nverts_c):
            coords[i, 0] -= mx
            coords[i, 1] -= my

        theta = np.empty(nverts_c, dtype=np.float64)
        for i in range(nverts_c):
            theta[i] = np.arctan2(coords[i, 1], coords[i, 0])
            if theta[i] < 0.0:
                theta[i] += 2.0 * np.pi

        order = np.argsort(theta)

        for i in range(nverts_c):
            v1 = verts[order[i]]
            v2 = verts[order[(i + 1) % nverts_c]]

            found = False
            for k in range(V_nverts.shape[1]):
                if V_nverts[v1, k] < 0:
                    break
                if V_nverts[v1, k] == v2:
                    found = True
                    break

            if found:
                e = _intersect_first(V_edges[v1], V_edges[v2])
                C_edges[c, i] = e
            else:
                C_edges[c, i] = -1

    return C_edges
