import numpy as np
from numba import jit, njit

# Helper functions to handle plotting and data processing


@jit
def fast_find_boundaries_subpixel(label_img):
    h, w = label_img.shape
    expanded_h, expanded_w = 2 * h - 1, 2 * w - 1
    boundaries = np.zeros((expanded_h, expanded_w), dtype=np.uint16)

    for i in range(h - 1):
        for j in range(w - 1):
            # Horizontal boundary
            if label_img[i, j] != label_img[i, j + 1]:
                boundaries[2 * i, 2 * j + 1] = 1
            
            # Vertical boundary
            if label_img[i, j] != label_img[i + 1, j]:
                boundaries[2 * i + 1, 2 * j] = 1 
            
            # Diagonal boundaries
            if label_img[i, j] != label_img[i + 1, j + 1]:
                boundaries[2 * i + 1, 2 * j + 1] = 1 
            
            if label_img[i, j + 1] != label_img[i + 1, j]:
                boundaries[2 * i + 1, 2 * j + 1] = 1 

    return boundaries


@njit
def intersect_first(a, b):
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
    max_edges = V_nverts.shape[1]

    C_edges = -np.ones((n_cells, max_edges), dtype=np.int32)

    for c in range(1, n_cells):
        # count valid verts
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
                e = intersect_first(V_edges[v1], V_edges[v2])
                C_edges[c, i] = e
            else:
                C_edges[c, i] = -1

    return C_edges


@njit(cache=True)
def _cross2d(a, b):
    # a: (2,)
    # b: (2,)
    return a[0] * b[1] - a[1] * b[0]


@njit(cache=True)
def _compute_new_vertex_position(rv, n):
    # rv: (2,)
    # n: (3, 2)

    # centro dei vicini
    center = np.empty(2, dtype=np.float64)
    center[0] = (n[0, 0] + n[1, 0] + n[2, 0]) / 3.0
    center[1] = (n[0, 1] + n[1, 1] + n[2, 1]) / 3.0

    # n_centered: (3, 2)
    n_centered = np.empty((3, 2), dtype=np.float64)
    for i in range(3):
        n_centered[i, 0] = n[i, 0] - center[0]
        n_centered[i, 1] = n[i, 1] - center[1]

    # theta_sort: (3,)
    theta_sort = np.empty(3, dtype=np.float64)
    for i in range(3):
        th = np.arctan2(n_centered[i, 1], n_centered[i, 0])
        theta_sort[i] = np.mod(th, 2.0 * np.pi)

    order = np.argsort(theta_sort)

    # n_sorted: (3, 2)
    n_sorted = np.empty((3, 2), dtype=np.float64)
    for i in range(3):
        n_sorted[i, 0] = n[order[i], 0]
        n_sorted[i, 1] = n[order[i], 1]

    # r = normalized(n_sorted - rv)
    r = np.empty((3, 2), dtype=np.float64)
    for i in range(3):
        dx = n_sorted[i, 0] - rv[0]
        dy = n_sorted[i, 1] - rv[1]
        norm = np.sqrt(dx * dx + dy * dy)
        if norm == 0.0:
            return rv.copy()
        r[i, 0] = dx / norm
        r[i, 1] = dy / norm

    z12 = _cross2d(r[0], r[1])
    z23 = _cross2d(r[1], r[2])
    z31 = _cross2d(r[2], r[0])

    d12 = r[0, 0] * r[1, 0] + r[0, 1] * r[1, 1]
    d23 = r[1, 0] * r[2, 0] + r[1, 1] * r[2, 1]
    d31 = r[2, 0] * r[0, 0] + r[2, 1] * r[0, 1]

    theta12 = np.mod(np.arctan2(z12, d12), 2.0 * np.pi)
    theta23 = np.mod(np.arctan2(z23, d23), 2.0 * np.pi)
    theta31 = np.mod(np.arctan2(z31, d31), 2.0 * np.pi)

    nrv = rv.copy()

    # rot90(x, y) = (-y, x)
    if theta12 > np.pi:
        e0x = n_sorted[0, 0] - n_sorted[1, 0]
        e0y = n_sorted[0, 1] - n_sorted[1, 1]
        perpx = -e0y
        perpy = e0x

        a0x = n_sorted[0, 0] - rv[0]
        a0y = n_sorted[0, 1] - rv[1]
        a2x = n_sorted[2, 0] - rv[0]
        a2y = n_sorted[2, 1] - rv[1]

        num = a0x * perpx + a0y * perpy
        den = a2x * perpx + a2y * perpy
        if den != 0.0:
            deltaR = num / den
            nrv[0] = rv[0] + 1.5 * deltaR * a2x
            nrv[1] = rv[1] + 1.5 * deltaR * a2y

    elif theta23 > np.pi:
        e1x = n_sorted[1, 0] - n_sorted[2, 0]
        e1y = n_sorted[1, 1] - n_sorted[2, 1]
        perpx = -e1y
        perpy = e1x

        a1x = n_sorted[1, 0] - rv[0]
        a1y = n_sorted[1, 1] - rv[1]
        a0x = n_sorted[0, 0] - rv[0]
        a0y = n_sorted[0, 1] - rv[1]

        num = a1x * perpx + a1y * perpy
        den = a0x * perpx + a0y * perpy
        if den != 0.0:
            deltaR = num / den
            nrv[0] = rv[0] + 1.5 * deltaR * a0x
            nrv[1] = rv[1] + 1.5 * deltaR * a0y

    elif theta31 > np.pi:
        e2x = n_sorted[2, 0] - n_sorted[0, 0]
        e2y = n_sorted[2, 1] - n_sorted[0, 1]
        perpx = -e2y
        perpy = e2x

        a2x = n_sorted[2, 0] - rv[0]
        a2y = n_sorted[2, 1] - rv[1]
        a1x = n_sorted[1, 0] - rv[0]
        a1y = n_sorted[1, 1] - rv[1]

        num = a2x * perpx + a2y * perpy
        den = a1x * perpx + a1y * perpy
        if den != 0.0:
            deltaR = num / den
            nrv[0] = rv[0] + 1.5 * deltaR * a1x
            nrv[1] = rv[1] + 1.5 * deltaR * a1y

    elif theta12 == np.pi:
        nrv[0] = rv[0] + 0.5 * r[2, 0]
        nrv[1] = rv[1] + 0.5 * r[2, 1]
    elif theta23 == np.pi:
        nrv[0] = rv[0] + 0.5 * r[0, 0]
        nrv[1] = rv[1] + 0.5 * r[0, 1]
    elif theta31 == np.pi:
        nrv[0] = rv[0] + 0.5 * r[1, 0]
        nrv[1] = rv[1] + 0.5 * r[1, 1]

    return nrv