import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def sx_grad(p1, p2, q1x, q1y, q2x, q2y, rx, ry):
    """
    Computes part of the jacobian of the energy function
    (the partial derivative with respect to x) analytically
    to improve the speed and accuracy of initial optimisation.
    """
    n = p1.shape[0]
    out = np.empty((n, 6), dtype=np.float64)

    t3 = p1 - p2
    t5 = p1 * q1x
    t6 = p2 * q2x
    t7 = rx * t3
    t2 = -t5 + t6 + t7

    t9 = p1 * q1y
    t10 = p2 * q2y
    t11 = ry * t3
    t4 = -t9 + t10 + t11

    t8 = t2 * t2
    t12 = t4 * t4
    t13 = t8 + t12
    t14 = 1.0 / np.power(t13, 1.5)
    t15 = 1.0 / np.sqrt(t13)

    t16 = q1x - rx
    t17 = q2x - rx
    q1y_ry = q1y - ry
    q2y_ry = q2y - ry

    out[:, 0] = p1 * t15 - p1 * t8 * t14
    out[:, 1] = -p1 * t2 * t4 * t14
    out[:, 2] = t15 * t16 - t2 * t14 * (t2 * t16 * 2.0 + t4 * q1y_ry * 2.0) * 0.5
    out[:, 3] = -p2 * t15 + p2 * t8 * t14
    out[:, 4] = p2 * t2 * t4 * t14
    out[:, 5] = -t15 * t17 + t2 * t14 * (t2 * t17 * 2.0 + t4 * q2y_ry * 2.0) * 0.5

    return out


@njit(cache=True, fastmath=False)
def sy_grad(p1, p2, q1x, q1y, q2x, q2y, rx, ry):
    """
    Computes part of the jacobian of the energy function
    (the partial derivative with respect to y) analytically
    to improve the speed and accuracy of initial optimisation.
    """
    n = p1.shape[0]
    out = np.empty((n, 6), dtype=np.float64)

    t3 = p1 - p2
    t5 = p1 * q1x
    t6 = p2 * q2x
    t7 = rx * t3
    t2 = -t5 + t6 + t7

    t8 = p1 * q1y
    t9 = p2 * q2y
    t10 = ry * t3
    t4 = -t8 + t9 + t10

    t11 = t2 * t2
    t12 = t4 * t4
    t13 = t11 + t12
    t14 = 1.0 / np.power(t13, 1.5)
    t15 = 1.0 / np.sqrt(t13)

    t16 = q1y - ry
    t17 = q2y - ry
    q1x_rx = q1x - rx
    q2x_rx = q2x - rx

    out[:, 0] = -p1 * t2 * t4 * t14
    out[:, 1] = p1 * t15 - p1 * t12 * t14
    out[:, 2] = t15 * t16 - t4 * t14 * (t4 * t16 * 2.0 + t2 * q1x_rx * 2.0) * 0.5
    out[:, 3] = p2 * t2 * t4 * t14
    out[:, 4] = -p2 * t15 + p2 * t12 * t14
    out[:, 5] = -t15 * t17 + t4 * t14 * (t4 * t17 * 2.0 + t2 * q2x_rx * 2.0) * 0.5

    return out


@njit(cache=True, fastmath=False)
def radius_grad_theta(p1, p2, q1x, q1y, q2x, q2y, t1, t2):
    """
    Computes the jacobian of the energy function
    (the partial derivative with respect to theta variables)
    analytically to improve the speed and accuracy of theta optimisation.
    """
    n = p1.shape[0]
    out = np.empty((n, 2), dtype=np.float64)

    t4 = p1 - p2
    t5 = q1x - q2x
    t6 = q1y - q2y
    t7 = 1.0 / (t4 * t4)
    t8 = t1 - t2
    t9 = t4 * t8
    t10 = t5 * t5
    t11 = t6 * t6
    t12 = t10 + t11
    t13 = t9 - p1 * p2 * t12
    t14 = 1.0 / np.sqrt(-t7 * t13)
    t15 = 1.0 / t4

    out[:, 0] = t14 * t15 * (-0.5)
    out[:, 1] = t14 * t15 * (0.5)

    return out


@njit(cache=True, fastmath=False)
def rx_grad(p1, p2, q1x, q2x, rx):
    """
    Computes part of the jacobian of the nonlinear constraint function
    (the partial derivative with respect to x) analytically
    to improve the speed and accuracy of initial optimisation.
    """
    n = p1.shape[0]
    out = np.empty((n, 6), dtype=np.float64)

    out[:, 0] = p1
    out[:, 1] = 0.0
    out[:, 2] = q1x - rx
    out[:, 3] = -p2
    out[:, 4] = 0.0
    out[:, 5] = -q2x + rx

    return out


@njit(cache=True, fastmath=False)
def ry_grad(p1, p2, q1y, q2y, ry):
    """
    Computes part of the jacobian of the nonlinear constraint function
    (the partial derivative with respect to y) analytically
    to improve the speed and accuracy of initial optimisation.
    """
    n = p1.shape[0]
    out = np.empty((n, 6), dtype=np.float64)

    out[:, 0] = 0.0
    out[:, 1] = p1
    out[:, 2] = q1y - ry
    out[:, 3] = 0.0
    out[:, 4] = -p2
    out[:, 5] = -q2y + ry

    return out


@njit(cache=True, fastmath=False)
def rho_x_grad(p1, p2, q1x, q2x):
    """
    Computes part of the jacobian of the energy function analytically
    to improve the speed and accuracy of the main optimisation step.
    """
    n = p1.shape[0]
    out = np.empty((n, 8), dtype=np.float64)

    t2 = p1 - p2
    t3 = 1.0 / t2
    t4 = 1.0 / (t2 * t2)
    t5 = p1 * q1x
    t6 = t5 - p2 * q2x

    out[:, 0] = p1 * t3
    out[:, 1] = 0.0
    out[:, 2] = 0.0
    out[:, 3] = q1x * t3 - t4 * t6
    out[:, 4] = -p2 * t3
    out[:, 5] = 0.0
    out[:, 6] = 0.0
    out[:, 7] = -q2x * t3 + t4 * t6

    return out


@njit(cache=True, fastmath=False)
def rho_y_grad(p1, p2, q1y, q2y):
    """
    Computes part of the jacobian of the energy function analytically
    to improve the speed and accuracy of the main optimisation step.
    """
    n = p1.shape[0]
    out = np.empty((n, 8), dtype=np.float64)

    t2 = p1 - p2
    t3 = 1.0 / t2
    t4 = 1.0 / (t2 * t2)
    t5 = p1 * q1y
    t6 = t5 - p2 * q2y

    out[:, 0] = 0.0
    out[:, 1] = p1 * t3
    out[:, 2] = 0.0
    out[:, 3] = q1y * t3 - t4 * t6
    out[:, 4] = 0.0
    out[:, 5] = -p2 * t3
    out[:, 6] = 0.0
    out[:, 7] = -q2y * t3 + t4 * t6

    return out


@njit(cache=True, fastmath=False)
def radius_grad(p1, p2, q1x, q1y, q2x, q2y, t1, t2):
    """
    Computes part of the jacobian of the energy function analytically
    to improve the speed and accuracy of the main optimisation step.
    """
    n = p1.shape[0]
    out = np.empty((n, 8), dtype=np.float64)

    t4 = p1 - p2
    t5 = q1x - q2x
    t6 = q1y - q2y
    t7 = 1.0 / (t4 * t4)
    t8 = t1 - t2
    t9 = t4 * t8
    t10 = t5 * t5
    t11 = t6 * t6
    t12 = t10 + t11
    t15 = p1 * p2 * t12
    t13 = t9 - t15
    t16 = t7 * t13
    t14 = 1.0 / np.sqrt(-t16)
    t17 = q1x * 2.0
    t18 = q2x * 2.0
    t19 = t17 - t18
    t20 = p1 * p2 * t7 * t14 * t19 * 0.5
    t21 = q1y * 2.0
    t22 = q2y * 2.0
    t23 = t21 - t22
    t24 = p1 * p2 * t7 * t14 * t23 * 0.5
    t25 = 1.0 / t4
    t26 = 1.0 / (t4 * t4 * t4)
    t27 = t13 * t26 * 2.0

    out[:, 0] = t20
    out[:, 1] = t24
    out[:, 2] = t14 * t25 * (-0.5)
    out[:, 3] = t14 * (t27 + t7 * (-t1 + t2 + p2 * t12)) * 0.5
    out[:, 4] = -t20
    out[:, 5] = -t24
    out[:, 6] = t14 * t25 * 0.5
    out[:, 7] = t14 * (t27 - t7 * (t1 - t2 + p1 * t12)) * (-0.5)

    return out


@njit(cache=True, fastmath=False)
def tension_projections_and_energy(v1, v2, tau_1, tau_2):
    # v1, v2, tau_1, tau_2: (Ne, 2)
    n = v1.shape[0]
    l1 = np.empty(n, dtype=np.float64)
    l2 = np.empty(n, dtype=np.float64)
    t1 = np.empty((n, 2), dtype=np.float64)
    t2 = np.empty((n, 2), dtype=np.float64)
    ip1 = np.empty(n, dtype=np.float64)
    ip2 = np.empty(n, dtype=np.float64)

    s = 0.0
    for i in range(n):
        x1 = v1[i, 0]
        y1 = v1[i, 1]
        x2 = v2[i, 0]
        y2 = v2[i, 1]

        n1 = np.sqrt(x1 * x1 + y1 * y1)
        n2 = np.sqrt(x2 * x2 + y2 * y2)
        l1[i] = n1
        l2[i] = n2

        tx1 = x1 / n1
        ty1 = y1 / n1
        tx2 = x2 / n2
        ty2 = y2 / n2

        t1[i, 0] = tx1
        t1[i, 1] = ty1
        t2[i, 0] = tx2
        t2[i, 1] = ty2

        a1 = tx1 * tau_1[i, 0] + ty1 * tau_1[i, 1]
        a2 = tx2 * tau_2[i, 0] + ty2 * tau_2[i, 1]
        ip1[i] = a1
        ip2[i] = a2
        s += a1 * a1 + a2 * a2

    E = 0.5 * (s / n)
    return l1, l2, t1, t2, ip1, ip2, E


@njit(cache=True, fastmath=False)
def theta_energy_core(rho, edgearc_x, edgearc_y, r):
    # rho: (Ne, 2)
    # edgearc_x, edgearc_y: (Ne, K)
    # r: (Ne,)
    n_edges = rho.shape[0]
    k = edgearc_x.shape[1]

    avg_d = np.empty(n_edges, dtype=np.float64)
    sumsq = 0.0

    for i in range(n_edges):
        rx = rho[i, 0]
        ry = rho[i, 1]
        rr = r[i]
        sdiff = 0.0
        ssq_row = 0.0

        for j in range(k):
            dx = rx - edgearc_x[i, j]
            dy = ry - edgearc_y[i, j]
            dmag = np.sqrt(dx * dx + dy * dy)
            diff = dmag - rr
            sdiff += diff
            ssq_row += diff * diff

        avg_d[i] = sdiff
        sumsq += ssq_row

    E = 0.5 * (sumsq / n_edges)
    return avg_d, E


def theta_radius_core_safe(dP, dT, b_theta, invalid_base):
    n_edges = dP.shape[0]
    r = np.zeros(n_edges, dtype=np.float64)
    ind_z = np.zeros(n_edges, dtype=np.bool_)
    invalid = invalid_base.copy()

    for i in range(n_edges):
        if invalid[i]:
            continue

        dp = dP[i]
        dt = dT[i]
        bt = b_theta[i]

        if not np.isfinite(dp) or not np.isfinite(dt) or not np.isfinite(bt):
            invalid[i] = True
            continue

        r_sq = (bt - (dp * dt)) / (dp * dp)
        if not np.isfinite(r_sq):
            invalid[i] = True
            continue

        if r_sq < 0.0:
            ind_z[i] = True
            r_sq = 0.0

        radius = np.sqrt(r_sq)
        if not np.isfinite(radius):
            invalid[i] = True
            continue

        r[i] = radius

    return r, ind_z, invalid


@njit(cache=True, fastmath=False)
def theta_energy_core_safe(rho, edgearc_x, edgearc_y, r, invalid):
    n_edges = rho.shape[0]
    k = edgearc_x.shape[1]

    avg_d = np.zeros(n_edges, dtype=np.float64)
    sumsq = 0.0
    invalid_count = 0

    for i in range(n_edges):
        if invalid[i]:
            invalid_count += 1
            continue

        rx = rho[i, 0]
        ry = rho[i, 1]
        rr = r[i]
        if not np.isfinite(rx) or not np.isfinite(ry) or not np.isfinite(rr):
            invalid_count += 1
            continue

        sdiff = 0.0
        ssq_row = 0.0
        edge_invalid = False

        for j in range(k):
            dx = rx - edgearc_x[i, j]
            dy = ry - edgearc_y[i, j]
            dmag_sq = dx * dx + dy * dy

            if not np.isfinite(dmag_sq):
                edge_invalid = True
                break

            dmag = np.sqrt(dmag_sq)
            diff = dmag - rr
            if not np.isfinite(diff):
                edge_invalid = True
                break

            sdiff += diff
            ssq_row += diff * diff

        if edge_invalid:
            invalid_count += 1
            continue

        avg_d[i] = sdiff
        sumsq += ssq_row

    E = 0.5 * (sumsq / n_edges)
    return avg_d, E, invalid_count


@njit(cache=True, fastmath=False)
def main_state_core(q, p, theta, c0, c1):
    # q: (Nc, 2)
    # p: (Nc,)
    # theta: (Nc,)
    # c0, c1: (Ne,)
    pc0 = p[c0]                                                  # (Ne,)
    pc1 = p[c1]                                                  # (Ne,)
    dP = pc0 - pc1                                               # (Ne,)
    dT = theta[c0] - theta[c1]                                   # (Ne,)
    dQ = q[c0, :] - q[c1, :]                                     # (Ne, 2)
    QL = np.sum(dQ * dQ, axis=1)                                 # (Ne,)
    q0 = q[c0, :]                                                # (Ne, 2)
    q1 = q[c1, :]                                                # (Ne, 2)
    rho = ((pc0[:, None] * q0) - (pc1[:, None] * q1)) / dP[:, None]  # (Ne, 2)
    r_sq = ((pc0 * pc1 * QL) - (dP * dT)) / (dP * dP)            # (Ne,)
    ind_z = r_sq <= 0.0                                          # (Ne,)
    r_sq = np.maximum(r_sq, 0.0)
    r = np.sqrt(r_sq)                                            # (Ne,)
    return pc0, pc1, dP, dT, dQ, QL, rho, r, ind_z


def main_state_core_safe(q, p, theta, c0, c1, dp_eps):
    n_edges = c0.shape[0]

    pc0 = p[c0]
    pc1 = p[c1]
    dP = pc0 - pc1
    dT = theta[c0] - theta[c1]
    dQ = q[c0, :] - q[c1, :]
    QL = np.sum(dQ * dQ, axis=1)

    rho = np.zeros((n_edges, 2), dtype=np.float64)
    r = np.zeros(n_edges, dtype=np.float64)
    ind_z = np.zeros(n_edges, dtype=np.bool_)
    invalid = np.zeros(n_edges, dtype=np.bool_)

    for i in range(n_edges):
        q0x = q[c0[i], 0]
        q0y = q[c0[i], 1]
        q1x = q[c1[i], 0]
        q1y = q[c1[i], 1]
        p0 = pc0[i]
        p1 = pc1[i]
        dp = dP[i]
        dt = dT[i]
        ql = QL[i]

        if (not np.isfinite(q0x) or not np.isfinite(q0y) or
                not np.isfinite(q1x) or not np.isfinite(q1y) or
                not np.isfinite(p0) or not np.isfinite(p1) or
                not np.isfinite(dp) or not np.isfinite(dt) or
                not np.isfinite(ql) or np.abs(dp) <= dp_eps):
            invalid[i] = True
            continue

        inv_dp = 1.0 / dp
        rho_x = ((p0 * q0x) - (p1 * q1x)) * inv_dp
        rho_y = ((p0 * q0y) - (p1 * q1y)) * inv_dp

        if not np.isfinite(rho_x) or not np.isfinite(rho_y):
            invalid[i] = True
            continue

        r_sq = ((p0 * p1 * ql) - (dp * dt)) * (inv_dp * inv_dp)
        if not np.isfinite(r_sq):
            invalid[i] = True
            continue

        if r_sq <= 0.0:
            ind_z[i] = True
            r_sq = 0.0

        radius = np.sqrt(r_sq)
        if not np.isfinite(radius):
            invalid[i] = True
            continue

        rho[i, 0] = rho_x
        rho[i, 1] = rho_y
        r[i] = radius

    return pc0, pc1, dP, dT, dQ, QL, rho, r, ind_z, invalid


@njit(cache=True, fastmath=False)
def nonlinear_constraint_core(pc0, pc1, dP, dT, QL):
    # all: (Ne,)
    return dP * dT - (pc0 * pc1 * QL)                            # (Ne,)


@njit(cache=True, fastmath=False)
def main_objective_core(rho, edgearc_x, edgearc_y, r):
    # rho: (Ne, 2)
    # edgearc_x, edgearc_y: (Ne, K)
    # r: (Ne,)
    n_edges = rho.shape[0]
    k = edgearc_x.shape[1]

    avg_d = np.empty(n_edges, dtype=np.float64)                  # (Ne,)
    dNormX = np.empty(n_edges, dtype=np.float64)                 # (Ne,)
    dNormY = np.empty(n_edges, dtype=np.float64)                 # (Ne,)

    sumsq = 0.0
    for i in range(n_edges):
        rx = rho[i, 0]
        ry = rho[i, 1]
        rr = r[i]

        sdiff = 0.0
        ssq = 0.0
        sx = 0.0
        sy = 0.0

        for j in range(k):
            dx = rx - edgearc_x[i, j]
            dy = ry - edgearc_y[i, j]
            dmag = np.sqrt(dx * dx + dy * dy)
            diff = dmag - rr

            sdiff += diff
            ssq += diff * diff

            inv = diff / dmag
            sx += dx * inv
            sy += dy * inv

        avg_d[i] = sdiff
        dNormX[i] = sx
        dNormY[i] = sy
        sumsq += ssq

    E = 0.5 * (sumsq / n_edges)
    return avg_d, dNormX, dNormY, E


@njit(cache=True, fastmath=False)
def main_objective_core_safe(rho, edgearc_x, edgearc_y, r, invalid):
    n_edges = rho.shape[0]
    k = edgearc_x.shape[1]

    avg_d = np.zeros(n_edges, dtype=np.float64)
    dNormX = np.zeros(n_edges, dtype=np.float64)
    dNormY = np.zeros(n_edges, dtype=np.float64)

    sumsq = 0.0
    invalid_count = 0

    for i in range(n_edges):
        if invalid[i]:
            invalid_count += 1
            continue

        rx = rho[i, 0]
        ry = rho[i, 1]
        rr = r[i]

        if not np.isfinite(rx) or not np.isfinite(ry) or not np.isfinite(rr):
            invalid_count += 1
            continue

        sdiff = 0.0
        ssq = 0.0
        sx = 0.0
        sy = 0.0
        edge_invalid = False

        for j in range(k):
            dx = rx - edgearc_x[i, j]
            dy = ry - edgearc_y[i, j]
            dmag_sq = dx * dx + dy * dy

            if not np.isfinite(dmag_sq):
                edge_invalid = True
                break

            dmag = np.sqrt(dmag_sq)
            diff = dmag - rr

            if not np.isfinite(diff):
                edge_invalid = True
                break

            sdiff += diff
            ssq += diff * diff

            if dmag > 0.0:
                inv = diff / dmag
                sx += dx * inv
                sy += dy * inv

        if edge_invalid:
            invalid_count += 1
            continue

        avg_d[i] = sdiff
        dNormX[i] = sx
        dNormY[i] = sy
        sumsq += ssq

    E = 0.5 * (sumsq / n_edges)
    return avg_d, dNormX, dNormY, E, invalid_count
