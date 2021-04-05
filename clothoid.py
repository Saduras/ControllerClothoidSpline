import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import fresnel
from scipy import optimize


def dot(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dot(v, v))


def angle(v1, v2):
    return math.acos(dot(v1, v2) / (length(v1) * length(v2)))


def normalize(v1):
    return v1/np.linalg.norm(v1)


def refelection_matrix(direction):
    theta = angle(direction, np.array([1, 0]))
    return np.array([[math.cos(2*theta), math.sin(2*theta)], [math.sin(2*theta), -math.cos(2*theta)]])


def fresnelIntegrals(t):
    r = (0.506 * t + 1.0) / (1.79 * t**2 + 2.045 * t + math.sqrt(2))
    a = 1 / (0.803 * t**3 + 1.886 * t**2 + 2.524 * t + 2)
    C = 0.5 - r * np.sin(0.5 * math.pi * (a - t**2))
    S = 0.5 - r * np.cos(0.5 * math.pi * (a - t**2))
    return (C, S)


def plotAngle(pos, alpha, length=0.3, color='red'):
    p0 = pos + np.array([1, 0]) * length
    p1 = pos + np.array([math.cos(alpha), math.sin(alpha)]) * length

    ts = np.arange(0, alpha, 0.01)
    xs = pos[0] + np.cos(ts) * length/2
    ys = pos[1] + np.sin(ts) * length / 2

    plt.plot([pos[0], p0[0]], [pos[1], p0[1]], color=color)
    plt.plot([pos[0], p1[0]], [pos[1], p1[1]], color=color)
    plt.plot(xs, ys, color=color)


def arclength(alpha):
    return math.sqrt(2 * alpha / math.pi)

# Based on A controlled clothoid spline
# http://www.lara.prd.fr/_media/users/franciscogarcia/a_controlled_clothoid_spline.pdf


def symmetric_blend(p0, p0_dir, p1):
    # this alpha is alpha/2 from the paper
    alpha = angle(p0_dir, p1 - p0)

    # length of vector from p0 to control point cp
    g = (length(p1 - p0) / 2) / math.cos(alpha)
    # control point
    cp = p0 + p0_dir * g

    # paper
    # C_p(theta) = 1/sqrt(2pi) int cos(u)/sqrt(u) du
    # theta = 1/2 pi * t^2
    # C_p((pi t^2)/2) = C_s(t)
    # t = sqrt((2 * theta)/pi)
    # C_p(theta) = C_s(sqrt((2 * theta)/pi)

    # scipy
    # C_s(z) = int cos((pi t^2)/2) dt

    # solve for a
    S, C = fresnel(arclength(alpha))
    a = g / (C + S * math.tan(alpha))

    # calculate x and y values
    ts = np.arange(0, arclength(alpha), 0.01)
    Ss, Cs = fresnel(ts)
    xs = a * Cs
    ys = a * Ss

    # reflect arc
    mid = (p1 - p0)/2
    ref_mat = refelection_matrix(mid - cp)
    ref_pts = p1.reshape(2, 1) + ref_mat @ np.array([xs, ys])
    ref_pts = np.flip(ref_pts, axis=1)

    xs = np.append(xs, ref_pts[0, :])
    ys = np.append(ys, ref_pts[1, :])

    return xs, ys, cp


def unsymmentric_blend(p0, v, p1):
    g = length(v - p0)
    h = length(v - p1)

    if (g < h):
        # flip p0 and p1
        tmp = p0
        p0 = p1
        p1 = tmp
        g = length(v - p0)
        h = length(v - p1)

    t0 = normalize(v - p0)
    t1 = normalize(v - p1)

    n0 = np.array([-t0[1], t0[0]])
    n1 = np.array([t1[1], -t1[0]])

    k = g / h
    alpha = math.pi - angle(t0, p1 - p0)

    S_a, C_a = fresnel(arclength(alpha))
    left = (k + math.cos(alpha)) / math.sin(alpha)
    right = C_a / S_a
    print(left, '<', right)
    if(left > right):
        raise ValueError('Invalid choice of v!')

    def f(theta):
        S_t, C_t = fresnel(arclength(theta))
        S_at, C_at = fresnel(arclength(alpha - theta))
        result = math.sqrt(theta) * (C_t * math.sin(alpha)
                                     - S_t * (k + math.cos(alpha)))
        result += math.sqrt(alpha - theta) * (S_at *
                                              (1 + k * math.cos(alpha))
                                              - k * C_at * math.sin(alpha))
        return result

    theta0 = optimize.bisect(f, 0, alpha)

    S_t0, C_t0 = fresnel(arclength(theta0))
    S_at0, C_at0 = fresnel(arclength(alpha - theta0))
    # Formular from paper
    # a0 * S_t0 + a0 * sqrt( (alpha - theta0)/theta0 ) * C_at0 * sin(alpha) - a0 * sqrt( (alpha - theta0)/theta0 ) * S_at0 * cos(alpha) = h * sin(alpha)
    # solve for a0
    # S_t0 + sqrt( (alpha - theta0)/theta0 ) * C_at0 * sin(alpha) - sqrt( (alpha - theta0)/theta0 ) * S_at0 * cos(alpha) = h * sin(alpha) / a0
    # (S_t0 + sqrt( (alpha - theta0)/theta0 ) * C_at0 * sin(alpha) - sqrt( (alpha - theta0)/theta0 ) * S_at0 * cos(alpha) ) / h * sin(alpha) = 1 / a0
    # a0 = h * sin(alpha) / (S_t0 + sqrt( (alpha - theta0)/theta0 ) * C_at0 * sin(alpha) - sqrt( (alpha - theta0)/theta0 ) * S_at0 * cos(alpha) )
    a0 = h * math.sin(alpha) / (S_t0 + math.sqrt((alpha - theta0)/theta0) * C_at0 *
                                math.sin(alpha) - math.sqrt((alpha - theta0) / theta0) * S_at0 * math.cos(alpha))
    a1 = a0 * math.sqrt((alpha - theta0) / theta0)

    # calculate x and y values
    ts = np.arange(0, arclength(theta0), 0.01)
    Ss, Cs = fresnel(ts)
    # xs = p0[0] + a0 * Cs
    # ys = p0[1] + a0 * Ss
    pts0 = np.reshape(p0, (2, 1)) + a0 * np.reshape(t0, (2, 1)) * np.reshape(Cs, (1, -1)) + \
        a0 * np.reshape(n0, (2, 1)) * np.reshape(Ss, (1, -1))

    ts = np.arange(0, arclength(alpha - theta0), 0.01)
    Ss, Cs = fresnel(ts)

    pts1 = np.reshape(p1, (2, 1)) + a1 * np.reshape(t1, (2, 1)) * np.reshape(Cs, (1, -1)) + \
        a1 * np.reshape(n1, (2, 1)) * np.reshape(Ss, (1, -1))
    pts1 = np.flip(pts1, axis=1)

    xs = np.append(pts0[0, :], pts1[0, :])
    ys = np.append(pts0[1, :], pts1[1, :])

    P = p0 + a0 * C_t0 * t0 + a0 * S_t0 * n0
    plt.scatter(P[0], P[1], color='red')

    P = p1 + a1 * C_at0 * t1 + a1 * S_at0 * n1
    plt.scatter(P[0], P[1], color='red')

    plotAngle(v, alpha)
    plotAngle(p0 + t0 * g/2, theta0, color='green')
    plotAngle(p1 + t1 * h/2, (alpha - theta0), color='blue')

    return xs, ys


p0 = np.array([0, 0])
p0_dir = np.array([1, 0])
p1 = np.array([2, 2])

# xs, ys, cp = symmetric_blend(p0, p0_dir, p1)


cp = p0_dir * 3
xs, ys = unsymmentric_blend(p0, cp, p1)

mid = (p1 - p0)/2

plt.plot(xs, ys)
# plt.plot([mid[0], cp[0]], [mid[1], cp[1]], color='green')
plt.scatter([p0[0], cp[0], p1[0]], [p0[1], cp[1], p1[1]])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('')
plt.gca().set_aspect('equal')
plt.show()
