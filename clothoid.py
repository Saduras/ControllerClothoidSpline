import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import fresnel


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
    arclength = math.sqrt(2 * alpha / math.pi)
    S, C = fresnel(arclength)
    a = g / (C + S * math.tan(alpha))

    # calculate x and y values
    ts = np.arange(0, arclength, 0.01)
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


p0 = np.array([0, 0])
p0_dir = np.array([1, 0])
p1 = np.array([1, 2.5])

xs, ys, cp = symmetric_blend(p0, p0_dir, p1)

mid = (p1 - p0)/2

plt.plot(xs, ys)
plt.plot([mid[0], cp[0]], [mid[1], cp[1]], color='green')
plt.scatter([p0[0], cp[0], p1[0]], [p0[1], cp[1], p1[1]])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('')
plt.gca().set_aspect('equal')
plt.show()
