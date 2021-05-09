import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import vg
from scipy.special import fresnel
from scipy import optimize


# Base on the paper "An Improved Euler Spiral Algorithm for Shape Completion", D. J. Walton and D. S. Meek (2008)
# https://www.researchgate.net/publication/4352650_An_Improved_Euler_Spiral_Algorithm_for_Shape_Completion

def arclength(alpha):
    return math.sqrt(2 * alpha / math.pi)


def angle(v1, v2):
    if v1.shape[0] == 2:
        v1 = np.append(v1, 0)
    if v2.shape[0] == 2:
        v2 = np.append(v2, 0)
    return vg.signed_angle(v1, v2, look=vg.basis.z, units='rad')


def rotate(v, alpha):
    w = np.array([0, 0])
    w[0] = v[0] * math.cos(alpha) - v[1] * math.sin(alpha)
    w[1] = v[0] * math.sin(alpha) + v[1] * math.cos(alpha)
    return w


def reverse(phi1, phi2, point1, D):
    D = -D
    point1 = point1 - D
    phi = phi1
    phi1 = -phi2
    phi2 = -phi
    return phi1, phi2, point1, D


def reflect(phi1, phi2):
    reflectFlag = True
    phi1 = -phi1
    phi2 = -phi2
    return reflectFlag, phi1, phi2


def fresnel2(theta):
    S, C = fresnel(arclength(theta))
    return np.sign(theta) * S, np.sign(theta) * C


def eval(theta, phi1, phi2, sign):
    S1, C1 = fresnel2(theta)
    S2, C2 = fresnel2(theta + phi1 + phi2)
    c = math.cos(theta + phi1)
    s = math.sin(theta + phi1)

    f = math.sqrt(2 * math.pi) * (c * (S2 - sign * S1) - s * (C2 - sign * C1))
    f_derv = math.sin(phi2)/(math.sqrt(theta + phi1 + phi2)) \
        + sign * math.sin(phi1)/(math.sqrt(theta)) \
        - math.sqrt(2 * math.pi)(s * (S2 - sign * S1) + c * (C2 - sign * C1))
    return f, f_derv


def solve(a, b, phi1, phi2, sign, tolerance, iterationLimit):
    theta = 0.5 * (a + b)
    fa, fa_deriv = eval(a, phi1, phi2, sign)
    fb, fb_deriv = eval(b, phi1, phi2, sign)
    f, f_deriv = eval(theta, phi1, phi2, sign)

    error = b - a
    iteration = 0
    while error > tolerance and iteration < iterationLimit:
        iteration += 1
        newtonFail = True

        if abs(f_deriv) > tolerance:
            theta_iter - theta - f/f_deriv
            delta = abs(theta - theta_iter)
            if theta_iter > a and theta_iter < b and delta < 0.5 * error:
                newtonFail = False
                theta = theta_iter
                error = delta

        if newtonFail:
            if fa * f < 0:
                b = theta
                fb = f
            else:
                a = theta
                fa = f
            theta = 0.5 * (a + b)
            error = b - a

        f, f_deriv = eval(theta, phi1, phi2, sign)
    # end while
    failFlag = iteration >= iterationLimit
    return theta, iteration, failFlag


def fitEuler(point1, tangent, d, phi1, phi2, tolerance, iterationLimit, reflectFlag):
    theta = 0
    sign = 1
    t1 = 0
    t2 = arclength(phi1 + phi2)
    S, C = fresnel(t2)
    h = S * math.cos(phi1) - C * math.sin(phi2)

    if phi1 > 0 and h <= 0:
        # C-shaped according to theorem 1
        if h > tolerance:
            # solution is theta = 0
            failFlag = False
        else:
            lam = (1 - math.cos(phi1))/(1 - math.cos(phi2))
            theta0 = (lam*lam)/(1 - lam*lam) * (phi1 + phi2)
            theta, iteration, failFlag = solve(
                0, theta0, phi1, phi2, sign, tolerance, iterationLimit)
    else:
        # S-shaped according to therems 2 and 3
        sign = -1
        theta0 = max(0, -phi1)
        theta1 = 0.5 * math.pi - phi1
        theta, iteration, failFlag = solve(
            theta0, theta1, phi1, phi2, sign, tolerance, iterationLimit)

    t1 = sign * arclength(theta)
    t2 = arclength(theta + phi + phi2)
    S1, C2 = fresnel(t1)
    S2, C2 = fresnel(t2)
    phi = phi1 + theta
    a = d/((S2 - S1) * sin(phi) + (C2 - C1) * cos(phi))

    if reflectFlag:
        T0 = rotate(tangent, phi)
        N0 = rotate(T0, -0.5 * math.pi)
    else:
        T0 = rotate(tangent, -phi)
        N0 = rotate(T0, 0.5 * phi)

    point0 = point1 - a(C1 * T0 + S1 * N0)
    return point0, T0, N0, a, t1, t2, iteration, failFlag


def completeShape(point1, tangent1, point2, tangent2, tolerance, iterationLimit):
    D = point2 - point1
    d = np.linalg.norm(D)
    if d <= tolerance:
        # degenerate case
        print("Degenerate case. Distance between the points is smaller than the tolerance.")
        return np.stack([point1, point2])
    else:
        phi1 = angle(tangent1, D)
        phi2 = angle(D, tangent2)
        if abs(phi1) > abs(phi2):
            phi1, phi2, point1, D = reverse(phi1, phi2, point1, D)

        reflectFlag = False
        if phi2 < 0:
            reflectFlag, phi1, phi2 = reflect(phi1, phi2)

        if (phi1 == 0 and phi2 == math.pi) \
                or (phi1 == math.pi and phi2 == 0) \
                or (phi1 == math.pi and phi2 == math.pi):
            # Ambiguous case; perturb tangent1 or tangent2 and repeat
            print("Ambiguous case; perturb tangent1 or tangent2 and repeat")
            return np.array([])
        elif (abs(phi1) <= tolerance) and (abs(phi2) <= tolerance):
            # draw a straight line segment from point1 to point2
            print("Straight line case.")
            return np.stack([point1, point2])
        elif abs(phi1 - phi2) <= tolerance:
            # draw a circular arc from point1 to point 2 whith
            # tangent vector tangent1 at point1
            # radius r=1/2 * d / |sin(phi1)|
            # and center = point1 + r * rotate(tangent1, 1/2 pi)
            print("Circular arc case.")
            radius = 1 / 2 * d / abs(math.sin(phi1))
            center = point1 + radius * \
                rotate(tangent1, np.sign(angle(tangent1, D)) * 1 / 2 * math.pi)
            alpha = angle(point1 - center, point2 - center)
            ts = np.arange(0.0, alpha + 0.1, 0.1)
            return np.stack(map(lambda beta: center + radius * np.array([math.cos(beta), math.sin(beta)]), ts))
        else:
            print("Euler spiral case.")
            point0, tangent0, normal0, a, t1, t2, iteration, failFlag = fitEuler(
                point1, D / d, d, phi1, phi2, tolerance, iterationLimit, reflectFlag)
            if failFlag:
                # repeat with larger values for tolerance and/or iterationLimit
                print(
                    "Failed. Repeat with larger values for tolerance and/or iterationLimit.")
                return np.array([])
            else:
                # draw Euler spiral from t1 to t2 with inflection point at point0,
                # unit tangent vector tangent0 at point0, unit normal vector normal0 at point0,
                # and scaling factor a.
                print("Draw Euler spiral.")


def testDegenerate():
    point1 = np.array([0, 0])
    tangent1 = np.array([1, 0])
    point2 = point1
    tangent2 = tangent1

    points = completeShape(point1, tangent1, point2, tangent2,
                           tolerance=0.1, iterationLimit=100)
    print(points)


def testAmbigous():
    point1 = np.array([0, 0])
    tangent1 = np.array([1, 0])
    point2 = np.array([2, 0])
    tangent2 = np.array([-1, 0])

    points = completeShape(point1, tangent1, point2, tangent2,
                           tolerance=0.1, iterationLimit=100)
    print(points)


def testStraightLine():
    point1 = np.array([0, 0])
    tangent1 = np.array([1, 0])
    point2 = np.array([2, 0])
    tangent2 = tangent1

    points = completeShape(point1, tangent1, point2, tangent2,
                           tolerance=0.1, iterationLimit=100)
    print(points)


def testCircular():
    point1 = np.array([0, 0])
    tangent1 = np.array([0, 1])
    point2 = np.array([-2, 0])
    tangent2 = np.array([0, -1])

    points = completeShape(point1, tangent1, point2, tangent2,
                           tolerance=0.1, iterationLimit=100)

    plt.plot(points[:, 0], points[:, 1])
    plt.gca().set_aspect('equal')
    plt.show()


def testEulerSpiral():
    point1 = np.array([0, 0])
    tangent1 = np.array([0, 1])
    point2 = np.array([-2, 0])
    tangent2 = np.array([-1, -1])

    points = completeShape(point1, tangent1, point2, tangent2,
                           tolerance=0.1, iterationLimit=100)

    plt.plot(points[:, 0], points[:, 1])
    plt.gca().set_aspect('equal')
    plt.show()


# testDegenerate()
# testAmbigous()
# testStraightLine()
# testCircular()
testEulerSpiral()
