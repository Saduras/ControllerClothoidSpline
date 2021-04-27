import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import vg
from scipy.special import fresnel
from scipy import optimize


# Base on the paper "An Improved Euler Spiral Algorithm for Shape Completion", D. J. Walton and D. S. Meek (2008)
# https://www.researchgate.net/publication/4352650_An_Improved_Euler_Spiral_Algorithm_for_Shape_Completion


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
    S, C = fresnel(arclength(alpha))
    return np.sign(theta) * C, np.sign(theta) * S


def fitEuler(point1, tangent, d, phi1, phi2, tolerance, iterationLimit, reflectFlag):
    print("TODO implement")
    return np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), 0.0, 0.0, 0.0, 0, True


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


# testDegenerate()
# testAmbigous()
# testStraightLine()
testCircular()
