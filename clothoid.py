import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import fresnel

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def normalize(v1):
    return v1/np.linalg.norm(v1)

def refelection_matrix(direction):
    theta = angle(direction, np.array([1, 0]))
    return np.array([[math.cos(2*theta), math.sin(2*theta)], [math.sin(2*theta), -math.cos(2*theta)]])

def fresnelIntegrals_2(t):

    x = np.abs(t) / math.sqrt(math.pi / 2)
    arg = math.pi * x**2 / 2
    s = np.sin(arg)
    c = np.sin(arg)

    if x > 4.4:
        x4 = x**4
        x3 = x**3
        x2 = 0.10132 - 0.154 / x4
        x1 = 0.3183099 - 0.0969 / x4

        cfrenl = 0.5 + x1 * s / x - x2 * c / x3
        sfrenl = 0.5 - x1 * c / x - x2 * s / x3

        if t < 0:
            cfrenl *= -1
            sfrenl *= -1
    else:
        a0 = x
        sum = x
        xmul = -((math.pi / 2)**2) * x**4
        an = a0
        nend = (x + 1) * 20

        for n in range(0, int(nend)):
            xnenn = (2 * n + 1) * (2 * n + 2) * (4 * n + 5)
            an1 = an * (4 * n + 1) * xmul / xnenn
            sum += an1
            an = an1

        cfrenl = sum

        a0 = (math.pi / 6) * x**3
        sum = a0
        an = a0
        nend = (x + 1) * 20

        for n in range(0, int(nend)):
            xnenn = (2 * n + 2) * (2 * n + 3) * (4 * n + 7)
            an1 = an * (4 * n + 3) * xmul / xnenn
            sum += an1
            an = an1
        

        sfrenl = sum

        if t < 0:
            cfrenl *= -1
            sfrenl *= -1

    return cfrenl, sfrenl

# void gGetFresnelIntegrals(float inT, rFloat outC, rFloat outS)
# {
# 	float r = (0.506f * inT + 1.0f) / (1.79f * gSquared(inT) + 2.045f*inT + gSqrt(2.0f));
# 	float a = 1.0f / (0.803f * (inT * inT * inT) + 1.886f * gSquared(inT) + 2.524f*inT + 2.0f);
# 	outC = 0.5f - r * gSin(0.5f*M_PI * (a - gSquared(inT)));
# 	outS = 0.5f - r * gCos(0.5f*M_PI * (a - gSquared(inT)));
# }
def fresnelIntegrals(t):
    r = (0.506 * t + 1.0) / (1.79 * t**2 + 2.045 * t + math.sqrt(2))
    a = 1 / (0.803 * t**3 + 1.886 * t**2 + 2.524 * t + 2)
    C = 0.5 - r * np.sin(0.5 * math.pi * (a - t**2))
    S = 0.5 - r * np.cos(0.5 * math.pi * (a - t**2))
    return (C, S)

# def clothoid_ode_rhs(state, s, kappa0, kappa1):
#     x, y, theta = state[0], state[1], state[2]
#     return np.array([np.cos(theta), np.sin(theta), kappa0 + kappa1*s])
# def eval_clothoid(x0,y0,theta0, kappa0, kappa1, s):
#     return odeint(clothoid_ode_rhs, np.array([x0,y0,theta0]), s, (kappa0, kappa1))

    
# x0,y0,theta0 = 0,0,0
# L = 10
# kappa0, kappa1 = 0, 0.2
# s = np.linspace(0, L, 1000)

# sol = eval_clothoid(x0, y0, theta0, kappa0, kappa1, s)

# xs, ys, thetas = sol[:,0], sol[:,1], sol[:,2] 
# plt.plot(xs, ys, lw=3);
# plt.xlabel('x (m)');
# plt.ylabel('y (m)');
# plt.title('An awesome Clothoid');
# plt.show()

# Based on A controlled clothoid spline
# http://www.lara.prd.fr/_media/users/franciscogarcia/a_controlled_clothoid_spline.pdf

p0 = np.array([0, 0])
p0_dir = np.array([1, 0])
p1 = np.array([1, 2.5])


# alpha = angle(cp - p0, p1 - cp)
# g = length(cp - p0)

# this alpha is alpha/2 from the paper
alpha = angle(p0_dir, p1 - p0)
print(math.degrees(alpha))
g = (length(p1 - p0)/2) / math.cos(alpha)

cp = p0 + p0_dir * g


# a * C(alpha) + a * S(alpha) tan(alpha) = g
# a =  g / (C(alpha) + S(alpha) * tan(alpha))
C, S = fresnelIntegrals(alpha)
C2, S2 = fresnelIntegrals_2(alpha)
S3, C3 = fresnel(alpha)
S3 /= (math.sqrt(2 * math.pi))
C3 /= (math.sqrt(2 * math.pi))
a = g / (C3 + S3 * math.tan(alpha))

ts = np.arange(0, 1.01, 0.01)
Cs, Ss = fresnelIntegrals(ts)
xs = a * Cs
ys = a * Ss

# xs = np.append(xs, p1[0])
# ys = np.append(ys, p1[1])

mid = (p1 - p0)/2

ref_mat = refelection_matrix(mid - cp)
ref_pts = p1.reshape(2,1) + ref_mat @ np.array([xs, ys])


plt.plot(xs, ys)
plt.plot(ref_pts[0,:], ref_pts[1,:])
plt.plot([mid[0], cp[0]], [mid[1], cp[1]], color='green')
plt.scatter([p0[0], cp[0], p1[0]], [p0[1], cp[1], p1[1]])
plt.xlabel('x (m)');
plt.ylabel('y (m)');
plt.title('');
plt.gca().set_aspect('equal')
plt.show()