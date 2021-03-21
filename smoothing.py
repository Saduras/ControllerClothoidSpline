import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial import geometric_slerp

def dotproduct(v1, v2):
       return sum((a*b) for a, b in zip(v1, v2))

def length(v):
       return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
       return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def normalize(v1):
       return v1/np.linalg.norm(v1)


start = np.array([0,0])
start_direction = normalize(np.array([0,-1]))

end = np.array([2,0])

# rotate by 90 degrees towards path
mid_dir = normalize(np.array([start_direction[1], -start_direction[0]]))
theta = angle(mid_dir, normalize(end-start))
radius = (length(end-start)/2) / math.cos(theta)

mid = start + mid_dir * radius

# hypothenuse = adjacent / cos A
alpha = angle(start_direction, end - start)
distance = (length(end - start)/2) / math.cos(alpha)

control_point = start + start_direction * distance
end_direction = normalize(end - control_point)

count = 36
beta = angle(start_direction, end_direction)
direction = start_direction

inc = 2*math.pi/(count - 2)
ts = np.arange(0, 2*math.pi + 2*inc, inc)
# directions = geometric_slerp(start_direction, end_direction, ts)

points = []
for i in range(0, count-1):
    points.append(mid + radius * np.array([math.cos(ts[i]), math.sin(ts[i])]))
points = np.array(points)

fig, ax = plt.subplots()

# Draw directions
ax.plot([start[0], start[0] + start_direction[0]], [start[1], start[1] + start_direction[1]], color='green')
ax.plot([end[0], end[0] + end_direction[0]], [end[1], end[1] + end_direction[1]], color='green')

ax.scatter(mid[0], mid[1], color='orange')
ax.plot(points[...,0], points[...,1], color='orange')

ax.set(title='Circle')
ax.grid()
plt.xlim(-2, 8)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal')

fig.savefig("test.png")
plt.show()
