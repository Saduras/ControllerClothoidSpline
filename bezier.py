import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

def dotproduct(v1, v2):
       return sum((a*b) for a, b in zip(v1, v2))

def length(v):
       return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
       return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def normalize(v1):
       return v1/np.linalg.norm(v1)

def bezier(t, p1, p2, p3):
       return (1-t)*(1-t)*p1 + 2*(1-t)*t*p2 + t*t*p3

fig, axes = plt.subplots(1, 2)

ax1 = axes[0]
ax1.set(title='Bezier with clamping')
ax1.grid()
ax1.set_xlim(-1, 3)
ax1.set_ylim(-1, 3)
ax1.set_aspect('equal')

start = np.array([0,0])
end = np.array([3,0])

colors = ['olive', 'green', 'orange', 'red', 'purple', 'blue', 'pink']
directions = [[1, -0.6], [1,0.3], [1,0.6], [1,1], [1,2], [0.1,1], [-0.5, 2]]

# draw straight line 
ax1.plot([start[0], end[0]], [start[1], end[1]], color='black')

# calculate and draw smoothed lines
for direction, color in zip(directions, colors):
       direction = normalize(np.array(direction))

       alpha = angle(direction, normalize(end - start))
       alpha = max(-math.pi/4, min(alpha, math.pi/4))
       l = (length(end - start) / 2) / math.cos(alpha) 
       middle = start + direction * l

       ts = np.arange(0, 1.01, 0.01).reshape(101,1);
       bs = bezier(ts, start.reshape(1,2), middle.reshape(1,2), end.reshape(1,2)).T

       ax1.plot([start[0], direction[0]], [start[1], direction[1]], color=color)
       ax1.plot([start[0], middle[0], end[0]], [start[1], middle[1], end[1]], color=color, alpha=0.1)
       ax1.scatter([middle[0]], [middle[1]], color=color, marker='x')
       ax1.plot(bs[0], bs[1], color=color, linestyle='--')



ax2 = axes[1]
ax2.set(title='Bezier without clamping')
ax2.grid()
ax2.set_xlim(-1, 3)
ax2.set_ylim(-1, 3)
ax2.set_aspect('equal')

# draw straight line 
ax2.plot([start[0], end[0]], [start[1], end[1]], color='black')

# calculate and draw smoothed lines
for direction, color in zip(directions, colors):
       direction = normalize(np.array(direction))

       alpha = angle(direction, normalize(end - start))
       l = (length(end - start) / 2) / math.cos(alpha) 
       middle = start + direction * l

       ts = np.arange(0, 1.01, 0.01).reshape(101,1);
       bs = bezier(ts, start.reshape(1,2), middle.reshape(1,2), end.reshape(1,2)).T

       ax2.plot([start[0], direction[0]], [start[1], direction[1]], color=color)
       ax2.plot([start[0], middle[0], end[0]], [start[1], middle[1], end[1]], color=color, alpha=0.1)
       ax2.scatter([middle[0]], [middle[1]], color=color, marker='x')
       ax2.plot(bs[0], bs[1], color=color, linestyle='--')



# direction = normalize(bs.T[4]- bs.T[3])
# start = np.array(bs.T[3])
# end = np.array([0,-4])

# alpha = angle(direction, normalize(end - start))
# # length = math.cos(angle) * length(end - start)
# c = length(end - start)/2
# l = math.sqrt(c*c + c*c)
# middle = start + direction * l
# middle = np.array([1.3, 0])

# ts = np.arange(0, 1.1, 0.1).reshape(11,1);
# bs = bezier(ts, start.reshape(1,2), middle.reshape(1,2), end.reshape(1,2)).T

# ax.plot([start[0], end[0]], [start[1], end[1]], color='black')
# ax.plot([start[0], middle[0], end[0]], [start[1], middle[1], end[1]], color='cyan')
# ax.plot(bs[0], bs[1], color='yellow')

plt.gca().set_aspect('equal')

fig.savefig("test.png")
plt.show()