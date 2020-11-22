from time import time
import numpy as np
import matplotlib.path as Path

# Ray tracing
def ray_tracing_method(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

lenpoly = 4
# polygon = [[np.sin(x)+0.5,np.cos(x)+0.5] for x in np.linspace(0,2*np.pi,lenpoly)[:-1]]
# polygon = [((np.sin(x)+0.5)**2, (np.cos(x)+0.5)**2) 
#            for x in np.linspace(0, 1000, lenpoly)[:-1]]
polygon = [(20,145), (12,100), (76,86), (90,90), (100, 120), (32,190)]

print(polygon)

M = 10000

check = []
for i in range(M):
    # x,y = np.random.random(), np.random.random()
    x,y = np.random.randint(300), np.random.randint(300)
    check.append((x, y))

start_time = time()
# Ray tracing
ins1 = []
for i in check:
    # x,y = np.random.random(), np.random.random()
    inside1 = ray_tracing_method(i[0], i[1], polygon)
    ins1.append(inside1)
# print(ray_tracing_method(21, 146, polygon))
# print(ray_tracing_method(21, 150, polygon))
print("Ray Tracing Elapsed time: " + str(time()-start_time))

# Matplotlib mplPath
start_time = time()
path = Path.Path(polygon)
ins2 = [] 
for i in check:
    # x,y = np.random.random(), np.random.random()
    # inside2 = path.contains_points([[i[0], i[1]]])
    inside2 = path.contains_point((i[0], i[1]))
    print(inside2)
    ins2.append(inside2)
# print(path.contains_points([[21, 146]]))
# print(path.contains_points([[21, 150]]))
print("Matplotlib contains_points Elapsed time: " + str(time()-start_time))

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

start_time = time()
ins3 = []
for i in check:
    # x, y = np.random.random(), np.random.random()
    point = Point(i[0], i[1])
    shapely_polygon = Polygon(polygon)
    inside3 = shapely_polygon.contains(point)
    ins3.append(inside3)
# print(shapely_polygon.contains(Point(21,146)))
# print(shapely_polygon.contains(Point(21,150)))
print('shapely elapsed time: ' + str(time()-start_time))


import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

verts = [
   (0., 0.),  # left, bottom
   (0., 1.),  # left, top
   (1., 1.),  # right, top
   (1., 0.),  # right, bottom
   (0., 0.),  # ignored
]

print(verts)

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]


# print(ins1)
# print(ins2)
# print(ins3)
print('Is all same : ', ins1==ins2==ins3)

# print('ins1')
# for x in range(len(ins1)):
#     if ins1[x] != ins2[x]:
#         print('ins2: ', check[x])
#     if ins1[x] != ins3[x]:
#         print('ins3: ', check[x])
print('ins2')
for x in range(len(ins2)):
    if ins2[x] != ins1[x]:
        print('ins1: ', check[x])
    if ins2[x] != ins3[x]:
        print('ins3: ', check[x])
# print('ins3')
# for x in range(len(ins3)):
#     if ins3[x] != ins2[x]:
#         print('ins2: ', check[x])
#     if ins3[x] != ins1[x]:
#         print('ins1: ', check[x])

# path = Path(polygon, codes)
# path = Path(polygon, closed=True)
path = Path(polygon, closed=False)

fig, ax = plt.subplots()
patch = patches.PathPatch(path, facecolor='orange', lw=2)
ax.add_patch(patch)
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
plt.show()

