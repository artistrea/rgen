import matplotlib.pyplot as plt
from pathlib import Path
from CubiCasa5k.floortrans.loaders.house import House
import cv2
import numpy as np
from skimage.draw import polygon

room_channel = 21

# p = Path("/home/artistreak/Downloads/cubicasa5k/high_quality/231")
p = Path("/home/artistreak/Downloads/cubicasa5k/high_quality_architectural/2090")
print("p.exists()", p.exists())
svg = p / "model.svg"
img = p / "F1_scaled.png"

fplan = cv2.imread(img)
fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
height, width, nchannel = fplan.shape

sample = House(str(svg.resolve()), height, width)

rsh_walls = np.zeros((sample.width, sample.height, 1))
walls = sample.wall_objs
wall_12 = sample.find_wall_by_id(12, sample.wall_objs)
print("wall_12.id",wall_12.id)
print("wall_12.Y", wall_12.Y)
print("wall_12.X", wall_12.X)

# print("WALLL")
for i, wall in enumerate(walls):
    # print("wall.end_points", wall.end_points)
    # print("wall.__dict__.keys()", wall.__dict__.keys())
    # print("wall.end_points", wall.end_points)
    if i == 11:
        print("wall.id",wall.id)
        print("wall.Y", wall.Y)
        print("wall.X", wall.X)
    rr, cc = polygon(wall.Y, wall.X)
    if rr.shape != wall.rr.shape or np.sum(rr != wall.rr):
        print("rr i", i)
    if cc.shape != wall.cc.shape or np.sum(cc != wall.cc):
        print("cc i", i)
    # print("wall.cc", wall.cc)
    # print("wall.rr", wall.rr)

    rsh_walls[wall.cc, wall.rr, 0] = 1
    # rsh_walls[wall.cc, wall.rr, 0] = i + 1

    # NOTE: min and max width are equal
    # didn't really dig deep enough to check which is true
    # print("wall.min_width", wall.min_width)
    # exit()

rsh_doors = np.zeros((width, height, 1))
doors = sample.door_objs
# print("################")
# print("################ DOOR")
for i, door in enumerate(doors):
    # print("door.__dict__.keys()", door.__dict__.keys())
    # print("door.end_points", door.end_points)
    # print("door.X", door.X)
    # print("door.Y", door.Y)
    # print("door.cc", door.cc)
    # print("door.rr", door.rr)
    break
    rsh_walls[door.cc, door.rr, 0] = 2
    # rsh_walls[door.cc, door.rr, 0] = i + 1 + len(walls)

    # NOTE: min and max width are equal
    # didn't really dig deep enough to check which is true
    # print("door.min_width", door.min_width)
    # exit()

rsh_windows = np.zeros((width, height, 1))
windows = sample.window_objs
# print("################")
# print("################ WINDOW")
for i, window in enumerate(windows):
    # print("window.__dict__.keys()", window.__dict__.keys())
    # print("window.end_points", window.end_points)
    # print("window.X", window.X)
    # print("window.Y", window.Y)
    # print("window.cc", window.cc)
    # print("window.rr", window.rr)
    break
    # print("window.end_points", window.end_points)
    # rsh_walls[window.cc, window.rr, 0] = i + 1 + len(walls) + len(doors)
    rsh_walls[window.cc, window.rr, 0] = 3

    # NOTE: min and max width are equal
    # didn't really dig deep enough to check which is true
    # print("window.min_width", window.min_width)
    # exit()

fig = plt.figure()
ax = fig.subplots()
ax.set_axis_off()
# rsh_walls
im = ax.imshow(
    rsh_walls
)
ax.set_title("ROOM")

fig = plt.figure()
ax = fig.subplots()
ax.set_axis_off()
# rsh_walls
im = ax.imshow(
    sample.wall_ids
)
ax.set_title("ROOM")

# print("np.unique(rsh_walls)", np.unique(rsh_walls))

# cv2.imshow("room cv2", rsh_walls)
# plt.imshow()
plt.show()
