import matplotlib.pyplot as plt
from pathlib import Path
from CubiCasa5k import House
import cv2

room_channel = 21

# p = Path("~/Downloads/cubicasa5k/high_quality/17")
p = Path("~/Downloads/cubicasa5k/high_quality_architectural/2090/")

svg = p / "model.svg"
img = p / "F1_scaled.png"

fplan = cv2.imread(img)
fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
height, width, nchannel = fplan.shape

sample = House(p, height, width)

room = sample['label'][room_channel]

fig = plt.figure(size=(20, 12))
ax = fig.subplots()
ax.set_axis_off()

im = ax.imshow(
    room
)
ax.set_title("ROOM")

plt.show()
