import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to file of extracted sources")
parser.add_argument("-s", "--shape", help="Data shape in format nx,ny")
args = parser.parse_args()

infile = args.input
shape = (int(args.shape.split(",")[0]), int(args.shape.split(",")[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

sources = np.load(infile, allow_pickle=True)
source_keys = sources.dtype.names

contour_keys = source_keys[1::5]

length = len(sources['EB0000 Flux'])

fig,ax = plt.subplots(figsize = (5,5))

# Set up inital polygon/contours:
artists = []
for k in contour_keys:
    if sources[k][0] == 0:
        arti = ax.fill([0],[0], c='C1')
        artists.append(arti)
    else:
        contour = sources[k][0]
        x = [c[0] for c in contour]
        y = [c[1] for c in contour]
        arti = ax.fill(x, y, c='C1')
        artists.append(arti)
ax.set_xlim(0, shape[0])
ax.set_ylim(0, shape[1])

def update(frame, sources, keys, artists, axes):
    e_arr = np.zeros((1,2))
    ax.set_title("Frame No.: " + str(frame))
    for i in range(len(keys)):
        if sources[keys[i]][frame] == 0:
            artists[i][0].set_xy(e_arr)
        else:
            contour = sources[keys[i]][frame]
            x = [c[0] for c in contour]
            y = [c[1] for c in contour]
            coords = np.zeros((len(x), 2))
            coords[:,0] = x
            coords[:,1] = y
            artists[i][0].set_xy(coords)
    return artists

ani = animation.FuncAnimation(fig=fig, func=partial(update, sources=sources, keys=contour_keys, artists=artists, axes=ax), frames=length, interval=30)
plt.show()
