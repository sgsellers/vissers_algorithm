import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to file of extracted sources")
args = parser.parse_args()

infile = args.input

import numpy as np
import matplotlib.pyplot as plt

sources = np.load(infile, allow_pickle=True)
source_keys = sources.dtype.names

center_keys = source_keys[0::5]
flux_keys = source_keys[2::5]
extent_keys = source_keys[3::5]

sumflux = np.zeros(len(sources[center_keys[0]]))
sumextent = np.zeros(len(sources[extent_keys[0]]))

for k in flux_keys:
    sumflux += sources[k]
for k in extent_keys:
    sumextent += sources[k]

sourceLength = np.zeros(len(flux_keys))
for i in range(len(flux_keys)):
    sourceLength[i] = len(sources[flux_keys[i]][sources[flux_keys[i]] != 0])

longest_5 = np.flip(np.argsort(sourceLength))[:5]

totalFluxPerSource = np.zeros(len(flux_keys))
for i in range(len(flux_keys)):
    totalFluxPerSource[i] = np.sum(sources[flux_keys[i]])

brightest_5 = np.flip(np.argsort(totalFluxPerSource))[:5]

greatestSourceExtent = np.zeros(len(flux_keys))
for i in range(len(extent_keys)):
    greatestSourceExtent[i] = np.nanmax(sources[extent_keys[i]])

largest_5 = np.flip(np.argsort(greatestSourceExtent))[:5]

# Plot the cumulative flux, extent, longest-lived 5, brightest 5, and largest 5
fig = plt.figure()
ax = fig.add_subplot(321)
ax.scatter(np.arange(len(sumflux)), sumflux, c='k', s=1)
ax.set_title("Cumulative Source Flux")
ax.set_ylabel("DN")
ax.set_xlabel("Frame No.")
ax.set_xlim(0,len(sumflux))

ax = fig.add_subplot(322)
ax.scatter(np.arange(len(sumextent)), sumextent, c='C0', s=1)
ax.set_title("Cumulative Source Extent")
ax.set_ylabel("No. Pixels")
ax.set_xlabel("Frame No.")
ax.set_xlim(0,len(sumflux))
ax.set_ylim(0, np.nanmax(greatestSourceExtent))

ax = fig.add_subplot(323)
for i in range(len(longest_5)):
    fl = sources[flux_keys[longest_5[i]]]
    ax.scatter(np.arange(len(fl)), fl, c='C'+str(i+1), s=1)
ax.set_title("Flux of 5 longest-lived sources")
ax.set_ylabel("DN")
ax.set_xlabel("Frame No.")
ax.set_xlim(0,len(sumflux))


ax = fig.add_subplot(324)
for i in range(len(brightest_5)):
    fl = sources[flux_keys[brightest_5[i]]]
    ax.scatter(np.arange(len(fl)), fl, c='C'+str(i+1), s=1)
ax.set_title("Flux of 5 brightest sources")
ax.set_ylabel("DN")
ax.set_xlabel("Frame No.")
ax.set_xlim(0,len(sumflux))

ax = fig.add_subplot(325)
for i in range(len(largest_5)):
    fl = sources[extent_keys[largest_5[i]]]
    ax.scatter(np.arange(len(fl)), fl, c='C'+str(i+1), s=1)
ax.set_title("Extent of 5 largest sources")
ax.set_ylabel("DN")
ax.set_xlabel("Frame No.")
ax.set_ylim(0, np.nanmax(greatestSourceExtent))
ax.set_xlim(0,len(sumflux))
plt.tight_layout()
plt.show()
