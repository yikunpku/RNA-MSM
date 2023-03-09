import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import numpy as np
import copy

probfile = sys.argv[1]
pngfile = sys.argv[2]
data = np.loadtxt(probfile)
# plot prob map
fig_size = [4, 3]
font = {'family': 'sans-serif',
        'sans-serif': ['Arial', ],
        'size': 6}
plt.rc('font', **font)
plt.rc('mathtext', fontset='stixsans')
plt.rc('lines', linewidth=.25)
# plt.rc('axes', grid=True)
# plt.rc('axes.grid', color=np.array([226,226,226])/255)

cmap = copy.copy(plt.get_cmap('jet'))
cmap.set_under([0,0,0,0])
vnorm = mpl.colors.Normalize(vmin=0.01, vmax=1)
fig, ax = plt.subplots(ncols=1,
                   nrows=1,
                   figsize=fig_size, squeeze=False)
plt.subplots_adjust(top=0.95, left=0.1, bottom=0.05)

ax1 = ax[0,0]
# scatter plot with color bar, with rows on y axis
draw = ax1.imshow(data, cmap=cmap, norm=vnorm,
        zorder=3,
        extent=(1, data.shape[1], data.shape[0], 1))
# ax1.set_zorder(1)
# ax1.grid(visible=True,
ax1.grid(
        color=np.array([221]*3)/255.,
        linestyle='-', linewidth=0.5)
cb = fig.colorbar(draw, shrink=0.95)

fig.savefig(pngfile, dpi=300)
fig.clf()
plt.close(fig)
