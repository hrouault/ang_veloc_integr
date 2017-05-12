# This file is part of veloc_integr which is a software which simulates angular
# velocity integration circuits
#
# Copyright © 2016 Howard Hughes Medical Institute
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 3. Neither the name of the organization nor the
# names of its contributors may be used to endorse or promote products
# derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY Howard Hughes Medical Institute ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL Howard Hughes Medical Institute BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
#

import numpy as np
from scipy.optimize import least_squares
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
mpl.rcParams['svg.fonttype'] = 'none'

vels = np.load('correlation.npy')

vels[1:, :] *= 180 / np.pi
vels[6, :] *= 100

plt.figure(figsize=(2.7, 2.7))
ax = plt.subplot()
plt.plot(vels[0, :1000] - 1.0, vels[1, :1000], lw=1.0)
plt.plot(vels[0, :1000] - 1.0, vels[6, :1000], lw=1.0)
plt.xlabel('time (s)')
plt.ylabel('velocity (º/s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
plt.tight_layout()
plt.savefig("vel_trace.svg", format="svg")

correlfun = np.correlate(vels[1, :], vels[6, :], 'same') / vels[1, :].size
correlfun2 = np.correlate(vels[6, :], vels[1, :], 'same') / vels[1, :].size
plt.figure(figsize=(2.3, 2.3))
ax = plt.subplot()
ax = plt.subplot()
N = correlfun.size
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([0.0, 2200])
vels1 = vels[0, 1:N // 2 + 1]
xaxis = np.concatenate((-vels1[::-1] + 1.0, vels[0, :N // 2 + 1] - 1.0))
plt.plot(xaxis, correlfun2, color='black', lw=1.0)
sel = np.logical_and(xaxis > -0.5, xaxis < 0.5)
plt.plot([0.0, 0.0], [-4000, 4000], '--', color='gray', lw=1.0)
plt.xlabel('time (s)')
plt.ylabel('Cross-correlation (º/s $\\times$ º/s)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
plt.tight_layout()
plt.savefig("correlation.svg", format="svg")
plt.show()

traj = np.load('diffs.npy')
traj = traj[:, ::100]

nbsteps = 20
dts = np.zeros(nbsteps)
varis = np.zeros(nbsteps)
ns = np.zeros(nbsteps)
alpha = np.log(traj.shape[1]) / nbsteps
for i in np.arange(0, nbsteps) + 1:
    shift = int(np.exp(alpha * i))
    idx = i - 1
    trajsub = traj[:, ::shift]
    dts[idx] = trajsub[0, 1] - trajsub[0, 0]
    diffs = (trajsub - np.roll(trajsub, 1, axis=1))[1:, 1:]
    ns[idx] = diffs.size
    varis[idx] = np.var(diffs, ddof=1)

plt.figure(figsize=(2.0, 2.0))
ax = plt.subplot()
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.get_yaxis().set_tick_params(which='both', direction='in', color='#808080')
ax.get_xaxis().set_tick_params(which='both', direction='in', color='#808080')
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
errup = varis * np.sqrt(2 / (ns - 1))
plt.ylabel('error variance (rad2)')
plt.xlabel('time (s)')


# fitting the power law


def fitfunc(p, t, y, ns):
    return (y - 2 * p[0] * np.power(t, 1.0) - p[1]) / (y / np.sqrt(ns - 1))


def fitfuncnore(p, t):
    return 2 * p[0] * np.power(t, 1.0) + p[1]


p0 = np.array([0.5, 3.0])  # Initial guess for the parameters
res_1 = least_squares(fitfunc, p0, args=(dts, varis, ns))

popti = res_1.x

print(popti)

p0 = np.array([0.4, 3.5])  # Initial guess for the parameters
# plot the fit
plt.plot(dts, fitfuncnore(popti, dts), lw=1.0)

# plot the experimental data
plt.plot(dts, varis, lw=1.0)
plt.fill_between(dts, varis + errup, varis - errup, alpha=0.5)

plt.tight_layout()
plt.savefig("diffusion.svg", format="svg")

plt.show()
