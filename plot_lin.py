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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import ScalarFormatter
import scipy.io as sio

mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['svg.fonttype'] = 'none'

linearity = np.load('linearity.npy')

thrlow = 5e-2
thrhigh = 2e1
linearity[linearity < thrlow] = thrlow
linearity[linearity > thrhigh] = thrhigh

nbeta = 10
limb1 = 1.0
limb2 = 30.0
blims = np.linspace(np.log(limb1), np.log(limb2), nbeta + 1)
limb1 = np.exp(1.5 * blims[0] - 0.5 * blims[1])
limb2 = np.exp(1.5 * blims[-1] - 0.5 * blims[-2])
blims = np.linspace(np.log(limb1), np.log(limb2), nbeta + 1)
betas = np.exp(blims)
nalpha = 10
lima1 = 6.0
lima2 = 20
alims = np.linspace(np.log(lima1), np.log(lima2), nbeta + 1)
lima1 = np.exp(1.5 * alims[0] - 0.5 * alims[1])
lima2 = np.exp(1.5 * alims[-1] - 0.5 * alims[-2])
alims = np.linspace(np.log(lima1), np.log(lima2), nbeta + 1)
alphas = np.exp(alims)

plt.figure(figsize=(3.0, 3.0))
ax = plt.subplot()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\alpha$')
plt.ylabel('$\\beta$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
logformatter = LogFormatter(labelOnlyBase=False)
scaformatter = ScalarFormatter()
scaformatter.set_scientific(False)
ax.yaxis.set_major_formatter(scaformatter)
ax.xaxis.set_major_formatter(scaformatter)

X, Y = np.meshgrid(alphas, betas)

im = plt.pcolor(alphas, betas, linearity,
                norm=LogNorm(vmin=linearity.min(), vmax=linearity.max()),
                cmap=plt.get_cmap('RdBu_r'))


cb = plt.colorbar()
plt.tight_layout()
plt.savefig("linearity.svg", format="svg")

plt.show()

sio.savemat('linearity_map',
            {'alphas': alphas, 'betas': betas, 'values': linearity})
