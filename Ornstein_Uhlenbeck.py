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
import numpy.random as rand

# Simulation of the Ornstein-Uhlenbeck process
# --------------------------------------------
#
# We simulate the following SDE
# d r_t = - \theta r_t dt + \sigma d W_t
#
# the code is inspired by the one on the wikipedia Euler Maruyama method


def veloc_trace(dt, t_end, seed=0):
    rand.seed(seed)

    theta = 1.0 / 0.12
    sigma = 50 * np.sqrt(2 * theta)

    # the stationary distribution is gaussian with standard deviation
    # \sigma / \sqrt{2 \theta}

    sqdt = np.sqrt(dt)

    N = int(t_end / dt)

    t = np.linspace(0.0, t_end, N)
    y = np.zeros(N)
    y_integr = np.zeros(N)

    # initialize with the stationary distribution
    y_integr[0] = 0.0
    for i in np.arange(N - 1):
        y[i + 1] = y[i] - theta * y[i] * dt + sigma * sqdt * rand.normal()
        y_integr[i + 1] = y_integr[i] + y[i] * dt

    return t, y, y_integr
