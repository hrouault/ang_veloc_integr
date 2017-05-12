# This file is part of veloc_integr is a software which simulates angular
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
from numpy import pi
from scipy.integrate import ode
from scipy.stats import vonmises
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['svg.fonttype'] = 'none'


def centerpeak(y):
    """finding the center of a peak
    """
    N = len(y)
    axis = np.linspace(0, 2 * pi, N, endpoint=False)
    sinseries = np.sin(axis)
    cosseries = np.cos(axis)
    cosc = y.dot(cosseries)
    sinc = y.dot(sinseries)
    pos = np.arctan2(sinc, cosc)
    if pos < 0:
        pos += 2 * pi

    return pos


def thr_lin(x):
    """Threshold linear function
    """
    x_cp = np.copy(x)
    x_cp[x_cp < 0] = 0
    return x_cp


def f_mat(t, y, net):
    """Dynamical system function
    """
    out = np.zeros(net.n_tot)

    nwt = net.n_wedgetot
    mat1 = net.fullconn[:nwt, nwt:]
    out[:nwt] = np.dot(mat1, y[nwt:]) + net.off_eb

    mat2 = net.fullconn[nwt:, :nwt]
    out[nwt:] = np.dot(mat2, y[:nwt]) + 1.0
    out[nwt:nwt + nts] += net.vel_left
    out[nwt + nts:] += net.vel_right

    out = thr_lin(out)

    out -= y

    out[:nwt] /= net.tau
    out[nwt:] /= net.tau / 1.2

    return out


class Network():

    """variable describing connectivity, etc"""

    def __init__(self, ebeff, a, b):
        # neuron time constant in seconds
        self.tau = 0.080

        self.alpha = a

        self.n_per_wedge = 3
        npw = self.n_per_wedge
        self.n_per_tile = 2
        npt = self.n_per_tile
        self.n_tile = 9
        nt = self.n_tile
        self.n_wedgetot = 2 * self.n_per_wedge * self.n_tile
        nwt = self.n_wedgetot
        self.n_tileside = self.n_tile * self.n_per_tile
        nts = self.n_tileside
        self.n_tot = nwt + 2 * nts

        self.ebeff = ebeff

        self.kappa = 12

        self.pos_wedge = np.linspace(0, 2 * pi, 2 * nt * npw, endpoint=False)
        self.pos_left = np.linspace(0, 2 * pi, nt * npt, endpoint=False)
        self.pos_right = np.linspace(0, 2 * pi, nt * npt, endpoint=False)

        xaxis = np.linspace(0, 2 * pi, nwt, endpoint=False)

        self.beta = b
        self.off_eb = -0.0001

        self.conn_pl_e = np.zeros((nts, nwt))
        self.conn_pr_e = np.zeros((nts, nwt))
        self.conn_e_pl = np.zeros((nwt, nts))
        self.conn_e_pr = np.zeros((nwt, nts))
        self.fullconn = np.zeros((nwt + 2 * nts, nwt + 2 * nts))

        for i in np.arange(nt):
            for j in np.arange(npw):
                for k in np.arange(npt):
                    netc = self.alpha / nwt
                    self.conn_pl_e[i * npt + k, 2 * i * npw + j] = netc
                    self.conn_pr_e[i * npt + k, 2 * i * npw + j + npw] = netc
        inh = self.beta / nwt
        self.conn_pl_e -= inh
        self.conn_pr_e -= inh

        for i in np.arange(nt):
            for j in np.arange(npt):
                view1 = self.conn_e_pl[:, i * npt + j]
                shifts = xaxis - (i + 1.35) * 2 * pi / nt + pi / nwt
                vonm = vonmises.pdf(shifts, self.kappa) / nts
                view1 += self.alpha * self.ebeff * vonm
                shifts = xaxis - (i + 0.5) * 2 * pi / nt + pi / nwt
                vonm = vonmises.pdf(shifts, self.kappa) / nts
                view1 += 0.5 * self.alpha * self.ebeff * vonm
                view2 = self.conn_e_pr[:, i * npt + j]
                shifts = xaxis - (i - 0.35) * 2 * pi / nt + pi / nwt
                vonm = vonmises.pdf(shifts, self.kappa) / nts
                view2 += self.alpha * self.ebeff * vonm
                shifts = xaxis - (i + 0.5) * 2 * pi / nt + pi / nwt
                vonm = vonmises.pdf(shifts, self.kappa) / nts
                view2 += 0.5 * self.alpha * self.ebeff * vonm

        self.fullconn[nwt:nwt + nts, :nwt] = self.conn_pl_e
        self.fullconn[nwt + nts:nwt + 2 * nts, :nwt] = self.conn_pr_e
        self.fullconn[:nwt, nwt:nwt + nts] = self.conn_e_pl
        self.fullconn[:nwt, nwt + nts:nwt + 2 * nts] = self.conn_e_pr

        self.vel_left = 0.0
        self.vel_right = 0.0


t0 = 0.0
t1 = 8.9
tstart = 8.5
dt = 0.05

nbvels = 200
velmax = 0.25

velsin = np.linspace(0, velmax, nbvels)

nbeta = 10
betas = np.exp(np.linspace(np.log(1.0), np.log(30), nbeta))
nalpha = 10
alphas = np.exp(np.linspace(np.log(6.0), np.log(20), nalpha))

linearity = np.zeros((nalpha, nbeta))

for k, b in enumerate(betas):
    for j, a in enumerate(alphas):
        velscoeflin = 1e3
        velscoefsat = 0

        net = Network(1.0, a, b)

        nwt = net.n_wedgetot
        nts = net.n_tileside

        y0 = np.zeros(net.n_wedgetot + 2 * net.n_tileside)
        y0[nwt // 2] = 0.1
        y0[nwt // 2 + 1] = 0.1
        y0[nwt // 2 - 1] = 0.1

        r = ode(f_mat).set_integrator('vode', method='adams')
        r.set_initial_value(y0, t0).set_f_params(net)

        # Checking if we are in the marginal phase.
        while r.successful() and r.t < 20.0:
            r.integrate(r.t + dt)
            if np.max(r.y) > 1e4:
                break
        if np.max(r.y) > 1e4:
            linearity[k, j] = 1e8
            continue
        maxprof = np.max(r.y[:nwt])
        minprof = np.min(r.y[:nwt])
        if (maxprof - minprof) / maxprof < 0.5:
            linearity[k, j] = 1e-8
            continue

        r.t = 0

        velsin = np.exp(np.linspace(-10.0, 20.0, nbvels))
        checklin = False
        checksat = False
        velsprof = np.zeros(velsin.size)
        for i, vr in enumerate(velsin):
            r.set_initial_value(y0, t0)
            net.vel_right = 0.0 * vr
            net.vel_left = vr

            while r.successful() and r.t < tstart:
                r.integrate(r.t + dt)

            pos1 = centerpeak(r.y[:nwt])
            time1 = r.t
            counts = 0
            while r.successful() and r.t < t1:
                r.integrate(r.t + dt)
                counts += 1

            pos2 = centerpeak(r.y[:nwt])
            time2 = r.t
            diff = np.abs(pos2 - pos1)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            vel = diff / (time2 - time1) * 180 / np.pi
            print(vel)
            velsprof[i] = vel

            if vel > 20 and vel < 60:
                velscoeflin = vel / vr
                checklin = True

            maxl = np.max(r.y[nwt + nts:])
            if maxl < 1e-3:
                velscoefsat = vel / vr
                break

        if not checklin:
            linearity[k, j] = 1e8
        else:
            linearity[k, j] = velscoefsat / velscoeflin

        print(a, b, linearity[k, j])

np.save('linearity.npy', linearity)
