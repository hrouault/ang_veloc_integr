# veloc_integr.py is a software which simulates angular velocity integration
# circuits
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
import numpy.random as rand
from scipy.integrate import ode
from scipy.stats import vonmises
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import Ornstein_Uhlenbeck as orn

mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.size'] = 10.0
mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
mpl.rcParams['svg.fonttype'] = 'none'


def fwhm(y):
    """Computing the full width at half maximum
    """

    size = len(y)
    maxval = np.max(y)
    minval = np.min(y)
    maxi = np.argmax(y)

    hm = (maxval + minval) / 2

    xaxis = np.linspace(0, 2 * pi, size + 1)
    ys = np.zeros(size + 1)
    ys[0:size] = np.roll(y, -maxi) - hm
    ys[-1] = ys[0]
    f = interp1d(xaxis, ys)
    x1 = brentq(f, 0, pi)
    x2 = brentq(f, pi, xaxis[-1])
    return x1 + 2 * pi - x2


def rewrap2pi(diff):
    """rewrap to have small differences
    """
    out = diff
    if np.abs(diff) > pi:
        out = 2 * pi - np.abs(diff)
        if diff > 0:
            out = -out
    return out


def unwrap_angles(angles):
    for a in range(angles.size - 1):
        diff = angles[a] - angles[a + 1]
        if np.abs(diff) > 3 * pi / 4:
            angles[a + 1:] += 2 * pi * np.sign(diff)


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

    def __init__(self, ebeff):
        # neuron time constant in seconds
        self.tau = 0.080

        self.alpha = 10.0

        self.n_per_wedge = 3
        npw = self.n_per_wedge
        self.n_per_tile = 1
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

        self.beta = 25.0
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
                view1 += self.alpha * vonm
                shifts = xaxis - (i + 0.5) * 2 * pi / nt + pi / nwt
                vonm = vonmises.pdf(shifts, self.kappa) / nts
                view1 += 0.5 * self.alpha * vonm
                view2 = self.conn_e_pr[:, i * npt + j]
                shifts = xaxis - (i - 0.35) * 2 * pi / nt + pi / nwt
                vonm = vonmises.pdf(shifts, self.kappa) / nts
                view2 += self.alpha * vonm
                shifts = xaxis - (i + 0.5) * 2 * pi / nt + pi / nwt
                vonm = vonmises.pdf(shifts, self.kappa) / nts
                view2 += 0.5 * self.alpha * vonm

                view1 *= self.ebeff
                view2 *= self.ebeff

        self.fullconn[nwt:nwt + nts, :nwt] = self.conn_pl_e
        self.fullconn[nwt + nts:nwt + 2 * nts, :nwt] = self.conn_pr_e
        self.fullconn[:nwt, nwt:nwt + nts] = self.conn_e_pl
        self.fullconn[:nwt, nwt + nts:nwt + 2 * nts] = self.conn_e_pr

        self.vel_left = 0.0
        self.vel_right = 0.0


net = Network(1.0)
y0 = np.zeros(net.n_wedgetot + 2 * net.n_tileside)
y0[net.n_wedgetot // 2] = 0.1
y0[net.n_wedgetot // 2 + 1] = 0.1
y0[net.n_wedgetot // 2 - 1] = 0.1

t0 = 0.0
t1 = 20.9
tstart = 20.5

dt = 0.004

nbvels = 18
nbvels = 30  # for higher resolution of the velocity curve
velmax = 0.06

r = ode(f_mat).set_integrator('vode', method='adams')
r.set_initial_value(y0, t0).set_f_params(net)

vels = np.zeros(nbvels)
velsin = np.linspace(0, velmax, nbvels)
maxeb = np.zeros(nbvels)
maxtl = np.zeros(nbvels)
maxtr = np.zeros(nbvels)
maxtineb = np.zeros(nbvels)
phaseeb = np.zeros(nbvels)
phasepbl = np.zeros(nbvels)
phasepbr = np.zeros(nbvels)
fwhmeb = np.zeros(nbvels)
fwhmtile = np.zeros(nbvels)

nwt = net.n_wedgetot
nts = net.n_tileside

tuning = np.zeros((nts, 2 * nbvels - 1))

vellow = 0.0
velhigh = 0.0

axis_w = np.linspace(-pi, pi, nwt, endpoint=False)
axis_t = np.linspace(-pi, pi, nts + 1)


def ani_update(i, lines):
    r.integrate(r.t + dt)
    lines[0].set_ydata(r.y[:nwt])
    lines[1].set_ydata(np.dot(net.conn_e_pl, r.y[nwt:nwt + nts]))
    lines[2].set_ydata(np.dot(net.conn_e_pr, r.y[nwt + nts:]))
    lines[3].set_ydata(r.y[nwt:nwt + nts])
    lines[4].set_ydata(np.dot(net.conn_pl_e, r.y[:nwt]) + 1.0)
    lines[5].set_ydata(r.y[nwt + nts:])
    lines[6].set_ydata(np.dot(net.conn_pr_e, r.y[:nwt]) + 1.0)
    print(r.t)
    return tuple(lines)


for i, vr in enumerate(velsin):
    if i < 3:
        t1 = 21.8
    elif i < 8:
        t1 = 21.3
    else:
        t1 = 20.9
    r.set_initial_value(y0, t0)
    print(vr)
    net.vel_right = 0.0 * vr
    net.vel_left = vr

    while r.successful() and r.t < tstart:
        r.integrate(r.t + dt)

    pos1 = centerpeak(r.y[:nwt])
    time1 = r.t
    counts = 0
    maxs = 0.0
    maxstr = 0.0
    maxstl = 0.0
    maxstineb = 0.0
    while r.successful() and r.t < t1:
        r.integrate(r.t + dt)
        maxs += np.max(r.y[:nwt])
        maxstl += np.max(r.y[nwt:nwt + nts])
        maxstr += np.max(r.y[nwt + nts:])
        maxstineb += np.max(np.dot(net.conn_e_pl, r.y[nwt:nwt + nts])
                            + np.dot(net.conn_e_pr, r.y[nwt + nts:]))
        counts += 1

    maxs /= counts
    maxstl /= counts
    maxstr /= counts
    maxstineb /= counts
    maxeb[i] = maxs
    maxtl[i] = maxstl
    maxtr[i] = maxstr
    maxtineb[i] = maxstineb

    pos2 = centerpeak(r.y[:nwt])
    time2 = r.t
    print(pos1, pos2, time1, time2)
    diff = np.abs(pos2 - pos1)
    if diff > pi:
        diff = 2 * pi - diff
    vels[i] = diff / (time2 - time1)
    print(vels[i])

    poseb = centerpeak(r.y[:nwt])
    postl = centerpeak(r.y[nwt:nwt + nts])
    postr = centerpeak(r.y[nwt + nts:])

    convol_left = np.dot(net.conn_e_pl, r.y[nwt:nwt + nts])
    convol_right = np.dot(net.conn_e_pr, r.y[nwt + nts:])

    pbl_epg = np.dot(net.conn_pl_e + net.beta / nwt, r.y[:nwt])
    pbr_epg = np.dot(net.conn_pr_e + net.beta / nwt, r.y[:nwt])

    pospbl = centerpeak(pbl_epg)
    pospbr = centerpeak(pbr_epg)

    posin = centerpeak(convol_left + convol_right)

    phaseeb[i] = rewrap2pi(poseb - posin)

    phasepbl[i] = rewrap2pi(pospbl - postl)
    phasepbr[i] = rewrap2pi(pospbr - postr)

    if np.max(r.y[nwt:nwt + nts]) < 1e-4:
        phasepbl[i] = 0
    if np.max(r.y[nwt + nts:]) < 1e-4:
        phasepbr[i] = 0

    fwhmeb[i] = fwhm(r.y[:nwt])
    fwhmtile[i] = fwhm(convol_right + convol_left)

    profile = r.y[nwt:nwt + nts]
    profilecontra = r.y[nwt + nts:]

    if i != 0:
        tuning[:, nbvels - 1 + i] = profile
        tuning[:, nbvels - 1 - i] = profilecontra
    else:
        tuning[:, nbvels - 1] = profile

    if i == 3 or i == nbvels - 14:
        plt.figure(figsize=(1.75, 1.35))
        ax = plt.subplot()
        ax.set_xticks([-pi, 0, pi])
        ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.get_yaxis().set_tick_params(direction='in', color='#808080')
        ax.get_xaxis().set_tick_params(direction='in', color='#808080')
        ax.spines['bottom'].set_color('#808080')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_color('#808080')
        ax.spines['left'].set_linewidth(1.0)

        maxi = np.argmax(r.y[nwt:nwt + nts])

        sc_shift = net.n_per_wedge * 2
        plt_convol_left = np.roll(convol_left, (-maxi + 5) * sc_shift)
        plt_convol_right = np.roll(convol_right, (-maxi + 5) * sc_shift)

        plt.plot(axis_w, np.roll(r.y[:nwt] / 2.0, (-maxi + 5) * sc_shift),
                 color='navy', lw=1.0)
        plt.plot(axis_w, plt_convol_left + plt_convol_right,
                 color='forestgreen', lw=1.0)
        plt.plot(axis_w, plt_convol_left, color='indianred', lw=1.0)
        plt.plot(axis_w, plt_convol_right, color='deepskyblue', lw=1.0)
        plt.xlabel('angle (rad)')
        plt.tight_layout()
        if i == 3:
            plt.savefig("prof_lowvel_eb.svg", format="svg")
            vellow = velsin[i]
        if i > 3:
            plt.savefig("prof_highvel_eb.svg", format="svg")
            velhigh = velsin[i]
        plt.figure(figsize=(1.8, 1.35))
        ax = plt.subplot()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.get_yaxis().set_tick_params(direction='in', color='#808080')
        ax.get_xaxis().set_tick_params(direction='in', color='#808080')
        ax.spines['bottom'].set_color('#808080')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_color('#808080')
        ax.spines['left'].set_linewidth(1.0)

        plt_epg_l = np.zeros(net.n_tileside + 1)
        plt_epg_l[:-1] = np.roll(pbl_epg, -maxi + 5)
        plt_epg_l[-1] = plt_epg_l[0]
        plt_epg_r = np.zeros(net.n_tileside + 1)
        plt_epg_r[:-1] = np.roll(pbr_epg, -maxi + 5)
        plt_epg_r[-1] = plt_epg_r[0]
        plt_pen_l = np.zeros(net.n_tileside + 1)
        plt_pen_l[:-1] = np.roll(r.y[nwt:nwt + nts], -maxi + 5)
        plt_pen_l[-1] = plt_pen_l[0]
        plt_pen_r = np.zeros(net.n_tileside + 1)
        plt_pen_r[:-1] = np.roll(r.y[nwt + nts:], -maxi + 5)
        plt_pen_r[-1] = plt_pen_r[0]

        ax.set_xticks([-pi, 0, pi])
        ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
        plt.plot(axis_t, plt_pen_l, color='indianred', lw=1.0)
        plt.plot(axis_t, plt_pen_r, color='deepskyblue', lw=1.0)
        plt.plot(axis_t, plt_epg_l, '--', color='navy', lw=1.0)
        plt.plot(axis_t, plt_epg_r, ':', color='navy', lw=1.0)
        plt.xlabel('angle (rad)')
        plt.tight_layout()
        if i == 3:
            plt.savefig("prof_lowvel_pb.svg", format="svg")
        if i > 3:
            plt.savefig("prof_highvel_pb.svg", format="svg")

velsin[0] = 1e-10
velscoefs = vels / velsin
vel_in_coef = velscoefs[8]
print(vel_in_coef)
vel_in_coef = 99.64
velaxis = vel_in_coef * velsin * 180 / pi

print('Velocity low: ', velaxis[3])
print('Velocity high: ', velaxis[nbvels - 14])

plt.figure(figsize=(4.4, 4.4))
ax = plt.subplot(221)
plt.plot(velaxis, vels * 180 / pi, color='black', lw=1.0)
plt.plot(velaxis, velaxis, color='black', lw=1.0)
plt.ylabel('bump velocity (º/s)')
plt.xlabel('rotational velocity (º/s)')
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
plt.tight_layout()


ax = plt.subplot(222)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
plt.plot(velaxis, maxeb / maxeb[0], color='navy', lw=1.0)
plt.plot(velaxis, maxtineb / maxtineb[0], color='green', lw=1.0)
plt.plot(velaxis, maxtr / maxtr[0], color='deepskyblue', lw=1.0)
plt.plot(velaxis, maxtl / maxtl[0], color='indianred', lw=1.0)
plt.ylabel('normalized bump amplitude')
plt.xlabel('rotational velocity (º/s)')

ax = plt.subplot(223)
maximg = np.max(np.abs(net.fullconn))
imconn = net.fullconn.T.repeat(5, axis=0).repeat(5, axis=1)
plt.imshow(imconn, clim=(-maximg, maximg),
           cmap=plt.get_cmap('Spectral_r'),
           extent=[0, 6 * pi, 0, 6 * pi],
           interpolation='nearest')
ax.set_yticks([0, 2 * pi, 4 * pi, 6 * pi])
ax.set_yticklabels([])
ax.set_xticks([0, 2 * pi, 4 * pi, 6 * pi])
ax.set_xticklabels([])
cb = plt.colorbar(fraction=0.042)
plt.tight_layout()
plt.savefig("general_quant.svg", format="svg")

sio.savemat('connectivity_map', {'connect': net.fullconn.T})

fig = plt.figure(figsize=(3.0, 3.0))
ax = fig.add_subplot('111')
plt.imshow(imconn, clim=(-maximg, maximg),
           cmap=plt.get_cmap('RdBu_r'),
           extent=[0, nwt + 2 * nts, 0, nwt + 2 * nts],
           interpolation='none')
ax.set_yticks([0, nts, 2 * nts, nwt + 2 * nts])
ax.set_yticklabels([])
ax.set_xticks([0, nwt, nwt + nts, nwt + 2 * nts])
ax.set_xticklabels([])
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
cb = plt.colorbar(fraction=0.042)
plt.tight_layout()
plt.savefig("connectivity.svg", format="svg")

plt.figure(figsize=(1.9, 2.6))
ax = plt.subplot()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
plt.plot(velaxis, -phasepbl * 180 / pi, color='indianred', lw=1.0)
plt.plot(velaxis, -phasepbr * 180 / pi, color='deepskyblue', lw=1.0)
plt.plot(velaxis, -phaseeb * 180 / pi, color='green', lw=1.0)
plt.ylabel('P-EN — E-PG phase difference (º)')
plt.xlabel('rotational velocity (º/s)')
plt.tight_layout()
plt.savefig("phase_diff.svg", format="svg")


plt.figure(figsize=(2.4, 2.4))
ax = plt.subplot()
tun_summed = np.sum(tuning, axis=0)
velmaxcoef = velmax * vel_in_coef * 180 / pi
xaxsum = np.linspace(-velmaxcoef, velmaxcoef, tun_summed.size)
plt.plot(xaxsum, tun_summed[::-1], color='deepskyblue', lw=1.0)
plt.plot(xaxsum, tun_summed, color='indianred', lw=1.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('P-EN firing rate (A. U.)')
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
plt.xlabel('rotational velocity (º/s)')
plt.tight_layout()
plt.savefig("pen_tuning_summed.svg", format="svg")

plt.show()


# Shi^TS simulations
nbshits = 80
amplishits = np.zeros(nbshits)
valsshits = np.linspace(0.0, 1.0, nbshits)
inte_pts = np.array([16, 40, 70])

plt.figure(figsize=(1.8, 2.1))
ax = plt.subplot(311)
for i, shits in enumerate(valsshits):
    net_shits = Network(shits)
    nwt = net_shits.n_wedgetot
    y0 = np.zeros(nwt + 2 * net_shits.n_tileside)
    y0[nwt // 2] = 0.1
    y0[nwt // 2 + 1] = 0.1
    y0[nwt // 2 - 1] = 0.1

    t0 = 0.0
    t1 = 40.0
    dt = 0.02

    r = ode(f_mat).set_integrator('vode', method='adams')
    r.set_initial_value(y0, t0).set_f_params(net_shits)

    while r.successful() and r.t < t1:
        r.integrate(r.t + dt)
    amplishits[i] = np.max(r.y[:nwt])

    xaxis = np.linspace(0, 2 * pi, nwt, endpoint=False)
    if i in inte_pts:
        if i == inte_pts[0]:
            ax = plt.subplot(313)
        if i == inte_pts[1]:
            ax = plt.subplot(312)
            ax.get_xaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
        if i == inte_pts[2]:
            ax = plt.subplot(311)
            ax.get_xaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_color('#808080')
        ax.spines['left'].set_color('#808080')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.set_xticks([0, pi, 2 * pi])
        ax.set_xticklabels(['$0$', '$\pi$', '2$\pi$'])
        ax.get_yaxis().set_tick_params(direction='in', color='#808080')
        ax.get_xaxis().set_tick_params(direction='in', color='#808080')
        ax.set_xlim((0, 2 * pi + 0.1))
        ax.set_ylim((0, 0.11))
        plt.plot(xaxis, r.y[:nwt])

plt.xlabel('angle (rad)')
plt.ylabel('A. U.')
plt.tight_layout()
plt.savefig("prof_shits.svg", format="svg")


plt.figure(figsize=(2.3, 2.6))
ax = plt.subplot()
plt.plot(valsshits, amplishits)
plt.plot(valsshits[inte_pts], amplishits[inte_pts], 'o')
plt.ylabel('E-PGs amplitude (A. U.)')
plt.xlabel('P-EN to E-PG synaptic efficacy')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
ax.get_yaxis().set_tick_params(direction='in', color='#808080', width=1.0)
ax.get_xaxis().set_tick_params(direction='in', color='#808080', width=1.0)
plt.tight_layout()
plt.savefig("shits_ampli.svg", format="svg")

plt.show()

# Dynamical velocity

y0 = np.zeros(nwt + 2 * net_shits.n_tileside)
y0[nwt // 2] = 0.1
y0[nwt // 2 + 1] = 0.1
y0[nwt // 2 - 1] = 0.1

net.vel_right = 0.0
net.vel_left = 0.0
r.set_initial_value(y0, 0.0).set_f_params(net)
while r.successful() and r.t < tstart:
    r.integrate(r.t + dt)

y0 = np.copy(r.y)


def velocity_track(input_vel, seed, n_samp, dt, y_start, net):
    """ Generate an integration for a varying velocity input
    :input_vel: numpy array for the input velocity
    :returns: the integrated track
    """

    # integrating the system
    v_left = thr_lin(-input_vel)
    v_right = thr_lin(input_vel)

    poseb = np.zeros(n_samp)
    r.set_initial_value(y_start, 0.0)
    tcur = 0.0
    for i_t in range(n_samp):
        net.vel_right = v_left[i_t]
        net.vel_left = v_right[i_t]
        tcur += dt
        r.integrate(tcur)
        poseb[i_t] = centerpeak(r.y[:nwt])
    return poseb


dt = 0.01

# small track for plotting
rand.seed(0)
t_end = 100.0
n_samp = int(t_end / dt)
tsamp, traj, traj_integr = orn.veloc_trace(dt, t_end,
                                           seed=rand.randint(100000))
traj *= pi / 180.0
traj_integr *= pi / 180.0
ang_out = velocity_track(traj / vel_in_coef,
                         rand.randint(100000), n_samp, dt, y0, net)
unwrap_angles(ang_out)

plt.figure(figsize=(3.2, 2.0))
ax = plt.subplot()
plt.plot(tsamp, traj_integr, color='black', lw=1.0)
plt.plot(tsamp, ang_out - pi, color='green', lw=1.0)
plt.xlabel('time (s)')
plt.ylabel('heading (rad)')
ax.set_yticks([0, pi])
ax.set_yticklabels(['$0$', '$\pi$'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
plt.tight_layout()
plt.savefig("WT_track.svg", format="svg")

net_shits_otrack = Network(0.55)
y0 = np.zeros(nwt + 2 * net_shits_otrack.n_tileside)
y0[nwt // 2] = 0.1
y0[nwt // 2 + 1] = 0.1
y0[nwt // 2 - 1] = 0.1

net_shits_otrack.vel_right = 0.0
net_shits_otrack.vel_left = 0.0
r.set_initial_value(y0, 0.0).set_f_params(net_shits_otrack)
while r.successful() and r.t < tstart:
    r.integrate(r.t + dt)

y0 = np.copy(r.y)
ang_out = velocity_track(traj / vel_in_coef,
                         rand.randint(100000), n_samp, dt, y0,
                         net_shits_otrack)
unwrap_angles(ang_out)

plt.figure(figsize=(4.0, 2.0))
ax = plt.subplot()
plt.plot(tsamp, traj_integr, color='black', lw=1.0)
plt.plot(tsamp, ang_out - pi, color='brown', lw=1.0)
plt.xlabel('time (s)')
plt.ylabel('heading (rad)')
ax.set_yticks([0, pi])
ax.set_yticklabels(['$0$', '$\pi$'])
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.get_yaxis().set_tick_params(direction='in', color='#808080')
ax.get_xaxis().set_tick_params(direction='in', color='#808080')
ax.spines['bottom'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['left'].set_color('#808080')
ax.spines['left'].set_linewidth(1.0)
plt.tight_layout()
plt.savefig("shits_track.svg", format="svg")


# full track for diffusion
r.set_initial_value(y0, 0.0).set_f_params(net)
rand.seed(0)
t_end = 4000.0
n_samp = int(t_end / dt)
nb_it = 5
diffs = np.zeros((nb_it, n_samp))
velsout = np.zeros((nb_it, n_samp - 1))
velsin = np.zeros((nb_it, n_samp - 1))
for i in np.arange(nb_it):
    tsamp, traj, traj_integr = orn.veloc_trace(dt, t_end,
                                               seed=rand.randint(100000))
    traj *= pi / 180.0
    traj_integr *= pi / 180.0
    ang_out = velocity_track(traj / vel_in_coef,
                             rand.randint(100000), n_samp, dt, y0, net)
    unwrap_angles(ang_out)
    diffs[i, :] = ang_out - pi - traj_integr

    # velocity correlation
    velsout[i, :] = ang_out[1:] - ang_out[:-1]
    velsin[i, :] = traj[:-1]


tsamp = tsamp[100:]
diffs = diffs[:, 100:]
tsampre = tsamp.reshape((1, tsamp.size))

velsin = velsin[:, 100:]
velsout = velsout[:, 100:]

stacked = np.vstack((tsampre, diffs))
stacked_vel = np.vstack((np.vstack((tsampre[:, :-1], velsin)), velsout))


np.save('diffs.npy', stacked)
np.save('correlation.npy', stacked_vel)
