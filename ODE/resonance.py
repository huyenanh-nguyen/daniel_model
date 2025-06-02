import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelmax, argrelextrema
from scipy.signal import find_peaks
from scipy.fft import fft

from onesidedcoupling import OnesidedCoupling

t_step = 0.01
t_last = 100 # 50h -> 1 point represent 1h
t = np.arange(0, 4000, t_step)
keep = int(t_last / t_step)
k = 0.1
gamma = 0.2
mu = 2
beta = 0.5
alpha = 0.63
alpha_up = np.arange(0,20, 0.1)
alpha_down = alpha_up[::-1]

par0 = 1,1,1,1

lilie = OnesidedCoupling(par0, t, keep, k, mu, gamma, alpha, beta)

reso_alpha = np.arange(0, 8, 0.1)[::-1]
omega = [np.sqrt(i) for i in reso_alpha]

# [Phasedifference]_____________________________________________________________________________________________________________________________________________________________________-
# # y-timeseries


# period_vdp = lilie.period(10)[1]
# time_amp_vdp = [t[i] for i in lilie.find_peaks_max()[1][0][-10:]]

# time_amp = [t[k] for k in [OnesidedCoupling(par0, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][0][-10:] for i in reso_alpha]]
# phaseamp = [2 * np.pi * (time_amp_vdp[0]-i[0])/period_vdp for i in time_amp]

# plt.plot(omega[3:], phaseamp[3:])
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.xlabel("$\omega$", fontsize = 30)
# plt.ylabel("$\\varphi$",fontsize = 30)
# plt.show()

# # [Resonance Curve]___________________________________________________________________________________________________________________________________________________________________________________-



amp = [np.mean(OnesidedCoupling(par0, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][1]['peak_heights'][-6:]) for i in reso_alpha]

# plt.plot(omega, amp)

# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.xlabel("$\omega _0$ in Hz", fontsize = 30)
# plt.ylabel("A in a.u.",fontsize = 30)
# plt.show()


# [Fourier]_____________________________________________________________________________________________________________________________________________________________________-

from scipy.fft import fft, fftfreq
import numpy as np

yf = np.fft.ifft(amp)

plt.plot(omega, yf)
plt.grid()
plt.show()

