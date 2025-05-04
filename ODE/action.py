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
t = np.arange(0, 5000, t_step)
keep = int(t_last / t_step)
x = 0.5
y = 1
q = 0
p = 1
par = x,y,p,q
k = 0.1
gamma = 0.1
mu = 0.1
beta = 0.2
alpha = 0.64
count = 6
lilie = OnesidedCoupling(par, t, keep, k, mu, gamma, alpha, beta)

xsol = lilie.x_solv()
ysol = lilie.y_solv()
psol = lilie.p_solv()
qsol = lilie.q_solv()

# selbstorganisation in der Zeit Werner Ebeling
rtol = 0.01
atol = 0.0000001
xsol_tol = lilie.duffvdpsolver_tolerance(rtol, atol)[:, 0]
xxsol_tol = lilie.duffvdpsolver_tolerance(0.1, atol)[:, 0]
xxxsol_tol = lilie.duffvdpsolver_tolerance(0.000001, atol)[:, 0] #  mit der Schrittweise ist es nahe am perfekten Wert, aber man kann um die 1/100 kürzer machen die Simulation (Für die Arnoldszunge ist es super praktisch die Zeit zu verkürzen)

# [Timeseries with different rtols]_________________________________________________________________________________________________________________________________

# xmax = lilie.maximumofplot()[0]
# ymax = lilie.maximumofplot()[1]
# x_amplitude = lilie.find_peaks_max()[0]
# y_amplitude = lilie.find_peaks_max()[1]

# # x-timeseries
# plt.plot(t, xsol, label = "rtol= auto")
# plt.plot(t, xsol_tol, label = "rtol= 0.01")
# plt.plot(t, xxsol_tol, label = "rtol= 0.1")
# plt.plot(t, xxxsol_tol, label = "rtol= 0.000001")
# # plt.plot([np.arange(0, t_last, t_step)[i] for i in x_amplitude[0]], x_amplitude[1]['peak_heights'], "x", label = "interpolated")
# # plt.plot([np.arange(0, t_last, t_step)[i] for i in x_amplitude[0]], x_amplitude[1]['peak_heights'], "x", label = "max peak")

# plt.ylabel("x in a.u.", fontsize = 20)
# x_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 16, loc = "upper right")
# plt.xlabel("t in s", fontsize = 20)

# plt.ylim([-3.5, 3.5])
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 16)
# print(x_title)
# plt.show()

# # y-timeseries
# plt.plot(np.arange(0, t_last, t_step), ysol, label = f"k: {k:.2f}")
# plt.plot([np.arange(0, t_last, t_step)[i] for i in y_amplitude[0]], y_amplitude[1]['peak_heights'], "x", label = "interpolated")
# plt.ylabel("y in a.u.", fontsize = 20)
# y_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 16, loc = "upper left")
# plt.xlabel("t in s", fontsize = 20)
# plt.ylim([-3.5, 3.5])
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 16)
# print(y_title)
# plt.show()

# [Timeseries]________________________________________________________________________________________________________________________________________________________________________

xmax = lilie.maximumofplot()[0]
ymax = lilie.maximumofplot()[1]
x_amplitude = lilie.find_peaks_max()[0]
y_amplitude = lilie.find_peaks_max()[1]

# x-timeseries
plt.plot(t[-keep:], xsol[-keep:], label = f"k: {k:.2f}")
# plt.plot([np.arange(0, t_last, t_step)[i] for i in x_amplitude[0]], x_amplitude[1]['peak_heights'], "x", label = "interpolated")
# plt.plot([np.arange(0, t_last, t_step)[i] for i in x_amplitude[0]], x_amplitude[1]['peak_heights'], "x", label = "max peak")

plt.ylabel("x in a.u.", fontsize = 20)
x_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 16, loc = "upper right")
plt.xlabel("t in s", fontsize = 20)

plt.ylim([-3.5, 3.5])
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
print(x_title)
plt.show()

# y-timeseries
plt.plot(t[-keep:], ysol[-keep:], label = f"k: {k:.2f}")
# plt.plot([np.arange(0, t_last, t_step)[i] for i in y_amplitude[0]], y_amplitude[1]['peak_heights'], "x", label = "interpolated")
plt.ylabel("y in a.u.", fontsize = 20)
y_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 16, loc = "upper left")
plt.xlabel("t in s", fontsize = 20)
plt.ylim([-3.5, 3.5])
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
print(y_title)
plt.show()

# [Timeseries with interpolation]__________________________________

# tx_plus = lilie.square_interpolation()[1][0]
# ty_plus = lilie.square_interpolation()[1][1]
# xing = lilie.square_interpolation()[0][0]
# ying = lilie.square_interpolation()[0][1]

# # x-timeseries
# plt.plot(np.arange(0, t_last, t_step), xsol, label = f"k: {k:.2f}")
# plt.plot(tx_plus, xing, "x", label = "find_peak")
# plt.ylabel("x in a.u.", fontsize = 20)
# x_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 16, loc = "upper right")
# plt.xlabel("t in ms", fontsize = 20)

# plt.ylim([-3.5, 3.5])
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 16)
# print(x_title)
# plt.show()

# # y-timeseries
# plt.plot(np.arange(0, t_last, t_step), ysol, label = f"k: {k:.2f}")
# plt.plot(ty_plus, ying, "x", label = "find_peak")
# plt.ylabel("y in a.u.", fontsize = 20)
# y_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 16, loc = "upper left")
# plt.xlabel("t in ms", fontsize = 20)
# plt.ylim([-3.5, 3.5])
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 16)
# print(y_title)
# plt.show()


# [Phasetime]____________________________________________________________________________________________________________________________________________________________________________________________________________-

# label = f"x$_0$ = {par[0]:.2f} \np$_0$ = {par[2]:.2f}"
# plt.plot(xsol[-keep:],psol[-keep:],label = label)
# plt.xlabel("x in a.u.",fontsize = 30)
# plt.ylabel("p in a.u.",fontsize = 30)
# # title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 16)
# plt.title("Phasenportraits X,P")
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 20)
# plt.show()

# label = f"y$_0$ = {par[1]:.2f} \nq$_0$ = {par[3]:.2f}"
# plt.plot(ysol[-keep:],qsol[-keep:],label = label)
# plt.xlabel("y in a.u.",fontsize = 30)
# plt.ylabel("q in a.u.",fontsize = 30)
# # title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 16)
# plt.title("Phasenportraits Y,Q")
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 20)
# plt.show()

# # [Resonance Curve]___________________________________________________________________________________________________________________________________________________________________________________-

# reso_alpha = np.arange(0.2, 4, 0.02)
# omega = [np.sqrt(i) for i in reso_alpha]
# # revers_omega = [i for i in reversed(omega)]

# amp = [np.mean(OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][1]['peak_heights'][-6:]) for i in reso_alpha]

# time_amp = [t[k] for k in [OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][0][-10:] for i in reso_alpha]]

# # reverse_amp = [np.mean(OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][1]['peak_heights'][-6:]) for i in reversed(reso_alpha)]
# plt.plot(omega, amp)
# # plt.plot(revers_omega, reverse_amp)
# plt.xlabel("$\omega _0$ in Hz", fontsize = 30)
# plt.ylabel("A in a.u.",fontsize = 30)
# plt.show()


# [Phasedifference]_____________________________________________________________________________________________________________________________________________________________________-
