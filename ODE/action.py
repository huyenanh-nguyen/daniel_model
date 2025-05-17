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
mu = 2
beta = 0.2
alpha = 3.8800000000000003
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
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
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
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.ylim([-3.5, 3.5])
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 16)
# print(y_title)
# plt.show()

# [Timeseries]________________________________________________________________________________________________________________________________________________________________________

# xmax = lilie.maximumofplot()[0]
# ymax = lilie.maximumofplot()[1]
# x_amplitude = lilie.find_peaks_max()[0]
# y_amplitude = lilie.find_peaks_max()[1]

# # x-timeseries
# plt.plot(t[-keep:], xsol[-keep:], label = f"k: {k:.2f}")
# # plt.plot([np.arange(0, t_last, t_step)[i] for i in x_amplitude[0]], x_amplitude[1]['peak_heights'], "x", label = "interpolated")
# # plt.plot([np.arange(0, t_last, t_step)[i] for i in x_amplitude[0]], x_amplitude[1]['peak_heights'], "x", label = "max peak")

# plt.ylabel("x in a.u.", fontsize = 20)
# x_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 16, loc = "upper right")
# plt.xlabel("t in s", fontsize = 20)
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.ylim([-3.5, 3.5])
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 16)
# print(x_title)
# plt.show()

# # y-timeseries
# plt.plot(t[-keep:], ysol[-keep:], label = f"k: {k:.2f}")
# # plt.plot([np.arange(0, t_last, t_step)[i] for i in y_amplitude[0]], y_amplitude[1]['peak_heights'], "x", label = "interpolated")
# plt.ylabel("y in a.u.", fontsize = 20)
# y_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 16, loc = "upper left")
# plt.xlabel("t in s", fontsize = 20)
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.ylim([-0.05, 0.05])
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 16)
# print(y_title)
# plt.show()

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
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
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
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
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
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
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
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.title("Phasenportraits Y,Q")
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 20)
# plt.show()

# # [Resonance Curve]___________________________________________________________________________________________________________________________________________________________________________________-

reso_alpha = np.arange(0.2, 8, 0.01) # 3.6, 10, 0.02
omega = [np.sqrt(i) for i in reso_alpha]
# revers_omega = [i for i in reversed(omega)]

amp = [np.mean(OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][1]['peak_heights'][-6:]) for i in reso_alpha]

# print(OnesidedCoupling(par, t, keep, k, mu, gamma, 3.9, beta).y_solv()[-keep:][OnesidedCoupling(par, t, keep, k, mu, gamma, 3.9, beta).maximumofplot()[1]])
# print(OnesidedCoupling(par, t, keep, k, mu, gamma, 3.9, beta).find_peaks_max())
# for i in reso_alpha:
#     plt.plot(t[-keep:], OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).y_solv()[-keep:]) 
#     # print(OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][0][-6:])
#     plt.plot([t[-keep:][u] for u in OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][0][-6:]],OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][1]['peak_heights'][-6:], "x", label = "$\\alpha$: " + str(round(i, 2))) 

# plt.ylabel("y in a.u.", fontsize = 20)
# plt.legend(fontsize = 16, loc = "upper left")
# plt.xlabel("t in ms", fontsize = 20)
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# # plt.ylim([-3.5, 3.5])
# plt.show()

# time_amp = [t[k] for k in [OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][0][-10:] for i in reso_alpha]]



# def lorenz(x, omega, gamma):
#     return 0.02 / np.sqrt((x**2 - omega ** 2)**2 + gamma ** 2 * x ** 2)

# reso_mu = np.arange(0.2, 10, 0.1)
# # # lorenz_sol = [lorenz(i, 1, 0.1) for i in omega]
# mu_amp = [np.mean(OnesidedCoupling(par, t, keep, k, i, gamma, 0.2, beta).find_peaks_max()[0][1]['peak_heights'][-10:]) for i in reso_mu]

# plt.plot(reso_mu, mu_amp)
# # # plt.plot(omega, lorenz_sol, label = "Lorentz Curve")
# # plt.plot(omega, amp)
# # plt.legend(fontsize = 20, loc = "upper right")
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# # plt.xlabel("$\omega _0$ in Hz", fontsize = 30)
# plt.xlabel("$\mu$ in a.u.", fontsize = 30)
# plt.ylabel("T in ms",fontsize = 30)
# plt.show()


# [Phasedifference]_____________________________________________________________________________________________________________________________________________________________________-
# y-timeseries
# plt.plot(t[-keep:], OnesidedCoupling(par, t, keep, k, mu, gamma, 3.86, beta).y_solv()[-keep:], label = "$\\alpha$: 3.86") 
# plt.plot(t[-keep:], OnesidedCoupling(par, t, keep, k, mu, gamma, 3.8800000000000003, beta).y_solv()[-keep:], label = "$\\alpha$: 3.88")
# plt.plot(t[-keep:], OnesidedCoupling(par, t, keep, k, mu, gamma, 3.9000000000000004, beta).y_solv()[-keep:], label = "$\\alpha$: 3.90")
# plt.plot(t[-keep:], OnesidedCoupling(par, t, keep, k, mu, gamma, 4, beta).y_solv()[-keep:], label = "$\\alpha$: 3.94")
# # plt.plot([np.arange(0, t_last, t_step)[i] for i in y_amplitude[0]], y_amplitude[1]['peak_heights'], "x", label = "interpolated")
# plt.ylabel("y in a.u.", fontsize = 20)
# y_title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 16, loc = "upper left")
# plt.xlabel("t in ms", fontsize = 20)
# plt.ylim([-0.009, 0.009])
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# # plt.figtext(0.99, 0.01, title,
# #         horizontalalignment="right",
# #         fontsize = 16)
# print(y_title)
# plt.show()

period_vdp = lilie.period(10)[1]
time_amp_vdp = [t[i] for i in lilie.find_peaks_max()[1][0][-10:]]

time_amp = [t[k] for k in [OnesidedCoupling(par, t, keep, k, mu, gamma, i, beta).find_peaks_max()[1][0][-10:] for i in reso_alpha]]
phaseamp = [2 * np.pi * (time_amp_vdp[0]-i[0])/period_vdp for i in time_amp]

plt.plot(omega[3:], phaseamp[3:])
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel("$\omega$", fontsize = 30)
plt.ylabel("$\\varphi$",fontsize = 30)
plt.show()