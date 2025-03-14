# linear coupling of Duffing Oscillator and Van der Pol Oscillator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelmax
from scipy.signal import find_peaks
from scipy.fft import fft
from linearcombi import LinearCoupling

# [dip at omega ~ 1]_________________________________________________________________________________________________________________________________
# [time]
# t_step = 0.01
# t_last = 250 # 50h -> 1 point represent 1h
# t = np.arange(3000, 5000, t_step)
# keep = int(t_last / t_step)

# # x_max(last 250 timepoints, k = 0) : np.int64(8550), np.int64(102), np.int64(15952), np.int64(1)

# x = 1
# y = 1
# q = 1
# p = 1
# par = x,y,p,q
# k = [0.01]
# gamma = 0.1
# mu = 0.1
# beta = 0.2
# alpha = 0.1

# index = 1 # 0 = x, 1 = y, 2 = p, 3 = q

# for i in range(len(k)):
#     lilie = LinearCoupling(par, t, keep, k[i], mu, gamma, alpha, beta)
#     period = lilie.period()
#     frequency = lilie.frequence()
#     omega = lilie.omegachen()


#     print("amplitude", [f"{np.mean(lilie.find_peaks_max()[k][1]['peak_heights']):.2f}" for k in range(4)])
#     print("k", f"{k[i]:.2f}")
#     print("period", f"{period[index]:.2f}")
#     print("frequency", f"{frequency[index]:.2f}")
#     print("omega", f"{omega[index]:.2f}")

# [kchanges]___________________________________________________________________________________________________________________________________________________________________________________________________
# nicht lineare phänomene characterisieren wollen wir omega^2 = alpha/m
t_step = 0.01
t_last = 250 # 50h -> 1 point represent 1h
t = np.arange(0, 5000, t_step)
keep = int(t_last / t_step)

x = 0.5
y = 1
q = 1
p = 0
par = x,y,p,q
k = [0.1]
gamma = 0.1
mu = 0.1
beta = 0.2
alpha = 1.1
x_sol = [LinearCoupling(par, t, keep, i, mu, gamma, alpha, beta).x_solv() for i in k]
y_sol = [LinearCoupling(par, t, keep, i, mu, gamma, alpha, beta).y_solv() for i in k]
p_sol = [LinearCoupling(par, t, keep, i, mu, gamma, alpha, beta).p_solv() for i in k]
q_sol = [LinearCoupling(par, t, keep, i, mu, gamma, alpha, beta).q_solv() for i in k]

for i in range(len(k)):
    plt.plot(np.arange(0, t_last, t_step), x_sol[i][-keep:], label = f"k: {k[i]:.2f}")
plt.ylabel("x in a.u.", fontsize = 30)
# title = "$\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, p$_0$ = "+ f"{par[2]:.2f}"

title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 30, loc = "upper right")
plt.xlabel("t in ms", fontsize = 30)
plt.ylim([-2.1, 2.1])
plt.figtext(0.99, 0.01, title,
        horizontalalignment="right",
        fontsize = 20)
plt.show()
for i in range(len(k)):
    plt.plot(np.arange(0, t_last, t_step), y_sol[i][-keep:], label = f"k: {k[i]:.2f}")
plt.ylabel("y in a.u.", fontsize = 30)
#title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f} y$_0$ = "+ f"{par[1]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 30, loc = "upper left")
plt.xlabel("t in ms", fontsize = 30)
plt.ylim([-2.1, 2.1])
plt.figtext(0.99, 0.01, title,
        horizontalalignment="right",
        fontsize = 20)
plt.show()
for i in range(len(k)):
    plt.plot(np.arange(0, t_last, t_step), p_sol[i][-keep:], label = f"k: {k[i]:.2f}")
plt.ylabel("p in a.u.", fontsize = 30)
# title = "$\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, p$_0$ = "+ f"{par[2]:.2f}"

title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 30, loc = "upper right")
plt.xlabel("t in ms", fontsize = 30)
plt.ylim([-2.1, 2.1])
plt.figtext(0.99, 0.01, title,
        horizontalalignment="right",
        fontsize = 20)
plt.show()
for i in range(len(k)):
    plt.plot(np.arange(0, t_last, t_step), q_sol[i][-keep:], label = f"k: {k[i]:.2f}")
plt.ylabel("q in a.u.", fontsize = 30)
#title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f} y$_0$ = "+ f"{par[1]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
title = "$\gamma$ = " + f"{gamma:.2f}, ß = " + f"{beta:.2f}, $\\alpha$ = " + f"{alpha:.2f}, $\mu$ = " + f"{mu:.2f}, x$_0$ = " + f"{par[0]:.2f}, y$_0$ = "+ f"{par[1]:.2f}, p$_0$ = "+ f"{par[2]:.2f}, q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 30, loc = "upper left")
plt.xlabel("t in ms", fontsize = 30)
plt.ylim([-2.1, 2.1])
plt.figtext(0.99, 0.01, title,
        horizontalalignment="right",
        fontsize = 20)
plt.show()