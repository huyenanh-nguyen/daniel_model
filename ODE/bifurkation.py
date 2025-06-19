import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelmax, argrelextrema
from scipy.signal import find_peaks
from scipy.fft import fft
import math
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from onesidedcoupling import OnesidedCoupling


t_step = 0.01
t_last = 100 # 50h -> 1 point represent 1h
t = np.arange(0, 4000, t_step)
keep = int(t_last / t_step)
k_up = np.arange(0.05,0.25, 0.005)
k_down = k_up[::-1]
gamma = 0.2
mu = 2
beta = 0.5
alpha = np.arange(0.01,0.5, 0.05)

def compute_amplitude(par, t, keep, k, mu, gamma, alpha, beta):
    amp = OnesidedCoupling(par, t, keep, k, mu, gamma, alpha, beta).find_peaks_max()[1][1]['peak_heights'][-10:]
    return amp



par0 = 1,1.4,1.4,1
amplitudes_up = []
amplitudes_down = []

for f in k_up:
    sol = OnesidedCoupling(par0, t, keep, f, mu, gamma, alpha[h], beta).duffvdpsolver()
    par0 = sol[-1]
    amplitudes_up.append(compute_amplitude(par0, t, keep, f, mu, gamma,alpha[h], beta))


par0 = sol[-1]

for j in k_down:
    sol = OnesidedCoupling(par0, t, keep, j, mu, gamma, alpha[h], beta).duffvdpsolver()
    par0 = sol[-1]  
    amplitudes_down.append(compute_amplitude(par0, t, keep, j, mu, gamma, alpha[h], beta))

for e,k in enumerate(k_up):
    try:
        plt.plot([k]*10, amplitudes_up[e], "o", color = "#283D3B", markersize= 3)
    except:
        None

for j,w in enumerate(k_down):
    try:
        plt.plot([w]*10, amplitudes_down[j],'o', color = "#D64D27",  markersize= 3, alpha = 0.8)
    except:
        None



plt.xlabel("k", fontsize = 20)
plt.ylabel("A$_{y}$ in a.u.", fontsize = 20)
plt.xticks(np.linspace(0.05,0.25, 5), fontsize = 18)
# plt.title(label = "$\\alpha$ = " + f"{alpha[h]:.4f}" + ", $\\omega$ = " + f"{np.sqrt(alpha[h]):.4f}", fontsize = 20)

plt.yticks(fontsize = 18)
plt.savefig("Bifurcation" +  ".png", dpi =  300, bbox_inches = "tight")
#plt.show()

