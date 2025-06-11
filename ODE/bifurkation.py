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
k = np.arange(0,30, 0.1)
gamma = 0.1
mu = 4
beta = 0.5
alpha = 0.1
T = OnesidedCoupling((1,1,1,1), t, keep, 1, mu, gamma, alpha, beta).period(10)[1]

t_start = 4000
poincarepoints = 1000 
times = t_start + np.arange(poincarepoints) * T

duffing = []

for j in k:
    sol = OnesidedCoupling((1,1,1,1), times, keep, j, mu, gamma, alpha, beta).y_solv()
    duffing.append(sol)

for e, j in enumerate(k):
    plt.plot([j]*poincarepoints, duffing[e],'k.', markersize=0.5)

plt.xlabel("k", fontsize = 25)
plt.ylabel("x (Poincaré Map)", fontsize = 25)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.savefig("Bifurkation" + ".png", dpi =  300, bbox_inches = "tight")
plt.show()
