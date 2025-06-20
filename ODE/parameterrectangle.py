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
k_up = np.arange(0.05,0.25, 0.001)
k_down = k_up[::-1]
gamma = 0.2
mu = 2
beta = 0.5
alpha = np.arange(0.01,0.5, 0.005)

def compute_amplitude(par, t, keep, k, mu, gamma, alpha, beta):
    amp = OnesidedCoupling(par, t, keep, k, mu, gamma, alpha, beta).peak()[1][1]['peak_heights'][-10:]

    return amp

def approx(listofnum, tol):
     mean = np.mean(listofnum)
     boolean = [abs(i - mean) / mean * 100 for i in listofnum]
     
     for num in boolean:
          if num > tol:
               return 0
          else:
               return mean



colours = ["#6CA5DE", "#F8D794", "#D15C64"]
cmap = ListedColormap(colours)

par0 = 1,1.4,1.4,1

q,y = np.meshgrid(k_up, alpha)
up = np.zeros_like(y)
down = np.zeros_like(y)


for l in range(len(alpha)):
        for m in range(len(k_up)):
            
            amp = approx(compute_amplitude(par0, t, keep, k_up[m], mu, gamma, alpha[l], beta), 0.1)
            up[l,m] = round(amp, 2)
    

plt.imshow(up, extent=[min(k_up),max(k_up),min(alpha),max(alpha)], cmap = cmap, origin='lower')
plt.xlabel("k in a.u.",fontsize = 25)
plt.ylabel("$\\omega$ in Hz",fontsize = 25)
plt.xticks( np.linspace(min(k_up),max(k_up), 5), fontsize = 18)
plt.yticks(np.linspace(min(alpha),max(alpha), 5),fontsize = 18)

labels = ["$\overline{A}_{10}$ =" + f"{np.nanmin(up):.2f}", "$\overline{A}_{10}$ =" + f"{np.nanmax(up):.2f}", "$\overline{A}_{10}$ = unknown"]
patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(len(colours))]
plt.legend(handles=patches, loc='upper right', fontsize = 14)

plt.savefig("parameterrectangle" + ".png", dpi =  300, bbox_inches = "tight")
        


