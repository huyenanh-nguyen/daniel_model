import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelmax, argrelextrema
from scipy.signal import find_peaks
from scipy.fft import fft
import math
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


from onesidedcoupling import OnesidedCoupling

t_step = 0.01
t_last = 100 # 50h -> 1 point represent 1h
t = np.arange(0, 4000, t_step)
keep = int(t_last / t_step)
k = 0.1
gamma = 0.2
mu = 2
beta = 0.5
alpha_up = np.arange(0,3, 0.01)
alpha_down = alpha_up[::-1]

def compute_amplitude(par, t, keep, k, mu, gamma, alpha, beta):
    amp = np.mean(OnesidedCoupling(par, t, keep, k, mu, gamma, alpha, beta).find_peaks_max()[1][1]['peak_heights'][-10:])
    # if math.isnan(amp):
    #     return 0
    
    # else:
    return amp


# [Hysteresis]_______________________________________________________________________________________

par0 = 1,1,1,1
amplitudes_up = []
amplitudes_down = []

for f in alpha_up:
    sol = OnesidedCoupling(par0, t, keep, k, mu, gamma, f, beta).duffvdpsolver()
    par0 = sol[-1]
    print(par0)
    amplitudes_up.append(compute_amplitude(par0, t, keep, k, mu, gamma, f, beta))


par0 = sol[-1]
for j in alpha_down:
    sol = OnesidedCoupling(par0, t, keep, k, mu, gamma, j, beta).duffvdpsolver()
    par0 = sol[-1]  
    print(par0)
    amplitudes_down.append(compute_amplitude(par0, t, keep, k, mu, gamma, j, beta))


plt.plot(np.sqrt(alpha_up), amplitudes_up, label="Increasing ω", color = "#283D3B")
plt.plot(np.sqrt(alpha_down), amplitudes_down, label="Decreasing ω", color = "#F8D794")
plt.xlabel("$\omega _0$ in Hz", fontsize = 25)
plt.ylabel("A in a.u.", fontsize = 25)
plt.axvline(x = np.sqrt(0.1), color = "r")
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 16)
plt.savefig("Hysteresis_withvertikalline" + ".png", dpi =  300, bbox_inches = "tight")
plt.show()


# [Basin of Attractors]______________________________________________________________________________________________________

# alph = [0.1]


# colours = ["#283D3B", "#F8D794"]

# cmap = ListedColormap(colours)
# for i in range(len(alph)):
#     x_par = np.arange(-3,3,0.1)
#     y_par = np.arange(-3,3,0.1)
#     p_par = np.arange(-3,3,0.1)
#     q_par = np.arange(-3,3, 0.1)
  

#     y,p = np.meshgrid(y_par,q_par)
#     attractor = np.zeros_like(y)
#     y_amplitude_matrix = np.zeros_like(y)

#     for l in range(len(q_par)):
#         for m in range(len(y_par)):

#             par0 = [x_par[m], y_par[m], p_par[l], q_par[l]]
#             y_amplitude = compute_amplitude(par0, t, keep, k, mu, gamma, alph[i], beta)
#             y_amplitude_matrix[l,m] = round(y_amplitude, 2)
    
#     # print(sol)
#     print(y_amplitude_matrix)
    
#     plt.imshow(y_amplitude_matrix, extent=[-3,3,-3,3], cmap = cmap)
#     plt.xlabel("y in a.u.",fontsize = 25)
#     plt.ylabel("q in a.u.",fontsize = 25)
#     plt.title(label = "$\\alpha$ = " + f"{alph[i]:.4f}" + ", $\\omega$ = " + f"{np.sqrt(alph[i]):.4f}", fontsize = 20)
#     plt.xticks(np.linspace(-3,3, 7), fontsize = 18)
#     plt.yticks(np.linspace(-3,3, 7),fontsize = 18)
    
#     labels = ["$\overline{A}_{10}$ =" + f"{np.nanmin(y_amplitude_matrix):.2f}", "$\overline{A}_{10}$ =" + f"{np.nanmax(y_amplitude_matrix):.2f}"]
#     patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(len(colours))]
#     plt.legend(handles=patches, loc='upper right', fontsize = 14)
    
#     plt.savefig("Basin_detailed" + str(i) + ".png", dpi =  300, bbox_inches = "tight")
            
