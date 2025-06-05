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
    return amp


# [Hysteresis]_______________________________________________________________________________________

# par0 = 1,1,1,1
# amplitudes_up = []
# amplitudes_down = []

# for f in alpha_up:
#     sol = OnesidedCoupling(par0, t, keep, k, mu, gamma, f, beta).duffvdpsolver()
#     par0 = sol[-1]
#     print(par0)
#     amplitudes_up.append(compute_amplitude(par0, t, keep, k, mu, gamma, f, beta))


# par0 = sol[-1]
# for j in alpha_down:
#     sol = OnesidedCoupling(par0, t, keep, k, mu, gamma, j, beta).duffvdpsolver()
#     par0 = sol[-1]  
#     print(par0)
#     amplitudes_down.append(compute_amplitude(par0, t, keep, k, mu, gamma, j, beta))


# plt.plot(np.sqrt(alpha_up), amplitudes_up, label="Increasing ω")
# plt.plot(np.sqrt(alpha_down), amplitudes_down, label="Decreasing ω")
# plt.xlabel("$\omega _0$ in Hz", fontsize = 30)
# plt.ylabel("A in a.u.", fontsize = 30)
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.legend(fontsize = 30)
# plt.show()


# [Basin of Attractors]______________________________________________________________________________________________________

alph = [0.05, 0.1, 0.15, 0.2]

colours = ["#283D3B", "#F8D794"]

cmap = ListedColormap(colours)
for i in range(len(alph)):
    x_par = np.arange(-2,2,0.1)
    y_par = np.arange(-2,2,0.1)
    p_par = np.arange(-2,2,0.1)
    q_par = np.arange(-2,2,0.1)
  

    y,p = np.meshgrid(y_par,p_par)
    attractor = np.zeros_like(y)
    y_amplitude_matrix = np.zeros_like(y)
    par0 = -1, -1, -1, -1

    sol = max(OnesidedCoupling(par0, t, keep, k, mu, gamma, alph[i], beta).duffvdpsolver()[-keep:, 1])

    for l in range(len(y_par)):
        for m in range(len(p_par)):

            par0 = [x_par[l], y_par[m], p_par[l], q_par[m]]
            y_amplitude = compute_amplitude(par0, t, keep, k, mu, gamma, alph[i], beta)
            y_amplitude_matrix[l,m] = round(y_amplitude, 2)
            
            if math.isclose(y_amplitude, sol, rel_tol=1e-1):
                attractor[l,m] = 1

            
            else: 
                attractor[l,m] = 0
    
    print(sol)
    print(y_amplitude_matrix)
    
    plt.imshow(attractor, extent=[-2,2,-2,2], cmap = cmap)
    plt.xlabel("y in a.u.",fontsize = 30)
    plt.ylabel("q in a.u.",fontsize = 30)
    plt.title(label = "$\\alpha$ = " + f"{alph[i]:.4f}" + ", $\\omega$ = " + f"{np.sqrt(alph[i]):.4f}", fontsize = 20)
    plt.xticks(np.linspace(-2,2, 5), fontsize = 20)
    plt.yticks(np.linspace(-2,2, 5),fontsize = 20)
    
    labels = [f"A = {attractor.min():.2f}", f"A = {attractor.max():.2f}"]
    patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(len(colours))]
    plt.legend(handles=patches, loc='upper right', fontsize = 16)
    
    plt.savefig("Basin_detailed" + f"{i}" + ".png", dpi =  300, bbox_inches = "tight")
            
