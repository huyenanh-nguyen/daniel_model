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
t = np.arange(0, 500, t_step)
keep = int(t_last / t_step)
k_up = np.arange(0.05,0.25, 0.005)
k_down = k_up[::-1]
gamma = 0.2
mu = 2
beta = 0.5
alpha = np.arange(0.01,0.5, 0.01)

def compute_amplitude(par, t, keep, k, mu, gamma, alpha, beta):
    amp = OnesidedCoupling(par, t, keep, k, mu, gamma, alpha, beta).peak()[1][1]['peak_heights'][-10:]

    return amp

def approx(listofnum, tol):
     mean = np.mean(listofnum)
     boolean = [abs(i - mean) / mean * 100 for i in listofnum]
     
     for num in boolean:
          if num > tol:
               return math.nan
          else:
               return mean



colours = ["#F8E854", "#43838C", "#433078"] # ["#F8E854", "#57AD82", "#43838C", "#433078"]
cmap = ListedColormap(colours)

par0 = 1,1.4,1.4,1

y,q = np.meshgrid(k_up, alpha)
up = np.zeros_like(y)
down = np.zeros_like(y)

index = []
for l in range(len(alpha)):
        for m in range(len(k_up)):
            sol = OnesidedCoupling(par0, t, keep, k_up[m], mu, gamma, alpha[l], beta).duffvdpsolver()
            par0 = sol[-1]
            amp = approx(compute_amplitude(par0, t, keep, k_up[m], mu, gamma, alpha[l], beta), 0.2)
            if math.isnan(amp) == True:
                index.append([l,m])
            up[l,m] = amp
    
from matplotlib.ticker import FormatStrFormatter
plt.imshow(up, extent=[min(k_up),max(k_up),min(alpha),max(alpha)], cmap = "viridis", origin='lower')
plt.xlabel("k in a.u.",fontsize = 25)
plt.ylabel("$\\omega$ in Hz",fontsize = 25)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks( np.linspace(round(min(k_up),2),round(max(k_up),2), 3), fontsize = 18)
plt.yticks(np.linspace(min(alpha),max(alpha), 5),fontsize = 18)
plt.colorbar()
# labels = ["small LC", "big LC", "unknown"]
# patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(len(colours))]
# plt.legend(handles=patches, loc='upper right', fontsize = 14)

# for i in range(0, up.shape[0]):
#     for j in range(0, up.shape[1]):
#         c = up[j,i]
#         plt.text(i, j, str(c), va='center', ha='center')

plt.savefig("parameterrectangle_up" + ".png", dpi =  300, bbox_inches = "tight")

print(index)   


