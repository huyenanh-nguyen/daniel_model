# linear coupling of Duffing Oscillator and Van der Pol Oscillator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelmax
from scipy.signal import find_peaks
from scipy.fft import fft

def linearduffingvdp(par, t, k, mu, gamma, alpha, beta):
    x,y,p,q = par
    dx = p
    dy = q
    dp = mu * (1-x**2)*p - x + k * (y-x)
    dq = - gamma * q - alpha * y - beta * y**3 + k * (x-y)

    return [dx,dy,dp,dq]

class LinearCoupling:
    """
    Coupling Duffing Oscillator with the Van der Pol Oscillator.
    Coupling is linear.

    I will couple those two equation in a non-linear way with the couple parameter k.

    Van der Pol:
        x" - mu * (1-x^2)x' + x = k(y - x)
    Duffing:
        y" + gamma * y' + alpha * y + beta y^3 = k(x - y)
    """

    def __init__(self, par, t, t_keep, k, mu, gamma, alpha, beta):
        self.par = par
        self.t = t
        self.t_keep = t_keep
        self.k = k
        self.mu = mu
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def duffvdpsolver(self):
        """solving coupled ODE f Duffing and VdP 

        Returns:
           List: index 0 -> x, Index 1 -> y, Index 2 -> p(vdp), Index 3 -> q(duffing)
        """
        par = self.par
        t = self.t
        k = self.k
        gamma = self.gamma
        alpha = self.alpha
        mu = self.mu
        beta = self.beta

        sol = odeint(linearduffingvdp, par, t, args = (k, mu, gamma, alpha, beta))

        return sol
    
    def x_solv(self):
        sol = self.duffvdpsolver()
        x_solv = sol[:,0]

        return x_solv
    
    def y_solv(self):
        sol = self.duffvdpsolver()
        y_solv = sol[:,1]

        return y_solv
    
    def p_solv(self):
        sol = self.duffvdpsolver()
        p_solv = sol[:,2]

        return p_solv
    
    def q_solv(self):
        sol = self.duffvdpsolver()
        q_solv = sol[:,3]

        return q_solv
    
    def maximumofplot(self):
        """
        return the maximum value of the plot

        Returns:
           List: index 0 -> x, Index 1 -> y, Index 2 -> p(vdp), Index 3 -> q(duffing)
        """

        sol = self.duffvdpsolver()

        maxima = [np.argmax(sol[:, i]) for i in range(sol.shape[1])]
        
        return maxima
    
    
    def find_peaks_max(self):
        """
        returning for all parameters the Index of the Peaks and the height where the peaks are.


        Returns:
            List: A list of the Parameters and within the list it returns the peak_index and the properties. Is there an X given, 
            there will be a dict with the key "peak_heights", and that is what we need for the amplitudes.\n
            So when i want to get the peak height of the x_plot or y_plot then i have to type:
            peaks[0][1]['peak_heights'] 
            0 -> x
            1 -> y
            2 -> p
            3 -> q

            second index is for accessing the peak height (aka amplitude).

            If i just want to know the index where to find the index, i will type:
            peaks[0][0]
        """
        sol = self.duffvdpsolver()
        maxima = self.maximumofplot()

        peaks = [find_peaks(sol[:,i], height=(-np.repeat(sol[:,i][maxima[i]], len(sol[:,i])), np.repeat(sol[:,i][maxima[i]], len(sol[:,i])))) for i in range(len(maxima))]
         
        return peaks
    

    def period(self):
        """
        Calculatig the period of the oscillator
        (Averaging the difference of two peaks through the giving time)
        unit is ms

        alpha ist dann omega^2 = k/m , m ist 1

        Returns:
            List: index 0 -> x, Index 1 -> y, Index 2 -> p(vdp), Index 3 -> q(duffing)
        """

        t = self.t
        sol = self.duffvdpsolver()
        index_peaks = [self.find_peaks_max()[i][0] for i in range(len(self.find_peaks_max()))]


        period = []

        for i in range(sol.shape[1]):
            peaks = [t[k] for k in index_peaks[i]]
            period.append(np.mean(np.diff(peaks)))

        return period
    
    def frequence(self):
        """
        calculating the frequency (1/T)
        Unit is mHz

        Returns:
            List: index 0 -> x, Index 1 -> y, Index 2 -> p(vdp), Index 3 -> q(duffing)
        """
        period = self.period()

        frequence = [1/ period[i] for i in range(len(period))]

        return frequence
    
    def omegachen(self):
        """
        calculating omega of the oscillator (2 * np.pi/T)
        unit is 1/ms

        Returns:
            List: index 0 -> x, Index 1 -> y, Index 2 -> p(vdp), Index 3 -> q(duffing)
        """
        period = self.period()

        omega = [(2 * np.pi )/ period[i] for i in range(len(period))]

        return omega

    


# alpha = 1

# resonazphänomene (ist der Ausblick was passiert wenn duffing 70 hz und vdp bei 140 hz schwingt)


# [find peaks]____________________________________________________________________________________________________________________________________________________________________________________________________________
# [time]
# t_step = 0.01
# t_last = 250 # 50h -> 1 point represent 1h
# t = np.arange(0, 5000, t_step)
# keep = int(t_last / t_step)


# x = 1
# y = 1
# q = 1
# p = 1
# par = x,y,p,q
# k = 5
# gamma = 0.1
# mu = 0.1
# beta = 0.2
# alpha = 2.5

# lilie = LinearCoupling(par, t,keep, k, mu, gamma, alpha, beta)
# xsol = lilie.x_solv()[:keep]
# ysol = lilie.y_solv()[:keep]
# psol = lilie.p_solv()[:keep]
# qsol = lilie.q_solv()[:keep]

# x_max = np.argmax(xsol)
# x_amplitude = find_peaks(xsol, height=(-np.repeat(xsol[x_max], keep), np.repeat(xsol[x_max], keep)))
# plt.plot([np.arange(0, t_last, t_step)[i] for i in x_amplitude[0]], x_amplitude[1]['peak_heights'], "x")
# plt.plot(np.arange(0, t_last, t_step), xsol)
# plt.plot(np.arange(0, t_last, t_step),-np.repeat(xsol[x_max], keep),":", color="gray")
# plt.plot(np.arange(0, t_last, t_step),np.repeat(xsol[x_max], keep),":", color="gray")
# title = "k = "+ f"{k:.2f} $\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.xlabel("t in ms", fontsize = 20)
# plt.ylabel("x in a.u.", fontsize = 20)
# plt.show()

# y_max = np.argmax(ysol)
# y_amplitude = find_peaks(ysol, height=(-np.repeat(ysol[:keep][y_max], keep), np.repeat(ysol[:keep][y_max], keep)))
# plt.plot([np.arange(0, t_last, t_step)[i] for i in y_amplitude[0]], y_amplitude[1]['peak_heights'], "x")
# plt.plot(np.arange(0, t_last, t_step), ysol)
# plt.plot(np.arange(0, t_last, t_step),-np.repeat(ysol[y_max], keep),":", color="gray")
# plt.plot(np.arange(0, t_last, t_step),np.repeat(ysol[y_max], keep),":", color="gray")
# title = "k = "+ f"{k:.2f} $\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.xlabel("t in ms", fontsize = 20)
# plt.ylabel("y in a.u.", fontsize = 20)
# plt.show()

# p_max = np.argmax(psol)
# p_amplitude = find_peaks(psol, height=(-np.repeat(psol[p_max], keep), np.repeat(psol[p_max], keep)))
# plt.plot([np.arange(0, t_last, t_step)[i] for i in p_amplitude[0]], p_amplitude[1]['peak_heights'], "x")
# plt.plot(np.arange(0, t_last, t_step), psol)
# plt.plot(np.arange(0, t_last, t_step),-np.repeat(psol[p_max], keep),":", color="gray")
# plt.plot(np.arange(0, t_last, t_step),np.repeat(psol[p_max], keep),":", color="gray")
# title = "k = "+ f"{k:.2f} $\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.xlabel("t in ms", fontsize = 20)
# plt.ylabel("p in a.u.", fontsize = 20)
# plt.show()

# q_max = np.argmax(qsol)
# q_amplitude = find_peaks(qsol, height=(-np.repeat(qsol[q_max], keep), np.repeat(qsol[q_max], keep)))
# plt.plot([np.arange(0, t_last, t_step)[i] for i in q_amplitude[0]], q_amplitude[1]['peak_heights'], "x")
# plt.plot(np.arange(0, t_last, t_step), qsol)
# plt.plot(np.arange(0, t_last, t_step),-np.repeat(qsol[q_max], keep),":", color="gray")
# plt.plot(np.arange(0, t_last, t_step),np.repeat(qsol[q_max], keep),":", color="gray")
# title = "k = "+ f"{k:.2f} $\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.xlabel("t in ms", fontsize = 20)
# plt.ylabel("q in a.u.", fontsize = 20)
# plt.show()


# [find omega, period]_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
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
# k = np.arange(0,2.5, 0.01)
# gamma = 0.1
# mu = 0.1
# beta = 0.2
# alpha = 2.5

# index = 3 # 0 = x, 1 = y, 2 = p, 3 = q

# for i in range(len(k)):
#     lilie = LinearCoupling(par, t, keep, k[i], mu, gamma, alpha, beta)
#     period = lilie.period()
#     frequency = lilie.frequence()
#     omega = lilie.omegachen()


    # print("amplitude", [f"{np.mean(lilie.find_peaks_max()[k][1]['peak_heights']):.2f}" for k in range(4)])
    # print("k", f"{k[i]:.2f}")
    # print("period", f"{period[index]:.2f}")
    # print("frequency", f"{frequency[index]:.2f}")
    # print("omega", f"{omega[index]:.2f}")


# [plotting values against time]_________________________________________________________________________________________________________________________

# [par changes]__________________________
# n = 3
# x = [np.random.uniform(-1,1) for i in range(n)]
# y = [np.random.uniform(-1,1) for i in range(n)]
# q = [np.random.uniform(-1,1) for i in range(n)]
# p = [np.random.uniform(-1,1) for i in range(n)]
        
# par = []
# for i in range(n):
#     par.append((x[i],y[i],p[i],q[i]))
# k = [2.5]
# gamma = 0.1
# mu = 0.1
# beta = 0.2
# alpha = 2.5
# lilie = LinearCoupling(par[0], t, keep, k, mu, gamma, alpha, beta)
# x_sol = lilie.x_solv()
# y_sol = lilie.y_solv()
# p_sol = lilie.p_solv()
# q_sol = lilie.q_solv()
# plt.plot(np.arange(0, t_last, t_step), x_sol[:keep], label = "x(t)")
# plt.plot(np.arange(0, t_last, t_step), y_sol[:keep], label = "y(t)")
# plt.plot(np.arange(0, t_last, t_step), p_sol[:keep], label = "p(t)")
# plt.plot(np.arange(0, t_last, t_step), q_sol[:keep], label = "q(t)")
# title = "k = " + f"{k:.2f} \n$\gamma$ = " + f"{gamma:.2f} \n$\mu$ = " + f"{mu:.2f} \nß =" + f"{beta:.2f} \nalpha = " + f"{alpha:.2f} \nx$_0$ = " + f"{par[0][0]:.2f} \ny$_0$ = "+ f"{par[0][1]:.2f} \np$_0$ = "+ f"{par[0][2]:.2f} \nq$_0$ = "+ f"{par[0][3]:.2f}"
# plt.legend()
# plt.xlabel("t in ms")
# plt.ylabel("[x,y,p,q] in a.u.")
# plt.text(0,1,title, fontsize = 10)
# plt.show()

# [kchanges]__________________________
# nicht lineare phänomene characterisieren wollen wir
t_step = 0.01
t_last = 250 # 50h -> 1 point represent 1h
t = np.arange(0, 5000, t_step)
keep = int(t_last / t_step)

x = 1
y = 1
q = 1
p = 1
par = x,y,p,q
k = [0.10, 10]
gamma = 0.01
mu = 0.1
beta = 0.2
alpha = 1
x_sol = [LinearCoupling(par, t, keep, i, mu, gamma, alpha, beta).x_solv() for i in k]
y_sol = [LinearCoupling(par, t, keep, i, mu, gamma, alpha, beta).y_solv() for i in k]
p_sol = [LinearCoupling(par, t, keep, i, mu, gamma, alpha, beta).p_solv() for i in k]
q_sol = [LinearCoupling(par, t, keep, i, mu, gamma, alpha, beta).q_solv() for i in k]

for i in range(len(k)):
    plt.plot(np.arange(0, t_last, t_step), x_sol[i][-keep:], label = f"k: {k[i]:.2f}")
plt.ylabel("x in a.u.", fontsize = 20)
title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 20, loc = "upper right")
plt.xlabel("t in ms", fontsize = 20)
plt.figtext(0.99, 0.01, title,
        horizontalalignment="right",
        fontsize = 16)
plt.show()
for i in range(len(k)):
    plt.plot(np.arange(0, t_last, t_step), y_sol[i][-keep:], label = f"k: {k[i]:.2f}")
plt.ylabel("y in a.u.", fontsize = 20)
title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 20, loc = "upper left")
plt.xlabel("t in ms", fontsize = 20)
plt.figtext(0.99, 0.01, title,
        horizontalalignment="right",
        fontsize = 16)
plt.show()
for i in range(len(k)):
    plt.plot(np.arange(0, t_last, t_step), p_sol[i][-keep:], label = f"k: {k[i]:.2f}")
plt.ylabel("p in a.u.", fontsize = 20)
title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 20, loc = "upper right")
plt.xlabel("t in ms", fontsize = 20)
plt.figtext(0.99, 0.01, title,
        horizontalalignment="right",
        fontsize = 16)
plt.show()
for i in range(len(k)):
    plt.plot(np.arange(0, t_last, t_step), q_sol[i][-keep:], label = f"k: {k[i]:.2f}")
plt.ylabel("q in a.u.", fontsize = 20)
title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
plt.legend(fontsize = 20, loc = "upper left")
plt.xlabel("t in ms", fontsize = 20)
plt.figtext(0.99, 0.01, title,
        horizontalalignment="right",
        fontsize = 16)
plt.show()

#
# [plotting Phasespace]_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# t_step = 0.01
# t_last = 100 # 50h -> 1 point represent 1h
# t = np.arange(0, 5000, t_step)
# keep = int(t_last / t_step)

# x = [1]
# y = [1]
# q = [1]
# p = [1]
# par = []
# for i in range(len(x)):
#     par.append((x[i],y[i],p[i],q[i]))

# k = 0
# gamma = 0.1
# mu = 0.1
# beta = 0.2
# alpha = 2.5
# x_sol = [LinearCoupling(i, t, keep, k, mu, gamma, alpha, beta).x_solv() for i in par]
# y_sol = [LinearCoupling(i, t, keep, k, mu, gamma, alpha, beta).y_solv() for i in par]
# p_sol = [LinearCoupling(i, t, keep, k, mu, gamma, alpha, beta).p_solv() for i in par]
# q_sol = [LinearCoupling(i, t, keep, k, mu, gamma, alpha, beta).q_solv() for i in par]

# for i in range(len(x)):
#     label = f"x$_0$ = {par[i][0]:.2f} \ny$_0$ = {par[i][1]:.2f}"
#     plt.plot(x_sol[i][-keep:],y_sol[i][-keep:],label = label)
# plt.xlabel("x in a.u.",fontsize = 20)
# plt.ylabel("y in a.u.",fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 20)
# plt.title("Phasenportraits X,Y")
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()

# for i in range(len(x)):
#     label = f"y$_0$ = {par[i][1]:.2f} \nq$_0$ = {par[i][3]:.2f}"
#     plt.plot(y_sol[i][-keep:],q_sol[i][-keep:],label = label)
# plt.xlabel("y in a.u.",fontsize = 20)
# plt.ylabel("q in a.u.",fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 20)
# plt.title("Phasenportraits Y,Q")
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()

# for i in range(len(x)):
#     label = f"x$_0$ = {par[i][0]:.2f} \np$_0$ = {par[i][2]:.2f}"
#     plt.plot(x_sol[i][-keep:],p_sol[i][-keep:],label = label)
# plt.xlabel("x in a.u.",fontsize = 20)
# plt.ylabel("p in a.u.",fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 20)
# plt.title("Phasenportraits X,P")
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()

# for i in range(len(x)):
#     label = f"p$_0$ = {par[i][2]:.2f} \nq$_0$ = {par[i][3]:.2f}"
#     plt.plot(p_sol[i][-keep:],q_sol[i][-keep:],label = label)
# plt.xlabel("p in a.u.",fontsize = 20)
# plt.ylabel("q in a.u.",fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 20)
# plt.title("Phasenportraits P,Q")
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()



"""
only the time intervall:
    t = np.arange(0, t_last, t_step)
if i want to see the first keep points (to include the transient phase):
    x_sol[:keep]

if i want to see the last keep points (to remove the transient phase):
    x_sol[-keep:]

but because singing and holding the voice range is quite short, i think the transient phase is quite important.
"""