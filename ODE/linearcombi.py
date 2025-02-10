# linear coupling of Duffing Oscillator and Van der Pol Oscillator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelmax
from scipy.signal import find_peaks

def linearduffingvdp(par, t, k, mu, gamma, alpha, beta):
    x,y,p,q = par
    dx = p
    dy = q
    dp = mu * (1-x**2)*p - x + k * (y-x)
    dq = gamma * q - alpha * y - beta * y**3 + k * (x-y)

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

    def __init__(self, par, t, k, mu, gamma, alpha, beta):
        self.par = par
        self.t = t
        self.k = k
        self.mu = mu
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def duffvdpsolver(self):
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
    




# [time]
t_step = 0.01
t_last = 100 # 50h -> 1 point represent 1h
t = np.arange(0, 5000, t_step)
keep = int(t_last / t_step)

x = 1
y = 1
q = 1
p = 1
par = x,y,p,q
k = 2.5
gamma = 0.1
mu = 0.1
beta = 0.2
alpha = 2.5

lilie = LinearCoupling(par, t, k, mu, gamma, alpha, beta)
maxi = np.argmax(lilie.q_solv()[:keep])
print(lilie.q_solv()[:keep][maxi])
amplitude = find_peaks(lilie.q_solv()[:keep], height=(-np.repeat(lilie.q_solv()[:keep][maxi], keep), np.repeat(lilie.q_solv()[:keep][maxi], keep)))

plt.plot([np.arange(0, t_last, t_step)[i] for i in amplitude[0]], amplitude[1]['peak_heights'], "x")
plt.plot(np.arange(0, t_last, t_step), lilie.q_solv()[:keep])
plt.plot(np.arange(0, t_last, t_step),-np.repeat(lilie.q_solv()[:keep][maxi], keep),":", color="gray")
plt.plot(np.arange(0, t_last, t_step),np.repeat(lilie.q_solv()[:keep][maxi], keep),":", color="gray")
plt.show()

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
# lilie = LinearCoupling(par[0], t, k, mu, gamma, alpha, beta)
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
# plt.xlabel("t in h")
# plt.ylabel("[x,y,p,q] in a.u.")
# plt.text(0,1,title, fontsize = 10)
# plt.show()

# [kchanges]__________________________

# x = 1
# y = 1
# q = 1
# p = 1
# par = x,y,p,q
# k = [5]
# gamma = 0.1
# mu = 0.1
# beta = 0.2
# alpha = 2.5
# x_sol = [LinearCoupling(par, t, i, mu, gamma, alpha, beta).x_solv() for i in k]
# y_sol = [LinearCoupling(par, t, i, mu, gamma, alpha, beta).y_solv() for i in k]
# p_sol = [LinearCoupling(par, t, i, mu, gamma, alpha, beta).p_solv() for i in k]
# q_sol = [LinearCoupling(par, t, i, mu, gamma, alpha, beta).q_solv() for i in k]

# for i in range(len(k)):
#     plt.plot(np.arange(0, t_last, t_step), x_sol[i][:keep], label = f"k: {k[i]:.2f}")
# plt.ylabel("x in a.u.", fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 20)
# plt.xlabel("t in h", fontsize = 20)
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()
# for i in range(len(k)):
#     plt.plot(np.arange(0, t_last, t_step), y_sol[i][:keep], label = f"k: {k[i]:.2f}")
# plt.ylabel("y in a.u.", fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 20)
# plt.xlabel("t in h", fontsize = 20)
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()
# for i in range(len(k)):
#     plt.plot(np.arange(0, t_last, t_step), p_sol[i][:keep], label = f"k: {k[i]:.2f}")
# plt.ylabel("p in a.u.", fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 20)
# plt.xlabel("t in h", fontsize = 20)
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()
# for i in range(len(k)):
#     plt.plot(np.arange(0, t_last, t_step), q_sol[i][:keep], label = f"k: {k[i]:.2f}")
# plt.ylabel("q in a.u.", fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} x$_0$ = " + f"{par[0]:.2f} y$_0$ = "+ f"{par[1]:.2f} p$_0$ = "+ f"{par[2]:.2f} q$_0$ = "+ f"{par[3]:.2f}"
# plt.legend(fontsize = 20)
# plt.xlabel("t in h", fontsize = 20)
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()

# [plotting Phasespace]_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# x = [1]
# y = [1]
# q = [1]
# p = [1]
# par = []
# for i in range(len(x)):
#     par.append((x[i],y[i],p[i],q[i]))

# k = 5
# gamma = 0.1
# mu = 0.1
# beta = 0.2
# alpha = 2.5
# x_sol = [LinearCoupling(i, t, k, mu, gamma, alpha, beta).x_solv() for i in par]
# y_sol = [LinearCoupling(i, t, k, mu, gamma, alpha, beta).y_solv() for i in par]
# p_sol = [LinearCoupling(i, t, k, mu, gamma, alpha, beta).p_solv() for i in par]
# q_sol = [LinearCoupling(i, t, k, mu, gamma, alpha, beta).q_solv() for i in par]

# for i in range(len(x)):
#     label = f"x = {par[i][0]:.2f} \ny = {par[i][1]:.2f}"
#     plt.plot(x_sol[i][:keep],y_sol[i][:keep],label = label)
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
#     label = f"y = {par[i][1]:.2f} \np = {par[i][2]:.2f}"
#     plt.plot(y_sol[i][:keep],p_sol[i][:keep],label = label)
# plt.xlabel("y in a.u.",fontsize = 20)
# plt.ylabel("p in a.u.",fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 20)
# plt.title("Phasenportraits Y,P")
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()

# for i in range(len(x)):
#     label = f"p = {par[i][2]:.2f} \nq = {par[i][3]:.2f}"
#     plt.plot(p_sol[i][:keep],q_sol[i][:keep],label = label)
# plt.xlabel("p in a.u.",fontsize = 20)
# plt.ylabel("q in a.u.",fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 20)
# plt.title("Phasenportraits P,Q")
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()

# for i in range(len(x)):
#     label = f"x = {par[i][0]:.2f} \nq = {par[i][3]:.2f}"
#     plt.plot(x_sol[i][:keep],q_sol[i][:keep],label = label)
# plt.xlabel("x in a.u.",fontsize = 20)
# plt.ylabel("q in a.u.",fontsize = 20)
# title = "$\gamma$ = " + f"{gamma:.2f} $\mu$ = " + f"{mu:.2f} ß =" + f"{beta:.2f} alpha = " + f"{alpha:.2f} k = " + f"{k:.2f}"
# plt.legend(fontsize = 20)
# plt.title("Phasenportraits X,Q")
# plt.figtext(0.99, 0.01, title,
#         horizontalalignment="right",
#         fontsize = 16)
# plt.show()