# linear coupling of Duffing Oscillator and Van der Pol Oscillator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelmax, argrelextrema
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.optimize import curve_fit

def linearduffingvdp(par : list, t : list, k : float, mu : float, gamma : float, alpha : float, beta : float):
    """Duffing Oscillator will get driven by the Van der Pol Oscillator.
    The coupling is linear and onesided.
        x -> Van der Pol Oscillator
        y -> Dufing Oscillator
        p -> Van der Pol Oscillator different Axis
        q -> Duffing Oscillator different Axis



    Args:
        par (List): Initial Condition x0, y0, p0, q0
        t (List): Timespan
        k (float): coupling strenght
        mu (float): non-linear damping constant
        gamma (_type_): damping constant
        alpha (_type_): linear restoring force
        beta (_type_): non linear restoring force

    Returns:
        List: ODEs for 
    """
    x,y,p,q = par
    dx = p
    dy = q
    dp = mu * (1-x**2)*p - x
    dq = - gamma * q - alpha * y - beta * y**3 + k * x

    return [dx,dy,dp,dq]

def quadraticinterpolation(t, a, b, c):
    return a * t**2 + b * t + c




class OnesidedCoupling:
    """
    Coupling Duffing Oscillator with the Van der Pol Oscillator.
    Coupling is linear.

    I will couple those two equation in a non-linear way with the couple parameter k.

    Van der Pol:
        x" - mu * (1-x^2)x' + x = 0
    Duffing:
        y" + gamma * y' + alpha * y + beta y^3 = kx
    
    We will then get for each oscillators two equations, because they both are equations 2nd order.
    
    Args:
        par (List): Initial Condition x0, y0, p0, q0
        t (List): Timespan
        k (float): coupling strenght
        mu (float): non-linear damping constant
        gamma (_type_): damping constant
        alpha (_type_): linear restoring force
        beta (_type_): non linear restoring force
    """

    def __init__(self, par : list, t : list, t_keep : int,  k : float, mu : float, gamma : float, alpha : float, beta : float):
        """_summary_

        Args:
            par (List): Initial Condition x0, y0, p0, q0
            t (List): Timespan
            t_keep (int): how many Timepoints to keep
            k (float): coupling strenght
            mu (float): non-linear damping constant
            gamma (_type_): damping constant
            alpha (_type_): linear restoring force
            beta (_type_): non linear restoring force

        """
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
    
    def duffvdpsolver_tolerance(self):
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

        sol = odeint(linearduffingvdp, par, t, args = (k, mu, gamma, alpha, beta), rtol= 0.01, atol= 0.00000001)

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
        np.argmax(xsol) for the maxima of the x-solution

        Returns:
           List: index 0 -> x, Index 1 -> y, Index 2 -> p(vdp), Index 3 -> q(duffing)
        """

        sol = self.duffvdpsolver()
        keep = self.t_keep

        maxima = [np.argmax(sol[:keep, i]) for i in range(len(self.par))]
        
        return maxima
    
    def minimumofplot(self):
        """
        return the mean minimum value of the plot

        Returns:
           List: index 0 -> x, Index 1 -> y, Index 2 -> p(vdp), Index 3 -> q(duffing)
        """

        sol = self.duffvdpsolver()
        keep = self.t_keep
        maxima = [np.mean(np.argmin(sol[:keep, i])) for i in range(sol.shape[1])]
        
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

            peak[i][0] will return the Index where the peaks are
        """
        sol = self.duffvdpsolver()
        maxima = self.maximumofplot()
        keep = self.t_keep

        peaks = []
        for i in range(len(maxima)):
            peaks.append(find_peaks(sol[:keep,i], height=(-np.repeat(sol[:keep,i][maxima[i]], keep), np.repeat(sol[:keep,i][maxima[i]], keep))))
        return peaks
    
    def peak_plusminus(self):
        """
        x-values are the time, where the peak is plus 1 and minus 1
        y-values are the fitting solutions to the timepoints

        Returns:
            List: two Indexes with n length of arrays -> depending on how many Peaks i have 
                  self.quadraticinterpolationofpeaks()
                  >>> [(t[i-1],t[i],t[i+1]), ...][(sol[i-1,u],sol[i, u],sol[i+1, u]), ...]
                      Index 0 -> time (and each for x,y,p,q)
                      Index 1 -> Solution (and each for x,y,p,q)
        """
        t = self.t
        sol = self.duffvdpsolver()
        keep = self.t_keep
        peaks = self.find_peaks_max()

        t_plusminuspeak = [[(t[i-1],t[i],t[i+1]) for i in peaks[u][0]] for u in range(len(self.par))]
        sol_plusminuspeak = [[(sol[i-1,u],sol[i, u],sol[i+1, u]) for i in peaks[u][0]] for u in range(len(self.par))]

        return t_plusminuspeak, sol_plusminuspeak
    

    def square_interpolation(self):
        t = self.peak_plusminus()[0]
        sol = self.peak_plusminus()[1]

        popt_sol = []
        y_value_peak = [] # interpolated y value where the peak is
        x_value_peak = []
        for i in range(len(self.par)):
            popt_inbetween = []
            y_peak_inbetween = []
            x_peak_inbetween = []
            for u in range(len(t[i])):
                popt, pcov = curve_fit(quadraticinterpolation, t[i][u], sol[i][u])
                max_interpolation = (- popt[1]) / (2 * popt[0])
                yuing = quadraticinterpolation(max_interpolation, popt[0], popt[1], popt[2])
                popt_inbetween.append(popt)
                y_peak_inbetween.append(yuing)
                x_peak_inbetween.append(max_interpolation)
            popt_sol.append(popt_inbetween)
            y_value_peak.append(y_peak_inbetween)
            x_value_peak.append(x_peak_inbetween)
        
        return y_value_peak, x_value_peak, popt_sol
        


    




