import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel
import lmfit
import requests

def histogram(data, filename, binwidth):
    '''
    plot a histogram of all the emissive decays via matplotlib;
    return the set of bin values and edges so we can fit them after
    '''
    # normalise so that the max intensity is 1
    (n, bins, patches)= plt.hist(data,
            bins=np.arange(0., np.max(data) + binwidth,
                binwidth), histtype="step", color='C0')
    plt.gca().set_ylabel("Counts")
    plt.gca().set_xlabel("Time (ps)")
    plt.gca().set_yscale('log')
    plt.gca().set_xlim([1.0, np.max(data)])
    plt.savefig(filename)
    plt.close()
    return n, bins

def Convol(x, h):
    X = np.fft.fft(x)
    H = np.fft.fft(h)
    return np.real(np.fft.ifft(X * H))

def monoexprisemodel(x, tau_1, a_1, y0, x0, irf):
    ymodel=np.zeros(x.size)
    t=x
    c=x0
    n=len(irf)
    irf_s11 = (1 - c + np.floor(c)) * np.roll(irf, int(np.floor(c)))
    irf_s22 = (c - np.floor(c)) * np.roll(irf, int(np.ceil(c)))
    irf_shift = irf_s11 + irf_s22
    irf_reshaped_norm=irf_shift/sum(irf_shift)
    ymodel = a_1 * np.exp(-x / float(tau_1))
    z=Convol(ymodel,irf_reshaped_norm)
    z+=y0
    return z

def biexprisemodel(x, tau_1, a_1, tau_2, a_2, y0, x0, irf):
    ymodel=np.zeros(x.size)
    t=x
    c=x0
    n=len(irf)
    irf_s11 = (1 - c + np.floor(c)) * np.roll(irf, int(np.floor(c)))
    irf_s22 = (c - np.floor(c)) * np.roll(irf, int(np.ceil(c)))
    irf_shift = irf_s11 + irf_s22
    irf_reshaped_norm=irf_shift/sum(irf_shift)
    ymodel = a_1 * np.exp(-x / float(tau_1))
    ymodel+= a_2 * np.exp(-x / float(tau_2))
    z=Convol(ymodel,irf_reshaped_norm)
    z+=y0
    return z

def dfda1(a_1, a_2, tau_1, tau_2):
    return (tau_1 * (a_1 + a_2) - (a_1 * tau_1 + a_2 * tau_2)/(a_1 + a_2)**2)

def dfda2(a_1, a_2, tau_1, tau_2):
    return (tau_2 * (a_1 + a_2) - (a_1 * tau_1 + a_2 * tau_2)/(a_1 + a_2)**2)

def dfdt1(a_1, a_2):
    return a_1 / (a_1 + a_2)

def dfdt2(a_1, a_2):
    return a_2 / (a_1 + a_2)

def error(fit):
    a_1 = fit.best_values["a_1"]
    a_2 = fit.best_values["a_2"]
    tau_1 = fit.best_values["tau_1"]
    tau_2 = fit.best_values["tau_2"]
    sigma = fit.covar[:4, :4]
    # order in fit.covar is tau_1, a_1, tau_2, a_2
    j = np.array([dfdt1(a_1, a_2), dfda1(a_1, a_2, tau_1, tau_2), dfdt2(a_1, a_2), dfda2(a_1, a_2, tau_1, tau_2)])
    m = np.matmul(j, sigma)
    error = np.sqrt(np.matmul(m, j.transpose()))
    return error
