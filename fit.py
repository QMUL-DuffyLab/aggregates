import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
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
    ymodel  = a_1 * np.exp(-x / float(tau_1))
    ymodel += a_2 * np.exp(-x / float(tau_2))
    z=Convol(ymodel,irf_reshaped_norm)
    z+=y0
    return z

def triexprisemodel(x, tau_1, a_1, tau_2, a_2, tau_3, a_3, y0, x0, irf):
    ymodel=np.zeros(x.size)
    t=x
    c=x0
    n=len(irf)
    irf_s11 = (1 - c + np.floor(c)) * np.roll(irf, int(np.floor(c)))
    irf_s22 = (c - np.floor(c)) * np.roll(irf, int(np.ceil(c)))
    irf_shift = irf_s11 + irf_s22
    irf_reshaped_norm=irf_shift/sum(irf_shift)
    ymodel  = a_1 * np.exp(-x / float(tau_1))
    ymodel += a_2 * np.exp(-x / float(tau_2))
    ymodel += a_3 * np.exp(-x / float(tau_3))
    z=Convol(ymodel,irf_reshaped_norm)
    z+=y0
    return z

def dfda1(a_1, a_2, a_3, tau_1, tau_2, tau_3):
    return ((tau_1 * (a_1 + a_2 + a_3)
            - (a_1 * tau_1 + a_2 * tau_2 + a_3 * tau_3))
            /(a_1 + a_2 + a_3)**2)

def dfda2(a_1, a_2, a_3, tau_1, tau_2, tau_3):
    return ((tau_2 * (a_1 + a_2 + a_3)
            - (a_1 * tau_1 + a_2 * tau_2 + a_3 * tau_3))
            /(a_1 + a_2 + a_3)**2)

def dfda3(a_1, a_2, a_3, tau_1, tau_2, tau_3):
    return ((tau_3 * (a_1 + a_2 + a_3)
            - (a_1 * tau_1 + a_2 * tau_2 + a_3 * tau_3))
            /(a_1 + a_2 + a_3)**2)

def dfdt1(a_1, a_2, a_3):
    return a_1 / (a_1 + a_2 + a_3)

def dfdt2(a_1, a_2, a_3):
    return a_2 / (a_1 + a_2 + a_3)

def dfdt3(a_1, a_2, a_3):
    return a_3 / (a_1 + a_2 + a_3)

def bi_error(fit):
    a_1 = fit.best_values["a_1"]
    a_2 = fit.best_values["a_2"]
    tau_1 = fit.best_values["tau_1"]
    tau_2 = fit.best_values["tau_2"]
    sigma = fit.covar[:4, :4]
    # order in fit.covar is tau_1, a_1, tau_2, a_2
    j = np.array([
        dfdt1(a_1, a_2, 0.),
        dfda1(a_1, a_2, 0., tau_1, tau_2, 0.),
        dfdt2(a_1, a_2, 0.),
        dfda2(a_1, a_2, 0., tau_1, tau_2, 0.)
        ])
    m = np.matmul(j, sigma)
    error = np.sqrt(np.matmul(m, j.transpose()))
    return error

def tri_error(fit):
    a_1 = fit.best_values["a_1"]
    a_2 = fit.best_values["a_2"]
    a_3 = fit.best_values["a_3"]
    tau_1 = fit.best_values["tau_1"]
    tau_2 = fit.best_values["tau_2"]
    tau_3 = fit.best_values["tau_3"]
    sigma = fit.covar[:6, :6]
    j = np.array([
        dfdt1(a_1, a_2, a_3),
        dfda1(a_1, a_2, a_3, tau_1, tau_2, tau_3),
        dfdt2(a_1, a_2, a_3),
        dfda2(a_1, a_2, a_3, tau_1, tau_2, tau_3),
        dfdt3(a_1, a_2, a_3),
        dfda3(a_1, a_2, a_3, tau_1, tau_2, tau_3)
        ])
    m = np.matmul(j, sigma)
    error = np.sqrt(np.matmul(m, j.transpose()))
    return error

def monofit(histvals, rates, xvals, irf, fluence, path, file_prefix):
    weights = 1/np.sqrt(histvals + 1)
    mod = Model(monoexprisemodel, independent_vars=('x', 'irf'))
    pars = mod.make_params(tau_1 = 1./rates.k_ann,
            a_1 = 1., y0 = 0., x0 = 0)
    pars['x0'].vary = True
    pars['y0'].vary = True
    try:
        result = mod.fit(histvals, params=pars, weights=weights, 
                method='leastsq', x=xvals, irf=irf)
        print(result.fit_report())
        res = result.best_values
        lifetime = res["tau_1"]
        error = result.params["tau_1"].stderr
        print("Lifetime (mono) = {} +/- {} ps".format(lifetime, error))
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.semilogy(xvals, histvals, label="hist")
        plt.semilogy(xvals, result.best_fit, label="fit")
        plt.subplot(2, 1, 2)
        plt.plot(xvals, result.residual, label="residuals")
        plt.savefig("{}/{}_mono_fit.pdf".format(path, file_prefix))
        plt.close()
    except ValueError:
        print("Monoexponential fit failed!")
        lifetime = np.nan
        error = np.nan
    except TypeError:
        print("Monoexponential fit couldn't estimate covariances!")
        lifetime = np.nan
        error = np.nan
    return [fluence, lifetime, error]

def bifit(histvals, rates, xvals, irf, fluence, path, file_prefix):
    weights = 1/np.sqrt(histvals + 1)
    mod = Model(biexprisemodel, independent_vars=('x', 'irf'))
    pars = mod.make_params(tau_1 = 1./rates.k_ann, a_1 = 1.,
            tau_2 = 1./rates.g_pool, a_2 = 1., y0 = 0., x0 = 0)
    pars['x0'].vary = True
    pars['y0'].vary = True
    try:
        result = mod.fit(histvals, params=pars, weights=weights, 
                method='leastsq', x=xvals, irf=irf)
        print(result.fit_report())
        res = result.best_values
        lifetime = ((res["a_1"] * res["tau_1"] + res["a_2"] * res["tau_2"])
                / (res["a_1"] + res["a_2"]))
        error = bi_error(result)
        print("Lifetime (bi) = {} +/- {} ps".format(lifetime, error))
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.semilogy(xvals, histvals, label="hist")
        plt.semilogy(xvals, result.best_fit, label="fit")
        plt.subplot(2, 1, 2)
        plt.plot(xvals, result.residual, label="residuals")
        plt.savefig("{}/{}_bi_fit.pdf".format(path, file_prefix))
        plt.close()
    except ValueError:
        print("Biexponential fit failed!")
        lifetime = np.nan
        error = np.nan
    except TypeError:
        print("Biexponential fit couldn't estimate covariances!")
        lifetime = np.nan
        error = np.nan
    return [fluence, lifetime, error]

def trifit(histvals, rates, xvals, irf, fluence, path, file_prefix):
    weights = 1/np.sqrt(histvals + 1)
    mod = Model(triexprisemodel, independent_vars=('x', 'irf'))
    pars = mod.make_params(tau_1 = 1./rates.k_ann, a_1 = 1.,
            tau_2 = 1./rates.g_pool, a_2 = 1., 
            tau_3=500., a_3 = 1., y0 = 0., x0 = 0)
    pars['x0'].vary = True
    pars['y0'].vary = True
    try:
        result = mod.fit(histvals, params=pars, weights=weights, 
                method='leastsq', x=xvals, irf=irf)
        print(result.fit_report())
        res = result.best_values
        lifetime = ((res["a_1"] * res["tau_1"] 
            + res["a_2"] * res["tau_2"]
            + res["a_3"] * res["tau_3"]) 
        / (res["a_1"] + res["a_2"] + res["a_3"]))
        error = tri_error(result)
        print("Lifetime (tri) = {} +/- {} ps".format(lifetime, error))
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.semilogy(xvals, histvals, label="hist")
        plt.semilogy(xvals, result.best_fit, label="fit")
        plt.subplot(2, 1, 2)
        plt.plot(xvals, result.residual, label="residuals")
        plt.savefig("{}/{}_tri_fit.pdf".format(path, file_prefix))
        plt.close()
    except ValueError:
        print("Triexponential fit failed!")
        lifetime = np.nan
        error = np.nan
    except TypeError:
        print("Triexponential fit couldn't estimate covariances!")
        lifetime = np.nan
        error = np.nan
    return [fluence, lifetime, error]
