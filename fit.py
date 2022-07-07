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
    pars = mod.make_params(tau_1 = 1./rates.g_pool,
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
        if error is None:
            error = np.nan
        r = open("{}/{}_mono_report.dat".format(path, file_prefix), "w")
        r.write(result.fit_report())
        r.write("\nLifetime (mono) = {} +/- {} ps".format(lifetime, error))
        r.close()
        print("Lifetime (mono) = {} +/- {} ps".format(lifetime, error))
        plt.figure()
        # plt.subplot(2, 1, 1)
        plt.semilogy(xvals, histvals, label="hist")
        plt.semilogy(xvals, result.best_fit, label="fit")
        plt.gca().set_xlim([0., 2000.])
        # plt.subplot(2, 1, 2)
        # plt.plot(xvals, result.residual, label="residuals")
        plt.savefig("{}/{}_mono_fit.pdf".format(path, file_prefix))
        plt.close()
    except ValueError:
        print("Monoexponential fit failed!")
        lifetime = np.nan
        error = np.nan
        result = None
    except TypeError:
        print("Monoexponential fit couldn't estimate covariances!")
        lifetime = np.nan
        error = np.nan
        result = None
    return (result, [fluence, lifetime, error])

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
        if error is None:
            error = np.nan
        r = open("{}/{}_bi_report.dat".format(path, file_prefix), "w")
        r.write(result.fit_report())
        r.write("\nLifetime (bi) = {} +/- {} ps".format(lifetime, error))
        r.close()
        print("Lifetime (bi) = {} +/- {} ps".format(lifetime, error))
        plt.figure()
        # plt.subplot(2, 1, 1)
        plt.semilogy(xvals, histvals, label="hist")
        plt.semilogy(xvals, result.best_fit, label="fit")
        plt.gca().set_xlim([0., 2000.])
        # plt.subplot(2, 1, 2)
        # plt.plot(xvals, result.residual, label="residuals")
        plt.savefig("{}/{}_bi_fit.pdf".format(path, file_prefix))
        plt.close()
    except ValueError:
        print("Biexponential fit failed!")
        lifetime = np.nan
        error = np.nan
        result = None
    except TypeError:
        print("Biexponential fit couldn't estimate covariances!")
        lifetime = np.nan
        error = np.nan
        result = None
    return (result, [fluence, lifetime, error])

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
        if error is None:
            error = np.nan
        r = open("{}/{}_tri_report.dat".format(path, file_prefix), "w")
        r.write(result.fit_report())
        r.write("\nLifetime (tri) = {} +/- {} ps".format(lifetime, error))
        r.close()
        print("Lifetime (tri) = {} +/- {} ps".format(lifetime, error))
        plt.figure()
        # plt.subplot(2, 1, 1)
        plt.semilogy(xvals, histvals, label="hist")
        plt.semilogy(xvals, result.best_fit, label="fit")
        plt.gca().set_xlim([0., 2000.])
        # plt.subplot(2, 1, 2)
        # plt.plot(xvals, result.residual, label="residuals")
        plt.savefig("{}/{}_tri_fit.pdf".format(path, file_prefix))
        plt.close()
    except ValueError:
        print("Triexponential fit failed!")
        lifetime = np.nan
        error = np.nan
        result = None
    except TypeError:
        print("Triexponential fit couldn't estimate covariances!")
        lifetime = np.nan
        error = np.nan
        result = None
    return (result, [fluence, lifetime, error])

def plot_fits(m, b, t, histvals, xvals, key, filename):
    fig, ax = plt.subplots()
    plt.semilogy(xvals, histvals, lw=2.0, color='k', marker='o', fillstyle='none', label="Counts")
    fluence = 0.
    if m[0] is not None:
        plt.semilogy(xvals, m[0].best_fit, label=r'Mono: $ \tau = $' + '{:4.2f}'.format(m[1][1]) + r'$ \pm $' + '{:4.2f}'.format(m[1][2]))
        fluence = m[1][0]
    if b[0] is not None:
        plt.semilogy(xvals, b[0].best_fit, label=r'Bi: $ \tau = $' + '{:4.2f}'.format(b[1][1]) + r'$ \pm $' + '{:4.2f}'.format(b[1][2]))
        fluence = b[1][0]
    if t[0] is not None:
        plt.semilogy(xvals, t[0].best_fit, label=r'Tri: $ \tau = $' + '{:4.2f}'.format(t[1][1]) + r'$ \pm $' + '{:4.2f}'.format(t[1][2]))
        fluence = t[1][0]

    plt.legend(prop={'size': 16})
    ax.set_xlim([0., 2000.])
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Counts (norm.)")
    plt.title("Model = {}, fluence = {:4.2E}".format(key, fluence))
    fig.savefig("{}_fits.pdf".format(filename))


'''
NB: could be possible to add one ExponentialModel at a time, name the parameters programatically
via e.g. "a{:1d}".format(n), then use parameter hints to add the lifetime via
set_param_hint('tau', expr='a{:1d} * tau{:1d}'.format(n)).
would have to add all the as and taus together though. can it do the derivative of this symbolically?
'''
# def tabulate(mono, bi, tri):
#     from lmfit import Parameter
#     '''take the results from each of the three fits and
#     sort them all out a bit.
#     how?
#     we could use the parameter class from lmfit and make a dict of them.
#     have tau/a_1 through 3, set them all to nan, then replace them all
#     with the corresponding values from each fit.
#     then do the same for the amplitude weighted lifetime (intensity weighted?)
#     and report the one with the lowest error along with its components.
#     '''
#     params = [
#             Parameter("tau_1", value=np.nan),
#             Parameter("a_1", value=np.nan),
#             Parameter("tau_2", value=np.nan),
#             Parameter("a_2", value=np.nan),
#             Parameter("tau_3", value=np.nan),
#             Parameter("a_3", value=np.nan),
#             Parameter("x0", value=np.nan),
#             Parameter("y0", value=np.nan),
#             ]
#     err = np.inf
#     min_err = 0
#     for i, fit in enumerate([mono, bi, tri]):
#         res = fit[0].best_values
#         # get the one with the smallest error
#         if fit[1][2] < err:
#             err = fit[1][2]
#             min_err = i
#         # this doesn't work yet but should look something like this
#         for key in res:
#             if key in params:
#                 params[key] = res[key]
