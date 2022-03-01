import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel
import lmfit

def histogram(data, filename, binwidth=50.):
    '''
    plot a histogram of all the emissive decays via matplotlib;
    return the set of bin values and edges so we can fit them after
    '''
    # normalise so that the max intensity is 1
    (n, bins, patches)= plt.hist(data,
            bins=np.arange(np.min(data), np.max(data) + binwidth,
                binwidth), histtype="step", color='C0')
    plt.gca().set_ylabel("Counts")
    plt.gca().set_xlabel("Time (ps)")
    plt.gca().set_yscale('log')
    plt.gca().set_xlim([0.0, 10000.0])
    plt.savefig(filename)
    plt.close()
    return n, bins

def saturation(x, amp, k):
    p = amp * (1 - np.exp(-k * x))
    p[np.where(x > 2. / k)] = 0.
    return p

def cutexp(x, amp, k):
    p = amp * np.exp(-k * x)
    # that 100 shouldn't be hardcoded lol
    p[np.where(x < 100.)] = 0.
    return p

def Convol(x, h):
    X = np.fft.fft(x)
    H = np.fft.fft(h)
    return np.real(np.fft.ifft(X * H))

def biexprisemodel(x, tau_1, a_1, tau_2, a_2, y0, x0, irf):
    ymodel=np.zeros(x.size)
    t=x
    c=x0
    n=len(irf)
    # not 100% sure what these lines are doing, honestly
    irf_s1=np.remainder(np.remainder(t-np.floor(c)-1, n)+n,n)
    irf_s11=(1-c+np.floor(c))*irf[irf_s1.astype(int)]
    irf_s2=np.remainder(np.remainder(t-np.ceil(c)-1,n)+n,n)
    irf_s22=(c-np.floor(c))*irf[irf_s2.astype(int)]
    irf_shift=irf_s11+irf_s22
    irf_reshaped_norm=irf_shift/sum(irf_shift)

    ymodel = a_1*np.exp(-(x)/tau_1)
    ymodel+= a_2*np.exp(-(x)/tau_2)
    z=Convol(ymodel,irf_reshaped_norm)
    z+=y0
    return z

def lm(no_exp, x, y, model, pulse_mu):
    ''' use lmfit to a mono or biexponential '''
    if no_exp == 1:
        exp1 = ExponentialModel(prefix='exp1')
        pars = exp1.make_params(exp1decay=1./model.g_pool,
                                exp1amplitude=np.max(y))
        mod = exp1
    if no_exp == 2:
        exp1 = ExponentialModel(prefix='exp1')
        pars = exp1.make_params(exp1decay=1./model.g_pool,
                                exp1amplitude=np.max(y))
        exp2 = ExponentialModel(prefix='exp2')
        pars.update(exp2.make_params(exp2decay=1./model.k_ann,
                                     exp2amplitude=np.max(y)))
        mod = exp1 + exp2
    if no_exp == 3:
        # exp2 = ExponentialModel(prefix='exp2')
        exp1 = ExponentialModel(prefix='exp1')
        pars = exp1.make_params(exp1decay=1./model.g_pool,
                                exp1amplitude=np.max(y))
        exp2 = lmfit.Model(cutexp, prefix='exp2')
        pars.update(exp2.make_params(exp2k=1./model.k_ann,
                                     exp2amp=np.max(y)))
        rise = lmfit.Model(saturation, prefix='rise')
        pars.update(rise.make_params(riseamp=np.max(y),
                                     risek=1./(2. * pulse_mu)))
        mod = exp1 + exp2 + rise
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    return out
