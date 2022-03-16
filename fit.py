import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel
import lmfit
import requests

def histogram(data, filename, binwidth=1.):
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
    plt.gca().set_xscale('log')
    plt.gca().set_xlim([1.0, np.max(data)])
    plt.savefig(filename)
    plt.close()
    return n, bins

def Convol(x, h):
    X = np.fft.fft(x)
    H = np.fft.fft(h)
    return np.real(np.fft.ifft(X * H))

def biexprisemodel(x, tau_1, a_1, tau_2, a_2, y0, x0, irf):
    ymodel=np.zeros(x.size)
    t=x
    c=x0
    n=len(irf)
    dt = x[1] - x[0] # assumes even spacing!
    # there's an issue here - if dt != 1 it doesn't seem to work right
    irf_s1 = np.remainder(np.remainder(t - np.floor(c) - dt, n) + n, n)
    irf_s11=(1-c+np.floor(c))*irf[irf_s1.astype(int)]
    irf_s2=np.remainder(np.remainder(t-np.ceil(c)-1,n)+n,n)
    irf_s22=(c-np.floor(c))*irf[irf_s2.astype(int)]
    irf_shift=irf_s11+irf_s22
    with open("out/irf_shift.dat", "a") as f:
        np.savetxt(f, irf_shift)
        f.write("\n")
    # irf_shift=irf
    irf_reshaped_norm=irf_shift/sum(irf_shift)
    ymodel = a_1 * np.exp(-x / float(tau_1))
    ymodel+= a_2 * np.exp(-x / float(tau_2))
    with open("out/ymodel.dat", "a") as f:
        np.savetxt(f, ymodel)
        f.write("\n")
    z=Convol(ymodel,irf_reshaped_norm)
    z+=y0
    with open("out/z.dat", "a") as f:
        np.savetxt(f, z)
        f.write("\n")

    return z

if __name__ == "__main__":
    url = requests.get('https://groups.google.com/group/lmfit-py/attach/73a983d40ad945b1/tcspcdatashifted.csv?part=0.1&authuser=0')    
    xvals,decay1,irf=np.loadtxt(url.iter_lines(),delimiter=',',unpack=True,dtype='float')
    np.savetxt("out/irf.dat", irf)
    mod = lmfit.Model(biexprisemodel, independent_vars=('x', 'irf'))
    pars = mod.make_params(tau_1=10,a_1=1000,tau_2=10,a_2=1000,y0=0,x0=10)
    pars['x0'].vary =True
    pars['y0'].vary =True
    weights = 1/np.sqrt(decay1 + 1)

    print(pars)

    # fit this model with weights, initial parameters
    result = mod.fit(decay1,params=pars,weights=weights,method='leastsq',x=xvals, irf=irf)

    # print results
    print(result.fit_report())

    # plot results
    plt.figure(5)
    plt.subplot(2,1,1)
    plt.semilogy(xvals,decay1,'r-',xvals,result.best_fit,'b')
    plt.subplot(2,1,2)
    plt.plot(xvals,result.residual)
    plt.savefig("out/exper.pdf")
