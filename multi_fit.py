# 22/08/22 - new fitting plan
import numpy as np
import sympy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

def Convol(x, h):
    X = np.fft.fft(x)
    H = np.fft.fft(h)
    return np.real(np.fft.ifft(X * H))

def lifetimes(n, names, bv, err, covar):
    '''
    given a set of n exponentials, construct the expressions for
    tau_amp and tau_int, calculate them with the best values `bv`,
    calculate the error on those, and return a dict with all the
    relevant information.
    '''
    strings = [[], [], []]
    # define ai, taui for i = 1, n
    sympy.symbols('tau:{:d}, a:{:d}'.format(n, n))
    # build up the expressions for ai, ai * taui, ai * taui^2
    for i in range(1, n + 1):
        for j in range(3):
            strings[j].append('a{:d} * tau{:d}**{:d}'.format(i, i, j))
    # turn the lists of strings into the relevant sympy expressions
    joined = [' + '.join(s) for s in strings]
    tau = [sympy.sympify(j, evaluate=False) for j in joined]
    # we need UnevaluatedExpr here otherwise sympy cancels the a1 for
    # the monoexponential fit and we never get out its value or error
    tau_amp = sympy.UnevaluatedExpr(tau[1]) / sympy.UnevaluatedExpr(tau[0])
    tau_int = sympy.UnevaluatedExpr(tau[2]) / sympy.UnevaluatedExpr(tau[1])
    # now start on relating these expressions to the fitted parameters
    j_amp = np.zeros(len(names))
    j_int = np.zeros(len(names))
    var = list(tau_int.free_symbols) # same for both amp and int
    tau_amp = tau_amp.doit()
    tau_int = tau_int.doit()
    # generate a list of tuples which tell sympy the values to substitute in
    repl = [(var[i], bv[str(var[i])]) for i in range(len(var))]
    # we're gonna return a dict which we turn into a pandas dataframe
    # then compare to find how many exponents gave the best fit
    d = {'n_exp': n}
    for i in range(len(var)):
        # dict key and index comparison require string representation
        s = str(var[i])
        # build up the dict as we go
        d[s] = bv[s]
        d[s + '_err'] = err[s]
        '''
        sympy doesn't order the variables ai, taui, but returns them as a set.
        they are ordered in curve_fit though - whatever order we put them in in,
        the covariance matrix etc is ordered the same way. so use `names` to find
        the right index to put the derivative in and use that.
        note that this also leaves the indices corresponding to x0 and y0 = 0,
        wherever they are in the list, so we don't need to worry about them.
        '''
        ind = np.nonzero(np.array(names) == s)[0][0]
        j_amp[ind] = sympy.diff(tau_amp, var[i]).subs(repl)
        j_int[ind] = sympy.diff(tau_int, var[i]).subs(repl)
    m_amp = np.matmul(j_amp, covar)
    m_int = np.matmul(j_int, covar)
    tau_amp_err = np.sqrt(np.matmul(m_amp, j_amp.transpose()))
    tau_int_err = np.sqrt(np.matmul(m_int, j_int.transpose()))
    d['tau_amp'] = tau_amp.subs(repl)
    d['tau_amp_err'] = tau_amp_err
    d['tau_int'] = tau_int.subs(repl)
    d['tau_int_err'] = tau_int_err
    print("n_exp = {:d}".format(n))
    print("tau_amp = {} +/- {} ps".format(tau_amp.subs(repl), tau_amp_err))
    print("tau_int = {} +/- {} ps".format(tau_int.subs(repl), tau_int_err))
    return d


def exp_model(x, *args):
    # args should be given as y0, x0, a_1, ..., a_n, tau_1, ..., tau_n
    print("exp_model: args = ", args)
    if (len(args) % 2 != 0):
        print("exp_model: number of args should be even - y0, x0, a_i, tau_i")
        return np.full(x.size, np.nan)
    if (len(args) == 0):
        print("exp_model: 0 arguments given - ???")
        return np.full(x.size, np.nan)
    y = np.zeros(x.size)
    t = x
    c = args[1]
    n_exp = (len(args) - 2) // 2
    for i in range(n_exp):
        y += args[i + 2] * np.exp(-x / float(args[i + 2 + (n_exp // 2)]))
    y+=args[0]
    return y

def reconv_model(X, *args):
    # args should be given as y0, x0, a_1, ..., a_n, tau_1, ..., tau_n
    x, irf = X
    if (len(args) % 2 != 0):
        print("reconv_model: number of args should be even - pairs of a_i, tau_i then y0, x0")
        return np.full(x.size, np.nan)
    if (len(args) == 0):
        print("reconv_model: 0 arguments given - ???")
        return np.full(x.size, np.nan)
    ymodel = np.zeros(x.size)
    t = x
    c = args[1]
    n = len(irf)
    irf_s11 = (1 - c + np.floor(c)) * np.roll(irf, int(np.floor(c)))
    irf_s22 = (c - np.floor(c)) * np.roll(irf, int(np.ceil(c)))
    irf_shift = irf_s11 + irf_s22
    irf_reshaped_norm=irf_shift/sum(irf_shift)
    # NB: need to figure out what exactly the difference is between the four lines above and this (from lifefit)
    #irf_shifted = (1 - irf_shift + np.floor(irf_shift)) * irf[np.fmod(np.fmod(channel - np.floor(irf_shift) - 1, n) + n, n).astype(int)] 
    #+ (irf_shift - np.floor(irf_shift)) * irf[np.fmod(np.fmod(channel - np.ceil(irf_shift) - 1, n) + n, n).astype(int)]
    n_exp = (len(args) - 2) // 2
    for i in range(n_exp):
        ymodel += args[i] * np.exp(-x / float(args[i + (n_exp // 2)])) 
    z=Convol(ymodel,irf_reshaped_norm)
    z+=args[0]
    return z

def fit(func, init, x, ydata, bounds):
    '''
    overall wrapper function. given a number n and a set of initial guesses
    for the time constants of the exponentials, this function generates the
    array p0 to pass to curve_fit, the ordered list of names which we use to
    calculate lifetimes later, then runs curve_fit, plots the result, calls
    the lifetime function to get amplitude- and intensity-weighted lifetimes,
    then returns a dictionary of the parameters and their errors along with
    popt from curve_fit, so we can plot all the fits together.
    '''
    n = len(init)
    a = 1. / n
    p0 = []
    names = []
    names.append("y0")
    names.append("x0")
    p0.append(0.) # y0
    p0.append(0.) # x0
    for i in range(n):
        names.append("a{:d}".format(i + 1))
        p0.append(a)
    for i in range(n):
        names.append("tau{:d}".format(i + 1))
        p0.append(init[i])
    sigma = np.zeros_like(ydata)
    c = np.max(ydata)
    for i in range(len(ydata)):
        c_err = ((1/(np.sqrt(c))) / c)**2
        b_err = ((1/np.sqrt(ydata[i]))/ydata[i])**2
        sigma[i] = (ydata[i] / c) * np.sqrt(b_err + c_err)
    ydata = ydata / np.max(ydata) # normalise for the fit
    #sigma = 1./np.sqrt(ydata + 1) # poisson errors on bin counts
    try:
        if bounds is not None:
            popt, pcov = curve_fit(func, x, ydata, p0=p0, sigma=sigma, bounds=bounds)
        else:
            popt, pcov = curve_fit(func, x, ydata, p0=p0, sigma=sigma)
        chisq = np.sum(((func(x, *popt) - ydata)/(sigma))**2)
        err = np.sqrt(np.diag(pcov))
        best_values = dict(zip(names, popt))
        errors = dict(zip(names, err))
        d = lifetimes(n, names, best_values, errors, pcov)
        print(d)
        for i in range(n):
            d['tau{:d}_init'.format(i + 1)] = init[i]
        d['x0'] = best_values['x0']
        d['x0_err'] = errors['x0']
        d['y0'] = best_values['y0']
        d['y0_err'] = errors['y0']
        d['chisq'] = chisq
        # now plot the fit
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(xdata, ydata, ls='--', label='data')
        plt.plot(xdata, func(x, *popt), label='fit')
        plt.legend()
        ax.set_yscale('log')
        plt.show()
        plt.close()
    except RuntimeError:
        d = {}
        popt = []
    return (d, popt)

mu = 100.
sigma = 50. / 2.355
hist = np.loadtxt("out/irrev/hex/0.85_1.31E+14_1.00_counts.dat")
ydata = hist[:, 2] + hist[:, 3]
xdata = hist[:, 0] + (hist[1, 0] - hist[0, 0])/2. # centre of bin
ydata = ydata / np.max(ydata)
irf = (1 / (sigma * np.sqrt(2. * np.pi))) * np.exp(-(xdata - mu)**2 / (np.sqrt(2.) * sigma)**2)
irf = irf / np.max(irf)

# min and max times we're concerned with (ns)
# for the experimental data this will require checking when the pulse turns on!
min_time = 1.
max_time = 10.

# cutoff time for the tail fit. again, check the curve first
cutoff = min_time + 0.7

# initial guesses for the time constants
taus = [5.]
p0 = [0., 0., 1., *taus]

# bounds for each of the time constants
lbs = tuple([0. for _ in range(len(p0))])
ubs = tuple([np.inf for _ in range(len(p0))])
bounds = [lbs, ubs]


'''
import the decay data. should be a two-column file with the first column
being the time and the second either being counts or "normalised" counts.
'''
path = "out/peter_data_refits/lekshmi_detergent"
decay_file = "{}/200au.txt".format(path)
decays = np.loadtxt(decay_file)
# if the decays aren't "normalised", "normalise" them
if (np.max(decays[:, 1]) > 1.):
    counts = decays[:, 1] / np.max(decays[:, 1])
    decays[:, 1] = counts

'''
now do the same for the IRF. I guess I should start printing out the IRF
in the code too, to make this all more consistent
'''
irf_file = "{}/irf.txt".format(path)
irf = np.loadtxt(irf_file)
if (np.max(irf[:, 1]) > 1.):
    counts = irf[:, 1] / np.max(irf[:, 1])
    irf[:, 1] = counts

# first do the tail fit
decays = decays[decays[:, 0] >= min_time]
decays = decays[decays[:, 0] <= max_time]

tail = decays[decays[:, 0] >= cutoff]
x = tail[:, 0]
y = tail[:, 1]
popt, pcov = curve_fit(exp_model, x, y, p0=p0, bounds=bounds)
print(popt)
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(x, y, ls='--', label='data')
plt.plot(x, exp_model(x, *popt), label='fit')
plt.legend()
ax.set_yscale('log')
plt.show()
plt.close()
