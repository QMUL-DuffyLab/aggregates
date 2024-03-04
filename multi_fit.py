# 22/08/22 - new fitting plan
import numpy as np
import os
import sympy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

def Convol(x, h):
    X = np.fft.fft(x)
    H = np.fft.fft(h)
    return np.real(np.fft.ifft(X * H))

def exp_model(x, *args):
    # args should be given as a_1, ..., a_n, tau_1, ..., tau_n
    if (len(args) % 2 != 0):
        print("exp_model: number of args should be even - y0, x0, a_i, tau_i")
        return np.full(x.size, np.nan)
    if (len(args) == 0):
        print("exp_model: 0 arguments given - ???")
        return np.full(x.size, np.nan)
    y = np.zeros(x.size)
    n_exp = (len(args)) // 2
    for i in range(n_exp):
        y += args[i] * np.exp(-x / float(args[i + (n_exp)]))
    return y

def reconv(X, *args):
    # args should be a_i, irf_shift
    x, irf, taus = X
    ymodel = np.zeros(x.size)
    irf_interp = np.interp(x, x - args[-1], irf)
    irf_reshaped_norm = irf_interp / np.sum(irf_interp)
    for i in range(len(args) - 1):
        ymodel += (args[i] * np.exp(-(x) / taus[i]))
    z=Convol(ymodel,irf_reshaped_norm)
    return z

def reconv_plus(X, *args):
    # args are a_i, a_extra, tau_extra, irf_shift
    x, irf, taus = X
    ymodel = np.zeros(x.size)
    irf_interp = np.interp(x, x - args[-1], irf)
    irf_reshaped_norm = irf_interp / np.sum(irf_interp)
    for i in range(len(args - 3)):
        ymodel += (args[i] * np.exp(-x / taus[i])) 
    # now add the extra one
    ymodel += args[-3] * np.exp(-x / args[-2])
    z=Convol(ymodel,irf_reshaped_norm)
    return z

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

def do_fit(filename, tau_init, irf_file=None,
        exp=False, pw=0.25, pm=0.4, time_unit="ns"):
    path = os.path.dirname(filename)
    (fluence, ext) = os.path.splitext(os.path.basename(filename))
    
    '''
    import the decay data. 
    if it's experimental it should be a two-column file with the first column
    being the time and the second either being counts or "normalised" counts.
    otherwise it's the histogram output by the fortran code: time, ann, pool, pq, q
    '''
    decays = np.loadtxt(filename, skiprows=2)
    if exp:
        times = decays[:, 2]
        counts = decays[:, 3]
        counts_norm = counts / np.max(counts)
    else:
        times = decays[:, 0]
        counts = decays[:, 2] + decays[:, 3]
        counts_norm = counts / np.max(counts)
    if time_unit == "ps":
        times = times / 1000.
    elif time_unit == "us":
        times = times * 1000.
    xyn = np.column_stack((times, counts, counts_norm))
    max_count_time = xyn[np.argmax(counts), 0]
    min_time = max_count_time - 1.
    max_time = 30.
    
    cutoff = max_count_time

    # now do the tail fit
    decays = xyn[xyn[:, 0] >= min_time]
    decays = xyn[xyn[:, 0] <= max_time]
    
    if min_time > 0.:
        xyn[:, 0] = xyn[:, 0] - min_time

    # errors for each count
    if np.max(xyn[:, 1]) > 1.:
        max_count = np.max(xyn[:, 1])
    else:
        print("Warning - assuming max_count = 10000. bin errors might be wrong")
        max_count = 10000. # arbitrary!
    sigma = np.zeros(xyn[:, 1].size)
    for i in range(len(xyn[:, 1])):
        if (xyn[i, 1] == 0.):
            count = 1.
        else:
            count = xyn[i, 1]
        sigma[i] = np.sqrt(1. / count + 1. / max_count)
        
    
    if exp:
        # check the x-axis is correct here!
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(xyn[:, 0], xyn[:, 1], label="decays")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Counts ('normalised')")
        ax.set_xlim([0., 20.])
        plt.legend()
        plt.title("Decay curve - checking the time adjustment")
        ax.set_yscale('log')
        plt.savefig("{}/{}_decays.pdf".format(path, fluence))
        plt.close()

    n_exp = len(tau_init)
    p0 = [*[1./n_exp for _ in range(n_exp)], *tau_init]
    names = [] # these will be needed to construct a dataframe later
    for i in range(n_exp):
        names.append("a{:d}".format(i + 1))
    for i in range(n_exp):
        names.append("tau{:d}".format(i + 1))
    # bounds for each of the time constants
    lbs = tuple([0. for _ in range(len(p0))])
    ubs = tuple([np.inf for _ in range(len(p0))])
    bounds = [lbs, ubs]

    tail = xyn[xyn[:, 0] >= cutoff]
    tail_sigma = sigma[xyn[:, 0] >= cutoff]
    x = tail[:, 0] - cutoff
    y = tail[:, 2]
    tail_popt, tail_pcov = curve_fit(exp_model, x, y, p0=p0,
            sigma=tail_sigma, bounds=bounds)
    print("popt tail: ",tail_popt)
    print(tail_pcov)
    tail_err = np.sqrt(np.diag(tail_pcov))
    print("errors:", tail_err)
    
    best_t = list(tail_popt[len(tail_popt)//2:])
    print("Time constant(s) from tail fit = ", best_t)
    best_a = list(tail_popt[:len(tail_popt)//2])
    print("Amplitude(s) from tail fit = ", best_a)
    # need this to plot the tail fit later
    bf = exp_model(xyn[:, 0], *tail_popt)
    
    '''
    now do the same for the IRF
    '''
    irf_norm = np.zeros(counts.size)
    if exp: # load in the experimental IRF - two column file
        irf = np.loadtxt(irf_file, skiprows=2)
        if (np.max(irf[:, 1]) > 1.):
            irf[:, 1] = irf[:, 1] / np.max(irf[:, 1])
        # assumes that the data and IRF are given in the same time units!
        if time_unit == "ps":
            irf[:, 0] = irf[:, 0] / 1000.
        elif time_unit == "us":
            irf[:, 0] = irf[:, 0] * 1000.
        irf_min_time = irf[np.argmax(irf[:, 1]), 0] - 1.
        irf_interp = np.interp(xyn[:, 0], irf[:, 0] - irf_min_time, irf[:, 1])
        irf_norm = irf_interp / np.max(irf_interp)
    else: # generate it
        sig = pw / 2.355
        irf_gen = ((1 / (sig * np.sqrt(2. * np.pi))) *
                np.exp(-(xyn[:, 0] - pm)**2 / (np.sqrt(2.) * sig)**2))
        irf_norm = irf_gen / np.max(irf_gen)
    
    # fit tail with IRF
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(x, y, ls='--', marker='o', label='Decays')
    plt.plot(x, exp_model(x, *tail_popt), label='Fit')
    plt.plot(xyn[:, 0] - cutoff, irf_norm, label='IRF')
    plt.legend(fontsize=32)
    # plt.title("Tail best fit - fluence = {}".format(fluence))
    plt.grid()
    # ax.set_yscale('log')
    ax.set_ylim([1e-5, 1.1])
    ax.set_xlim([-1., 5.])
    ax.set_xticks([0., 1., 2., 3., 4., 5.])
    ax.set_ylabel("Counts (normalised)")
    ax.set_xlabel("Time (ns)")
    plt.tight_layout()
    plt.savefig("{}/{}_tail_fit_{}.pdf".format(path, fluence, n_exp))
    # plt.show()
    plt.close()

    """
    now we need to do something horrible!
    generate an empty array with the same length as x and irf, and fill
    the first n_exp elements with our time constants. 
    we do this because we want to keep them fixed in the subsequent fit, but:
      - you can't just pass (X, irf, *best_t) because
        the arrays have to have the same shape (?)
      - you can't wrap it in lambda X, *best_t because
        then the reconvolution function is passed
        with two extra arguments, for some reason
    This second point is something internal to curve_fit;
    just doing f = lambda X, *best_t: 
    print(len((X, *best_t, *best_a, x0, irf_shift))) returns 5, but when you
    do that in curve_fit it returns 7. 
    no idea why. bug? something to do with self?
    """
    
    taus = np.zeros(len(xyn[:, 0]))
    for i in range(n_exp):
        taus[i] = best_t[i]

    irf_shift = 0.0
    X = (xyn[:, 0], irf_norm, taus)
    p0 = [*best_a, irf_shift]
    lbs = tuple([0. for _ in range(len(best_a))] + [-max_time])
    ubs = tuple([np.inf for _ in range(len(best_a))] + [max_time])
    bounds = [lbs, ubs]
    popt, pcov = curve_fit(reconv,
            X, xyn[:, 2], p0=p0, sigma=sigma, bounds=bounds)
    bf = reconv(X, *popt)

    print("best fit for amps, irf_shift: ", popt)
    print("cov: ", pcov)

    # now we set up the lifetime calculations
    err = np.sqrt(np.diag(pcov))
    print("err: ", err)
    best_values = dict(zip(names, np.concatenate((popt[:n_exp], best_t))))
    errors = dict(zip(names, np.concatenate((err[:n_exp], tail_err[n_exp:]))))
    # amplitudes and time constants are fitted separately
    # so their covariance is necessarily zero
    cov = np.block([
        [pcov[:n_exp, :n_exp], np.zeros((n_exp, n_exp))],
        [np.zeros((n_exp, n_exp)), tail_pcov[n_exp:, n_exp:]]
    ])
    d = lifetimes(n_exp, names, best_values, errors, cov)
    d["n_exp"] = n_exp
    d["irf_shift"] = popt[-1]
    d["irf_shift_err"] = err[-1]
    d["cutoff"] = cutoff
    print(d)
    fig, axs = plt.subplots(2, 1, figsize=(12,8))
    plt.suptitle("{}: ".format(fluence) + r'$ \tau_{\text{amp.}} = $'
            + "{:4.2f} +/- {:4.2f} ns".format(d["tau_amp"], d["tau_amp_err"]))
    axs[0].plot(xyn[:, 0], xyn[:, 2], ls='--', marker='o', label='Decays')
    axs[0].plot(xyn[:, 0], bf, label='fit')
    plot_file = "{}/{}_reconv_{}.pdf".format(path, fluence, str(n_exp))
    axs[0].legend()
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_ylim([0., 1.1])
    axs[0].set_ylabel("Counts")
    axs[0].set_xlabel("Time (ns)")
    axs[0].set_xlim([-1., 10.])
    axs[1].plot(xyn[:, 0], xyn[:, 2], ls='--', marker='o', label='Decays')
    axs[1].plot(xyn[:, 0], reconv(X, *popt), label='fit')
    axs[1].set_yscale('log')
    axs[1].set_ylim([1e-2, 1.5])
    axs[1].set_ylabel("Counts")
    axs[1].set_xlabel("Time (ns)")
    axs[1].set_xlim([-1., 10.])
    fig.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    # now just the linear plot for paper
    fig, ax = plt.subplots(figsize=(12,8))
    mpl.rcParams['font.size'] = 36
    # plt.suptitle("{}: ".format(fluence) + r'$ \tau_{\text{amp.}} = $'
    #         + "{:4.2f} +/- {:4.2f} ns".format(d["tau_amp"], d["tau_amp_err"]))
    ax.plot(xyn[:, 0], xyn[:, 2], ls='--', marker='o', label='Decays')
    ax.plot(xyn[:, 0], bf, label='Fit')
    plot_file = "{}/{}_reconv_paper_{}.pdf".format(path, fluence, str(n_exp))
    ax.legend(fontsize=36)
    ax.grid(True)
    ax.set_ylim([0., 1.1])
    ax.set_ylabel("Counts (normalised)")
    ax.set_xlabel("Time (ns)")
    ax.set_xlim([-1., 10.])
    # ax.tick_params(labelsize=32)
    fig.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((.12, .45, .84, .48))
    mpl.rcParams['font.size'] = 36
    plt.plot(xyn[:, 0], xyn[:, 2] * max_count, ls='--', marker='o', label='Decays')
    plt.plot(xyn[:, 0], bf * max_count, label='Fit')
    plot_file = "{}/{}_resid_{}.pdf".format(path, fluence, str(n_exp))
    frame1.set_ylim([0., max_count * 1.1])
    frame1.set_yticklabels(["0", "5", "10"])
    frame1.set_ylabel(r'Counts $ \times 10^3  $')
    frame1.set_xlim([0., 10.])
    frame1.set_xticklabels([])
    plt.gca().grid(True)
    plt.legend()
    # NB - picoquant documentation says it uses "weighted residuals"
    # but gives no more info than that. i've used population weighted
    # residuals here and the results are on the order of 1, generally
    diff = max_count * sigma * (xyn[:, 2] - bf)
    plt.legend()
    frame2 = fig1.add_axes((.12, .15, .84, .3))
    frame2.set_ylim([-3, 3])
    frame2.set_yticks([-2., 0., 2.])
    frame2.set_ylabel("Residuals")
    # frame2.set_yticklabels(["-0.01", "0", "0.01"])
    frame2.set_xlim([0., 10.])
    frame2.set_xlabel("Time (ns)")
    plt.plot(xyn[:, 0], diff, marker='o')
    plt.gca().grid(True)
    # ax.legend(fontsize=36)
    # ax.grid(True)
    # ax.tick_params(labelsize=32)
    fig.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    chisq = np.sum((max_count * (xyn[:, 2] - bf))**2 * (sigma**2))
    redchisq = chisq / (np.size(xyn[:, 2]) - (2 * n_exp + 1))
    print("Chi-squared = {}".format(chisq))
    print("Reduced chi-squared = {}".format(redchisq))

    df = pd.DataFrame(d, index=[0])
    df.to_csv("{}/{}_fit_info_{}.csv".format(path, fluence, str(n_exp)))
    return (d, np.column_stack((xyn[:, 0], xyn[:, 2], bf)))

if __name__ == "__main__":
    
    decay_file = "out/fast_entropic/hex/0.80_0.05_counts.dat"
    # irf_file = "out/peter_data_refits/lekshmi_detergent/second_batch/IRF 1MHz.dat"
    do_fit(decay_file, [1.0, 3.0], exp=False, pm=0.35, pw=0.09, time_unit="ps")
