#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess
from lmfit import Model
from scipy import signal
import fit
from trimer import Aggregate, theoretical_aggregate
from kmc import Pulse, Rates, Iteration

if __name__ == "__main__":
    start_time = time.monotonic()
    if (len(sys.argv) > 1):
        fit_only = sys.argv[1]
    else:
        fit_only = False
    r = 5.
    lattice_type = "hex"
    n_iter = 8 # 434 trimers for honeycomb
    max_count = 5000
    binwidth = 25.
    rho_quenchers = 0.0
    # fluences given here as photons per pulse per unit area - 485nm
    fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
            1.9E14, 3.22E14, 6.12E14, 9.48E14]
    # fluences = [3.03E13, 6.24E13, 1.31E14,
    #         1.9E14, 3.22E14, 6.12E14, 9.48E14]
    # fluences = [3.22E14, 6.12E14, 9.48E14]
    # annihilation, pool decay, pq decay, q decay
    rates_dict = {
     'lut_eet': Rates(20., 3800., 3800., 14., 
         7., 1., 20., np.inf, 50., [False, True, True, False], True, True),
     'schlau_cohen': Rates(20., 3800., 3800., 14., 
         7., 1., 0.4, 0.4, 50., [False, True, True, False], True, True)
     }
    rates_key = 'lut_eet'
    rates = rates_dict[rates_key]
    mono_tau = []
    bi_tau = []
    tri_tau = []
    pulse = Pulse(fwhm=50., mu=100.)
    plt.subplots()
    for fluence in fluences:
        plt.plot(pulse.ft * fluence, label=r'f = {:5.3e}'.format(fluence))
    plt.grid()
    plt.gca().set_ylabel("Intensity")
    plt.gca().set_xlabel("Time (ps)")
    plt.legend()
    plt.savefig("out/pulses.pdf")
    plt.close()
    lifetimes = []
    for fluence in fluences:
        print("Fluence = {:4.2E}, n_iterations = {:d}".format(
            fluence, n_iterations))
        path = "out/{}/{}".format(rates_key, lattice_type)
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:3.2f}_{:4.2E}".format(
                rho_quenchers, fluence)
        decay_filename = "{}/{}_decays.dat".format(path, file_prefix)
        emission_filename = "{}/{}_emissions.dat".format(path, file_prefix)

        if not fit_only:
            # note - second parameter here is the nn cutoff. set to 0 to
            # disable excitation hopping between trimers
            verbose = False
            agg = theoretical_aggregate(r, 0., lattice_type, n_iter)
            n_es = []
            means = []
            stddevs = []
            emission_means = []
            emission_stddevs = []
            yields = []
            it = Iteration(agg, rates, pulse,
                    rho_quenchers,
                    path, fluence, binwidth, max_count,
                    verbose=verbose)
            subprocess.run(['./f_iter', it.params_file], check=True)

        decays = np.loadtxt(decay_filename)
        emissions = decays[np.where(decays[:, 1] > 1), 0].flatten()
        tau = np.mean(decays[:, 0])
        sigma_tau = np.std(decays[:, 0])
        print("Total μ, σ: ", tau, sigma_tau)
        np.savetxt("{}/{}_total_mean_std.dat".format(path, file_prefix),
                [tau, sigma_tau])
        decay_pd = pd.DataFrame(decays, columns=["Time (ps)", "Decay type"])
        typedict = {1.: "Ann.", 2.: "Pool", 3.: "PQ", 4: "Q"}
        decay_pd = decay_pd.replace({"Decay type": typedict})
        ax = sns.histplot(data=decay_pd, x="Time (ps)", hue="Decay type",
                element="step", fill=False)
        ax.set_yscale('log')
        # ax.set_xlim((0., 800.))
        plt.axvline(x=tau, ls="--", c='k')
        plt.savefig("{}/{}_plot.pdf".format(path, file_prefix))
        plt.close()

        # matplotlib histogram - output bins and vals for lmfit
        histvals, histbins = fit.histogram(emissions,
                "{}/{}_emission_histogram.pdf".format(path, file_prefix),
                binwidth)
        xvals = histbins[:-1] + (np.diff(histbins) / 2.)
        print("xvals = ", xvals)
        histvals = histvals / np.max(histvals)
        np.savetxt("{}/{}_histvals.dat".format(path, file_prefix), histvals)
        np.savetxt("{}/{}_histbins.dat".format(path, file_prefix), histbins)
        long_gauss = 1. / (pulse.sigma * np.sqrt(2. * np.pi)) * \
            np.exp(- (xvals - pulse.mu)**2 \
            / (np.sqrt(2.) * pulse.sigma)**2)
        long_gauss = long_gauss/np.max(long_gauss)
        histvals = histvals / np.max(histvals)
        np.savetxt("out/long_gauss.dat", long_gauss)
        # func = fit.monoexprisemodel(xvals, 1./rates.g_pool, 1., 0., 0, long_gauss)
        # quit()
        # this is in the SO question - are these weights necessary? why?
        weights = 1/np.sqrt(histvals + 1)
        if fluence > 1E14:
            mod = Model(fit.biexprisemodel, independent_vars=('x', 'irf'))
            pars = mod.make_params(tau_1 = 1./rates.k_ann, a_1 = 1.,
                    tau_2 = 1./rates.g_pool, a_2 = 1., y0 = 0., x0 = 0)
        else:
            mod = Model(fit.monoexprisemodel, independent_vars=('x', 'irf'))
            pars = mod.make_params(tau_1 = 1./rates.g_pool, a_1 = 1., y0 = 0., x0 = 0)
        pars['x0'].vary = True
        pars['y0'].vary = True
        try:
            result = mod.fit(histvals, params=pars, weights=weights, method='leastsq', x=xvals, irf=long_gauss)
            print(result.fit_report())
            res = result.best_values
            if fluence > 1E14:
                lifetime = ((res["a_1"] * res["tau_1"] + res["a_2"] * res["tau_2"])
                        / (res["a_1"] + res["a_2"]))
            else:
                lifetime = res["tau_1"]
            lifetimes.append(lifetime)
            print("Lifetime = {} ps".format(lifetime))
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.semilogy(xvals, histvals, label="hist")
            plt.semilogy(xvals, result.best_fit, label="fit")
            plt.subplot(2, 1, 2)
            plt.plot(xvals, result.residual, label="residuals")
            plt.savefig("{}/{}_fit.pdf".format(path, file_prefix))
            plt.close()
        except ValueError:
            print("fit failed!")

    end_time = time.monotonic()
    print("Total time elapsed: {}".format((end_time - start_time)))
    np.savetxt("{}/lifetimes.dat".format(path), np.array(lifetimes))
    subprocess.run(['python', 'plot_tau.py', '{}/{:d}_{:3.2f}'.format(path, n_iterations, rho_quenchers)], check=True)
