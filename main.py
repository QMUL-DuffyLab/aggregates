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
    max_count = 10000
    binwidth = 25.
    rho_quenchers = 0.0
    # fluences given here as photons per pulse per unit area - 485nm
    fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
            1.9E14, 3.22E14, 6.12E14, 9.48E14]
    # fluences = [6.12E14]
    # annihilation, pool decay, pq decay, q decay
    rates_dict = {
     'lut_eet': Rates(20., 3600., 3600., 14., 
         7., 1., 20., np.inf, 50., [False, True, True, False], True, True),
     'schlau_cohen': Rates(20., 3600., 3600., 14., 
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
        print("Fluence = {:4.2E}".format(
            fluence))
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

        hist = np.loadtxt("{}/{}_counts.dat".format(path, file_prefix))
        xvals = hist[:, 0] + ((hist[0, 1] - hist[0, 0]) / 2.)
        histvals = hist[:, 2] + hist[:, 3]
        long_gauss = 1. / (pulse.sigma * np.sqrt(2. * np.pi)) * \
            np.exp(- (xvals - pulse.mu)**2 \
            / (np.sqrt(2.) * pulse.sigma)**2)
        long_gauss = long_gauss/np.max(long_gauss)
        histvals = histvals / np.max(histvals)

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
    subprocess.run(['python', 'plot_tau.py', '{}'.format(path, rho_quenchers)], check=True)
