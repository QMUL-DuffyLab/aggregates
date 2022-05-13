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
from trimer import check_neighbours, Aggregate, theoretical_aggregate
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
    if lattice_type == "honeycomb":
        n_iter = n_iter / 2 # 2 atom basis
    max_count = 10000
    binwidth = 25.
    rho_quenchers = 0.0
    # fluences given here as photons per pulse per unit area - 485nm
    fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
            1.9E14, 3.22E14, 6.12E14, 9.48E14]
    # annihilation, pool decay, pq decay, q decay
    rates_dict = {
     'hopping_only': Rates(5., 3600., 3600., 14., np.inf, np.inf,
         np.inf, np.inf, 50., [False, True, True, False], True, True),
     'lut_eet': Rates(20., 3600., 3600., 14.,
         7., 1., 20., np.inf, 50., [False, True, True, False], True, True),
     'schlau_cohen': Rates(20., 3600., 3600., 14.,
         7., 1., 0.4, np.inf, 50., [False, True, True, False], True, True),
     'mennucci': Rates(20., 3600., 3600., 14.,
         7., 1., 29., 43., 50., [False, True, True, False], True, True),
     'holzwarth': Rates(20., 3600., 3600., 833.,
         180., 550., 260., 3300., 50., [False, True, False, False], True, True),
     'exciton': Rates(20., 3600., 40., 40.,
         7., 1., 1000., 1000., 50., [False, True, False, False], True, True),
     }
    rates_key = 'hopping_only'
    path = "out/{}/{}".format(rates_key, lattice_type)
    # for rates in rates_dict:
    #     for lattice in ["line", "square", "hex", "honeycomb"]:
    rates = rates_dict[rates_key]
    pulse = Pulse(fwhm=50., mu=100.)
    run = False # set to true to run, false to generate input files
    mt = open("{}/{:3.2f}_mono_tau.dat".format(path, rho_quenchers), "w")
    bt = open("{}/{:3.2f}_bi_tau.dat".format(path, rho_quenchers), "w")
    tt = open("{}/{:3.2f}_tri_tau.dat".format(path, rho_quenchers), "w")
    for fluence in fluences:
        print("Fluence = {:4.2E}".format(
            fluence))
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:3.2f}_{:4.2E}".format(
                rho_quenchers, fluence)

        if not fit_only:
            verbose = False
            # note - second parameter here is the nn cutoff. set to 0 to
            # disable excitation hopping between trimers
            agg = theoretical_aggregate(r, 2.01 * r, lattice_type, n_iter)
            it = Iteration(agg, rates, pulse,
                    rho_quenchers,
                    path, fluence, binwidth, max_count,
                    verbose=verbose)
            if (run):
                subprocess.run(['./f_iter', it.params_file], check=True)

        hist = np.loadtxt("{}/{}_counts.dat".format(path, file_prefix))
        xvals = hist[:, 0] + ((hist[0, 1] - hist[0, 0]) / 2.)
        histvals = hist[:, 2] + hist[:, 3]
        long_gauss = 1. / (pulse.sigma * np.sqrt(2. * np.pi)) * \
            np.exp(- (xvals - pulse.mu)**2 \
            / (np.sqrt(2.) * pulse.sigma)**2)
        long_gauss = long_gauss/np.max(long_gauss)
        histvals = histvals / np.max(histvals)
        mono_tau = fit.monofit(histvals, rates, xvals,
            long_gauss, fluence, path, file_prefix)
        bi_tau = fit.bifit(histvals, rates, xvals,
            long_gauss, fluence, path, file_prefix)
        tri_tau = fit.trifit(histvals, rates, xvals,
            long_gauss, fluence, path, file_prefix)
        # horrible way of doing this. but allows us to look at
        # partially finished runs
        np.savetxt(mt, np.array(mono_tau).reshape(1, 3))
        np.savetxt(bt, np.array(bi_tau).reshape(1, 3))
        np.savetxt(tt, np.array(tri_tau).reshape(1, 3))

    end_time = time.monotonic()
    print("Total time elapsed: {}".format((end_time - start_time)))
    subprocess.run(['python', 'plot_tau.py', '{}'.format(path)], check=True)
