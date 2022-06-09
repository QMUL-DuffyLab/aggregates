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
    run = True # set to true to run, false to generate input files
    r = 5.
    lattice_type = "square"
    n_max = 200 # set to 8 for hex/honeycomb, 10 for square, 110 or so for line!
    max_count = 10000
    binwidth = 25.
    rho_quenchers = 0.85
    # fluences given here as photons per pulse per unit area - 485nm
    fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
            1.9E14, 3.22E14, 6.12E14, 9.48E14]
    # annihilation, pool decay, pq decay, q decay
    hop = 25.
    chl_decay = 3600.
    lut_decay = 10.
    ann = 50.
    rates_dict = {
     'hopping_only': Rates(hop, chl_decay, chl_decay, lut_decay, np.inf, np.inf,
         np.inf, np.inf, ann, [False, True, True, False], True, True),
     'lut_eet': Rates(hop, chl_decay, chl_decay, lut_decay,
         5., 1., 20., np.inf, ann, [False, True, True, False], True, True),
     'schlau_cohen': Rates(hop, chl_decay, chl_decay, lut_decay,
         5., 1., 0.4, np.inf, ann, [False, True, True, False], True, True),
     'mennucci': Rates(hop, chl_decay, chl_decay, lut_decay,
         5., 1., 20., 20., ann, [False, True, True, False], True, True),
     'holzwarth': Rates(hop, chl_decay, chl_decay, 833.,
         180., 550., 260., 3300., ann, [False, True, False, False], True, True),
     'exciton': Rates(hop, chl_decay, 40., 40.,
         5., 1., 1000., 1000., ann, [False, True, False, False], True, True),
     }
    rates_key = 'lut_eet'
    path = "out/{}/{}".format(rates_key, lattice_type)
    os.makedirs(path, exist_ok=True)
    # for rates in rates_dict:
    #     for lattice in ["line", "square", "hex", "honeycomb"]:
    rates = rates_dict[rates_key]
    pulse = Pulse(fwhm=50., mu=100.)
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
            agg = theoretical_aggregate(r, 2.01 * r, lattice_type, n_max)
            quit()
            it = Iteration(agg, rates, pulse,
                    rho_quenchers,
                    path, fluence, binwidth, max_count,
                    verbose=verbose)
            if (run):
                subprocess.run(['./f_iter', it.params_file], check=True)

        if os.path.isfile("{}/{}_counts.dat".format(path, file_prefix)):
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

    mt.close()
    bt.close()
    tt.close()
    end_time = time.monotonic()
    print("Total time elapsed: {}".format((end_time - start_time)))
    subprocess.run(['python', 'plot_tau.py', '{}'.format(path), '{:3.2f}'.format(rho_quenchers)], check=True)
