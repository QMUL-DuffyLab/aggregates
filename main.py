#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
from lmfit import Model
from scipy import signal
import fit
import defaults
from trimer import check_neighbours, Aggregate, theoretical_aggregate
from kmc import Pulse, Rates, Iteration

if __name__ == "__main__":
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(description="set up aggregate simualtion")
    # required arguments
    parser.add_argument('-q', '--rho_q', type=float, required=True,
            help=r'Density of quenchers \rho_q')
    parser.add_argument('-l', '--lattice', type=str, required=True,
            choices=['line', 'square', 'honeycomb', 'hex'],
            help="Lattice type. Options are: 'hex', 'honeycomb', 'square', 'line'")
    parser.add_argument('-m', '--model', type=str, required=True,
            choices=[*defaults.rates_dict],
            help='''
Quenching model to use. Options are:
'hop_only', 'irrev', 'rev', 'fast_irrev', 'slow', 'exciton'.
See defaults.py for specific numbers''')
    # optional arguments
    parser.add_argument('-pr', '--protein_radius', type=float,
            default=defaults.protein_r,
            help="Radius of the protein (nm) - only relevant when packing real aggregates")
    parser.add_argument('-n', '--n_trimers', type=int, default=defaults.n_trimers,
            help="Number of trimers (approximate) to put in the aggregate")
    parser.add_argument('-mc', '--max_count', type=int, default=defaults.max_count,
            help="Maximum count for a given bin in the histogram")
    parser.add_argument('-bw', '--binwidth', type=float, default=defaults.binwidth,
            help="Binwidth for the histogram")
    parser.add_argument('-pw', '--pulsewidth', type=float, default=defaults.pulse_fwhm,
            help="Full width half maximum of the pulse (ps)")
    parser.add_argument('-pm', '--pulse_mean', type=float, default=defaults.pulse_mu,
            help="Peak time of the pulse (ps)")
    parser.add_argument('-f', '--fluences', type=float, nargs='+',
            default=defaults.fluences,
            help='''
Fluence(s) to use for the pulse (photons per pulse per cm^2).
Note that we consider a 485nm laser here; if this changes, the cross-section
per trimer will also need to be changed.
''')
    parser.add_argument('--fit_only', action=argparse.BooleanOptionalAction,
            help="Pass to disable running the code and just re-fit the data")
    parser.add_argument('--files_only', action=argparse.BooleanOptionalAction,
            help="Pass to disable the running and the fits and just generate input files")

    args = parser.parse_args()
    print(args)
    # annihilation, pool decay, pq decay, q decay
    path = "out/{}/{}".format(args.model, args.lattice)
    os.makedirs(path, exist_ok=True)
    rates = defaults.rates_dict[args.model]
    pulse = Pulse(fwhm=args.pulsewidth, mu=args.pulse_mean)
    if not args.files_only:
        mt = open("{}/{:3.2f}_mono_tau.dat".format(path, args.rho_q), "w")
        bt = open("{}/{:3.2f}_bi_tau.dat".format(path, args.rho_q), "w")
        tt = open("{}/{:3.2f}_tri_tau.dat".format(path, args.rho_q), "w")
        # st = open("{}/{:3.2f}_tau_min_error.dat".format(path, args.rho_q), "w")
    for fluence in args.fluences:
        print("Fluence = {:4.2E}".format(
            fluence))
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:3.2f}_{:4.2E}".format(
                args.rho_q, fluence)

        if not args.fit_only:
            verbose = False
            # note - second parameter here is the nn cutoff. set to 0 to
            # disable excitation hopping between trimers
            agg = theoretical_aggregate(args.protein_radius,
                    2.01 * args.protein_radius, args.lattice, args.n_trimers)
            it = Iteration(agg, rates, pulse, args.rho_q,
                    path, fluence, args.binwidth, args.max_count,
                    verbose=verbose)
            if not args.files_only:
                # this doesn't work because anaconda overwrites all my aliases and conflicts with the system mpi i installed myself, because it's fucking useless
                # subprocess.run(['mpirun', '-np', ' 4', './f_iter', it.params_file], check=True)
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
            np.savetxt(mt, np.array(mono_tau[1]).reshape(1, 3))
            np.savetxt(bt, np.array(bi_tau[1]).reshape(1, 3))
            np.savetxt(tt, np.array(tri_tau[1]).reshape(1, 3))
            fit.plot_fits(mono_tau, bi_tau, tri_tau, histvals,
                    xvals, args.model, "{}/{}".format(path, file_prefix))

    if not args.files_only:
        mt.close()
        bt.close()
        tt.close()
        subprocess.run(['python', 'plot_tau.py', '{}'.format(path), '{:3.2f}'.format(args.rho_q)], check=True)

    end_time = time.monotonic()
    print("Total time elapsed: {}".format((end_time - start_time)))
