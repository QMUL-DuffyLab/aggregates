#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import argparse
from lmfit import Model
from scipy import signal
import fit
import multi_fit
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
'detergent', 'hop_only', 'irrev', 'rev', 'fast_irrev', 'slow', 'exciton'.
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
    parser.add_argument('--detergent', action=argparse.BooleanOptionalAction,
            help="Run for detergent case")

    args = parser.parse_args()
    if args.detergent:
        args.model = "hop_only"
        args.rho_q = 0.0
    print(args)
    path = "out/{}/{}".format(args.model, args.lattice)
    os.makedirs(path, exist_ok=True)
    rates = defaults.rates_dict[args.model]
    rates.print()
    pulse = Pulse(fwhm=args.pulsewidth, mu=args.pulse_mean)
    if not args.files_only:
        mt = open("{}/{:3.2f}_mono_tau.dat".format(path, args.rho_q), "w")
        bt = open("{}/{:3.2f}_bi_tau.dat".format(path, args.rho_q), "w")
        tt = open("{}/{:3.2f}_tri_tau.dat".format(path, args.rho_q), "w")
    fluences = args.fluences
    for fluence in fluences:
        n_per_t = defaults.xsec * fluence
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:3.2f}_{:3.2f}_{:5.2f}_".format(
                args.rho_q, n_per_t, args.pulsewidth)
        print("Prefix = {}".format(file_prefix))

        if not args.fit_only:
            verbose = False
            # note - second parameter here is the nn cutoff. set to 0 to
            # disable excitation hopping between trimers
            if args.detergent:
                agg = theoretical_aggregate(args.protein_radius,
                        0. * args.protein_radius, args.lattice, args.n_trimers)
            else:
                agg = theoretical_aggregate(args.protein_radius,
                        2.01 * args.protein_radius, args.lattice, args.n_trimers)
            it = Iteration(agg, rates, pulse, args.rho_q,
                    path, file_prefix, n_per_t, args.binwidth, args.max_count,
                    verbose=verbose)
            if not args.files_only:
                subprocess.run(['which', 'mpirun'], check=True)
                subprocess.run(['/usr/lib64/openmpi/bin/mpirun', '-np', '4', './f_iter', it.params_file], check=True)
                # non-MPI run
                # subprocess.run(['./f_iter', it.params_file], check=True)

        count_file = "{}/{}counts.dat".format(path, file_prefix)
        if os.path.isfile(count_file):
            try:
                mono_fit = multi_fit.do_fit(count_file, [3.6], exp=False,
                        pw = args.pulsewidth / 1000.,
                        pm = args.pulse_mean / 1000.,
                        time_unit = 'ps')
            except RuntimeError:
                print("monoexponential fit didn't work")
                mono_fit = {}
            try:
                bi_fit = multi_fit.do_fit(count_file, [0.3, 3.6], exp=False,
                        pw = args.pulsewidth / 1000.,
                        pm = args.pulse_mean / 1000.,
                        time_unit = 'ps')
            except RuntimeError:
                print("biexponential fit didn't work")
                bi_fit = {}
            df = pd.DataFrame([mono_fit, bi_fit])
            df.to_csv("{}/{}fit_info.csv".format(path, file_prefix))
            print("Mono fit: fit_info = ", mono_fit)
            print("Bi fit: fit_info = ", bi_fit)
            tau_init = [1./rates.g_pool, 1./rates.k_ann, 500.]
            """
               do some stuff with the dataframe (set up to do the
               multi scatter plot as a function of n_per_t)
            """
            # horrible way of doing this. but allows us to look at
            # partially finished runs
            # np.savetxt(mt, np.array(mono_tau[1]).reshape(1, 3))
            # np.savetxt(bt, np.array(bi_tau[1]).reshape(1, 3))
            # np.savetxt(tt, np.array(tri_tau[1]).reshape(1, 3))
            # fit.plot_fits(mono_tau, bi_tau, tri_tau, histvals,
            #     xvals, args.model, "{}/{}".format(path, file_prefix))
    if not args.files_only:
        mt.close()
        bt.close()
        tt.close()

    # subprocess.run(['python', 'plot_tau.py', '{}'.format(path),
    #     '{:3.2f}'.format(args.rho_q)], check=True)
    end_time = time.monotonic()
    print("Total time elapsed: {}".format((end_time - start_time)))
