#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import argparse
from lmfit import Model
import multi_fit
import defaults
from trimer import check_neighbours, Aggregate, theoretical_aggregate
from kmc import Pulse, Rates, Iteration

if __name__ == "__main__":
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
            description="set up aggregate simulation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
'detergent', 'hop_only', 'slow_entropic', 'medium_entropic', 'fast_entropic',
'slow_non-entropic', 'medium_non-entropic', 'fast_non-entropic',
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
    # make sure the fortran is compiled and up to date
    subprocess.run(['mpifort',
        '-O2', 'iteration.f90', '-o', './agg_mc'], check=True)
    path = "out/{}/{}".format(args.model, args.lattice)
    os.makedirs(path, exist_ok=True)
    rates = defaults.rates_dict[args.model]
    rates.print()
    pulse = Pulse(fwhm=args.pulsewidth, mu=args.pulse_mean)
    fluences = args.fluences
    for fluence in fluences:
        n_per_t = defaults.xsec * fluence
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:3.2f}_{:3.2f}_{:5.2f}_".format(
                args.rho_q, n_per_t, args.pulsewidth)
        print("Prefix = {}".format(file_prefix))

        if not args.fit_only:
            verbose = False
            # note - second parameter here is the nn cutoff
            agg = theoretical_aggregate(args.protein_radius,
                    2.01 * args.protein_radius, args.lattice, args.n_trimers)
            it = Iteration(agg, rates, pulse, args.rho_q,
                    path, file_prefix, n_per_t, args.binwidth, args.max_count,
                    verbose=verbose)
            if not args.files_only:
                subprocess.run(['mpirun',
                    '-np', '4', './agg_mc', it.params_file], check=True)
                # non-MPI run
                # subprocess.run(['./f_iter', it.params_file], check=True)

        count_file = "{}/{}counts.dat".format(path, file_prefix)
        if os.path.isfile(count_file):
            try:
                (mono_d, mono_fit) = multi_fit.do_fit(count_file,
                        [1/rates.g_pool],
                        exp=False,
                        pw = args.pulsewidth / 1000.,
                        pm = args.pulse_mean / 1000.,
                        time_unit = 'ps')
            except RuntimeError:
                print("monoexponential fit didn't work")
                mono_d = {}
                mono_fit = np.empty(1)
            try:
                (bi_d, bi_fit) = multi_fit.do_fit(count_file,
                        [1/rates.k_ann, 1/rates.g_pool],
                        exp=False,
                        pw = args.pulsewidth / 1000.,
                        pm = args.pulse_mean / 1000.,
                        time_unit = 'ps')
            except RuntimeError:
                print("biexponential fit didn't work")
                bi_d = {}
                bi_fit = np.empty(1)
            df = pd.DataFrame([mono_d, bi_d])
            print("Mono fit: fit_info = ", mono_d)
            print("Bi fit: fit_info = ", bi_d)
            tau_init = [1./rates.g_pool, 1./rates.k_ann, 500.]
            if args.fit_only:
                fig, ax = plt.subplots(figsize=(8,6))
                ax.set_ylabel("Counts")
                ax.set_xlabel("Time (ns)")
                ax.set_xlim([-1., 10.])
                ax.plot(mono_fit[:, 0], mono_fit[:, 1], ls='--', marker='o',
                        lw=3., label='decays')
                if mono_fit.size > 0:
                    ax.plot(mono_fit[:, 0], mono_fit[:, 2], lw=2.5, label='mono')
                if bi_fit.size > 0:
                    ax.plot(bi_fit[:, 0], bi_fit[:, 2], lw=2.5, label='bi')
                fig.tight_layout()
                plt.grid()
                plt.legend()
                plt.show()
                best_n = input("Best fit - 1 or 2 exponentials?")
                df["best_fit"] = best_n
            df.to_csv("{}/{}fit_info.csv".format(path, file_prefix))
    # subprocess.run(['python', 'plot_tau.py', '{}'.format(path),
    #     '{:3.2f}'.format(args.rho_q)], check=True)
    end_time = time.monotonic()
    print("Total time elapsed: {}".format((end_time - start_time)))
