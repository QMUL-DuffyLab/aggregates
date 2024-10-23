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
from rates import Pulse, Rates, Iteration

if __name__ == "__main__":
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
            description="set up aggregate simulation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # required arguments
    parser.add_argument('-q', '--rho_q', type=float, required=True,
            help=r'Density of quenchers \rho_q')
    parser.add_argument('-name', type=str, required=True,
            help="Name for the output folder")
    # optional arguments
    parser.add_argument('-l', '--lattice', type=str, default='hex',
            choices=['line', 'square', 'honeycomb', 'hex'],
            help="Lattice type. Options are: 'hex', 'honeycomb', 'square', 'line'")
    parser.add_argument('-t', '--transfer', type=float,
            default=defaults.t_pq_q,
            help=f"Transfer time from pre-quencher to quencher (in picoseconds).")
    parser.add_argument('-o', '--omega', type=float,
            default=defaults.omega,
            help=f"Omega: entropic penalty for transfer to the pre-quencher.")
    parser.add_argument('-d', '--chl_decay', type=float,
            default=defaults.chl_decay,
            help=f"Chlorophyll decay time (in picoseconds).")
    parser.add_argument('-r', '--protein_radius', type=float,
            default=defaults.protein_r,
            help="Radius of the protein (nm) - only relevant when packing real aggregates")
    parser.add_argument('-n', '--n_trimers', type=int, default=defaults.n_trimers,
            help="Number of trimers (approximate) to put in the aggregate")
    parser.add_argument('-pr', '--prefix', type=str, default="",
            help="Optional folder prefix")
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
    parser.add_argument('--fit', action=argparse.BooleanOptionalAction,
            help="Pass to disable running the code and just re-fit the data")
    parser.add_argument('--files_only', action=argparse.BooleanOptionalAction,
            help="Pass to disable the running and the fits and just generate input files")

    args = parser.parse_args()
    print(args)
    rates = Rates(defaults.hop, args.chl_decay, args.chl_decay,
            defaults.q_decay, args.omega, defaults.t_intra,
            args.transfer, defaults.t_q_pq,
            defaults.ann, defaults.emissive, True, True)
    # make sure the fortran is compiled and up to date
    subprocess.run(['mpifort',
        '-O2', 'iteration.f90', '-o', './agg_mc'], check=True)
    path = os.path.join("out", args.name, args.lattice)
    if len(args.prefix) > 0:
        path = os.path.join(path, args.prefix)
    os.makedirs(path, exist_ok=True)
    rates.print()
    pulse = Pulse(fwhm=args.pulsewidth, mu=args.pulse_mean)
    fluences = args.fluences
    for fluence in fluences:
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:3.2f}_{:6.4e}_{:5.2f}_".format(
                args.rho_q, fluence, args.pulsewidth)
        print("Prefix = {}".format(file_prefix))

        # internally the fortran uses the excitation density to
        # fix the amplitude of the pulse. long story
        n_per_t = defaults.xsec * fluence
        if not args.fit:
            verbose = False
            # note - second parameter here is the nn cutoff
            agg = theoretical_aggregate(args.protein_radius,
                    2.01 * args.protein_radius, args.lattice, args.n_trimers)
            it = Iteration(agg, rates, pulse, args.rho_q,
                    path, file_prefix, n_per_t, args.binwidth, args.max_count,
                    verbose=verbose)
            if not args.files_only:
                for i in range(defaults.n_repeats):
                    subprocess.run(['mpirun',
                        '-np', f'{defaults.n_procs:1d}',
                        './agg_mc', it.params_file,
                        f"{i:d}"], check=True)

        for i in range(defaults.n_repeats):
            count_file = os.path.join(path, f"{file_prefix}{i:1d}_counts.dat")
            print(f"Fitting {count_file}")
            if os.path.isfile(count_file):
                n_max = defaults.n_max
                tau_init = defaults.tau_init
                dicts = []
                fits = []
                for j in range(1, n_max + 1):
                    try:
                        (d, fit) = multi_fit.do_fit(count_file,
                                tau_init[:j],
                                exp=False,
                                pw = args.pulsewidth / 1000.,
                                pm = args.pulse_mean / 1000.,
                                time_unit = 'ps')
                    except RuntimeError:
                        print("n_exp = {:2d} fit didn't work".format(j))
                        d = {}
                        fit = np.empty(1)
                    dicts.append(d)
                    fits.append(fit)
                    print("n_exp = {:2d}: fit_info = ".format(j), d)
                df = pd.DataFrame(dicts)
                # once a sweep of fluences is run, run again with option
                # --fit_only; this lets you pick the best visual fits
                if args.fit:
                    fig, ax = plt.subplots(figsize=(8,6))
                    ax.set_ylabel("Counts")
                    ax.set_xlabel("Time (ns)")
                    ax.set_xlim([-1., 10.])
                    ax.plot(fits[0][:, 0], fits[0][:, 1], ls='--', marker='o',
                            lw=3., label='decays')
                    for j, fit in enumerate(fits):
                        if fit.size > 1:
                            ax.plot(fit[:, 0], fit[:, 2], lw=2.5,
                                    label='n = {:2d}'.format(j + 1))
                    fig.tight_layout()
                    plt.grid()
                    plt.legend()
                    fig.savefig(os.path.join(path,
                        f"{file_prefix}_{i:1d}_fits.pdf")) 
                    # best_n = input("Best fit - how many exponentials?")
                    # df["best_fit"] = best_n
                df.to_csv(os.path.join(path,
                    f"{file_prefix}_{i:1d}_fit_info.csv"))
    end_time = time.monotonic()
    print("Total time elapsed: {}".format((end_time - start_time)))
