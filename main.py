#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess
from scipy import signal
import fit
from trimer import Aggregate, theoretical_aggregate
from kmc import Pulse, Model, Iteration


if __name__ == "__main__":
    start_time = time.monotonic()
    fit_only = False
    r = 5.
    lattice_type = "hex"
    n_iter = 8 # 434 trimers for honeycomb
    n_iterations = 1000
    rho_quenchers = 0.0
    # fluences given here as photons per pulse per unit area - 485nm
    fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
            1.9E14, 3.22E14, 6.12E14, 9.48E14]
    # fluences = [6.12E14, 9.48E14]
    # annihilation, pool decay, pq decay, q decay
    model_dict = {
     'lut_eet': Model(20., 3800., 3800., 14., 
         7., 1., 20., np.inf, 700., [False, True, True, False], True, True),
     'schlau_cohen': Model(20., 3800., 3800., 14., 
         7., 1., 0.4, 0.4, 700., [False, True, True, False], True, True)
     }
    model_key = 'lut_eet'
    model = model_dict[model_key]
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
    for fluence in fluences:
        print("Fluence = {:4.2E}, n_iterations = {:d}".format(
            fluence, n_iterations))
        path = "out/{}/{}".format(model_key, lattice_type)
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:d}_{:3.2f}_{:4.2E}".format(
                n_iterations, rho_quenchers, fluence)
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
            # decay_file     = open(decay_filename, mode='w')
            # emissions_file = open(emission_filename, mode='w')
            it = Iteration(agg, model, pulse,
                    rho_quenchers, n_iterations,
                    path, fluence, verbose=verbose)
            subprocess.run(['./f_iter', it.params_file], check=True)
        '''
        tau is just a straight estimation of everything
        mean of means and mean of emissive means reported separately
        statistics of these????? are they valid things to report???
        '''
        decays = np.loadtxt(decay_filename)
        # emissions = np.loadtxt(emission_filename)
        emissions = decays[np.where(decays[:, 1] > 1), 0].flatten()
        tau = np.mean(decays[:, 0])
        sigma_tau = np.std(decays[:, 0])
        print("Total μ, σ: ", tau, sigma_tau)
        np.savetxt("{}/{}_total_mean_std.dat".format(path, file_prefix),
                [tau, sigma_tau])

        '''
        NB: latex will work in column names and captions (e.g.
        in typedict below, if it were needed). just have e.g. 0: r'$ \sigma $',
        '''
        # print(emissions)
        decay_pd = pd.DataFrame(decays, columns=["Time (ps)", "Decay type"])
        # print(decay_pd.to_string())
        # typedict = {"Ann": 0., "Pool": 1., "PQ":, 2., "Q": 3.}
        typedict = {1.: "Ann.", 2.: "Pool", 3.: "PQ", 4: "Q"}
        decay_pd = decay_pd.replace({"Decay type": typedict})
        ax = sns.histplot(data=decay_pd, x="Time (ps)", hue="Decay type",
                element="step", fill=False)
        ax.set_yscale('log')
        plt.axvline(x=tau, ls="--", c='k')
        plt.savefig("{}/{}_plot.pdf".format(path, file_prefix))
        plt.close()

        ax = sns.histplot(data=decays[:, 0], element="step",
                          binwidth=25., fill=False)
        ax.set_xlabel("Time (ps)")
        plt.savefig("{}/{}_hist.pdf".format(path, file_prefix))
        plt.close()

        # matplotlib histogram - output bins and vals for lmfit
        histvals, histbins = fit.histogram(emissions,
                "{}/{}_emission_histogram.pdf".format(path, file_prefix))
        x = histbins[:-1] + (np.diff(histbins) / 2.)
        histvals = histvals / np.max(histvals)
        np.savetxt("{}/{}_histvals.dat".format(path, file_prefix), histvals)
        np.savetxt("{}/{}_histbins.dat".format(path, file_prefix), histbins)
        long_gauss = 1. / (pulse.sigma * np.sqrt(2. * np.pi)) * \
            np.exp(- (x - pulse.mu)**2 \
            / (np.sqrt(2.) * pulse.sigma)**2)
        long_gauss = long_gauss/np.max(long_gauss)
        conv = np.convolve(long_gauss, histvals)
        deconv, remainder = signal.deconvolve(histvals, long_gauss)
        print(deconv)
        plt.plot(x, long_gauss, label='Gaussian')
        plt.plot(x, histvals, label='histogram')
        plt.plot(conv, label='convolution')
        plt.gca().set_xscale('log')
        plt.legend()
        plt.savefig("{}/{}_conv.pdf".format(path, file_prefix))
        plt.close()
        plt.plot(deconv, label='deconvolution')
        plt.savefig("{}/{}_deconv.pdf".format(path, file_prefix))
        plt.close()

        # x = x[np.where(x > 200.)]
        # histvals = histvals[np.where(x > 200.)]
        histvals = histvals / np.max(histvals)
        try:
            mono_fit = fit.lm(1, x, histvals, model, 1./pulse.mu)
            print(mono_fit.fit_report())
            fig = mono_fit.plot(xlabel="Time (ps)", ylabel="Counts")
            axes = fig.gca()
            # axes.set_xscale('log')
            axes.set_yscale('log')
            axes.set_xlim((10., 10000.))
            axes.set_ylim((0.01, 1.5))
            plt.grid()
            plt.savefig("{}/{}_mono.pdf".format(path, file_prefix))
            plt.close()
            comps = mono_fit.best_values
            print(comps)
            mono_tau.append([fluence, comps['exp1decay']])
            print("Mono-exponential <tau> = {:8.3f}".format(comps['exp1decay']))
        except ValueError:
            print("Mono-exponential fit failed!")
            pass
        try:
            bi_fit = fit.lm(2, x, histvals, model, 1./pulse.mu)
            print(bi_fit.fit_report())
            fig = bi_fit.plot(xlabel="Time (ps)", ylabel="Counts")
            axes = fig.gca()
            # axes.set_xscale('log')
            axes.set_yscale('log')
            axes.set_xlim((10., 10000.))
            axes.set_ylim((0.01, 1.5))
            plt.grid()
            plt.savefig("{}/{}_bi.pdf".format(path, file_prefix))
            plt.close()
            comps = bi_fit.best_values
            avg_tau = (comps['exp1decay'] * comps['exp1amplitude'] \
                + comps['exp2decay'] * comps['exp2amplitude']) \
                / (comps['exp1amplitude'] + comps['exp2amplitude'])
            bi_tau.append([fluence, avg_tau])
            print("Bi-exponential <tau> = {:8.3f}".format(avg_tau))
        except ValueError:
            print("Bi-exponential fit failed!")
            pass
        try:
            tri_fit = fit.lm(3, x, histvals, model, 1./pulse.mu)
            print(tri_fit.fit_report())
            fig = tri_fit.plot(xlabel="Time (ps)", ylabel="Counts")
            axes = fig.gca()
            # axes.set_xscale('log')
            axes.set_yscale('log')
            axes.set_xlim((10., 10000.))
            axes.set_ylim((0.01, 1.5))
            plt.grid()
            plt.savefig("{}/{}_tri.pdf".format(path, file_prefix))
            plt.close()
            comps = tri_fit.best_values
            avg_tau = (comps['exp1decay'] * comps['exp1amplitude'] \
                + comps['exp2k'] * comps['exp2amp']) \
                / (comps['exp1amplitude'] + comps['exp2amp'])
            tri_tau.append([fluence, avg_tau])
            print("Tri-exponential <tau> = {:8.3f}".format(avg_tau))
        except ValueError:
            print("Tri-exponential fit failed!")
            pass
    end_time = time.monotonic()
    print("Total time elapsed: {}".format((end_time - start_time)))

    np.savetxt("{}/mono_tau.dat".format(path), np.array(mono_tau))
    np.savetxt("{}/bi_tau.dat".format(path), np.array(bi_tau))
    np.savetxt("{}/tri_tau.dat".format(path), np.array(tri_tau))
    subprocess.run(['python', 'plot_tau.py', '{}/{:d}_{:3.2f}'.format(path, n_iterations, rho_quenchers)], check=True)
