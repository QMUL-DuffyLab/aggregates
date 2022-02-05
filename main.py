#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from trimer import Aggregate, theoretical_aggregate
from kmc import Pulse, Model, Iteration

def histogram(data, filename, binwidth=25.):
    '''
    plot a histogram of all the emissive decays via matplotlib;
    return the set of bin values and edges so we can fit them after
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    (n, bins, patches)= plt.hist(data,
            bins=np.arange(np.min(data), np.max(data) + binwidth,
                binwidth), histtype="step", color='C0')
    plt.gca().set_ylabel("Counts")
    plt.gca().set_xlabel("Time (ps)")
    plt.savefig(filename)
    plt.close()
    return n, bins

def lm(no_exp, x, y, model):
    from lmfit.models import ExponentialModel
    ''' use lmfit to a mono or biexponential '''
    exp1 = ExponentialModel(prefix='exp1')
    pars = exp1.make_params(exp1decay=1./model.g_pool,
                            exp1amplitude=np.max(y))
    mod = exp1
    if no_exp == 2:
        exp2 = ExponentialModel(prefix='exp2')
        pars.update(exp2.make_params(exp2decay=1./model.k_ann,
                                     exp2amplitude=np.max(y)))
        mod = mod + exp2
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    return out
    
if __name__ == "__main__":
    r = 5.
    lattice_type = "hex"
    n_iter = 8 # 434 trimers for honeycomb
    n_iterations = 1000
    rho_quenchers = 0.0
    # fluences given here as photons per pulse per unit area - 485nm
    fluences = [6.07E12, 3.03E13, 6.24E13, 1.31E14,
            1.9E14, 3.22E14, 6.12E14, 9.48E14]
    # annihilation, pool decay, pq decay, q decay
    model_dict = {
     'lut_eet': Model(20., 3800., 3800., 14., 
         7., 1., 20., np.inf, 48., [False, True, True, False], True, True),
     'schlau_cohen': Model(20., 3800., 3800., 14., 
         7., 1., 0.4, 0.4, 48., [False, True, True, False], True, True)
     }
    model_key = 'lut_eet'
    model = model_dict[model_key]
    mono_tau = []
    bi_tau = []
    pulse = Pulse(fwhm=50., mu=100.)
    for fluence in fluences:
        print("Fluence = {:4.2e}, n_iterations = {:d}".format(
            fluence, n_iterations))
        path = "out/{}/{}".format(model_key, lattice_type)
        os.makedirs(path, exist_ok=True)
        file_prefix = "{:d}_{:3.2f}_{:4.2e}".format(
                n_iterations, rho_quenchers, fluence)
        decay_filename = "{}/{}_decays.dat".format(path, file_prefix)
        emission_filename = "{}/{}_emissions.dat".format(path, file_prefix)

        # note - second parameter here is the nn cutoff. set to 0 to
        # disable excitation hopping between trimers
        agg = theoretical_aggregate(r, 0., lattice_type, n_iter)
        n_es = []
        means = []
        stddevs = []
        emission_means = []
        emission_stddevs = []
        yields = []
        decay_file     = open(decay_filename, mode='w')
        emissions_file = open(emission_filename, mode='w')
        for i in range(n_iterations):
            verbose = False
            emissions = []
            it = Iteration(agg, model, pulse, i,
                    rho_quenchers, 0, fluence, verbose=verbose)
            n_es.append(len(it.loss_times))
            for k in range(len(it.loss_times)):
                print("{:1.5e} {:1d}".format(it.loss_times[k], it.decay_type[k]), 
                        file=decay_file)
                if model.emissive[it.decay_type[k]] is True:
                    print("{:1.5e}".format(it.loss_times[k]), file=emissions_file)
                    emissions.append(it.loss_times[k])
            means.append(np.mean(it.loss_times))
            stddevs.append(np.std(it.loss_times))
            emission_means.append(np.mean(emissions))
            emission_stddevs.append(np.std(emissions))
            yields.append(emission_means[-1]/means[-1])
            if verbose is True:
                print("Iteration {:d}".format(i))
                print("=== μ, σ ===")
                print(means[-1], stddevs[-1])
                print("=== EMISSION μ, σ ===")
                print(emission_means[-1], emission_stddevs[-1])
            else:
                width = os.get_terminal_size().columns - 20
                print("\rProgress: [{0}{1}] {2}%".format(
                    '█'*int((i + 1) * width/n_iterations),
                    ' '*int(width - ((i + 1) * width/n_iterations)),
                    int((i + 1) * 100 / n_iterations)), end='')

        print() # newline after progress bar
        decay_file.close()
        emissions_file.close()
        '''
        tau is just a straight estimation of everything
        mean of means and mean of emissive means reported separately
        statistics of these????? are they valid things to report???
        '''
        decays = np.loadtxt(decay_filename)
        emissions = np.loadtxt(emission_filename)
        tau = np.mean(decays[:, 0])
        sigma_tau = np.std(decays[:, 0])
        print("Total μ, σ: ", tau, sigma_tau)
        print("μ, σ of means: ", np.mean(means),
                np.std(means))
        print("μ, σ of emission means: ", np.mean(emission_means),
                np.std(emission_means))
        print("μ, σ of excitation numbers: ", np.mean(n_es),
                np.std(n_es))
        print("Average fraction of excited trimers ρ_exc: ",
                np.mean(n_es) / len(it.aggregate.trimers))

        np.savetxt("{}/{}_total_mean_std.dat".format(path, file_prefix),
                [tau, sigma_tau])
        np.savetxt("{}/{}_total_emission_mean_std.dat".format(path, file_prefix),
                [np.mean(emissions), np.std(emissions)])
        np.savetxt("{}/{}_n_es.dat".format(path, file_prefix), n_es)
        np.savetxt("{}/{}_means.dat".format(path, file_prefix), means)
        np.savetxt("{}/{}_stddevs.dat".format(path, file_prefix), stddevs)
        np.savetxt("{}/{}_emission_means.dat".format(path, 
            file_prefix), emission_means)
        np.savetxt("{}/{}_emission_stddevs.dat".format(path, 
            file_prefix), emission_stddevs)

        '''
        NB: latex will work in column names and captions (e.g.
        in typedict below, if it were needed). just have e.g. 0: r'$ \sigma $',
        '''
        decay_pd = pd.DataFrame(decays, columns=["Time (ps)", "Decay type"])
        # typedict = {"Ann": 0., "Pool": 1., "PQ":, 2., "Q": 3.}
        typedict = {0.: "Ann.", 1.: "Pool", 2.: "PQ", 3: "Q"}
        decay_pd = decay_pd.replace({"Decay type": typedict})
        ax = sns.histplot(data=decay_pd, x="Time (ps)", hue="Decay type",
                element="step", fill=False)
        plt.axvline(x=tau, ls="--", c='k')
        plt.savefig("{}/{}_plot.pdf".format(path, file_prefix))
        plt.close()

        ax = sns.histplot(data=emissions, element="step",
                          binwidth=25., fill=False)
        ax.set_xlabel("Time (ps)")
        plt.savefig("{}/{}_hist.pdf".format(path, file_prefix))
        plt.close()

        # matplotlib histogram - output bins and vals for lmfit
        # emissions or all decays? who tf knows :)
        histvals, histbins = histogram(emissions,
                "{}/{}_hist_mpl.pdf".format(path, file_prefix))
        x = histbins[:-1] + (np.diff(histbins) / 2.)
        try:
            mono_fit = lm(1, x, histvals, model)
            print(mono_fit.fit_report())
            fig = mono_fit.plot(xlabel="Time (ps)", ylabel="Counts")
            axes = fig.gca()
            axes.set_yscale('log')
            ax.set_ylim((1., np.max(x)))
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
            bi_fit = lm(2, x, histvals, model)
            print(bi_fit.fit_report())
            fig = bi_fit.plot(xlabel="Time (ps)", ylabel="Counts")
            axes = fig.gca()
            axes.set_yscale('log')
            ax.set_ylim((1., np.max(x)))
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
        
    np.savetxt("{}/mono_tau.dat".format(path), np.array(mono_tau))
    np.savetxt("{}/bi_tau.dat".format(path), np.array(bi_tau))