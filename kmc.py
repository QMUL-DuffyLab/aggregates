#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import cv2
from trimer import Aggregate

class Pulse():
    def __init__(self, fwhm, mu):
        self.fwhm = fwhm # ps
        self.mu = mu
        self.sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))
        self.t = np.arange(0., 2. * mu, 1.)
        self.ft = 1. / (self.sigma * np.sqrt(2. * np.pi)) * \
    np.exp(- (self.t - self.mu)**2 / (np.sqrt(2.) * self.sigma)**2)

class Rates():
    '''
    Simple container for a set of relevant rates - tau_x is the time constant
    of process x. Give the times in ps! As you can see below __init__
    takes the reciprocal to give a rate in ps^{-1} which is used throughout.
    emissive is a bool list I use to track which of the decays are emissive.
    '''
    def __init__(self, tau_hop, tau_a, tau_p, tau_q,
            t_a_p, t_p_a, t_p_q, t_q_p, t_annihilation,
            emissive):
        self.tau_hop  = tau_hop
        self.hop      = 1. / tau_hop
        self.g_a      = 1. / tau_a
        self.g_p      = 1. / tau_p
        self.g_q      = 1. / tau_q
        self.k_a_p    = 1. / t_a_p
        self.k_p_a    = 1. / t_p_a
        self.k_p_q    = 1. / t_p_q
        self.k_q_p    = 1. / t_q_p
        self.k_ann    = 1. / t_annihilation
        self.emissive = emissive
    def print(self):
        print("Rates - all given in ps^{-1}:")
        print("k_hop = {:6.4f}".format(self.hop))
        print("g_a = {:6.4f}".format(self.g_a))
        print("g_p = {:6.4f}".format(self.g_p))
        print("g_q = {:6.4f}".format(self.g_q))
        print("k_a_p = {:6.4f}".format(self.k_a_p))
        print("k_p_a = {:6.4f}".format(self.k_p_a))
        print("k_p_q = {:6.4f}".format(self.k_p_q))
        print("k_q_p = {:6.4f}".format(self.k_q_p))
        print("k_ann = {:6.4f}".format(self.k_ann))
        print("emissive decays: {}".format(self.emissive))

class Setup():
    '''
    Take an aggregate - either one constructed from an experimental image
    or a generated one - populate it with a fraction of quenchers and a
    set of excitons (the number of excitons is dependent on fluence), and
    run kinetic Monte Carlo until all the excitons are gone.
    Record the intervals between exciton losses and whether these were
    emissive or not.
    '''
    def __init__(self, aggregate, model, pulse, rho_quenchers,
            path, prefix, n_per_t, binwidth, max_count,
            verbose=False):
        if verbose:
            self.output = sys.stdout
        else:
            self.output = open(os.devnull, "w")
        self.aggregate = aggregate
        self.model = model
        self.pulse = pulse
        self.n_per_t = n_per_t
        self.rho_quenchers = rho_quenchers
        self.n_sites = len(self.aggregate.trimers)
        self.base_rates = self.rate_setup()
        self.neighbours = np.zeros((self.n_sites,
            self.max_neighbours), dtype=int)
        for i in range(self.n_sites):
            for j in range(len(self.aggregate.trimers[i].get_neighbours())):
                self.neighbours[i][j] = \
                        self.aggregate.trimers[i].get_neighbours()[j].index
        self.write_arrays(path, binwidth, max_count, prefix)
            
    def rate_setup(self):
        '''
        generate a big numpy array where each row is the set
        of all possible transition rates for the corresponding trimer.
        the order's arbitrary, but has to be followed in the Metropolis
        and kinetic Monte Carlo simulations below; the `rate_calc` and
        `move` functions carry out the process based on which rate is picked.
        the final two sets of rates are for the pre-quenching
        and quenching states - we assume these states are connected to
        certain trimers at random, but that they are identical for each.
        '''
        self.max_neighbours = np.max(np.fromiter((len(x.get_neighbours())
                     for x in self.aggregate.trimers), int))
        self.base_rates = np.zeros((self.n_sites + 2, self.max_neighbours + 6),\
                dtype=float)
        for i in range(self.n_sites + 2):
            t = self.base_rates[i].copy()
            # first element is generation
            # second is stimulated emission
            if i < self.n_sites:
                # antenna
                n_neigh = len(self.aggregate.trimers[i].get_neighbours())
                for j in range(n_neigh):
                    t[j + 2] = self.model.hop
                t[self.max_neighbours + 3] = self.model.k_a_p
                t[self.max_neighbours + 4] = self.model.g_a
                t[self.max_neighbours + 5] = self.model.k_ann
            elif i == self.n_sites:
                '''
                now the pre-quencher and quencher rates. these rows
                have to be the same size (no ragged arrays in numpy)
                even though there are fewer processes; the rates are put
                at the end so that the rate_calc code can do the same
                thing for each index as much as possible (the
                decay rate has to be multiplied by n, etc. etc.)
                '''
                if (self.rho_quenchers != 0):
                    # pre-quencher
                    t[self.max_neighbours + 2] = self.model.k_p_a
                    t[self.max_neighbours + 3] = self.model.k_p_q
                    t[self.max_neighbours + 4] = self.model.g_p
                    t[self.max_neighbours + 5] = self.model.k_ann
            elif i == self.n_sites + 1:
                # quencher
                if (self.rho_quenchers != 0):
                    t[self.max_neighbours + 3] = self.model.k_q_p
                    t[self.max_neighbours + 4] = self.model.g_q
                    t[self.max_neighbours + 5] = self.model.k_ann
            print(t)
            self.base_rates[i] = t
        return self.base_rates

    def write_arrays(self, path, binwidth, max_count, prefix):
        '''
        write out parameters, rates and neighbours for the fortran code
        '''
        self.params_file = "{}/{}params".format(path, prefix)
        neighbours_file = "{}/neighbours.dat".format(path)
        rates_file = "{}/base_rates.dat".format(path)
        # base_rates is a 1-d array in the fortran code
        # so no need to worry about fortran ordering
        np.savetxt(rates_file, self.base_rates.flatten())
        # + 1 here because fortran is 1-indexed
        np.savetxt(neighbours_file,
                self.neighbours.flatten(order='F') + 1, fmt='%6d')
        with open(self.params_file, 'w') as f:
            f.write("{:d}\n".format(self.n_sites))
            f.write("{:d}\n".format(self.max_neighbours))
            f.write("{:f}\n".format(self.rho_quenchers))
            f.write("{:f}\n".format(self.n_per_t))
            f.write("{:f}\n".format(self.pulse.mu))
            f.write("{:f}\n".format(self.pulse.fwhm))
            f.write("{:f}\n".format(binwidth))
            f.write("{:d}\n".format(max_count))
            # [1:-1] here because fortran doesn't like parsing the [ ]
            f.write(str(self.model.emissive)[1:-1])
            f.write("\n")
            f.write(rates_file)
            f.write("\n")
            f.write(neighbours_file)
            f.write("\n")
            f.write(prefix)
