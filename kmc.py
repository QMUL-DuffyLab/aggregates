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
    of process x. e.g. we assume "pool" chlorophylls decay on a timescale
    of a few nanoseconds, whereas the decay of a carotenoid quencher might
    be 10-15ps. Give these in ps!
    emissive is a bool list I use to track which of the decays are emissive.
    Not currently useful but might be.
    '''
    def __init__(self, tau_hop, tau_pool, tau_pq, tau_q,
            t_po_pq, t_pq_po, t_pq_q, t_q_pq, t_annihilation,
            emissive, decay_on_pq, ann_on_pq):
        self.tau_hop        = tau_hop
        self.hop            = 1. / tau_hop
        self.g_pool         = 1. / tau_pool
        self.g_pq           = 1. / tau_pq
        self.g_q            = 1. / tau_q
        self.k_po_pq        = 1. / t_po_pq
        self.k_pq_po        = 1. / t_pq_po
        self.k_pq_q         = 1. / t_pq_q
        self.k_q_pq         = 1. / t_q_pq
        self.k_ann          = 1. / t_annihilation
        self.emissive       = emissive
        self.decay_on_pq    = decay_on_pq
        self.ann_on_pq      = ann_on_pq
    def print(self):
        print("Rates - all given in ps^{-1}:")
        print("k_hop = {:6.4f}".format(self.hop))
        print("g_pool = {:6.4f}".format(self.g_pool))
        print("g_pq = {:6.4f}".format(self.g_pq))
        print("g_q = {:6.4f}".format(self.g_q))
        print("k_po_pq = {:6.4f}".format(self.k_po_pq))
        print("k_pq_po = {:6.4f}".format(self.k_pq_po))
        print("k_pq_q = {:6.4f}".format(self.k_pq_q))
        print("k_q_pq = {:6.4f}".format(self.k_q_pq))
        print("k_ann = {:6.4f}".format(self.k_ann))
        print("emissive decays: {}".format(self.emissive))

class Iteration():
    import numpy as np
    '''
    Take an aggregate - either one constructed from an experimental image
    or a generated one - populate it with a fraction of quenchers and a
    set of excitons (the number of excitons is dependent on fluence), and
    run kinetic Monte Carlo until all the excitons are gone.
    Record the intervals between exciton losses and whether these were
    emissive or not.
    '''
    def __init__(self, aggregate, model, pulse, rho_quenchers,
            path, prefix, fluence, binwidth, max_count,
            verbose=False, draw_frames=False):
        if verbose:
            self.output = sys.stdout
        else:
            self.output = open(os.devnull, "w")
        self.aggregate = aggregate
        self.model = model
        self.pulse = pulse
        self.fluence = fluence
        self.rho_quenchers = rho_quenchers
        self.n_sites = len(self.aggregate.trimers)
        self.base_rates = self.rate_setup()
        self.write_arrays(path, binwidth, max_count, prefix)
            
    def rate_setup(self):
        '''
        generate a list of numpy arrays where each array is the set
        of all possible transition rates for the corresponding trimer.
        the order's arbitrary, but has to be followed in the Metropolis
        and kinetic Monte Carlo simulations below; the function `move`
        carries out the process based on which of these rates we pick.
        note that the final two sets of rates are for the pre-quenching
        and quenching states - we assume these states are connected to
        certain trimers at random, but that they are identical for each
        '''
        self.max_neighbours = np.max(np.fromiter((len(x.get_neighbours()) 
                     for x in self.aggregate.trimers), int))
        self.base_rates = np.zeros((self.n_sites + 2, self.max_neighbours + 5),\
                dtype=float)
        for i in range(self.n_sites + 2):
            t = self.base_rates[i].copy()
            # first element is generation
            if i < self.n_sites:
                # trimer (pool)
                n_neigh = len(self.aggregate.trimers[i].get_neighbours())
                for j in range(n_neigh):
                    t[self.max_neighbours - j] = self.model.hop
                t[self.max_neighbours + 2] = self.model.k_po_pq
                t[self.max_neighbours + 3] = self.model.g_pool
                t[self.max_neighbours + 4] = self.model.k_ann
            elif i == self.n_sites:
                if (self.rho_quenchers != 0):
                    # pre-quencher
                    t[self.max_neighbours + 1] = self.model.k_pq_po
                    t[self.max_neighbours + 2] = self.model.k_pq_q
                    t[self.max_neighbours + 3] = self.model.g_pq
                    t[self.max_neighbours + 4] = self.model.k_ann
            elif i == self.n_sites + 1:
                # quencher
                if (self.rho_quenchers != 0):
                    t[self.max_neighbours + 2] = self.model.k_q_pq
                    t[self.max_neighbours + 3] = self.model.g_q
                    t[self.max_neighbours + 4] = self.model.k_ann
            self.base_rates[i] = t
        return self.base_rates

    def write_arrays(self, path, binwidth, max_count, prefix):
        self.params_file = "{}/{}params".format(path, prefix)
        neighbours_file = "{}/neighbours.dat".format(path)
        rates_file = "{}/base_rates.dat".format(path)
        np.savetxt(rates_file, self.base_rates.flatten())
        neighbours = np.zeros((self.n_sites, self.max_neighbours), dtype=int)
        for i in range(self.n_sites):
            for j in range(len(self.aggregate.trimers[i].get_neighbours())):
                # add 1 here because fortran's 1-indexed!
                # also the array's np.zeros - need to distinguish
                neighbours[i][self.max_neighbours - (j + 1)] = self.aggregate.trimers[i].get_neighbours()[j].index + 1
        np.savetxt(neighbours_file, neighbours.flatten(order='F'), fmt='%6d')
        with open(self.params_file, 'w') as f:
            f.write("{:d}\n".format(self.n_sites))
            f.write("{:d}\n".format(self.max_neighbours))
            f.write("{:f}\n".format(self.rho_quenchers))
            f.write("{:f}\n".format(self.fluence))
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
