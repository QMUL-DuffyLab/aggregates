#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class Rates(hopping_time, Gamma_pool, Gamma_pq, Gamma_q, k_po_pq, k_pq_po, k_pq_q, k_q_pq, k_annihilation):
    '''
    take a set of rates and normalise them to the hopping rate:
    e.g. a fast pre-quencher to quencher rate of 400fs^{-1}
    compared to a hopping rate of 20ps^{-1} gives a rate of 20/0.4 = 50.
    Give all the rates in ps!!!
    '''
    self.hop = hopping_time
    self.g_pool = Gamma_pool / self.hop
    self.g_pq = Gamma_pq / self.hop
    self.g_q = Gamma_q / self.hop
    self.k_po_pq = k_po_pq / self.hop
    self.k_pq_po = k_pq_po / self.hop
    self.k_pq_q = k_pq_q / self.hop
    self.k_q_pq = k_q_pq / self.hop
    self.k_ann = k_annihilation / self.hop

class Iteration():
    def __init__(self, aggregate, rates, seed, n_steps, n_excitations):
        self.aggregate = aggregate
        self.rates = rates
        self.rng = np.random.default_rng(seed=seed)
        self.currently_occupied = []
        self.n_st = n_steps
        self.n_e = n_excitations
        # +2 for the pre-quencher and quencher
        self.n_sites = len(self.aggregate.trimers) + 2
        self.n_i = np.zeros(self.n_sites)
        self.transitions = np.zeros((self.n_sites, 1))
        self.transition_calc()

    def transition_calc(self):
        for i in range(self.n_sites):
            if i <= self.n_sites - 2:
                # trimer (pool)
                for j in range(len(self.aggregate.trimers[i].get_neighbours())):
                    self.transitions[i].append(rates.hop, axis=1)
                self.transitions[i].append(rates.k_po_pq, axis=1)
                self.transitions[i].append(rates.g_pool, axis=1)
                self.transitions[i].append(rates.k_ann, axis=1)
            elif i == self.n_sites - 1:
                # pre-quencher
                self.transitions[i].append(rates.k_pq_po, axis=1)
                self.transitions[i].append(rates.k_pq_q, axis=1)
                self.transitions[i].append(rates.g_pq, axis=1)
            elif i == self.n_sites:
                # quencher
                self.transitions[i].append(rates.k_q_pq, axis=1)
                self.transitions[i].append(rates.g_q, axis=1)
    
    def mc_setup(self, n_excitations):
        for i in range(n_excitations):
            # only generate excitations on the pools, not quenchers
            self.n_i[self.rng.integers(low=0,
                high=len(self.aggregate.trimers))] += 1

    def mc_step(self):
        '''
        first draft of one MC step
        '''
        start = rng.integers(low=0, high=n)
        # here we assume we're on a trimer (in the pool)
        # assume that the excitation is equally likely to hop
        # to one of the trimer's neighbours or to the pre-quencher
        neighbours = aggregate.trimers[start].get_neighbours()
        selected = rng.choice(neighbours + ["P"])
        if (selected == "P"):
            # pre-quencher
            l = rates.k_po_pq
            prob = l * np.exp(-l * t)
        else:
            # hop to another trimer




