#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from trimer import Aggregate, theoretical_aggregate

class Rates():
    '''
    take a set of rates and normalise them to the hopping rate:
    e.g. a fast pre-quencher to quencher rate of 400fs^{-1}
    compared to a hopping rate of 20ps^{-1} gives a rate of 20/0.4 = 50.
    Give all the rates in ps!!!
    '''
    def __init__(self, hopping_time, Gamma_pool, Gamma_pq, Gamma_q, k_po_pq, k_pq_po, k_pq_q, k_q_pq, k_annihilation):
        np.seterr(divide='ignore')
        self.hop = 1.
        self.g_pool  = (1. / Gamma_pool) * hopping_time
        self.g_pq    = (1. / Gamma_pq) * hopping_time
        self.g_q     = (1. / Gamma_q) * hopping_time
        self.k_po_pq = (1. / k_po_pq) * hopping_time
        self.k_pq_po = (1. / k_pq_po) * hopping_time
        self.k_pq_q  = (1. / k_pq_q) * hopping_time
        self.k_q_pq  = (1. / k_q_pq) * hopping_time
        self.k_ann   = (1. / k_annihilation) * hopping_time

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
        self.n_i = np.zeros(self.n_sites, dtype=np.uint8)
        self.previous_pool = np.zeros(self.n_e, dtype=np.int8) - 1
        self.transitions = [[]] * self.n_sites
        self.transition_calc()
        self.t = 0. # probably need to write a proper init function
        self.kmc_setup()
        print("Init: excitations at trimers: ", self.n_i.nonzero()[0])
        for i in range(self.n_st):
            self.kmc_step()
            print("after step {}, time = {:8.3e}, excitations at trimers: ".format(i, self.t), self.n_i.nonzero()[0])
            if i // 10 == 0:
                self.draw("components/frame_{:03d}.jpg".format(i))

    def transition_calc(self):
        print(self.n_sites)
        for i in range(self.n_sites):
            t = []
            if i < self.n_sites - 2:
                # trimer (pool)
                for j in range(len(self.aggregate.trimers[i].get_neighbours())):
                    t.append(rates.hop)
                t.append(rates.k_po_pq)
                t.append(rates.g_pool)
                t.append(rates.k_ann)
            elif i == self.n_sites - 2:
                # pre-quencher
                t.append(rates.k_pq_po)
                t.append(rates.k_pq_q)
                t.append(rates.g_pq)
            elif i == self.n_sites - 1:
                # quencher
                t.append(rates.k_q_pq)
                t.append(rates.g_q)
            self.transitions[i] = np.array(t)
            print(i, self.transitions[i])
    
    def kmc_setup(self):
        for i in range(self.n_e):
            # only generate excitations on the pools, not quenchers
            choice = self.rng.integers(low=0,
                high=len(self.aggregate.trimers))
            self.n_i[choice] += 1
            self.currently_occupied.append(choice)

    def kmc_step(self):
        '''
        first draft of one kMC step
        extremely inefficient
        '''
        if np.sum(self.n_i) == 0:
            print("no excitations left!")
            return
        currently_occupied = []
        # the [0] is because nonzero() returns a tuple
        # and the list of nonzero indices is the first bit
        occupied_sites = self.n_i.nonzero()[0]
        for i in range(len(occupied_sites)):
            for j in range(self.n_i[occupied_sites[i]]):
                currently_occupied.append(occupied_sites[i])
        print("Currently occupied = {}".format(currently_occupied))
        i = self.rng.integers(low=0, high=len(currently_occupied))
        rand1 = self.rng.random()
        rand2 = self.rng.random()
        print("{}: occupations at sites {}, occupation numbers {}".format(i, self.n_i.nonzero()[0], self.n_i[np.nonzero(self.n_i)]))
        ind = currently_occupied[i]
        if ind < self.n_sites - 2:
            # pool
            n = self.n_i[ind]
            if n >= 2:
                (q, k_tot) = select_process(self.transitions[ind], rand1)
            else:
                (q, k_tot) = select_process(self.transitions[ind][:-1], rand1)
            '''
            q is a number on [0, len(self.transitions[ind]]
            but can only be len(self.transitions[ind]) if n >= 2.
            so we can safely just put in all the cases here without
            worrying about accidentally subtracting 2 and getting
            negative populations
            '''
            if (q == len(self.transitions[ind]) - 1):
                # annihilation
                print("po ann from trimer {}".format(ind))
                self.n_i[ind] -= 2
            elif (q == len(self.transitions[ind]) - 2):
                # decay
                print("po decay from trimer {}".format(ind))
                self.n_i[ind] -= 1
            elif (q == len(self.transitions[ind]) - 3):
                # hop to pre-quencher
                print("po->pq from trimer {}".format(ind))
                self.n_i[ind] -= 1
                self.n_i[-2]  += 1
                self.previous_pool[i] = ind
            else:
                # hop to neighbour
                print(q, ind, [self.aggregate.trimers[ind].get_neighbours()[j].index for j in range(len(self.aggregate.trimers[ind].get_neighbours()))])
                nn = self.aggregate.trimers[ind].get_neighbours()[q].index
                print("neighbour: {} to {}".format(ind, nn))
                self.n_i[ind] -= 1
                self.n_i[self.aggregate.trimers[ind].get_neighbours()[q].index] += 1
        elif ind == self.n_sites - 2:
            '''
            NB: in principle it's possible for multiple excitations
            to land on the pre-quencher/quencher at the same time;
            I don't currently deal with annihilation on those.
            '''
            # pre-quencher
            (q, k_tot) = select_process(self.transitions[ind], rand1)
            if (q == 0):
                # hop back to pool
                print("pq->po, previous_pool = {}".format(self.previous_pool))
                self.n_i[ind] -= 1
                self.n_i[self.previous_pool[i]] += 1
            elif (q == 1):
                # hop to quencher
                print("pq->q")
                self.n_i[ind] -= 1
                self.n_i[-1] = 1
            elif (q == 2):
                # decay
                print("pq decay")
                self.n_i[ind] -= 1
                self.previous_pool[i] = -1
        elif ind == self.n_sites - 1:
            # quencher
            (q, k_tot) = select_process(self.transitions[ind], rand1)
            if (q == 0):
                # hop back to pre-quencher
                print("q->pq")
                self.n_i[ind] -= 1
                self.n_i[-2] += 1
            elif (q == 1):
                # decay
                print("q decay")
                self.n_i[ind] -= 1
                self.previous_pool[i] = -1
        self.t -= np.log(rand2 / k_tot)

    def draw(self, filename):
        '''
        draw the system after one step.
        doesn't do pre quencher or quencher yet
        '''
        xmax = np.max([np.abs(t.x) for t in self.aggregate.trimers])
        scale = 4
        img = np.zeros((2 * scale * int(xmax + 4. * r) + 200,
            2 * scale * int(xmax + 4. * r) + 200, 3), np.uint8)
        for i, t in enumerate(self.aggregate.trimers):
            cv2.circle(img, (int(scale * (t.y + xmax + 2. * r)),
                int(scale * (t.x + xmax + 2. * r))), 
                int(scale * t.r), (255, 255, 255), -1)
            if (self.n_i[i] != 0):
                # coloured circle to indicate exciton
                cv2.circle(img, (int(scale * (t.y + xmax + 2. * r)),
                    int(scale * (t.x + xmax + 2. * r))), 
                    int(scale * t.r), (26, 0, 153), -1)
                cv2.putText(img, "{:1d}".format(self.n_i[i]),
                        (scale * int(t.y + xmax + 2 * r - 1),
                    scale * int(t.x + xmax + 2 * r + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
        # pre-quencher
        pq_colour = (0, 94, 20)
        cv2.rectangle(img, pt1=(2 * scale * int(xmax + 4 * r) + 50, 100), 
            pt2=(2 * scale * int(xmax + 4 * r) + 100, 150),
            color=pq_colour, thickness=-1)
        cv2.putText(img, "{:2d}".format(self.n_i[-2]),
                (2 * scale * int(xmax + 4 * r) + 55, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
        cv2.putText(img, "PQ",
                (2 * scale * int(xmax + 4 * r) + 55, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, pq_colour, 3)
        # quencher
        q_colour = (60, 211, 242)
        cv2.rectangle(img, pt1=(2 * scale * int(xmax + 4 * r) + 50, 200), 
            pt2=(2 * scale * int(xmax + 4 * r) + 100, 250),
            color=q_colour, thickness=-1)
        cv2.putText(img, "{:2d}".format(self.n_i[-1]),
                (2 * scale * int(xmax + 4 * r) + 55, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
        cv2.putText(img, "Q",
                (2 * scale * int(xmax + 4 * r) + 55, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, q_colour, 3)
        cv2.imwrite(filename, img)

def select_process(k_p, rand):
    '''
    BKL algorithm for KMC.
    choose which configuration to jump to given a set of transition rates
    to those configurations from the current one, and a random number in [0,1].
    '''
    k_p_s = np.cumsum(k_p)
    k_tot = k_p_s[-1]
    i = 0
    while rand * k_tot > k_p_s[i]:
        i += 1
    return (i, k_tot)

if __name__ == "__main__":
    r = 5.
    lattice_type = "hex"
    n_iter = 5

    rates = Rates(25., 4000., 4000., 14., 7., 1., 20., np.inf, 0.5)
    agg = theoretical_aggregate(r, 2.5*r, lattice_type, n_iter)
    it = Iteration(agg, rates, 0, 100, 20)
