#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import cv2
from trimer import Aggregate, theoretical_aggregate

class Rates():
    '''
    take a set of rates and normalise them to the hopping rate:
    e.g. a fast pre-quencher to quencher rate of 400fs^{-1}
    compared to a hopping rate of 20ps^{-1} gives a rate of 20/0.4 = 50.
    Give as equivalent times in ps!
    '''
    def __init__(self, tau_hop, tau_pool, tau_pq, tau_q,
            t_po_pq, t_pq_po, t_pq_q, t_q_pq, t_annihilation):
        self.tau_hop = tau_hop
        self.hop = 1.
        self.g_pool  = self.tau_hop / tau_pool
        self.g_pq    = self.tau_hop / tau_pq
        self.g_q     = self.tau_hop / tau_q
        self.k_po_pq = self.tau_hop / t_po_pq
        self.k_pq_po = self.tau_hop / t_pq_po
        self.k_pq_q  = self.tau_hop / t_pq_q
        self.k_q_pq  = self.tau_hop / t_q_pq
        self.k_ann   = self.tau_hop / t_annihilation

class Iteration():
    def __init__(self, aggregate, rates, seed,
            n_steps, n_excitations, verbose=False):
        if verbose:
            self.output = sys.stdout
        else:
            self.output = open(os.devnull, "w")
        self.aggregate = aggregate
        self.rates = rates
        self.n_st = n_steps
        self.n_e = n_excitations
        self.n_current = self.n_e
        self.rng = np.random.default_rng(seed=seed)
        self.currently_occupied = np.zeros(self.n_e, dtype=np.int)
        # + 2 for the pre-quencher and quencher
        self.n_sites = len(self.aggregate.trimers) + 2
        self.n_i = np.zeros(self.n_sites, dtype=np.uint8)
        self.previous_pool = np.zeros(self.n_e, dtype=np.int)
        self.transitions = np.array(self.transition_calc())
        self.t = 0. # probably need to write a proper init function
        self.ti = [0.]
        # four ways to lose population: annihilation, decay from a
        # chl pool (trimer), decay from pre-quencher, decay from quencher
        self.loss_times = [[] for _ in range(4)]
        self.kmc_setup()
        for i in range(self.n_st):
            print("Step {}, time = {:8.3e}".format(i, self.t))
            if i % 20 == 0:
                self.draw("frames/{:03d}.jpg".format(i))
            status = self.kmc_step()
            if status < 0:
                print("Nonzero status returned by kmc_step() - breaking")
                break

    def transition_calc(self):
        print(self.n_sites)
        self.transitions = [[] for _ in range(self.n_sites)]
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
            print(i, self.transitions[i], file=self.output)
        return self.transitions
    
    def kmc_setup(self):
        for i in range(self.n_e):
            # only generate excitations on the pools, not quenchers
            choice = self.rng.integers(low=0,
                high=len(self.aggregate.trimers))
            self.n_i[choice] += 1
            self.currently_occupied[i] = choice

    def kmc_step(self):
        '''
        first draft of one kMC step
        extremely inefficient
        '''
        if self.n_current == 0:
            print("no excitations left!")
            return -1
        # currently_occupied = []
        # annihilation, pool decay, pq decay, q decay
        pop_loss = [False] * 4
        # the [0] is because nonzero() returns a tuple
        # and the list of nonzero indices is the first bit
        # occupied_sites = self.n_i.nonzero()[0]
        # for i in range(len(occupied_sites)):
        #     for j in range(self.n_i[occupied_sites[i]]):
        #         currently_occupied.append(occupied_sites[i])
        print("Currently occupied = {}".format(self.currently_occupied), file=self.output)
        i = self.rng.integers(low=0, high=self.n_current)
        rand1 = self.rng.random()
        rand2 = self.rng.random()
        print(self.n_current, i)
        ind = self.currently_occupied[np.where(self.currently_occupied >= 0)][i]
        print(ind)
        i = np.nonzero(self.currently_occupied == ind)[0][0]
        print(i)
        if ind < self.n_sites - 2:
            # pool
            n = self.n_i[ind]
            if n >= 2:
                (q, k_tot) = select_process(self.transitions[ind], rand1)
            else:
                (q, k_tot) = select_process(self.transitions[ind][:-1], rand1)
            print("q = {}, kp = {}".format(q, self.transitions[ind]), file=self.output)
            if (q == len(self.transitions[ind]) - 1):
                # annihilation
                print("po ann from trimer {}".format(ind), file=self.output)
                self.n_i[ind] -= 1
                self.n_current -= 1
                self.currently_occupied[i] = -1
                pop_loss[0] = True
            elif (q == len(self.transitions[ind]) - 2):
                # decay
                print("po decay from trimer {}".format(ind), file=self.output)
                self.n_i[ind] -= 1
                self.n_current -= 1
                self.currently_occupied[i] = -1
                pop_loss[1] = True
            elif (q == len(self.transitions[ind]) - 3):
                # hop to pre-quencher
                print("po->pq from trimer {}".format(ind), file=self.output)
                self.n_i[ind] -= 1
                self.n_i[-2]  += 1
                self.currently_occupied[i] = self.n_sites - 2
                self.previous_pool[i] = ind
                print("previous pool = {}".format(self.previous_pool), file=self.output)
            else:
                # hop to neighbour
                nn = self.aggregate.trimers[ind].get_neighbours()[q].index
                print(q, ind, i, nn,
                        [self.aggregate.trimers[ind].get_neighbours()[j].index 
                        for j in range(len(
                        self.aggregate.trimers[ind].get_neighbours()))], 
                        file=self.output)
                print("neighbour: {} to {}".format(ind, nn), file=self.output)
                self.n_i[ind] -= 1
                self.n_i[nn] += 1
                self.currently_occupied[i] = nn
        elif ind == self.n_sites - 2:
            '''
            NB: no annihilation here - in principle every trimer is
            connected to a pre-quencher and through that to a quencher;
            this is just a convenient way of bookkeeping
            '''
            # pre-quencher
            (q, k_tot) = select_process(self.transitions[ind], rand1)
            if (q == 0):
                # hop back to pool
                # excitations on the pre-quencher are indistinguishable:
                # pick one at random from previous_pool and put it there
                # with previous_pool.pop() it'd be first in first out
                pp = self.previous_pool[np.where(self.previous_pool >= 0)]
                choice = self.rng.integers(low=0, high=len(pp))
                # choice = self.rng.integers(low=0, high=len(self.previous_pool))
                self.n_i[ind] -= 1
                self.n_i[pp[choice]] += 1
                self.currently_occupied[i] = pp[choice]
                self.previous_pool[i] = -1
                print("pq->po: ind {}, choice {}, previous pool = {}".format(
                    ind, choice, self.previous_pool), file=self.output)
                print("pq->po after delete: previous pool = {}".format(
                    self.previous_pool), file=self.output)
            elif (q == 1):
                # hop to quencher
                print("pq->q", file=self.output)
                self.n_i[ind] -= 1
                self.n_i[-1] = 1
                self.currently_occupied[i] = self.n_sites - 1
            elif (q == 2):
                # decay
                print("pq decay", file=self.output)
                self.n_i[ind] -= 1
                self.currently_occupied[i] = -1
                self.n_current -= 1
                # choice = self.rng.integers(low=0, high=self.n_i[-2])
                # pp = self.previous_pool[np.where(self.previous_pool >= 0)]
                self.previous_pool[i] = -1
                print("previous pool = {}".format(self.previous_pool)
                        , file=self.output)
                pop_loss[2] = True
        elif ind == self.n_sites - 1:
            # quencher
            (q, k_tot) = select_process(self.transitions[ind], rand1)
            if (q == 0):
                # hop back to pre-quencher
                print("q->pq", file=self.output)
                self.n_i[ind] -= 1
                self.n_i[-2] += 1
            elif (q == 1):
                # decay
                print("q decay", file=self.output)
                self.n_i[ind] -= 1
                self.n_current -= 1
                self.previous_pool[i] = -1
                self.currently_occupied[i] = -1
                # pp = self.previous_pool[np.where(self.previous_pool >= 0)]
                # choice = self.rng.integers(low=0, high=len(pp))
                # self.previous_pool[self.previous_pool == pp[choice]] = -1
                print("previous pool = {}".format(self.previous_pool),
                        file=self.output)
                pop_loss[3] = True
        # i think this is correct???? need to figure out
        self.t -= np.log(rand2 / (k_tot * self.rates.tau_hop))
        self.ti.append(self.t)
        if any(pop_loss):
            # add this time to the relevant stat
            # we only do one move at a time, so only one of pop_loss
            # can be true at any one time; hence it's safe to do [0][0]
            self.loss_times[np.nonzero(pop_loss)[0][0]].append(self.t)
        return 0

    def draw(self, filename):
        '''
        draw the current state of the system.
        pretty ugly to do this manually in opencv
        imo. but it's quick and it works
        '''
        font = cv2.FONT_HERSHEY_DUPLEX
        xmax = np.max([np.abs(t.x) for t in self.aggregate.trimers])
        scale = 4
        pic_side = (2 * scale * int(xmax + 4 * r))
        # the + 200 is to add space to put pre-quencher,
        # quencher, time and population details in
        img = np.zeros((pic_side + 200,
            pic_side + 200, 3), np.uint8)
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
                        (scale * int(t.y + xmax + 1.75 * r),
                    scale * int(t.x + xmax + 2.25 * r)),
                        font, 0.75, (255, 255, 255), 3)
        # pre-quencher
        pq_colour = (0, 94, 20)
        cv2.rectangle(img, pt1=(pic_side + 5, 100), 
            pt2=(pic_side + 55, 150),
            color=pq_colour, thickness=-1)
        cv2.putText(img, "{:2d}".format(self.n_i[-2]),
                (pic_side + 5, 135),
                font, 1, (255, 255, 255), 2)
        cv2.putText(img, "PQ",
                (pic_side + 15, 90),
                font, 0.75, pq_colour, 3)
        # quencher
        q_colour = (60, 211, 242)
        cv2.rectangle(img, pt1=(pic_side + 5, 200), 
            pt2=(pic_side + 55, 250),
            color=q_colour, thickness=-1)
        cv2.putText(img, "{:2d}".format(self.n_i[-1]),
                (pic_side + 5, 235),
                font, 1, (255, 255, 255), 2)
        cv2.putText(img, "Q",
                (pic_side + 20, 190),
                font, 0.75, q_colour, 3)
        cv2.putText(img, "t = {:6.2f}ps".format(self.t),
                (pic_side + 5, 300),
                font, 0.75, (255, 255, 255), 2)
        cv2.putText(img, "n = {:02d}".format(np.sum(self.n_i)),
                (pic_side + 5, 350),
                font, 0.75, (255, 255, 255), 2)
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
    lattice_type = "honeycomb"
    n_iter = 5
    n_iterations = 2

    rates = Rates(5., 4000., 4000., 14., 7., 1., 20., np.inf, 0.5)
    agg = theoretical_aggregate(r, 2.5*r, lattice_type, n_iter)
    loss_times = []
    for i in range(n_iterations):
        it = Iteration(agg, rates, i, 1000, 20, True)
        print(it.ti)
        print(it.loss_times)
        loss_times.append(it.loss_times)
    loss_times = np.array(loss_times, dtype=object)
    print("Annihilations", np.transpose(loss_times)[0])
    print("Pool decays", np.transpose(loss_times)[1])
    print("Pre-quencher decays", np.transpose(loss_times)[2])
    print("Quencher decays", np.transpose(loss_times)[3])
