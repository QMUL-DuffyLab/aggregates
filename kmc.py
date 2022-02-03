#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from trimer import Aggregate, theoretical_aggregate

class Pulse():
    def __init__(self, fwhm, mu):
        self.fwhm = fwhm # ps
        self.mu = mu
        self.sigma = fwhm / 2. * np.sqrt(2. * np.log(2.))
        self.t = np.arange(0., 2. * mu, 1.)
        self.ft = 1. / (self.sigma * np.sqrt(2. * np.pi)) * \
    np.exp(- (self.t - self.mu)**2 / (np.sqrt(2.) * self.sigma)**2)

class Model():
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

class Iteration():
    import numpy as np
    import cv2
    '''
    Take an aggregate - either one constructed from an experimental image
    or a generated one - populate it with a fraction of quenchers and a
    set of excitons (the number of excitons is dependent on fluence), and
    run kinetic Monte Carlo until all the excitons are gone.
    Record the intervals between exciton losses and whether these were
    emissive or not.
    '''
    def __init__(self, aggregate, model, pulse, seed, rho_quenchers,
            n_steps, fluence, verbose=False, draw_frames=False):
        if verbose:
            self.output = sys.stdout
        else:
            self.output = open(os.devnull, "w")
        self.aggregate = aggregate
        self.model = model
        self.pulse = pulse
        self.fluence = fluence
        self.n_st = n_steps
        self.rng = np.random.default_rng(seed=seed)
        self.rho_quenchers = rho_quenchers
        # pre-quencher and quencher
        self.n_sites = len(self.aggregate.trimers) + 2
        self.n_i = np.zeros(self.n_sites, dtype=np.uint8)
        self.quenchers = np.full(len(self.aggregate.trimers), False, dtype=bool)
        self.quencher_setup(self.rho_quenchers)
        self.transitions = self.transition_calc()
        self.pq = []
        self.q = []
        self.t = 0.
        self.t_tot = 0.
        self.n_current = 0
        self.loss_times = []
        # four ways to lose population: annihilation, decay from a
        # chl pool (trimer), decay from pre-quencher, decay from quencher
        self.decay_type = []
        if seed == 0:
            self.draw("frames/init_{:03d}.jpg".format(seed))
        if self.n_st > 0:
            for i in range(self.n_st):
                print("Step {}, time = {:8.3e}, t_tot = {:8.3e}".format(i, self.t, self.t_tot))
                if draw_frames:
                    if i % 100 == 0:
                        self.draw("frames/{:03d}_{:03d}.jpg".format(i, seed))
                self.kmc_step()
        else:
            i = 0
            res = 0
            while self.t_tot < 2. * self.pulse.mu:
                self.mc_step(1.)
            while res == 0:
                res = self.kmc_step()
                i += 1
                if draw_frames:
                    if i % 100 == 0:
                        self.draw("frames/{:03d}_{:03d}.jpg".format(i, seed))
        self.loss_times = np.array(self.loss_times)
        self.decay_type = np.array(self.decay_type)
            
    def quencher_setup(self, rho_quench):
        '''
        randomly allocate quenchers based on ρ_q
        '''
        self.n_q = int(len(self.aggregate.trimers) * rho_quench)
        for i in range(self.n_q):
            choice = self.rng.integers(low=0,
                high=len(self.aggregate.trimers))
            while self.quenchers[choice]:
                choice = self.rng.integers(low=0,
                    high=len(self.aggregate.trimers))
            self.quenchers[choice] = True

    def transition_calc(self):
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
        self.transitions = np.zeros((self.n_sites, self.max_neighbours + 5),\
                dtype=float)
        for i in range(self.n_sites):
            t = self.transitions[i].copy()
            # first element is null rate, second is generation
            if i < self.n_sites - 2:
                # trimer (pool)
                n_neigh = len(self.aggregate.trimers[i].get_neighbours())
                for j in range(n_neigh):
                    t[self.max_neighbours - (j - 1)] = model.hop
                if self.quenchers[i]:
                    t[self.max_neighbours + 2] = model.k_po_pq
                else:
                    t[self.max_neighbours + 2] = 0.
                t[self.max_neighbours + 3] = model.g_pool
                t[self.max_neighbours + 4] = model.k_ann
            elif i == self.n_sites - 2:
                # pre-quencher
                t[self.max_neighbours + 1] = model.k_pq_po
                t[self.max_neighbours + 2] = model.k_pq_q
                t[self.max_neighbours + 3] = model.g_pq
                t[self.max_neighbours + 4] = model.k_ann
            elif i == self.n_sites - 1:
                # quencher
                t[self.max_neighbours + 2] = model.k_q_pq
                t[self.max_neighbours + 3] = model.g_q
                t[self.max_neighbours + 4] = model.k_ann
            self.transitions[i] = t
            print(i, self.transitions[i], file=self.output)
        return self.transitions

    def update_rates(self, rates, index, n, t):
        '''
        the base set of rates calculated in transition_calc are actually
        population-dependent for the most part; photon absorption is also
        time-dependent. this function takes the time and the population
        and updates the rates for a given trimer accordingly
        '''
        if t < 2 * self.pulse.mu:
            # allow for nothing to happen!
            rates[0] = 0.
            # generation term non-zero
            t_index = int(t)
            ft = self.pulse.ft[t_index]
            # σ @ 480nm \approx 1.1E-14
            # sigma_ratio = σ_{se} / σ
            xsec = 1.1E-14
            sigma_ratio = 3.
            if ((1 + sigma_ratio) * n) <= 24:
                rates[1] = xsec * self.fluence * ft * \
                (24 - (1 + sigma_ratio) * n)
                # xsec * fluence is the average number we expect
                # one trimer to absorb over the whole pulse, so
                # xsec * fluence / 2 * mu should give the rate
                # per photon?
                # rates[1] = (xsec * self.fluence / 2 * self.pulse.mu) \
                #         * ((24 - (1 + sigma_ratio) * n) / 24.)
        for k in range(2, len(rates) - 1):
            rates[k] *= n # account for number of excitations
        # annihilation can happen on the pre-quencher or quencher,
        # in principle. but only if they're on the same trimer!
        # check this here
        if index == self.n_sites - 2:
            (uniques, counts) = np.unique(self.pq, return_counts=True)
            n_max = np.max(counts)
            ann_fac = n_max * (n_max - 1) / 2.
        elif index == self.n_sites - 1:
            (uniques, counts) = np.unique(self.q, return_counts=True)
            n_max = np.max(counts)
            ann_fac = n_max * (n_max - 1) / 2.
        else:
            ann_fac = n * (n - 1) / 2.
        rates[-1] *= ann_fac
        return rates

    def move(self, i, q, rates, pop_loss):
        '''
        this function is the guts of both the Metropolis and kinetic
        Monte Carlo code below - take a site (whichever trimer we picked
        at random), an index into the rate array (the index determines
        which move we're doing, based on the order in transition_calc above),
        and carry out the corresponding process. Note that the possible
        processes are different for the pre-quencher and quencher as above.
        pop_loss is updated to tell me if any loss of population occurred
        as a result, and if so what type.
        '''
        if i < self.n_sites - 2:
            # pool
            # if (q == 0):
                # nothing
                # print("nothing")
            if (q == 1):
                # generation
                self.n_i[i] += 1
                self.n_current += 1
                print("generation on {}".format(i), file=self.output)
            if (q == len(rates) - 1):
                # annihilation
                print("po ann from trimer {}".format(i),
                        file=self.output)
                self.n_i[i] -= 1
                self.n_current -= 1
                pop_loss[0] = True
            if (q == len(rates) - 2):
                # decay
                print("po decay from trimer {}".format(i),
                        file=self.output)
                self.n_i[i] -= 1
                self.n_current -= 1
                pop_loss[1] = True
            if (q == len(rates) - 3):
                # hop to pre-quencher
                print("po->pq from trimer {}".format(i),
                        file=self.output)
                self.n_i[i] -= 1
                self.n_i[-2]  += 1
                self.pq.append(i) # keep track of which trimer it came from
                print("i = {} -> pq".format(i), file=self.output)
            if (q > 1 and q < self.max_neighbours + 1):
                # hop to neighbour
                nn = self.aggregate.trimers[i].get_neighbours()[q - 1].index
                print(q, i, nn,
                        [self.aggregate.trimers[i].get_neighbours()[p].index 
                        for p in range(len(
                        self.aggregate.trimers[i].get_neighbours()))], 
                        file=self.output)
                print("neighbour: {} to {}".format(i, nn), file=self.output)
                self.n_i[i] -= 1
                self.n_i[nn] += 1
        elif i == self.n_sites - 2:
            '''
            pre-quencher
            note that excitations can be created here, too
            need to fix the annihilation behaviour on this and the quencher,
            since atm excitations can annihilate on these even if they didn't
            come from the same trimer!!!
            '''
            # if (q == 0):
                # print("nothing")
            if (q == 1):
                self.n_i[i] += 1
                self.n_current += 1
                print("generation on pq", file=self.output)
                # choose a random trimer for it to hop to
                choice = self.rng.integers(low=0, high=self.n_sites - 2)
                self.pq.append(choice)
            if (q == len(rates) - 4):
                # hop back to pool
                # excitations on the pre-quencher are indistinguishable:
                # pick one at random from pq and put it back
                # with pq.pop() it'd be first in first out
                choice = self.rng.integers(low=0, high=len(self.pq))
                print("pq->po: choice {} of {}, previous = {}".format(
                    choice, len(self.pq), self.pq[choice]),
                    file=self.output)
                self.n_i[i] -= 1
                self.n_i[self.pq[choice]] += 1
                self.pq.remove(choice)
                print("pq->po after delete: pq = {}".format(
                    self.pq), file=self.output)
            elif (q == len(rates) - 3):
                # hop to quencher
                print("pq->q", file=self.output)
                choice = self.rng.integers(low=0, high=len(self.pq))
                self.n_i[i] -= 1
                self.n_i[-1] += 1
                self.q.append(self.pq[choice])
                self.pq.remove(choice)
            elif (q == len(rates) - 2):
                # decay
                print("pq decay", file=self.output)
                choice = self.rng.integers(low=0, high=len(self.pq))
                self.n_i[i] -= 1
                print("previous = {}".format(self.pq[choice]), 
                        file=self.output)
                self.pq.remove(choice)
                pop_loss[2] = True
            elif (q == len(rates) - 1):
                # annihilation
                print("pq ann", file=self.output)
                '''
                annihilation can only occur here if two excitons are on
                the same quenching trimer. because i only consider one
                pre-quencher since they're all identical, this means we
                have to be careful about annihilation! first, find which
                trimer(s) have multiple excitons currently on the pre-quencher.
                then pick one of those multiples and remove the first exciton
                on that trimer.
                '''
                (uniques, counts) = np.unique(self.pq, return_counts=True)
                multiples = np.nonzero(counts > 1)[0]
                choice = self.rng.integers(low=0, high=len(multiples))
                self.n_i[i] -= 1
                # uniques[multiples][choice] gives us which of the trimers
                # with multiple excitons on the pre-quencher is annihilating.
                # np.where()[0][0] gives us the first index of that trimer
                # on self.pq
                index = np.where(self.pq == uniques[multiples][choice])[0][0]
                print("previous = {}".format(self.pq[index]), file=self.output)
                self.pq.remove(index)
                pop_loss[0] = True
        elif i == self.n_sites - 1:
            '''
            quencher
            we can generate excitations here too in principle
            '''
            # if (q == 0):
            #     print("nothing")
            if (q == 1):
                self.n_i[i] += 1
                self.n_current += 1
                print("generation on q", file=self.output)
                choice = self.rng.integers(low=0, high=self.n_sites - 2)
                self.q.append(choice)
            if (q == len(rates) - 3):
                # hop back to pre-quencher
                print("q->pq", file=self.output)
                choice = self.rng.integers(low=0, high=len(self.q))
                self.n_i[i] -= 1
                self.n_i[-2] += 1
                self.pq.append(self.q[choice])
                self.q.remove(choice)
            elif (q == len(rates) - 2):
                # decay
                print("quencher decay, n = {}".format(self.n_i[i]),
                        file=self.output)
                choice = self.rng.integers(low=0, high=len(self.q))
                self.n_i[i] -= 1
                print("previous chl was = {}".format(self.q[choice]),
                        file=self.output)
                self.q.remove(choice)
                pop_loss[3] = True
            elif (q == len(rates) - 1):
                # annihilation
                print("quencher annihilation", file=self.output)
                (uniques, counts) = np.unique(self.q, return_counts=True)
                multiples = np.nonzero(counts > 1)[0]
                choice = self.rng.integers(low=0, high=len(multiples))
                self.n_i[i] -= 1
                index = np.where(self.q == uniques[multiples][choice])[0][0]
                # uniques[multiples][choice] gives us which of the trimers
                # with multiple excitons on the pre-quencher is annihilating.
                # np.where()[0][0] gives us the first index of that trimer
                # on self.pq
                print("previous chl was = {}".format(self.q[index]),
                        file=self.output)
                self.q.remove(index)
                pop_loss[0] = True

    def mc_step(self, dt):
        '''
        for a given time step, we can calculate the probability of a Poisson
        process with the associated rate happening within that time step.
        (this is just an exponential distribution).
        these can be set as the acceptance probabilities for Metropolis,
        since we don't have to worry about detailed balance (this is not an
        equilibrium process!). we loop over trimers, propose a move at random
        (absorption, hop, decay, annihilation if n > 1),
        then run it through Metropolis.
        '''
        if self.rho_quenchers != 0.:
            n_attempts = self.n_sites
        else:
            n_attempts = self.n_sites - 2
        for i in range(n_attempts):
            # annihilation, pool decay, pq decay, q decay
            pop_loss = [False for _ in range(4)]
            trimer = self.rng.integers(low=0, high=n_attempts)
            rates = self.transitions[trimer].copy()
            rates = self.update_rates(rates, trimer,
                    self.n_i[trimer], self.t_tot)
            # print(rates)
            probs = np.fromiter((rate * np.exp(-rate * dt) for rate in rates),
                dtype=float) # acceptance probabilities for Metropolis
            # ignore moves with zero rate
            choice = self.rng.integers(low=0, high=np.count_nonzero(probs))
            proposed_move = np.nonzero(probs)[0][choice]
            rand = self.rng.random()
            if (rand < probs[proposed_move]):
                # carry out the move
                self.move(trimer, proposed_move, rates, pop_loss)
                print('Move accepted. index = {:d}, p = {:f}, '\
                        'rand = {:f}, t_tot = {:6.3f}, '\
                        'n_current = {:d}'.format(proposed_move,
                            probs[proposed_move], rand, self.t_tot,
                            self.n_current), file=self.output)
                # NB: not sure this is correct for Metropolis!
                # need to think about it :|
                if any(pop_loss):
                    # add this time to the relevant stat
                    # we only do one move at a time, so only one of pop_loss
                    # can be true at any one time; hence it's safe to do [0][0]
                    decay_type = np.nonzero(pop_loss)[0][0]
                    print("decay type = {}".format(decay_type),
                            file=self.output)
                    print("loss time = {}".format(self.t), file=self.output)
                    self.loss_times.append(self.t)
                    self.decay_type.append(decay_type)
                    # zero the time to get time between decays!
                    self.t = 0.
        self.t += dt
        self.t_tot += dt
    
    def kmc_step(self):
        '''
        after the pulse we have no exciton generation - switch to
        kinetic Monte Carlo to simulate annihilations and decays
        '''
        if self.n_current == 0 and self.t_tot > 2. * self.pulse.mu:
            return -1
        for j in range(self.n_sites):
            pop_loss = [False for _ in range(4)]
            if (self.rho_quenchers != 0.):
                i = self.rng.integers(low=0, high=self.n_sites)
            else:
                i = self.rng.integers(low=0, high=self.n_sites - 2)
            rand1 = self.rng.random()
            rand2 = self.rng.random()
            print(self.n_current, i, file=self.output)
            n = self.n_i[i]
            rates = self.transitions[i].copy()
            print("before rates, n, t_tot: ", rates, self.n_i[i], self.t_tot,
                    file=self.output)
            rates = self.update_rates(rates, i, self.n_i[i], self.t_tot)
            print("after rates: ", rates, file=self.output)
            if np.any(rates):
                (q, k_tot) = self.bkl(rates, rand1)
                self.move(i, q, rates, pop_loss)
                self.t -= 1./ (k_tot) * np.log(rand2)
                self.t_tot += self.t
                if any(pop_loss):
                    # add this time to the relevant stat
                    # we only do one move at a time, so only one of pop_loss
                    # can be true at any one time; hence it's safe to do [0][0]
                    decay_type = np.nonzero(pop_loss)[0][0]
                    print("decay type = {}".format(decay_type),
                            file=self.output)
                    print("loss time = {}".format(self.t), file=self.output)
                    self.loss_times.append(self.t)
                    self.decay_type.append(decay_type)
                    # zero the time to get time between decays!
                    self.t = 0.
            else:
                print("all rates zero!", file=self.output)
                continue
        return 0

    def draw(self, filename):
        import cv2
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
            if self.quenchers[i]:
                colour = (232, 139, 39)
            else:
                colour = (255, 255, 255)
            cv2.circle(img, (int(scale * (t.y + xmax + 2. * r)),
                int(scale * (t.x + xmax + 2. * r))), 
                int(scale * t.r), colour, -1)
            if (self.n_i[i] != 0):
                # coloured circle to indicate exciton
                cv2.circle(img, (int(scale * (t.y + xmax + 2. * r)),
                    int(scale * (t.x + xmax + 2. * r))), 
                    int(scale * 0.75 * t.r), (26, 0, 153), -1)
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

    def bkl(self, k_p, rand):
        '''
        BKL algorithm for KMC.
        choose which configuration to jump to given a set of transition rates
        to those configurations from the current one, and a random number in [0,1].
        '''
        k_p_s = np.cumsum(k_p)
        k_tot = k_p_s[-1]
        i = 0
        process = 0
        # the check here that k_p_s[i] > k_p_s[i - 1] allows for
        # zeroes in the rates without picking spurious processes
        while rand * k_tot > k_p_s[i]:
            i += 1
            if k_p_s[i] > k_p_s[i - 1]:
                process += 1
        return (i, k_tot)

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
    # lazy - make a different main file to loop over models and fluences
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
        sj = pulse.ft * 1.1E-14 * fluence
        print("Integral over pulse = {}".format(np.sum(sj)))
        plt.plot(sj, label='{:8.3e}'.format(fluence))

    plt.legend()
    plt.savefig("out/fluences.pdf") 

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
